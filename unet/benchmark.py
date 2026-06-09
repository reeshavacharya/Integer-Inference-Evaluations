"""Unified benchmark runner for UNet float, INT8, INT32, FixedPoint32, and FixedPoint64 inference.

This script benchmarks trained UNet models over the deterministic 10% test splits.
It relies exclusively on offline compiled states (_int8.pth, _int32.pth) and
calibration dictionaries for integer evaluation.

Supported flags:
- --bench: benchmark one dataset (Skin-Lesion, Flood, Brain-MRI-Seg, BUSI)
- --num_data: number of test images to benchmark (defaults to full 10% split)
- --mode {fp32, int8, int32, fxp32, fxp64}: benchmark one inference mode
"""

import argparse
import importlib.util
import json
import os
import sys
from typing import Optional

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INT8_DIR = os.path.join(THIS_DIR, "INT8")
INT32_DIR = os.path.join(THIS_DIR, "INT32")
FP32_DIR = os.path.join(THIS_DIR, "FixedPoint32")
FP64_DIR = os.path.join(THIS_DIR, "FixedPoint64")
for p in (THIS_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)

def _load_module(module_name: str, file_path: str, prepend_dir: str):
    if not os.path.exists(file_path):
        return None
    saved_path = list(sys.path)
    previous_utils = sys.modules.get("utils")
    try:
        sys.path.insert(0, prepend_dir)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = saved_path
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils

import u_net as train_mod

# Dynamically load modules (Fail gracefully if a pipeline isn't built yet)
int8_utils = _load_module("unet_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR)
int8_inference = _load_module("unet_int8_inference", os.path.join(INT8_DIR, "inference.py"), INT8_DIR)

int32_utils = _load_module("unet_int32_utils", os.path.join(INT32_DIR, "utils.py"), INT32_DIR)
int32_inference = _load_module("unet_int32_inference", os.path.join(INT32_DIR, "inference.py"), INT32_DIR)

fp64_utils = _load_module("unet_fp64_utils", os.path.join(FP64_DIR, "utils.py"), FP64_DIR)
fp64_inference = _load_module("unet_fp64_inference", os.path.join(FP64_DIR, "inference.py"), FP64_DIR)

fp32_utils = _load_module("unet_fp32_utils", os.path.join(FP32_DIR, "utils.py"), FP32_DIR)
fp32_inference = _load_module("unet_fp32_inference", os.path.join(FP32_DIR, "inference.py"), FP32_DIR)

BENCHMARK_DATASETS = [
    "Skin-Lesion",
    "Flood",
    "Brain-MRI-Seg",
    "BUSI",
]

def _normalize_bench_name(name: str) -> str:
    key = name.strip().upper().replace("_", "-")
    if key == "SKIN-LESION": return "Skin-Lesion"
    if key == "FLOOD": return "Flood"
    if key == "BRAIN-MRI-SEG": return "Brain-MRI-Seg"
    if key == "BUSI": return "BUSI"
    raise ValueError(f"Unknown benchmark dataset: {name}")

def _model_path_for_dataset(dataset_name: str, activation: str = "relu") -> str:
    if dataset_name == "Skin-Lesion": return f"best_unet5_{activation}_skin_lesion.pth"
    if dataset_name == "Flood": return f"best_unet5_{activation}_flood.pth"
    if dataset_name == "Brain-MRI-Seg": return f"best_unet5_{activation}_brain_mri_seg.pth"
    if dataset_name == "BUSI": return f"best_unet5_{activation}_busi.pth"
    raise ValueError(f"Unknown dataset: {dataset_name}")

def _ensure_checkpoint(dataset_name: str, activation: str = "relu"):
    model_path = _model_path_for_dataset(dataset_name, activation)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Checkpoint missing for {dataset_name} ({activation}): {model_path}")

def _get_test_loader(dataset_name: str, batch_size: int = 64):
    _, _, test_loader = train_mod.setup_data(
        train_data=dataset_name, batch_size=batch_size, image_size=256, num_workers=0
    )
    return test_loader

def _build_model(dataset_name: str, activation: str = "relu") -> torch.nn.Module:
    model = train_mod.UNet(n_class=1, activation=activation) if hasattr(train_mod, 'UNet') and 'activation' in train_mod.UNet.__init__.__code__.co_varnames else train_mod.UNet(n_class=1)
    model_path = _model_path_for_dataset(dataset_name, activation)
    state = torch.load(model_path, map_location="cpu")
    if len(state) > 0 and list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

def _accumulate_confusion(logits: torch.Tensor, masks: torch.Tensor):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    tp = (preds_flat * masks_flat).sum().item()
    tn = ((1 - preds_flat) * (1 - masks_flat)).sum().item()
    fp = (preds_flat * (1 - masks_flat)).sum().item()
    fn = ((1 - preds_flat) * masks_flat).sum().item()
    return tp, tn, fp, fn

def _confusion_to_metrics(tp: float, tn: float, fp: float, fn: float):
    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    return {"dice": dice, "iou": iou, "acc": acc, "f1": f1}

def _float_metrics(model: torch.nn.Module, loader, dataset_name: str, num_data: Optional[int]):
    total_samples = len(loader.dataset)
    target_samples = total_samples if num_data is None else min(num_data, total_samples)
    print(f"[bench][{dataset_name}][fp32] Starting benchmark over {target_samples}/{total_samples} samples.")

    tp = tn = fp = fn = 0.0
    seen = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader, 1):
            if seen >= target_samples: break
            remaining = target_samples - seen
            if images.size(0) > remaining:
                images, masks = images[:remaining], masks[:remaining]

            logits = model(images)
            b_tp, b_tn, b_fp, b_fn = _accumulate_confusion(logits, masks)
            tp += b_tp; tn += b_tn; fp += b_fp; fn += b_fn
            seen += images.size(0)

            print(f"[bench][{dataset_name}][fp32] Batch {batch_idx}: processed {seen}/{target_samples} samples.")

    return _confusion_to_metrics(tp, tn, fp, fn)

def _run_offline_int_metrics(loader, dataset_name: str, num_data: Optional[int], mode: str, activation: str = "relu"):
    """Unified offline runner for both int8 and int32 precision models."""
    total_samples = len(loader.dataset)
    target_samples = total_samples if num_data is None else min(num_data, total_samples)
    print(f"[bench][{dataset_name}][{mode}] Starting benchmark over {target_samples}/{total_samples} samples.")

    if mode == "int8":
        mod_utils, mod_inf, state_suffix, dtype, base_dir = int8_utils, int8_inference, "_int8.pth", torch.uint8, THIS_DIR
    elif mode == "int32":
        mod_utils, mod_inf, state_suffix, dtype, base_dir = int32_utils, int32_inference, "_int32.pth", torch.uint32, THIS_DIR
    else:
        raise ValueError(f"Unsupported integer mode: {mode}")

    if mod_utils is None or mod_inf is None:
        print(f"[-] Code for {mode} not found. Skipping.")
        return None

    # Load State
    model_name = _model_path_for_dataset(dataset_name, activation)
    state_path = os.path.join(base_dir, model_name.replace(".pth", state_suffix))
    if not os.path.exists(state_path):
        print(f"[-] Missing offline checkpoint: {state_path}. Please run export_{mode}_model.py first.")
        return None
    int_state = torch.load(state_path, map_location="cpu")

    # Load Calibration
    calib_filename = f"{dataset_name.lower().replace(' ', '_').replace('-', '_')}_{activation}_calibration.json"
    calib_path = os.path.join(THIS_DIR, "calibration", calib_filename)
    if not os.path.exists(calib_path):
        print(f"[-] Missing calibration: {calib_path}")
        return None
    with open(calib_path, "r") as f:
        calib_data = json.load(f)
    activation_ranges = calib_data.get("layers", calib_data)

    tp = tn = fp = fn = 0.0
    seen = 0

    # Inline accurate requantization (prevents dependency on missing utils.py math)
    def _requantize(q_tensor, zp_old, zp_new, s_old, s_new):
        M = s_old / s_new
        q_f = q_tensor.to(torch.float32) - zp_old
        max_val = 255 if dtype == torch.uint8 else 4294967295
        return torch.clamp(torch.round(q_f * M) + zp_new, 0, max_val).to(dtype)

    for batch_idx, (images, masks) in enumerate(loader, 1):
        if seen >= target_samples: break
        remaining = target_samples - seen
        if images.size(0) > remaining:
            images, masks = images[:remaining], masks[:remaining]

        scale_in = int_state["meta"]["in_scale"]
        zp_in = int_state["meta"]["in_zp"]
        
        # Use headroom constraints for int32 if defined
        try:
            q_x = mod_utils.quantize_tensor(images, scale_in, zp_in, dtype=dtype, num_bits=8 if mode=="int8" else 16)
        except TypeError:
            q_x = mod_utils.quantize_tensor(images, scale_in, zp_in, dtype=dtype)

        with torch.no_grad():
            act_kwargs = {"apply_activation": True, "activation": activation} if mode == "int32" else {"apply_relu": True}
            no_act_kwargs = {"apply_activation": False, "activation": "none"} if mode == "int32" else {"apply_relu": False}

            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e11"], zp_in, **act_kwargs)
            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e12"], z, **act_kwargs)
            q_e12, s_e12, z_e12 = q_x, s, z
            B, C, H, W = q_x.shape
            if mode == "int32":
                q_x = q_x.to(torch.int32).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5))
            else:
                q_x = q_x.to(torch.int64).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5)).to(dtype)

            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e21"], z, **act_kwargs)
            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e22"], z, **act_kwargs)
            q_e22, s_e22, z_e22 = q_x, s, z
            B, C, H, W = q_x.shape
            if mode == "int32":
                q_x = q_x.to(torch.int32).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5))
            else:
                q_x = q_x.to(torch.int64).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5)).to(dtype)

            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e31"], z, **act_kwargs)
            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e32"], z, **act_kwargs)
            q_e32, s_e32, z_e32 = q_x, s, z
            B, C, H, W = q_x.shape
            if mode == "int32":
                q_x = q_x.to(torch.int32).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5))
            else:
                q_x = q_x.to(torch.int64).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5)).to(dtype)

            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e41"], z, **act_kwargs)
            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e42"], z, **act_kwargs)
            q_e42, s_e42, z_e42 = q_x, s, z
            B, C, H, W = q_x.shape
            if mode == "int32":
                q_x = q_x.to(torch.int32).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5))
            else:
                q_x = q_x.to(torch.int64).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5)).to(dtype)

            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e51"], z, **act_kwargs)
            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["e52"], z, **act_kwargs)

            q_x, s_up1, z_up1 = mod_inf.run_integer_layer(q_x, int_state["upconv1"], z, **no_act_kwargs)
            s_cat1 = activation_ranges["d11"]["in_scale"]; z_cat1 = activation_ranges["d11"]["in_zero_point"]
            q_x_aligned = _requantize(q_x, z_up1, z_cat1, s_up1, s_cat1)
            q_skip_aligned = _requantize(q_e42, z_e42, z_cat1, s_e42, s_cat1)
            q_x, s, z = mod_inf.run_integer_layer(torch.cat([q_x_aligned, q_skip_aligned], dim=1), int_state["d11"], z_cat1, **act_kwargs)
            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["d12"], z, **act_kwargs)

            q_x, s_up2, z_up2 = mod_inf.run_integer_layer(q_x, int_state["upconv2"], z, **no_act_kwargs)
            s_cat2 = activation_ranges["d21"]["in_scale"]; z_cat2 = activation_ranges["d21"]["in_zero_point"]
            q_x_aligned = _requantize(q_x, z_up2, z_cat2, s_up2, s_cat2)
            q_skip_aligned = _requantize(q_e32, z_e32, z_cat2, s_e32, s_cat2)
            q_x, s, z = mod_inf.run_integer_layer(torch.cat([q_x_aligned, q_skip_aligned], dim=1), int_state["d21"], z_cat2, **act_kwargs)
            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["d22"], z, **act_kwargs)

            q_x, s_up3, z_up3 = mod_inf.run_integer_layer(q_x, int_state["upconv3"], z, **no_act_kwargs)
            s_cat3 = activation_ranges["d31"]["in_scale"]; z_cat3 = activation_ranges["d31"]["in_zero_point"]
            q_x_aligned = _requantize(q_x, z_up3, z_cat3, s_up3, s_cat3)
            q_skip_aligned = _requantize(q_e22, z_e22, z_cat3, s_e22, s_cat3)
            q_x, s, z = mod_inf.run_integer_layer(torch.cat([q_x_aligned, q_skip_aligned], dim=1), int_state["d31"], z_cat3, **act_kwargs)
            q_x, s, z = mod_inf.run_integer_layer(q_x, int_state["d32"], z, **act_kwargs)

            q_x, s_up4, z_up4 = mod_inf.run_integer_layer(q_x, int_state["upconv4"], z, **no_act_kwargs)
            s_cat4 = activation_ranges["d41"]["in_scale"]; z_cat4 = activation_ranges["d41"]["in_zero_point"]
            q_x_aligned = _requantize(q_x, z_up4, z_cat4, s_up4, s_cat4)
            q_skip_aligned = _requantize(q_e12, z_e12, z_cat4, s_e12, s_cat4)
            q_x, s, z = mod_inf.run_integer_layer(torch.cat([q_x_aligned, q_skip_aligned], dim=1), int_state["d41"], z_cat4, **act_kwargs)
        if mode == "fp32":
            logits = model(images)
            print(f"DEBUG FP32: logits min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}")
            b_tp, b_tn, b_fp, b_fn = _accumulate_confusion(logits, masks)
        else:
            q_out, final_s, final_z = mod_inf.run_integer_layer(q_x, int_state["outconv"], z, **no_act_kwargs)

            q_out_float = q_out.to(torch.float32)
            logits = final_s * (q_out_float - final_z)

            print(f"DEBUG INT32: logits min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}, z={final_z}")
            b_tp, b_tn, b_fp, b_fn = _accumulate_confusion(logits, masks)
        tp += b_tp; tn += b_tn; fp += b_fp; fn += b_fn
        seen += images.size(0)

        print(f"[bench][{dataset_name}][{mode}] Batch {batch_idx}: processed {seen}/{target_samples} samples.")

    return _confusion_to_metrics(tp, tn, fp, fn)

def _mode_suffix(mode: Optional[str]) -> str:
    return "all_modes" if mode is None else mode.replace("-", "_")

def benchmark_single(dataset_name: str, num_data: Optional[int], mode: Optional[str], activation: str, batch_size: int = 64):
    print(f"\n[bench] Dataset: {dataset_name} | Activation: {activation}")
    try:
        _ensure_checkpoint(dataset_name, activation)
    except RuntimeError as e:
        print(f"[-] {e}")
        return None

    loader = _get_test_loader(dataset_name, batch_size)
    model = _build_model(dataset_name, activation)
    selected_modes = [mode] if mode is not None else ["fp32", "int8", "int32", "fxp32", "fxp64"]
    
    per_dataset = {}
    if "fp32" in selected_modes:
        m = _float_metrics(model, loader, dataset_name, num_data)
        if m: per_dataset["fp32"] = m
    if "int8" in selected_modes:
        m = _run_offline_int_metrics(loader, dataset_name, num_data, "int8", activation)
        if m: per_dataset["int8"] = m
    if "int32" in selected_modes:
        m = _run_offline_int_metrics(loader, dataset_name, num_data, "int32", activation)
        if m: per_dataset["int32"] = m

    summary = []
    for m, vals in per_dataset.items():
        summary.append(f"{m}(dice={vals['dice']:.4f}, iou={vals['iou']:.4f}, acc={vals['acc']:.4f})")
    print(f"[bench] {dataset_name} ({activation}): " + " | ".join(summary))

    return per_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default=None, help="Benchmark a single dataset")
    parser.add_argument("--num_data", type=int, default=None, help="Number of test samples")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for benchmark test loader")
    parser.add_argument("--mode", type=str, choices=["fp32", "int8", "int32", "fxp32", "fxp64"], default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu", "leaky_relu"])
    args = parser.parse_args()

    single_name = _normalize_bench_name(args.bench) if args.bench else None
    targets = [single_name] if single_name else BENCHMARK_DATASETS
    target_activations = [args.activation] if args.activation else ["relu", "gelu", "leaky_relu"]

    results_base_dir = os.path.join(THIS_DIR, "benchmark-results")
    os.makedirs(results_base_dir, exist_ok=True)

    mode_part = _mode_suffix(args.mode)

    for dataset_name in targets:
        ds_part = dataset_name.lower().replace(" ", "-")
        ds_dir = os.path.join(results_base_dir, ds_part)
        os.makedirs(ds_dir, exist_ok=True)

        for activation in target_activations:
            metrics = benchmark_single(dataset_name, args.num_data, args.mode, activation, args.batch_size)
            
            if metrics:
                results_file = os.path.join(ds_dir, f"benchmark_results_{activation}_{mode_part}.json")
                with open(results_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"\nSaved {results_file} with:")
                summary = []
                for m, m_dict in metrics.items():
                    summary.append(f"{m}(dice={m_dict['dice']:.4f}, iou={m_dict['iou']:.4f}, acc={m_dict['acc']:.4f})")
                print(f"  {dataset_name} ({activation}): " + " | ".join(summary))