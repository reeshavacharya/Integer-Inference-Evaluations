"""Unified benchmark runner for UNet float, INT8, and FixedPoint64 inference.

This script benchmarks trained UNet models over the deterministic 10% test
splits created by u_net.py.

Supported flags:
- --bench: benchmark one dataset (defaults to all datasets)
- --num_data: number of test images to benchmark (defaults to full 10% split)
- --mode {int,fixed-point,floating-point}: benchmark one inference mode
  (defaults to all 3 modes)
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
FP64_DIR = os.path.join(THIS_DIR, "FixedPoint64")
for p in (THIS_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module(module_name: str, file_path: str, prepend_dir: str):
    """Load a module from an explicit file path with controlled import precedence."""
    saved_path = list(sys.path)
    previous_utils = sys.modules.get("utils")
    try:
        sys.path.insert(0, prepend_dir)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create module spec for {file_path}")
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
int8_utils = _load_module("unet_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR)
int8_inference = _load_module(
    "unet_int8_inference", os.path.join(INT8_DIR, "inference.py"), INT8_DIR
)
fp64_utils = _load_module(
    "unet_fp64_utils", os.path.join(FP64_DIR, "utils.py"), FP64_DIR
)
fp64_inference = _load_module(
    "unet_fp64_inference", os.path.join(FP64_DIR, "inference.py"), FP64_DIR
)


BENCHMARK_DATASETS = [
    "Skin-Lesion",
    "Flood",
    "Brain-MRI-Seg",
    "BUSI",
]


def _disable_heavy_debug_logs():
    class _NoOpList(list):
        def append(self, item):  # type: ignore[override]
            return None

    int8_inference.debug_trace = {
        "input": None,
        "layers": _NoOpList(),
        "pooling": _NoOpList(),
    }
    fp64_inference.debug_trace = {
        "input": None,
        "layers": _NoOpList(),
        "pooling": _NoOpList(),
    }


def _normalize_bench_name(name: str) -> str:
    key = name.strip().upper().replace("_", "-")
    if key == "SKIN-LESION":
        return "Skin-Lesion"
    if key == "FLOOD":
        return "Flood"
    if key == "BRAIN-MRI-SEG":
        return "Brain-MRI-Seg"
    if key == "BUSI":
        return "BUSI"
    raise ValueError(f"Unknown benchmark dataset: {name}")


def _model_path_for_dataset(dataset_name: str) -> str:
    if dataset_name == "Skin-Lesion":
        return "best_unet5_skin_lesion.pth"
    if dataset_name == "Flood":
        return "best_unet5_flood.pth"
    if dataset_name == "Brain-MRI-Seg":
        return "best_unet5_brain_mri_seg.pth"
    if dataset_name == "BUSI":
        return "best_unet5_busi.pth"
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _ensure_checkpoint(dataset_name: str):
    model_path = _model_path_for_dataset(dataset_name)
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Checkpoint missing for {dataset_name}: {model_path}. Train this dataset first."
        )


def _get_test_loader(dataset_name: str, batch_size: int = 4):
    _, _, test_loader = train_mod.setup_data(
        train_data=dataset_name,
        batch_size=batch_size,
        image_size=256,
        num_workers=0,
    )
    return test_loader


def _build_model(dataset_name: str) -> torch.nn.Module:
    model = int8_inference.UNet(num_classes=1)
    model_path = _model_path_for_dataset(dataset_name)
    state = torch.load(model_path, map_location="cpu")
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
    print(
        f"[bench][{dataset_name}][floating-point] Starting benchmark over {target_samples}/{total_samples} samples."
    )

    tp = tn = fp = fn = 0.0
    seen = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader, 1):
            if seen >= target_samples:
                break

            remaining = target_samples - seen
            if images.size(0) > remaining:
                images = images[:remaining]
                masks = masks[:remaining]

            logits = model(images)
            b_tp, b_tn, b_fp, b_fn = _accumulate_confusion(logits, masks)
            tp += b_tp
            tn += b_tn
            fp += b_fp
            fn += b_fn
            seen += images.size(0)

            left = max(target_samples - seen, 0)
            print(
                f"[bench][{dataset_name}][floating-point] Batch {batch_idx}: processed {seen}/{target_samples} samples, remaining {left}."
            )

    return _confusion_to_metrics(tp, tn, fp, fn)


def _int_metrics(model: torch.nn.Module, loader, dataset_name: str, num_data: Optional[int]):
    total_samples = len(loader.dataset)
    target_samples = total_samples if num_data is None else min(num_data, total_samples)
    print(
        f"[bench][{dataset_name}][int] Starting benchmark over {target_samples}/{total_samples} samples."
    )

    tp = tn = fp = fn = 0.0
    seen = 0

    for batch_idx, (images, masks) in enumerate(loader, 1):
        if seen >= target_samples:
            break

        remaining = target_samples - seen
        if images.size(0) > remaining:
            images = images[:remaining]
            masks = masks[:remaining]

        int8_inference.activation_ranges.clear()
        handles = int8_inference.register_hooks(model)
        with torch.no_grad():
            _ = model(images)
        for h in handles:
            h.remove()

        in_range = int8_inference.activation_ranges["conv1"]
        pseudo_in_tensor = torch.tensor([in_range["in_min"], in_range["in_max"]])
        scale_in, zp_in = int8_utils.get_quantization_params(pseudo_in_tensor, num_bits=8)
        q_x = int8_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)

        cfg = int8_inference._get_layer_config(model)

        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv1"], "conv1", scale_in, zp_in, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv2"], "conv2", s, z, apply_relu=True
        )
        q_e12, s_e12, z_e12 = q_x, s, z
        q_x = int8_inference.pool_uint8(q_x, name="pool1")

        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv3"], "conv3", s, z, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv4"], "conv4", s, z, apply_relu=True
        )
        q_e22, s_e22, z_e22 = q_x, s, z
        q_x = int8_inference.pool_uint8(q_x, name="pool2")

        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv5"], "conv5", s, z, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv6"], "conv6", s, z, apply_relu=True
        )
        q_e32, s_e32, z_e32 = q_x, s, z
        q_x = int8_inference.pool_uint8(q_x, name="pool3")

        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv7"], "conv7", s, z, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv8"], "conv8", s, z, apply_relu=True
        )
        q_e42, s_e42, z_e42 = q_x, s, z
        q_x = int8_inference.pool_uint8(q_x, name="pool4")

        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv9"], "conv9", s, z, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv10"], "conv10", s, z, apply_relu=True
        )

        q_x, s_up1, z_up1, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["upconv1"], "upconv1", s, z, apply_relu=False
        )
        s_cat1, z_cat1 = int8_inference.get_concat_quantization_params(
            int8_inference.activation_ranges, "upconv1", "conv8"
        )
        M0_up1, shift_up1 = int8_utils.compute_requantize_multiplier(s_up1, s_cat1)
        M0_e42, shift_e42 = int8_utils.compute_requantize_multiplier(s_e42, s_cat1)
        q_x_aligned = int8_utils.requantize_tensor(q_x, z_up1, z_cat1, M0_up1, shift_up1)
        q_skip_aligned = int8_utils.requantize_tensor(q_e42, z_e42, z_cat1, M0_e42, shift_e42)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_cat, cfg["conv11"], "conv11", s_cat1, z_cat1, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv12"], "conv12", s, z, apply_relu=True
        )

        q_x, s_up2, z_up2, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["upconv2"], "upconv2", s, z, apply_relu=False
        )
        s_cat2, z_cat2 = int8_inference.get_concat_quantization_params(
            int8_inference.activation_ranges, "upconv2", "conv6"
        )
        M0_up2, shift_up2 = int8_utils.compute_requantize_multiplier(s_up2, s_cat2)
        M0_e32, shift_e32 = int8_utils.compute_requantize_multiplier(s_e32, s_cat2)
        q_x_aligned = int8_utils.requantize_tensor(q_x, z_up2, z_cat2, M0_up2, shift_up2)
        q_skip_aligned = int8_utils.requantize_tensor(q_e32, z_e32, z_cat2, M0_e32, shift_e32)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_cat, cfg["conv13"], "conv13", s_cat2, z_cat2, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv14"], "conv14", s, z, apply_relu=True
        )

        q_x, s_up3, z_up3, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["upconv3"], "upconv3", s, z, apply_relu=False
        )
        s_cat3, z_cat3 = int8_inference.get_concat_quantization_params(
            int8_inference.activation_ranges, "upconv3", "conv4"
        )
        M0_up3, shift_up3 = int8_utils.compute_requantize_multiplier(s_up3, s_cat3)
        M0_e22, shift_e22 = int8_utils.compute_requantize_multiplier(s_e22, s_cat3)
        q_x_aligned = int8_utils.requantize_tensor(q_x, z_up3, z_cat3, M0_up3, shift_up3)
        q_skip_aligned = int8_utils.requantize_tensor(q_e22, z_e22, z_cat3, M0_e22, shift_e22)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_cat, cfg["conv15"], "conv15", s_cat3, z_cat3, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv16"], "conv16", s, z, apply_relu=True
        )

        q_x, s_up4, z_up4, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["upconv4"], "upconv4", s, z, apply_relu=False
        )
        s_cat4, z_cat4 = int8_inference.get_concat_quantization_params(
            int8_inference.activation_ranges, "upconv4", "conv2"
        )
        M0_up4, shift_up4 = int8_utils.compute_requantize_multiplier(s_up4, s_cat4)
        M0_e12, shift_e12 = int8_utils.compute_requantize_multiplier(s_e12, s_cat4)
        q_x_aligned = int8_utils.requantize_tensor(q_x, z_up4, z_cat4, M0_up4, shift_up4)
        q_skip_aligned = int8_utils.requantize_tensor(q_e12, z_e12, z_cat4, M0_e12, shift_e12)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_cat, cfg["conv17"], "conv17", s_cat4, z_cat4, apply_relu=True
        )
        q_x, s, z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["conv18"], "conv18", s, z, apply_relu=True
        )

        q_out, final_s, final_z, _, _, _ = int8_inference.run_integer_layer(
            q_x, cfg["outconv"], "outconv", s, z, apply_relu=False
        )

        q_out_float = q_out.to(torch.float32)
        logits = final_s * (q_out_float - final_z)
        b_tp, b_tn, b_fp, b_fn = _accumulate_confusion(logits, masks)
        tp += b_tp
        tn += b_tn
        fp += b_fp
        fn += b_fn
        seen += images.size(0)

        left = max(target_samples - seen, 0)
        print(
            f"[bench][{dataset_name}][int] Batch {batch_idx}: processed {seen}/{target_samples} samples, remaining {left}."
        )

    return _confusion_to_metrics(tp, tn, fp, fn)


def _fixed_point_metrics(model: torch.nn.Module, loader, dataset_name: str, num_data: Optional[int]):
    total_samples = len(loader.dataset)
    target_samples = total_samples if num_data is None else min(num_data, total_samples)
    print(
        f"[bench][{dataset_name}][fixed-point] Starting benchmark over {target_samples}/{total_samples} samples."
    )

    tp = tn = fp = fn = 0.0
    seen = 0

    for batch_idx, (images, masks) in enumerate(loader, 1):
        if seen >= target_samples:
            break

        remaining = target_samples - seen
        if images.size(0) > remaining:
            images = images[:remaining]
            masks = masks[:remaining]

        cfg = fp64_inference._get_layer_config(model)
        q_x = fp64_utils.quantize_fixed_point(images)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv1"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv2"], apply_relu=True)
        q_e12 = q_x
        q_x = fp64_inference.pool_fixed_point(q_x)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv3"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv4"], apply_relu=True)
        q_e22 = q_x
        q_x = fp64_inference.pool_fixed_point(q_x)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv5"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv6"], apply_relu=True)
        q_e32 = q_x
        q_x = fp64_inference.pool_fixed_point(q_x)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv7"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv8"], apply_relu=True)
        q_e42 = q_x
        q_x = fp64_inference.pool_fixed_point(q_x)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv9"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv10"], apply_relu=True)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["upconv1"], apply_relu=False)
        q_cat = torch.cat([q_x, q_e42], dim=1)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_cat, cfg["conv11"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv12"], apply_relu=True)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["upconv2"], apply_relu=False)
        q_cat = torch.cat([q_x, q_e32], dim=1)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_cat, cfg["conv13"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv14"], apply_relu=True)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["upconv3"], apply_relu=False)
        q_cat = torch.cat([q_x, q_e22], dim=1)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_cat, cfg["conv15"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv16"], apply_relu=True)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["upconv4"], apply_relu=False)
        q_cat = torch.cat([q_x, q_e12], dim=1)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_cat, cfg["conv17"], apply_relu=True)
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(q_x, cfg["conv18"], apply_relu=True)

        q_out, _, _ = fp64_inference.run_static_fixed_point_layer(
            q_x, cfg["outconv"], apply_relu=False
        )

        logits = fp64_utils.dequantize_fixed_point(q_out)
        b_tp, b_tn, b_fp, b_fn = _accumulate_confusion(logits, masks)
        tp += b_tp
        tn += b_tn
        fp += b_fp
        fn += b_fn
        seen += images.size(0)

        left = max(target_samples - seen, 0)
        print(
            f"[bench][{dataset_name}][fixed-point] Batch {batch_idx}: processed {seen}/{target_samples} samples, remaining {left}."
        )

    return _confusion_to_metrics(tp, tn, fp, fn)


def benchmark(dataset_names=None, num_data: Optional[int] = None, mode: Optional[str] = None):
    _disable_heavy_debug_logs()

    targets = dataset_names or BENCHMARK_DATASETS
    selected_modes = [mode] if mode is not None else ["floating-point", "int", "fixed-point"]

    results = {}

    for name in targets:
        print(f"\n[bench] Dataset: {name}")
        _ensure_checkpoint(name)

        loader = _get_test_loader(name)
        model = _build_model(name)

        per_dataset = {}
        if "floating-point" in selected_modes:
            per_dataset["floating-point"] = _float_metrics(model, loader, name, num_data)
        if "int" in selected_modes:
            per_dataset["int"] = _int_metrics(model, loader, name, num_data)
        if "fixed-point" in selected_modes:
            per_dataset["fixed-point"] = _fixed_point_metrics(model, loader, name, num_data)

        results[name] = per_dataset
        summary = []
        for m, vals in per_dataset.items():
            summary.append(
                f"{m}(dice={vals['dice']:.4f}, iou={vals['iou']:.4f}, acc={vals['acc']:.4f}, f1={vals['f1']:.4f})"
            )
        print(f"[bench] {name}: " + " | ".join(summary))

    return results


def _mode_suffix(mode: Optional[str]) -> str:
    if mode is None:
        return "all_modes"
    return mode.replace("-", "_")


def _results_filename(single_dataset_name: Optional[str], mode: Optional[str]) -> str:
    mode_part = _mode_suffix(mode)
    if single_dataset_name is None:
        return f"benchmark_results_{mode_part}.json"
    ds_part = single_dataset_name.lower().replace("-", "_")
    return f"benchmark_results_{ds_part}_{mode_part}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help="Benchmark a single dataset: Skin-Lesion, Flood, Brain-MRI-Seg, BUSI",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=None,
        help="Number of test samples to benchmark per dataset. If omitted, benchmarks full 10%% test split.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["int", "fixed-point", "floating-point"],
        default=None,
        help="Benchmark a specific mode. If omitted, benchmarks all 3 modes.",
    )
    args = parser.parse_args()

    if args.bench is None:
        targets = None
        single_name = None
    else:
        single_name = _normalize_bench_name(args.bench)
        targets = [single_name]

    metrics = benchmark(dataset_names=targets, num_data=args.num_data, mode=args.mode)
    results_file = _results_filename(single_name, args.mode)
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {results_file} with:")
    for ds, vals in metrics.items():
        summary = []
        for m, metrics_dict in vals.items():
            summary.append(
                f"{m}(dice={metrics_dict['dice']:.4f}, iou={metrics_dict['iou']:.4f}, acc={metrics_dict['acc']:.4f}, f1={metrics_dict['f1']:.4f})"
            )
        print(f"  {ds}: " + " | ".join(summary))
