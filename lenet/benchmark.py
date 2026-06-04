"""Unified benchmark runner for LeNet float, INT8, and FixedPoint64 inference.

This script benchmarks trained LeNet models over the deterministic 10% test
splits created by lenet5.py.

Supported flags:
- --bench: benchmark one dataset (defaults to all datasets)
- --num_data: number of test images to benchmark (defaults to full 10% split)
- --mode {fp32,int,int32,fxp32,fxp64}: benchmark one inference mode
    (defaults to all 5 modes)
"""

import argparse
import importlib.util
import json
import os
import sys
from typing import Optional

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        return value.item()
    return value


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INT8_DIR = os.path.join(THIS_DIR, "INT8")
FP64_DIR = os.path.join(THIS_DIR, "FixedPoint64")
INT32_DIR = os.path.join(THIS_DIR, "INT32")
FP32_DIR = os.path.join(THIS_DIR, "FixedPoint32")
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


import lenet5 as train_mod
int8_utils = _load_module("lenet_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR)
int8_inference = _load_module(
    "lenet_int8_inference", os.path.join(INT8_DIR, "inference.py"), INT8_DIR
)
fp64_utils = _load_module(
    "lenet_fp64_utils", os.path.join(FP64_DIR, "utils.py"), FP64_DIR
)
fp64_inference = _load_module(
    "lenet_fp64_inference", os.path.join(FP64_DIR, "inference.py"), FP64_DIR
)
int32_utils = _load_module("lenet_int32_utils", os.path.join(INT32_DIR, "utils.py"), INT32_DIR)
int32_inference = _load_module(
    "lenet_int32_inference", os.path.join(INT32_DIR, "inference.py"), INT32_DIR
)
fp32_utils = _load_module(
    "lenet_fp32_utils", os.path.join(FP32_DIR, "utils.py"), FP32_DIR
)
fp32_inference = _load_module(
    "lenet_fp32_inference", os.path.join(FP32_DIR, "inference.py"), FP32_DIR
)


BENCHMARK_DATASETS = {
    "MNIST",
    "CIFAR10",
    "Brain-MRI",
    "NIH-CHEST",
    "OCTMNIST",
    "BloodMNIST",
    "OrganAMNIST",
}


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
    int32_inference.debug_trace = {
        "input": None,
        "layers": _NoOpList(),
        "pooling": _NoOpList(),
    }
    fp32_inference.debug_trace = {
        "input": None,
        "layers": _NoOpList(),
        "pooling": _NoOpList(),
    }


def _normalize_bench_name(name: str) -> str:
    name_upper = name.upper()
    if name_upper == "CIFR10":
        return "CIFAR10"
    if name_upper in ("BRAIN-MRI", "BRAIN_MRI"):
        return "Brain-MRI"
    if name_upper in ("NIH-CHEST", "NIH_CHEST"):
        return "NIH-CHEST"
    if name_upper == "ORGANAMNIST":
        return "OrganAMNIST"
    if name_upper == "PNEUMONIAMNIST":
        return "PneumoniaMNIST"
    if name_upper == "MNIST":
        return "MNIST"
    if name_upper == "CIFAR10":
        return "CIFAR10"
    if name_upper == "OCTMNIST":
        return "OCTMNIST"
    if name_upper == "BLOODMNIST":
        return "BloodMNIST"
    raise ValueError(f"Unknown benchmark dataset: {name}")


def _train_dataset_for_checkpoint(dataset_name: str) -> None:
    train_data_flag = dataset_name

    print(
        f"[train] Missing checkpoint for {dataset_name}. Training with --train_data {train_data_flag}."
    )

    args = argparse.Namespace(
        batch_size=64,
        learning_rate=1e-3,
        train_data=train_data_flag,
        data_dir="",
        in_channels=1,
    )

    if train_data_flag == "MNIST":
        args.data_dir = train_mod.DATA_MNIST_DIR
    elif train_data_flag == "Brain-MRI":
        args.data_dir = train_mod.DATA_BRAIN_MRI_DIR
    elif train_data_flag == "NIH-CHEST":
        args.data_dir = train_mod.DATA_NIH_CHEST_XRAY_DIR
    elif train_data_flag == "OCTMNIST":
        args.data_dir = train_mod.DATA_OCTMNIST_DIR
    elif train_data_flag == "BloodMNIST":
        args.data_dir = train_mod.DATA_BLOODMNIST_DIR
    elif train_data_flag == "OrganAMNIST":
        args.data_dir = train_mod.DATA_ORGANAMNIST_DIR
    elif train_data_flag in ("CIFR10", "CIFAR10"):
        args.data_dir = train_mod.DATA_CIFAR10_DIR
    else:
        raise ValueError(f"Unsupported train_data flag: {train_data_flag}")

    train_mod.datasetDownloader(train_data_flag)
    train_mod.main(args)


def _ensure_checkpoint(dataset_name: str) -> None:
    cfg = int32_inference._resolve_infer_config(dataset_name)
    model_path = cfg["model_path"]

    if os.path.exists(model_path):
        return

    _train_dataset_for_checkpoint(dataset_name)

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Checkpoint still missing after training for {dataset_name}: {model_path}"
        )


def _get_test_loader(dataset_name: str, batch_size: Optional[int] = None) -> DataLoader:
    cfg = int32_inference._resolve_infer_config(dataset_name)
    effective_batch_size = batch_size if batch_size is not None else cfg.get("eval_batch_size", 64)

    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None
    setup_result = cfg["setup_fn"](batch_size=effective_batch_size)

    if (
        isinstance(setup_result, tuple)
        and len(setup_result) >= 3
        and hasattr(setup_result[2], "dataset")
    ):
        loader = setup_result[2]
        train_mod.validate_loader_preprocessing(loader, dataset_name, stage="benchmark")
        return loader

    if train_mod.test_loader is not None:
        train_mod.validate_loader_preprocessing(
            train_mod.test_loader, dataset_name, stage="benchmark"
        )
        return train_mod.test_loader

    raise RuntimeError(f"Could not resolve test loader for dataset: {dataset_name}")


def _build_model(dataset_name: str) -> torch.nn.Module:
    cfg = int32_inference._resolve_infer_config(dataset_name)
    model = cfg["model"]
    state = torch.load(cfg["model_path"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_batch_metrics(outputs: torch.Tensor, labels: torch.Tensor):
    # Fix MedMNIST [N, 1] shape by squeezing it to [N]
    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(-1)
        
    # Ensure it's evaluated as integer indices
    labels = labels.long()

    # Now it properly separates Multi-label (NIH) from Multi-class (MedMNIST)
    if labels.dim() > 1:
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct = (preds == labels).sum().item()
        total = labels.numel()
    else:
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    return correct, total


def _is_multilabel(dataset_name: str) -> bool:
    return bool(int32_inference._resolve_infer_config(dataset_name)["is_multilabel"])


def _is_medmnist(dataset_name: str) -> bool:
    name = dataset_name.upper()
    return name in ["OCTMNIST", "BLOODMNIST", "ORGANAMNIST", "PNEUMONIAMNIST"]

def _compute_auc_for_outputs(targets_np, outputs_np):
    if outputs_np.ndim == 2 and outputs_np.shape[1] == 2:
        if targets_np.ndim == 2 and targets_np.shape[1] == 2:
            y_true = targets_np[:, 1]
        else:
            y_true = targets_np.ravel()
        y_score = outputs_np[:, 1]
        return roc_auc_score(y_true, y_score)
    return roc_auc_score(targets_np, outputs_np, multi_class="ovr", average="macro")


def _format_metric_value(dataset_name: str, value) -> str:
    if isinstance(value, dict):
        return f"AUC={value['AUC']:.4f}, ACC={value['ACC']:.2f}%"
    if _is_multilabel(dataset_name):
        return f"{value:.4f}"
    return f"{value:.2f}%"


def _float_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
) -> float:
    is_multi = _is_multilabel(dataset_name)
    is_medmnist = _is_medmnist(dataset_name)
    
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][fp32] Starting benchmark over {target_images}/{total_images} images.")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, 1):
            if processed_images >= target_images:
                break

            remaining = target_images - processed_images
            if images.size(0) > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            outputs = model(images)
            
            if is_multi:
                all_targets.append(labels.detach().cpu())
                all_outputs.append(torch.sigmoid(outputs).detach().cpu())
            elif is_medmnist:
                all_targets.append(labels.detach().cpu())
                all_outputs.append(torch.softmax(outputs, dim=1).detach().cpu())
                c, t = _compute_batch_metrics(outputs, labels)
                correct += c
                total += t
            else:
                c, t = _compute_batch_metrics(outputs, labels)
                correct += c
                total += t

            processed_images += images.size(0)
            left = max(target_images - processed_images, 0)
            print(f"[bench][{dataset_name}][fp32] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}.")

    if is_multi:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        return roc_auc_score(targets, outputs, average="macro")
    elif is_medmnist:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        auc = _compute_auc_for_outputs(targets, outputs)
        acc = 100.0 * correct / max(total, 1)
        return {"AUC": auc, "ACC": acc}
    else:
        return 100.0 * correct / max(total, 1)


def _integer_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
):
    is_multi = _is_multilabel(dataset_name)
    is_medmnist = _is_medmnist(dataset_name)
    
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][int] Starting benchmark over {target_images}/{total_images} images.")

    # Load configuration and the offline compiled integer dictionary
    cfg = int32_inference._resolve_infer_config(dataset_name)
    int8_model_path = cfg["model_path"].replace(".pth", "_int8.pth")
    
    if not os.path.exists(int8_model_path):
        raise FileNotFoundError(
            f"Missing compiled model: {int8_model_path}. "
            f"Please run export_int8_model.py first."
        )
    
    int8_state = torch.load(int8_model_path, map_location="cpu")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    # Extract global input boundaries
    scale_in = int8_state["meta"]["in_scale"]
    zp_in = int8_state["meta"]["in_zp"]

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        # 1. Quantize Input Image
        q_x = int8_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)

        # ---------------------------------------------------------
        # 2. Execute Convs using int8_state (Offline Dictionary)
        # ---------------------------------------------------------
        q_x, s_out, z_out = int8_inference.run_integer_layer(
            q_x, int8_state["conv1"], "conv1", zp_in, apply_relu=True, is_conv=True
        )
        q_x = int8_inference.avg_pool_uint8(q_x, name="bench_pool_after_conv1")

        q_x, s_out, z_out = int8_inference.run_integer_layer(
            q_x, int8_state["conv2"], "conv2", z_out, apply_relu=True, is_conv=True
        )
        q_x = int8_inference.avg_pool_uint8(q_x, name="bench_pool_after_conv2")

        # Flatten for Dense Layers
        q_x = q_x.view(q_x.size(0), -1)

        # ---------------------------------------------------------
        # 3. Execute FCs using int8_state (Offline Dictionary)
        # ---------------------------------------------------------
        q_x, s_out, z_out = int8_inference.run_integer_layer(
            q_x, int8_state["fc1"], "fc1", z_out, apply_relu=True, is_conv=False
        )
        q_x, s_out, z_out = int8_inference.run_integer_layer(
            q_x, int8_state["fc2"], "fc2", z_out, apply_relu=True, is_conv=False
        )

        q_out, final_s, final_z = int8_inference.run_integer_layer(
            q_x, int8_state["fc3"], "fc3", z_out, apply_relu=False, is_conv=False
        )

        # ---------------------------------------------------------
        # 4. Output Dequantization & Metric Evaluation
        # ---------------------------------------------------------
        int_logits = q_out.to(torch.float32)
        dequantized_logits = final_s * (int_logits - final_z)

        # Route evaluation based on dataset type
        if is_multi:
            # NIH-CHEST (Sigmoid AUROC)
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.sigmoid(dequantized_logits).detach().cpu())
        elif is_medmnist:
            # MedMNIST (Softmax OVR AUROC + Accuracy)
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.softmax(dequantized_logits, dim=1).detach().cpu())
            c, t = _compute_batch_metrics(dequantized_logits, labels)
            correct += c
            total += t
        else:
            # Standard Datasets (Accuracy Only)
            c, t = _compute_batch_metrics(dequantized_logits, labels)
            correct += c
            total += t

        processed_images += images.size(0)
        left = max(target_images - processed_images, 0)
        print(f"[bench][{dataset_name}][int] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}.")

    # ---------------------------------------------------------
    # 5. Final Metric Aggregation
    # ---------------------------------------------------------
    if is_multi:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        return roc_auc_score(targets, outputs, average="macro")
    elif is_medmnist:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        auc = _compute_auc_for_outputs(targets, outputs)
        acc = 100.0 * correct / max(total, 1)
        return {"AUC": auc, "ACC": acc}
    else:
        return 100.0 * correct / max(total, 1)


def _fxp64_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
):
    is_multi = _is_multilabel(dataset_name)
    is_medmnist = _is_medmnist(dataset_name)
    
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][fxp64] Starting benchmark over {target_images}/{total_images} images.")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        # 1. Quantize input to 64-bit Fixed Point
        q_x = fp64_utils.quantize_fixed_point(images)
        
        # 2. Extract configuration (This now returns tuples of (layer, bn_layer))
        cfg = fp64_inference._get_layer_config(model)

        # ---------------------------------------------------------
        # Execute Convs (The tuple natively folds BN if it exists)
        # ---------------------------------------------------------
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(
            q_x, cfg["conv1"], apply_relu=True, is_conv=True
        )
        q_x = fp64_inference.avg_pool_fixed_point(q_x)

        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(
            q_x, cfg["conv2"], apply_relu=True, is_conv=True
        )
        q_x = fp64_inference.avg_pool_fixed_point(q_x)

        # Flatten for Dense Layers
        q_x = q_x.view(q_x.size(0), -1)

        # ---------------------------------------------------------
        # Execute FCs (The tuple passes (linear_layer, None))
        # ---------------------------------------------------------
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(
            q_x, cfg["fc1"], apply_relu=True, is_conv=False
        )
        q_x, _, _ = fp64_inference.run_static_fixed_point_layer(
            q_x, cfg["fc2"], apply_relu=True, is_conv=False
        )

        q_out, _, _ = fp64_inference.run_static_fixed_point_layer(
            q_x, cfg["fc3"], apply_relu=False, is_conv=False
        )

        # 3. Dequantize and evaluate
        dequantized_logits = fp64_utils.dequantize_fixed_point(q_out)

        if is_multi:
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.sigmoid(dequantized_logits).detach().cpu())
        elif is_medmnist:
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.softmax(dequantized_logits, dim=1).detach().cpu())
            c, t = _compute_batch_metrics(dequantized_logits, labels)
            correct += c
            total += t
        else:
            c, t = _compute_batch_metrics(dequantized_logits, labels)
            correct += c
            total += t

        processed_images += images.size(0)
        left = max(target_images - processed_images, 0)
        print(f"[bench][{dataset_name}][fxp64] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}.")

    if is_multi:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        return roc_auc_score(targets, outputs, average="macro")
    elif is_medmnist:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        auc = _compute_auc_for_outputs(targets, outputs)
        acc = 100.0 * correct / max(total, 1)
        return {"AUC": auc, "ACC": acc}
    else:
        return 100.0 * correct / max(total, 1)


def _int32_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
    activation: str,
):
    is_multi = _is_multilabel(dataset_name)
    is_medmnist = _is_medmnist(dataset_name)

    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][int32] Starting benchmark over {target_images}/{total_images} images.")

    cfg = int32_inference._resolve_infer_config(dataset_name, activation)
    int32_model_path = os.path.join(THIS_DIR, cfg["model_path"].replace(".pth", "_int32.pth"))

    if not os.path.exists(int32_model_path):
        raise FileNotFoundError(
            f"Missing compiled model: {int32_model_path}. "
            f"Please run INT32/export_int32_model.py first."
        )

    int32_state = torch.load(int32_model_path, map_location="cpu")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    scale_in = int32_state["meta"]["in_scale"]
    zp_in = int32_state["meta"]["in_zp"]

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        q_x = int32_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint32)

        q_x, s_out, z_out = int32_inference.run_integer_layer(
            q_x, int32_state["conv1"], "conv1", zp_in, apply_relu=True, is_conv=True, act_name=activation
        )
        q_x = int32_inference.avg_pool_uint32(q_x, name="bench_pool_after_conv1")

        q_x, s_out, z_out = int32_inference.run_integer_layer(
            q_x, int32_state["conv2"], "conv2", z_out, apply_relu=True, is_conv=True, act_name=activation
        )
        q_x = int32_inference.avg_pool_uint32(q_x, name="bench_pool_after_conv2")

        q_x = q_x.view(q_x.size(0), -1)

        q_x, s_out, z_out = int32_inference.run_integer_layer(
            q_x, int32_state["fc1"], "fc1", z_out, apply_relu=True, is_conv=False, act_name=activation
        )
        q_x, s_out, z_out = int32_inference.run_integer_layer(
            q_x, int32_state["fc2"], "fc2", z_out, apply_relu=True, is_conv=False, act_name=activation
        )

        q_out, final_s, final_z = int32_inference.run_integer_layer(
            q_x, int32_state["fc3"], "fc3", z_out, apply_relu=False, is_conv=False, act_name=activation
        )

        int_logits = q_out.to(torch.float64)
        dequantized_logits = final_s * (int_logits - final_z)

        if is_multi:
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.sigmoid(dequantized_logits).detach().cpu())
        elif is_medmnist:
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.softmax(dequantized_logits, dim=1).detach().cpu())
            c, t = _compute_batch_metrics(dequantized_logits, labels)
            correct += c
            total += t
        else:
            c, t = _compute_batch_metrics(dequantized_logits, labels)
            correct += c
            total += t

        processed_images += images.size(0)
        left = max(target_images - processed_images, 0)
        print(f"[bench][{dataset_name}][int32] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}.")

    if is_multi:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        return roc_auc_score(targets, outputs, average="macro")
    elif is_medmnist:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        auc = _compute_auc_for_outputs(targets, outputs)
        acc = 100.0 * correct / max(total, 1)
        return {"AUC": auc, "ACC": acc}
    else:
        return 100.0 * correct / max(total, 1)


def _fxp32_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
):
    is_multi = _is_multilabel(dataset_name)
    is_medmnist = _is_medmnist(dataset_name)

    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][fxp32] Starting benchmark over {target_images}/{total_images} images.")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        q_x = fp32_utils.quantize_fixed_point(images)
        cfg = fp32_inference._get_layer_config(model)

        q_x, _, _ = fp32_inference.run_static_fixed_point_layer(
            q_x, cfg["conv1"], apply_relu=True, is_conv=True
        )
        q_x = fp32_inference.avg_pool_fixed_point(q_x)

        q_x, _, _ = fp32_inference.run_static_fixed_point_layer(
            q_x, cfg["conv2"], apply_relu=True, is_conv=True
        )
        q_x = fp32_inference.avg_pool_fixed_point(q_x)

        q_x = q_x.view(q_x.size(0), -1)

        q_x, _, _ = fp32_inference.run_static_fixed_point_layer(
            q_x, cfg["fc1"], apply_relu=True, is_conv=False
        )
        q_x, _, _ = fp32_inference.run_static_fixed_point_layer(
            q_x, cfg["fc2"], apply_relu=True, is_conv=False
        )

        q_out, _, _ = fp32_inference.run_static_fixed_point_layer(
            q_x, cfg["fc3"], apply_relu=False, is_conv=False
        )

        dequantized_logits = fp32_utils.dequantize_fixed_point(q_out)

        if is_multi:
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.sigmoid(dequantized_logits).detach().cpu())
        elif is_medmnist:
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.softmax(dequantized_logits, dim=1).detach().cpu())
            c, t = _compute_batch_metrics(dequantized_logits, labels)
            correct += c
            total += t
        else:
            c, t = _compute_batch_metrics(dequantized_logits, labels)
            correct += c
            total += t

        processed_images += images.size(0)
        left = max(target_images - processed_images, 0)
        print(f"[bench][{dataset_name}][fxp32] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}.")

    if is_multi:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        return roc_auc_score(targets, outputs, average="macro")
    elif is_medmnist:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        auc = roc_auc_score(targets, outputs, multi_class="ovr", average="macro")
        acc = 100.0 * correct / max(total, 1)
        return {"AUC": auc, "ACC": acc}
    else:
        return 100.0 * correct / max(total, 1)


def benchmark(dataset_names=None, num_data: Optional[int] = None, mode: Optional[str] = None, activation: str = "relu"):
    _disable_heavy_debug_logs()

    targets = dataset_names or BENCHMARK_DATASETS
    selected_modes = [mode] if mode is not None else ["fp32", "int", "int32", "fxp32", "fxp64"]

    results = {}
    for name in targets:
        print(f"\n[bench] Dataset: {name}")
        _ensure_checkpoint(name)

        loader = _get_test_loader(name)
        model = _build_model(name)

        per_dataset = {}
        if "fp32" in selected_modes:
            fp_acc = _float_accuracy(model, loader, name, num_data)
            per_dataset["fp32"] = fp_acc
        if "int" in selected_modes:
            int_acc = _integer_accuracy(model, loader, name, num_data)
            per_dataset["int"] = int_acc
        if "fxp64" in selected_modes:
            fxp64_acc = _fxp64_accuracy(model, loader, name, num_data)
            per_dataset["fxp64"] = fxp64_acc
        if "int32" in selected_modes:
            int32_acc = _int32_accuracy(model, loader, name, num_data, activation=activation)
            per_dataset["int32"] = int32_acc
        if "fxp32" in selected_modes:
            fxp32_acc = _fxp32_accuracy(model, loader, name, num_data)
            per_dataset["fxp32"] = fxp32_acc

        results[name] = per_dataset
        stats = " | ".join([f"{k}={_format_metric_value(name, v)}" for k, v in per_dataset.items()])
        print(f"[bench] {name}: {stats}")

    return results


def _mode_suffix(mode: Optional[str]) -> str:
    if mode is None:
        return "all_modes"
    return mode.replace("-", "_")


def _results_filename(single_dataset_name: Optional[str], mode: Optional[str], activation: str) -> str:
    mode_part = _mode_suffix(mode)

    if single_dataset_name is None:
        return f"benchmark_results_{mode_part}_{activation}.json"
    
    ds_part = single_dataset_name.lower().replace("-", "_")
    return f"benchmark_results_{ds_part}_{mode_part}_{activation}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help=(
            "Benchmark a single dataset: MNIST, CIFAR10, Brain-MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST"
        ),
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=None,
        help="Number of test images to benchmark per dataset. If omitted, benchmarks full 10%% test split.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fp32", "int", "int32", "fxp32", "fxp64"],
        default=None,
        help="Benchmark a specific mode. If omitted, benchmarks all 5 modes.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function to evaluate (mainly used for int32 in this runner)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    targets = None
    if args.bench:
        targets = [_normalize_bench_name(args.bench)]

    final_results = benchmark(targets, args.num_data, args.mode, args.activation)

    bench_root = os.path.join(THIS_DIR, "benchmark-results")
    os.makedirs(bench_root, exist_ok=True)

    first_saved_path = None
    for ds, vals in final_results.items():
        ds_part = ds.lower().replace("-", "_")
        per_ds_dir = os.path.join(bench_root, ds_part)
        os.makedirs(per_ds_dir, exist_ok=True)
        per_file = _results_filename(ds, args.mode, args.activation)
        per_path = os.path.join(per_ds_dir, per_file)
        with open(per_path, "w") as pf:
            json.dump(_json_safe({ds: vals}), pf, indent=2)
        if first_saved_path is None:
            first_saved_path = per_path

    print(f"\nSaved benchmark results under {bench_root}:")
    for ds, vals in final_results.items():
        stats = " | ".join([f"{k}={_format_metric_value(ds, v)}" for k, v in vals.items()])
        print(f"  {ds}: {stats}")
