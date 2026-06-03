"""Unified benchmark runner for ResNet float, INT8, INT32, FXP32, and FXP64 inference.

This script benchmarks trained ResNet models over the deterministic 10% test
splits created by resnet18.py.

Supported flags:
- --bench: benchmark one dataset (defaults to all datasets)
- --num_data: number of test images to benchmark (defaults to full 10% split)
- --mode {fp32,int8,int32,fxp32,fxp64}: benchmark one inference mode
    (defaults to all 5 modes)
- --activation {relu,gelu,leaky_relu}: the activation function to evaluate
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


def _compute_auc_for_outputs(targets_np, outputs_np):
    """Compute AUC for MedMNIST outputs that may be binary (N,2) or multiclass.

    For binary (two-column) outputs treat column 1 as the positive-class
    probability and collapse one-hot targets accordingly.
    """
    # Binary two-column outputs -> use positive class probabilities
    if outputs_np.ndim == 2 and outputs_np.shape[1] == 2:
        if targets_np.ndim == 2 and targets_np.shape[1] == 2:
            y_true = targets_np[:, 1]
        else:
            y_true = targets_np.ravel()
        y_score = outputs_np[:, 1]
        return roc_auc_score(y_true, y_score)

    # Otherwise fall back to multiclass AUC
    return roc_auc_score(targets_np, outputs_np, multi_class="ovr", average="macro")


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

def _is_medmnist(dataset_name: str) -> bool:
    name = dataset_name.upper()
    return name in ["OCTMNIST", "BLOODMNIST", "ORGANAMNIST", "PNEUMONIAMNIST"]

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


import resnet18 as train_mod
int8_utils = _load_module("resnet_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR)
int8_inference = _load_module(
    "resnet_int8_inference", os.path.join(INT8_DIR, "inference.py"), INT8_DIR
)
fp64_utils = _load_module(
    "resnet_fp64_utils", os.path.join(FP64_DIR, "utils.py"), FP64_DIR
)
fp64_inference = _load_module(
    "resnet_fp64_inference", os.path.join(FP64_DIR, "inference.py"), FP64_DIR
)
int32_utils = _load_module("resnet_int32_utils", os.path.join(INT32_DIR, "utils.py"), INT32_DIR)
int32_inference = _load_module(
    "resnet_int32_inference", os.path.join(INT32_DIR, "inference.py"), INT32_DIR
)
fp32_utils = _load_module(
    "resnet_fp32_utils", os.path.join(FP32_DIR, "utils.py"), FP32_DIR
)
fp32_inference = _load_module(
    "resnet_fp32_inference", os.path.join(FP32_DIR, "inference.py"), FP32_DIR
)


BENCHMARK_DATASETS = [
    "MNIST",
    "CIFAR10",
    "Brain-MRI",
    "NIH-CHEST",
    "OCTMNIST",
    "BloodMNIST",
    "OrganAMNIST",
    "PneumoniaMNIST"
]


def _disable_heavy_debug_logs():
    class _NoOpList(list):
        def append(self, item):
            return None

    int8_inference.debug_trace = {"input": None, "layers": _NoOpList(), "pooling": _NoOpList()}
    fp64_inference.debug_trace = {"input": None, "layers": _NoOpList(), "pooling": _NoOpList()}
    int32_inference.debug_trace = {"input": None, "layers": _NoOpList(), "pooling": _NoOpList()}
    fp32_inference.debug_trace = {"input": None, "layers": _NoOpList(), "pooling": _NoOpList()}


def _normalize_bench_name(name: str) -> str:
    name_upper = name.upper()
    if name_upper == "CIFR10": return "CIFAR10"
    if name_upper == "BRAIN-MRI": return "Brain-MRI"
    if name_upper == "NIH-CHEST": return "NIH-CHEST"
    if name_upper == "MNIST": return "MNIST"
    if name_upper == "CIFAR10": return "CIFAR10"
    if name_upper == "OCTMNIST": return "OCTMNIST"
    if name_upper == "BLOODMNIST": return "BloodMNIST"
    if name_upper == "ORGANAMNIST": return "OrganAMNIST"
    if name_upper == "PNEUMONIAMNIST": return "PneumoniaMNIST"
    raise ValueError(f"Unknown benchmark dataset: {name}")


def _train_dataset_for_checkpoint(dataset_name: str, activation: str) -> None:
    train_data_flag = dataset_name
    batch_size = 64

    print(f"[train] Missing checkpoint for {dataset_name}. Training with --train_data {train_data_flag} ({activation}).")

    args = argparse.Namespace(
        batch_size=batch_size,
        learning_rate=1e-3,
        train_data=train_data_flag,
        in_channels=1,
        data_dir="",
        activation=activation,
    )

    if train_data_flag == "MNIST": args.data_dir = train_mod.DATA_MNIST_DIR
    elif train_data_flag == "Brain-MRI": args.data_dir = train_mod.DATA_BRAIN_MRI_DIR
    elif train_data_flag in ("CIFR10", "CIFAR10"): args.data_dir = train_mod.DATA_CIFAR10_DIR
    elif train_data_flag == "OCTMNIST": args.data_dir = train_mod.DATA_OCTMNIST_DIR
    elif train_data_flag == "PneumoniaMNIST": args.data_dir = train_mod.DATA_PNEUMONIAMNIST_DIR
    elif train_data_flag == "BloodMNIST": args.data_dir = train_mod.DATA_BLOODMNIST_DIR
    elif train_data_flag == "OrganAMNIST": args.data_dir = train_mod.DATA_ORGANAMNIST_DIR
    else: raise ValueError(f"Unsupported train_data flag: {train_data_flag}")

    train_mod.datasetDownloader(train_data_flag)
    train_mod.main(args)


def _ensure_checkpoint(dataset_name: str, activation: str) -> None:
    cfg = int8_inference._resolve_infer_config(dataset_name, activation)
    model_path = cfg["model_path"]

    if os.path.exists(model_path):
        return

    _train_dataset_for_checkpoint(dataset_name, activation)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Checkpoint still missing after training for {dataset_name}: {model_path}")


def _get_test_loader(dataset_name: str, activation: str, batch_size: Optional[int] = None) -> DataLoader:
    cfg = int8_inference._resolve_infer_config(dataset_name, activation)
    effective_batch_size = batch_size if batch_size is not None else cfg["eval_batch_size"]

    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None
    setup_result = cfg["setup_fn"](batch_size=effective_batch_size)

    if isinstance(setup_result, tuple) and len(setup_result) >= 3 and hasattr(setup_result[2], "dataset"):
        loader = setup_result[2]
        train_mod.validate_loader_preprocessing(loader, dataset_name, stage="benchmark")
        return loader

    if train_mod.test_loader is not None:
        train_mod.validate_loader_preprocessing(train_mod.test_loader, dataset_name, stage="benchmark")
        return train_mod.test_loader

    raise RuntimeError(f"Could not resolve test loader for dataset: {dataset_name}")


def _build_model(dataset_name: str, activation: str) -> torch.nn.Module:
    cfg = int8_inference._resolve_infer_config(dataset_name, activation)
    model = cfg["model"]
    state = torch.load(cfg["model_path"], map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_batch_metrics(outputs: torch.Tensor, labels: torch.Tensor):
    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(-1)
        
    labels = labels.long()

    if labels.dim() > 1:
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct = (preds == labels).sum().item()
        total = labels.numel()
    else:
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    return correct, total

def _is_multilabel(dataset_name: str, activation: str) -> bool:
    return bool(int8_inference._resolve_infer_config(dataset_name, activation)["is_multilabel"])


def _format_metric_value(dataset_name: str, activation: str, value) -> str:
    if isinstance(value, dict):
        return f"AUC={value['AUC']:.4f}, ACC={value['ACC']:.2f}%"
    if _is_multilabel(dataset_name, activation):
        return f"{value:.4f}"
    return f"{value:.2f}%"


def _float_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
    activation: str,
):
    is_multi = _is_multilabel(dataset_name, activation)
    is_medmnist = _is_medmnist(dataset_name)
    
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][{activation}][fp32] Starting benchmark over {target_images}/{total_images} images.")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, 1):
            if processed_images >= target_images:
                break

            remaining = target_images - processed_images
            if images.size(0) > remaining:
                images, labels = images[:remaining], labels[:remaining]

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
            print(f"[bench][{dataset_name}][{activation}][fp32] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}.")

    if is_multi:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()
        return roc_auc_score(targets, outputs, average="macro")
    elif is_medmnist:
        targets = torch.cat(all_targets, dim=0).numpy()
        outputs = torch.cat(all_outputs, dim=0).numpy()

        # Handle binary MedMNIST (e.g. PneumoniaMNIST) where outputs are (N,2)
        # and labels may be one-hot (N,2). For binary AUC compute using the
        # probability of the positive class (column 1) and a 1-D y_true.
        auc = _compute_auc_for_outputs(targets, outputs)

        acc = 100.0 * correct / max(total, 1)
        return {"AUC": auc, "ACC": acc}
    else:
        return 100.0 * correct / max(total, 1)


def _int8_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
    activation: str,
):
    is_multi = _is_multilabel(dataset_name, activation)
    is_medmnist = _is_medmnist(dataset_name)
    
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][{activation}][int8] Starting benchmark over {target_images}/{total_images} images.")

    cfg = int8_inference._resolve_infer_config(dataset_name, activation)
    int8_model_path = cfg["model_path"].replace(".pth", "_int8.pth")
    
    if not os.path.exists(int8_model_path):
        raise FileNotFoundError(f"Missing compiled model: {int8_model_path}. Please run export_int8_model.py first.")
    
    int8_state = torch.load(int8_model_path, map_location="cpu")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    scale_in = int8_state["meta"]["in_scale"]
    zp_in = int8_state["meta"]["in_zp"]

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images, labels = images[:remaining], labels[:remaining]

        q_x = int8_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)
        q_x, s_out, z_out = int8_inference.run_integer_conv_block(q_x, int8_state["conv1"], zp_in, apply_act=True, act_name=activation)

        for layer_idx in range(1, 5):
            for block_idx in range(2):
                prefix = f"layer{layer_idx}_block{block_idx}"
                q_x, s_out, z_out = int8_inference.run_integer_basic_block(q_x, int8_state[prefix], z_out, s_out, act_name=activation)

        fc_in_scale = int8_state["fc"]["scale_in"]
        fc_in_zp = int8_state["fc"]["zp_in"]
        
        q_pooled = int8_utils.integer_max_pool2d(q_x)
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, final_s, final_z = int8_inference.run_integer_fc(q_fc_in, int8_state["fc"], fc_in_zp)

        int_logits = q_out.to(torch.float32)
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
        print(f"[bench][{dataset_name}][{activation}][int8] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}.")

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
    is_multi = _is_multilabel(dataset_name, activation)
    is_medmnist = _is_medmnist(dataset_name)

    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][{activation}][int32] Starting benchmark over {target_images}/{total_images} images.")

    cfg = int32_inference._resolve_infer_config(dataset_name, activation)
    int32_model_path = cfg["model_path"].replace(".pth", "_int32.pth")

    if not os.path.exists(int32_model_path):
        raise FileNotFoundError(
            f"Missing compiled model: {int32_model_path}. Please run INT32/export_int32_model.py first."
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
            images, labels = images[:remaining], labels[:remaining]

        q_x = int32_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint32)
        q_x, s_out, z_out = int32_inference.run_integer_conv_block(q_x, int32_state["conv1"], zp_in, apply_act=True, act_name=activation)

        for layer_idx in range(1, 5):
            for block_idx in range(2):
                prefix = f"layer{layer_idx}_block{block_idx}"
                q_x, s_out, z_out = int32_inference.run_integer_basic_block(q_x, int32_state[prefix], z_out, s_out, act_name=activation)

        fc_in_scale = int32_state["fc"]["scale_in"]
        fc_in_zp = int32_state["fc"]["zp_in"]

        q_pooled = int32_utils.integer_max_pool2d(q_x)
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, final_s, final_z = int32_inference.run_integer_fc(q_fc_in, int32_state["fc"], fc_in_zp)

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

def _fxp64_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
    activation: str,
):
    is_multi = _is_multilabel(dataset_name, activation)
    is_medmnist = _is_medmnist(dataset_name)
    
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][{activation}][fxp64] Starting benchmark over {target_images}/{total_images} images.")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images, labels = images[:remaining], labels[:remaining]

        q_x = fp64_utils.quantize_fixed_point(images)

        q_x = fp64_inference.run_static_fixed_point_conv_block(q_x, model.conv1, model.bn1, apply_act=True, act_name=activation)

        for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in stage:
                q_x = fp64_inference.run_static_fixed_point_basic_block(q_x, block, act_name=activation)

        q_pooled = fp64_utils.fixed_point_max_pool2d(q_x)
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, _, _ = fp64_inference.run_static_fixed_point_fc(q_fc_in, model.fc)
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


def _fxp32_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
    activation: str,
):
    is_multi = _is_multilabel(dataset_name, activation)
    is_medmnist = _is_medmnist(dataset_name)

    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][{activation}][fxp32] Starting benchmark over {target_images}/{total_images} images.")

    global_gelu_lut = None
    if activation == "gelu":
        lut_path = os.path.join(FP32_DIR, "gelu_q15_16_lut.pt")
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"Missing LUT file: {lut_path}. Run lut.py first.")
        global_gelu_lut = torch.load(lut_path, map_location="cpu")

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images, labels = images[:remaining], labels[:remaining]

        q_x = fp32_utils.quantize_fixed_point(images)

        q_x = fp32_inference.run_static_fixed_point_conv_block(q_x, model.conv1, model.bn1, apply_act=True, act_name=activation, lut_dict=global_gelu_lut)

        for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in stage:
                q_x = fp32_inference.run_static_fixed_point_basic_block(q_x, block, act_name=activation, lut_dict=global_gelu_lut)

        q_pooled = fp32_utils.fixed_point_max_pool2d(q_x)
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, _, _ = fp32_inference.run_static_fixed_point_fc(q_fc_in, model.fc)
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
        print(f"[bench][{dataset_name}][{activation}][fxp32] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}.")

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


def benchmark(dataset_names=None, num_data: Optional[int] = None, batch_size: int = 64, mode: Optional[str] = None, activation: str = "relu"):
    _disable_heavy_debug_logs()

    targets = dataset_names or BENCHMARK_DATASETS
    selected_modes = [mode] if mode is not None else ["fp32", "int8", "int32", "fxp32", "fxp64"]

    results = {}
    for name in targets:
        print(f"\n[bench] Dataset: {name}")
        _ensure_checkpoint(name, activation)

        loader = _get_test_loader(name, activation, batch_size=batch_size)
        model = _build_model(name, activation)

        per_dataset = {}
        if "fp32" in selected_modes:
            fp_acc = _float_accuracy(model, loader, name, num_data, activation)
            per_dataset["fp32"] = fp_acc
        if "int8" in selected_modes:
            int8_acc = _int8_accuracy(model, loader, name, num_data, activation)
            per_dataset["int8"] = int8_acc
        if "int32" in selected_modes:
            int32_acc = _int32_accuracy(model, loader, name, num_data, activation)
            per_dataset["int32"] = int32_acc
        if "fxp32" in selected_modes:
            fxp32_acc = _fxp32_accuracy(model, loader, name, num_data, activation)
            per_dataset["fxp32"] = fxp32_acc
        if "fxp64" in selected_modes:
            fxp64_acc = _fxp64_accuracy(model, loader, name, num_data, activation)
            per_dataset["fxp64"] = fxp64_acc

        results[name] = per_dataset
        stats = " | ".join(
            [f"{k}={_format_metric_value(name, activation, v)}" for k, v in per_dataset.items()]
        )
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
        "--NIH-CHEST",
        dest="nih_chest",
        action="store_true",
        help="Benchmark only NIH-CHEST using the custom test split",
    )
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
        choices=["fp32", "int8", "int32", "fxp32", "fxp64"],
        default=None,
        help="Benchmark a specific mode. If omitted, benchmarks all 5 modes.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function the model was trained with"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    args = parser.parse_args()
    
    
    if args.nih_chest:
        targets = ["NIH-CHEST"]
        single_name = "NIH-CHEST"
    elif args.bench is None:
        targets = None
        single_name = None
    else:
        single_name = _normalize_bench_name(args.bench)
        targets = [single_name]

    metrics = benchmark(dataset_names=targets, num_data=args.num_data, batch_size=args.batch_size, mode=args.mode, activation=args.activation)
    results_file = _results_filename(single_name, args.mode, args.activation)

    # Save all benchmark outputs under resnet/benchmark-results/
    bench_root = os.path.join(THIS_DIR, "benchmark-results")
    os.makedirs(bench_root, exist_ok=True)

    first_saved_path = None
    for ds, vals in metrics.items():
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
    for ds, vals in metrics.items():
        stats = " | ".join([f"{k}={_format_metric_value(ds, args.activation, v)}" for k, v in vals.items()])
        print(f"  {ds}: {stats}")