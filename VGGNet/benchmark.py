"""Unified benchmark runner for VGGNet float, INT8, INT32, and FixedPoint inference.

This script benchmarks trained VGG19 models over the deterministic test
splits created by vgg19.py.

Supported flags:
- --bench: benchmark one dataset (defaults to all datasets)
- --num_data: number of test images to benchmark (defaults to full test split)
- --mode {fp32,int8,int32,fxp32,fxp64}: benchmark one inference mode
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INT8_DIR = os.path.join(THIS_DIR, "INT8")
INT32_DIR = os.path.join(THIS_DIR, "INT32")
FP64_DIR = os.path.join(THIS_DIR, "FixedPoint64")
FP32_DIR = os.path.join(THIS_DIR, "FixedPoint32")

for path in (THIS_DIR,):
    if path not in sys.path:
        sys.path.insert(0, path)


def _is_medmnist(dataset_name: str) -> bool:
    return dataset_name.upper() in ["OCTMNIST", "BLOODMNIST", "ORGANAMNIST"]


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


import vgg19 as train_mod

int8_utils = _load_module(
    "vgg_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR
)
int8_inference = _load_module(
    "vgg_int8_inference", os.path.join(INT8_DIR, "inference.py"), INT8_DIR
)
int32_utils = _load_module(
    "vgg_int32_utils", os.path.join(INT32_DIR, "utils.py"), INT32_DIR
)
int32_inference = _load_module(
    "vgg_int32_inference", os.path.join(INT32_DIR, "inference.py"), INT32_DIR
)
fp64_utils = _load_module(
    "vgg_fp64_utils", os.path.join(FP64_DIR, "utils.py"), FP64_DIR
)
fp64_inference = _load_module(
    "vgg_fp64_inference", os.path.join(FP64_DIR, "inference.py"), FP64_DIR
)
fp32_utils = _load_module(
    "vgg_fp32_utils", os.path.join(FP32_DIR, "utils.py"), FP32_DIR
)
fp32_inference = _load_module(
    "vgg_fp32_inference", os.path.join(FP32_DIR, "inference.py"), FP32_DIR
)


BENCHMARK_DATASETS = [
    "MNIST",
    "CIFAR10",
    "Brain-MRI",
    "NIH-CHEST",
    "OCTMNIST",
    "BloodMNIST",
    "OrganAMNIST",
    "PneumoniaMNIST",
]


def _disable_heavy_debug_logs():
    class _NoOpList(list):
        def append(self, item):  # type: ignore[override]
            return None

    if hasattr(int8_inference, "debug_trace"):
        int8_inference.debug_trace = {
            "input": None,
            "layers": _NoOpList(),
            "pooling": _NoOpList(),
        }
    if hasattr(int32_inference, "debug_trace"):
        int32_inference.debug_trace = {
            "input": None,
            "layers": _NoOpList(),
            "pooling": _NoOpList(),
        }
    if hasattr(fp64_inference, "debug_trace"):
        fp64_inference.debug_trace = {
            "input": None,
            "layers": _NoOpList(),
            "pooling": _NoOpList(),
        }
    if hasattr(fp32_inference, "debug_trace"):
        fp32_inference.debug_trace = {
            "input": None,
            "layers": _NoOpList(),
            "pooling": _NoOpList(),
        }


def _normalize_bench_name(name: str) -> str:
    name_upper = name.upper()
    if name_upper == "CIFR10":
        return "CIFAR10"
    if name_upper == "MNIST":
        return "MNIST"
    if name_upper == "CIFAR10":
        return "CIFAR10"
    if name_upper == "BRAIN-MRI":
        return "Brain-MRI"
    if name_upper == "NIH-CHEST":
        return "NIH-CHEST"
    if name_upper == "OCTMNIST":
        return "OCTMNIST"
    if name_upper == "BLOODMNIST":
        return "BloodMNIST"
    if name_upper == "ORGANAMNIST":
        return "OrganAMNIST"
    if name_upper == "PNEUMONIAMNIST":
        return "PneumoniaMNIST"
    raise ValueError(f"Unknown benchmark dataset: {name}")


def _is_multilabel(dataset_name: str) -> bool:
    return dataset_name.upper() == "NIH-CHEST"


def _format_metric_value(dataset_name: str, value) -> str:
    if isinstance(value, dict):
        return f"AUC={value['AUC']:.4f}, ACC={value['ACC']:.2f}%"
    if _is_multilabel(dataset_name):
        return f"{value:.4f}"
    return f"{value:.2f}%"


def _ensure_checkpoint(dataset_name: str, activation: str) -> None:
    cfg = int8_inference._resolve_infer_config(dataset_name)
    slug = dataset_name.lower().replace(" ", "_").replace("-", "_")
    
    if dataset_name == "NIH-CHEST":
        model_path = os.path.abspath(os.path.join(THIS_DIR, f"best_vgg19_{activation}_NIH_Chest_XRay.pth"))
    else:
        model_path = os.path.abspath(os.path.join(THIS_DIR, f"best_vgg19_{activation}_{slug}.pth"))

    if os.path.exists(model_path):
        return

    raise FileNotFoundError(
        f"Missing checkpoint for {dataset_name}: {model_path}. Train the model first."
    )


def _get_test_loader(dataset_name: str, batch_size: Optional[int] = None) -> DataLoader:
    cfg = int8_inference._resolve_infer_config(dataset_name)
    effective_batch_size = (
        batch_size if batch_size is not None else cfg["eval_batch_size"]
    )

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


def _build_model(dataset_name: str, activation: str) -> torch.nn.Module:
    cfg = int8_inference._resolve_infer_config(dataset_name)
    model = cfg["model"]
    model.activation = activation
    
    slug = dataset_name.lower().replace(" ", "_").replace("-", "_")
    if dataset_name == "NIH-CHEST":
        model_path = os.path.abspath(os.path.join(THIS_DIR, f"best_vgg19_{activation}_NIH_Chest_XRay.pth"))
    else:
        model_path = os.path.abspath(os.path.join(THIS_DIR, f"best_vgg19_{activation}_{slug}.pth"))

    state = torch.load(model_path, map_location="cpu")
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


def _format_metric_value(dataset_name: str, value) -> str:
    if isinstance(value, dict):
        return f"AUC={value['AUC']:.4f}, ACC={value['ACC']:.2f}%"
    if _is_multilabel(dataset_name):
        return f"{value:.4f}"
    return f"{value:.2f}%"


def _load_int8_state(dataset_name: str, activation: str):
    cfg = int8_inference._resolve_infer_config(dataset_name)
    slug = dataset_name.lower().replace(" ", "_").replace("-", "_")
    if dataset_name == "NIH-CHEST":
        int8_model_path = os.path.abspath(os.path.join(THIS_DIR, f"best_vgg19_{activation}_NIH_Chest_XRay_int8.pth"))
    else:
        int8_model_path = os.path.abspath(os.path.join(THIS_DIR, f"best_vgg19_{activation}_{slug}_int8.pth"))

    if not os.path.exists(int8_model_path):
        raise FileNotFoundError(
            f"Missing compiled model: {int8_model_path}. Please run INT8/export_int8_model.py first."
        )

    return torch.load(int8_model_path, map_location="cpu")


def _float_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
):
    is_multi = _is_multilabel(dataset_name)
    is_medmnist = _is_medmnist(dataset_name)
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(
        f"[bench][{dataset_name}][fp32] Starting benchmark over {target_images}/{total_images} images."
    )

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
            print(
                f"[bench][{dataset_name}][fp32] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}."
            )

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

    return 100.0 * correct / max(total, 1)


def _int8_accuracy(
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

    print(
        f"[bench][{dataset_name}][int8] Starting benchmark over {target_images}/{total_images} images."
    )

    int8_state = _load_int8_state(dataset_name, activation)
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

        conv_indices = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49]
        maxpool_indices = [6, 13, 26, 39, 52]

        s_out, z_out = scale_in, zp_in
        for conv_idx in conv_indices:
            layer_name = f"features_{conv_idx}"
            apply_pool = (conv_idx + 3) in maxpool_indices
            q_x, s_out, z_out = int8_inference.run_integer_layer(
                q_x,
                int8_state[layer_name],
                z_out,
                layer_type="conv",
                apply_relu=True,
                apply_maxpool=apply_pool,
                activation=activation,
            )

        fc_in_scale = int8_state["classifier_0"]["scale_in"]
        fc_in_zp = int8_state["classifier_0"]["zp_in"]
        q_pooled = int8_utils.integer_adaptive_avg_pool(
            q_x, z_out, s_out, fc_in_zp, fc_in_scale, output_size=(7, 7)
        )
        q_fc_in = torch.flatten(q_pooled, 1)

        s_out, z_out = fc_in_scale, fc_in_zp
        for i, fc_idx in enumerate([0, 3, 6]):
            layer_name = f"classifier_{fc_idx}"
            is_last = i == 2
            q_fc_in, s_out, z_out = int8_inference.run_integer_layer(
                q_fc_in,
                int8_state[layer_name],
                z_out,
                layer_type="linear",
                apply_relu=not is_last,
                apply_maxpool=False,
                activation=activation,
            )

        int_logits = q_fc_in.to(torch.float32)
        dequantized_logits = s_out * (int_logits - z_out)

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
        print(
            f"[bench][{dataset_name}][int8] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}."
        )

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

    print(
        f"[bench][{dataset_name}][int32] Starting benchmark over {target_images}/{total_images} images."
    )

    slug = dataset_name.lower().replace(" ", "_").replace("-", "_")
    int32_model_path = os.path.join(THIS_DIR, f"best_vgg19_{activation}_{slug}_int32.pth")

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

        conv_indices = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49]
        maxpool_indices = [6, 13, 26, 39, 52]

        s_out, z_out = scale_in, zp_in
        for conv_idx in conv_indices:
            layer_name = f"features_{conv_idx}"
            apply_pool = (conv_idx + 3) in maxpool_indices
            layer_data = int32_state[layer_name]
            if activation == "gelu":
                layer_data["gelu_lut"] = int32_state[f"{layer_name}_gelu_lut"]
                
            q_x, s_out, z_out = int32_inference.run_integer_layer(
                q_x,
                layer_data,
                z_out,
                layer_type="conv",
                apply_relu=True,
                apply_maxpool=apply_pool,
                activation=activation,
            )

        q_pooled = torch.nn.functional.adaptive_max_pool2d(q_x.to(torch.float32), (7, 7)).to(torch.int32)
        q_fc_in = torch.flatten(q_pooled, 1)

        fc_in_scale = int32_state["classifier_0"]["scale_in"]
        fc_in_zp = int32_state["classifier_0"]["zp_in"]

        s_out, z_out = fc_in_scale, fc_in_zp
        for i, fc_idx in enumerate([0, 3, 6]):
            layer_name = f"classifier_{fc_idx}"
            is_last = i == 2
            layer_data = int32_state[layer_name]
            if not is_last and activation == "gelu":
                layer_data["gelu_lut"] = int32_state[f"{layer_name}_gelu_lut"]
                
            q_fc_in, s_out, z_out = int32_inference.run_integer_layer(
                q_fc_in,
                layer_data,
                z_out,
                layer_type="linear",
                apply_relu=not is_last,
                apply_maxpool=False,
                activation=activation,
            )

        int_logits = q_fc_in.to(torch.float64)
        dequantized_logits = s_out * (int_logits - z_out)

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
        print(
            f"[bench][{dataset_name}][int32] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}."
        )

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

    print(
        f"[bench][{dataset_name}][fxp64] Starting benchmark over {target_images}/{total_images} images."
    )

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images, labels = images[:remaining], labels[:remaining]

        q_x = fp64_utils.quantize_fixed_point(images)
        q_x = fp64_inference.run_fixed_point_features(q_x, model.features)
        q_pooled = fp64_utils.fixed_point_adaptive_avg_pool2d(q_x, output_size=(7, 7))
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)
        q_out = fp64_inference.run_fixed_point_classifier(q_fc_in, model.classifier)
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
        print(
            f"[bench][{dataset_name}][fxp64] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}."
        )

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

    print(
        f"[bench][{dataset_name}][fxp32] Starting benchmark over {target_images}/{total_images} images."
    )

    all_targets, all_outputs = [], []
    correct, total, processed_images = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images, labels = images[:remaining], labels[:remaining]

        q_x = fp32_utils.quantize_fixed_point(images)
        q_x = fp32_inference.run_fixed_point_features(q_x, model.features)
        q_pooled = fp32_utils.fixed_point_adaptive_avg_pool2d(q_x, output_size=(7, 7))
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)
        q_out = fp32_inference.run_fixed_point_classifier(q_fc_in, model.classifier)
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
        print(
            f"[bench][{dataset_name}][fxp32] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}."
        )

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

    return 100.0 * correct / max(total, 1)


def benchmark(
    dataset_names=None, num_data: Optional[int] = None, mode: Optional[str] = None, activation: str = "relu"
):
    _disable_heavy_debug_logs()

    targets = dataset_names or BENCHMARK_DATASETS
    selected_modes = (
        [mode] if mode is not None else ["fp32", "int8", "int32", "fxp32", "fxp64"]
    )

    results = {}
    for name in targets:
        print(f"\n[bench] Dataset: {name} | Activation: {activation}")
        _ensure_checkpoint(name, activation)

        loader = _get_test_loader(name)
        model = _build_model(name, activation)

        per_dataset = {}
        if "fp32" in selected_modes:
            per_dataset["fp32"] = _float_accuracy(
                model, loader, name, num_data
            )
        if "int8" in selected_modes:
            per_dataset["int8"] = _int8_accuracy(model, loader, name, num_data, activation)
        if "int32" in selected_modes:
            per_dataset["int32"] = _int32_accuracy(model, loader, name, num_data, activation)
        if "fxp32" in selected_modes:
            per_dataset["fxp32"] = _fxp32_accuracy(model, loader, name, num_data)
        if "fxp64" in selected_modes:
            per_dataset["fxp64"] = _fxp64_accuracy(
                model, loader, name, num_data
            )

        results[name] = per_dataset
        stats = " | ".join(
            [f"{k}={_format_metric_value(name, v)}" for k, v in per_dataset.items()]
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
        return f"benchmark_results_vgg19_{mode_part}_{activation}.json"
    ds_part = single_dataset_name.lower().replace("-", "_")
    
    # Save into the specific dataset directory
    results_dir = os.path.join(THIS_DIR, "benchmark-results", ds_part)
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"benchmark_results_{ds_part}_{mode_part}_{activation}.json")


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
        help="Benchmark a single dataset: MNIST, CIFAR10, Brain-MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=None,
        help="Number of test images to benchmark per dataset. If omitted, benchmarks the full test split.",
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
        help="Activation function to use",
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

    metrics = benchmark(dataset_names=targets, num_data=args.num_data, mode=args.mode, activation=args.activation)
    results_file = _results_filename(single_name, args.mode, args.activation)
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {results_file} with:")
    for ds, vals in metrics.items():
        stats = " | ".join(
            [f"{k}={_format_metric_value(ds, v)}" for k, v in sorted(vals.items())]
        )
        print(f"  {ds}: {stats}")
