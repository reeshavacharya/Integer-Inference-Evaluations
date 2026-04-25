"""Unified benchmark runner for ResNet float, INT8, and FixedPoint64 inference.

This script benchmarks trained ResNet models over the deterministic 10% test
splits created by resnet18.py.

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
from torch.utils.data import DataLoader


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


BENCHMARK_DATASETS = [
    "MNIST",
    "CIFAR10",
    "Brain-MRI",
    "CHEST",
    "Brain-Cancer",
    "Breast-Cancer",
    "Cervical-Cancer",
    "Kidney-Cancer",
    "Lung-And-Colon-Cancer",
    "Lymphoma-Cancer",
    "Oral-Cancer",
]

MULTI_CANCER_DATASETS = {
    "Brain-Cancer",
    "Breast-Cancer",
    "Cervical-Cancer",
    "Kidney-Cancer",
    "Lung-And-Colon-Cancer",
    "Lymphoma-Cancer",
    "Oral-Cancer",
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


def _normalize_bench_name(name: str) -> str:
    name_upper = name.upper()
    if name_upper == "CIFR10":
        return "CIFAR10"
    if name_upper == "BRAIN-MRI":
        return "Brain-MRI"
    if name_upper == "BRAIN-CANCER":
        return "Brain-Cancer"
    if name_upper == "BREAST-CANCER":
        return "Breast-Cancer"
    if name_upper == "CERVICAL-CANCER":
        return "Cervical-Cancer"
    if name_upper == "KIDNEY-CANCER":
        return "Kidney-Cancer"
    if name_upper == "LUNG-AND-COLON-CANCER":
        return "Lung-And-Colon-Cancer"
    if name_upper == "LYMPHOMA-CANCER":
        return "Lymphoma-Cancer"
    if name_upper == "ORAL-CANCER":
        return "Oral-Cancer"
    if name_upper == "MNIST":
        return "MNIST"
    if name_upper == "CIFAR10":
        return "CIFAR10"
    if name_upper == "CHEST":
        return "CHEST"
    raise ValueError(f"Unknown benchmark dataset: {name}")


def _train_dataset_for_checkpoint(dataset_name: str, multi_trained: bool) -> bool:
    if dataset_name in MULTI_CANCER_DATASETS:
        if multi_trained:
            return multi_trained
        train_data_flag = "Multi-Cancer"
    else:
        train_data_flag = dataset_name

    print(
        f"[train] Missing checkpoint for {dataset_name}. Training with --train_data {train_data_flag}."
    )

    args = argparse.Namespace(
        batch_size=64,
        learning_rate=1e-3,
        train_data=train_data_flag,
        in_channels=1,
        data_dir="",
    )

    if train_data_flag == "MNIST":
        args.data_dir = train_mod.DATA_MNIST_DIR
    elif train_data_flag == "Brain-MRI":
        args.data_dir = train_mod.DATA_BRAIN_MRI_DIR
    elif train_data_flag in ("CIFR10", "CIFAR10"):
        args.data_dir = train_mod.DATA_CIFAR10_DIR
    elif train_data_flag == "CHEST":
        args.data_dir = train_mod.DATA_CHEST_DIR
    elif train_data_flag == "Multi-Cancer":
        args.data_dir = train_mod.DATA_MULTI_CANCER_DIR
    else:
        raise ValueError(f"Unsupported train_data flag: {train_data_flag}")

    train_mod.datasetDownloader(train_data_flag)
    train_mod.main(args)

    return train_data_flag == "Multi-Cancer" or multi_trained


def _ensure_checkpoint(dataset_name: str, multi_trained: bool) -> bool:
    cfg = int8_inference._resolve_infer_config(dataset_name)
    model_path = cfg["model_path"]

    if os.path.exists(model_path):
        return multi_trained

    multi_trained = _train_dataset_for_checkpoint(dataset_name, multi_trained)

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Checkpoint still missing after training for {dataset_name}: {model_path}"
        )
    return multi_trained


def _get_test_loader(dataset_name: str, batch_size: int = 64) -> DataLoader:
    cfg = int8_inference._resolve_infer_config(dataset_name)

    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None
    setup_result = cfg["setup_fn"](batch_size=batch_size)

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
    cfg = int8_inference._resolve_infer_config(dataset_name)
    model = cfg["model"]
    state = torch.load(cfg["model_path"], map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_batch_metrics(outputs: torch.Tensor, labels: torch.Tensor):
    if labels.dim() > 1:
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct = (preds == labels).sum().item()
        total = labels.numel()
    else:
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    return correct, total


def _float_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
) -> float:
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(
        f"[bench][{dataset_name}][floating-point] Starting benchmark over {target_images}/{total_images} images."
    )

    correct = 0.0
    total = 0
    processed_images = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, 1):
            if processed_images >= target_images:
                break

            remaining = target_images - processed_images
            if images.size(0) > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            outputs = model(images)
            c, t = _compute_batch_metrics(outputs, labels)
            correct += c
            total += t
            processed_images += images.size(0)

            left = max(target_images - processed_images, 0)
            print(
                f"[bench][{dataset_name}][floating-point] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}."
            )

    return 100.0 * correct / max(total, 1)


def _integer_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
) -> float:
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(f"[bench][{dataset_name}][int] Starting benchmark over {target_images}/{total_images} images.")

    correct = 0.0
    total = 0
    processed_images = 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        int8_inference.activation_ranges.clear()
        handles = int8_inference.register_hooks(model)
        with torch.no_grad():
            _ = model(images)
        for h in handles:
            h.remove()

        in_range = int8_inference.activation_ranges["conv1"]
        pseudo_in_tensor = torch.tensor(
            [in_range["in_min"], in_range["in_max"]], dtype=torch.float32
        )
        scale_in, zp_in = int8_utils.get_quantization_params(pseudo_in_tensor, num_bits=8)
        q_x = int8_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)

        q_x, s_out, z_out = int8_inference.run_integer_conv_block(
            q_x,
            model.conv1,
            model.bn1,
            "conv1_relu",
            scale_in,
            zp_in,
            apply_relu=True,
        )

        for layer_idx, stage in enumerate(
            [model.layer1, model.layer2, model.layer3, model.layer4], 1
        ):
            for block_idx, block in enumerate(stage):
                prefix = f"layer{layer_idx}_block{block_idx}"
                q_x, s_out, z_out = int8_inference.run_integer_basic_block(
                    q_x, block, prefix, s_out, z_out
                )

        fc_in_range = int8_inference.activation_ranges["fc"]
        pseudo_fc_in = torch.tensor(
            [fc_in_range["in_min"], fc_in_range["in_max"]], dtype=torch.float32
        )
        s_fc_in, z_fc_in = int8_utils.get_quantization_params(pseudo_fc_in, num_bits=8)

        q_pooled = int8_utils.integer_global_avg_pool2d(
            q_x, z_out, s_out, z_fc_in, s_fc_in
        )
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, final_s, final_z, _, _, _ = int8_inference.run_integer_fc(
            q_fc_in, model.fc, "fc", s_fc_in, z_fc_in
        )

        int_logits = q_out.to(torch.float32)
        dequantized_logits = final_s * (int_logits - final_z)

        c, t = _compute_batch_metrics(dequantized_logits, labels)
        correct += c
        total += t
        processed_images += images.size(0)

        left = max(target_images - processed_images, 0)
        print(
            f"[bench][{dataset_name}][int] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}."
        )

    return 100.0 * correct / max(total, 1)


def _fixed_point_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_name: str,
    num_data: Optional[int],
) -> float:
    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)

    print(
        f"[bench][{dataset_name}][fixed-point] Starting benchmark over {target_images}/{total_images} images."
    )

    correct = 0.0
    total = 0
    processed_images = 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        q_x = fp64_utils.quantize_fixed_point(images)

        q_x = fp64_inference.run_static_fixed_point_conv_block(
            q_x, model.conv1, model.bn1, apply_relu=True
        )

        for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in stage:
                q_x = fp64_inference.run_static_fixed_point_basic_block(q_x, block)

        q_pooled = fp64_utils.fixed_point_global_avg_pool2d(q_x)
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, _, _ = fp64_inference.run_static_fixed_point_fc(q_fc_in, model.fc)
        dequantized_logits = fp64_utils.dequantize_fixed_point(q_out)

        c, t = _compute_batch_metrics(dequantized_logits, labels)
        correct += c
        total += t
        processed_images += images.size(0)

        left = max(target_images - processed_images, 0)
        print(
            f"[bench][{dataset_name}][fixed-point] Batch {batch_idx}: processed {processed_images}/{target_images} images, remaining {left}."
        )

    return 100.0 * correct / max(total, 1)


def benchmark(dataset_names=None, num_data: Optional[int] = None, mode: Optional[str] = None):
    _disable_heavy_debug_logs()

    targets = dataset_names or BENCHMARK_DATASETS
    selected_modes = [mode] if mode is not None else ["floating-point", "int", "fixed-point"]

    results = {}
    multi_trained = False

    for name in targets:
        print(f"\n[bench] Dataset: {name}")
        multi_trained = _ensure_checkpoint(name, multi_trained)

        loader = _get_test_loader(name)
        model = _build_model(name)

        per_dataset = {}
        if "floating-point" in selected_modes:
            fp_acc = _float_accuracy(model, loader, name, num_data)
            per_dataset["floating-point"] = fp_acc
        if "int" in selected_modes:
            int_acc = _integer_accuracy(model, loader, name, num_data)
            per_dataset["int"] = int_acc
        if "fixed-point" in selected_modes:
            fxp_acc = _fixed_point_accuracy(model, loader, name, num_data)
            per_dataset["fixed-point"] = fxp_acc

        results[name] = per_dataset
        stats = " | ".join([f"{k}={v:.2f}%" for k, v in per_dataset.items()])
        print(f"[bench] {name}: {stats}")

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
        help=(
            "Benchmark a single dataset: MNIST, CIFAR10, Brain-MRI, CHEST, "
            "Brain-Cancer, Breast-Cancer, Cervical-Cancer, Kidney-Cancer, "
            "Lung-And-Colon-Cancer, Lymphoma-Cancer, Oral-Cancer"
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
        stats = " | ".join([f"{k}={v:.2f}%" for k, v in vals.items()])
        print(f"  {ds}: {stats}")

