"""Generate dataset-wide calibration ranges for INT8 inference.

Writes a {dataset}_calibration.json containing per-layer min/max and
quantization parameters (scale, zero_point) for the activation tensors
collected across the entire deterministic 10% test split.

Usage: python3 calibration.py --dataset MNIST
    python3 calibration.py --NIH-CHEST
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

CALIBRATION_DATASETS = [
    "MNIST",
    "CIFAR10",
    "Brain-MRI",
    "NIH-CHEST",
    "OCTMNIST",
    "BloodMNIST",
    "OrganAMNIST",
]


def _load_module(module_name: str, file_path: str, prepend_dir: str):
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


def _get_test_loader(cfg, batch_size: int = 1):
    # Prevent stale globals in resnet18 from interfering
    import resnet18 as train_mod

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
        train_mod.validate_loader_preprocessing(loader, cfg["display"], stage="calibration")
        return loader

    if train_mod.test_loader is not None:
        train_mod.validate_loader_preprocessing(train_mod.test_loader, cfg["display"], stage="calibration")
        return train_mod.test_loader

    raise RuntimeError(f"Could not resolve test loader for dataset: {cfg['display']}")


def main(dataset: str, batch_size: int = 1, out_dir: Optional[str] = None):
    int8_inference = _load_module(
        "resnet_int8_inference",
        os.path.join(INT8_DIR, "inference.py"),
        INT8_DIR,
    )
    int8_utils = _load_module(
        "resnet_int8_utils",
        os.path.join(INT8_DIR, "utils.py"),
        INT8_DIR,
    )
    
    import resnet18 as train_mod

    # Single-model calibration for supported datasets
    cfg = int8_inference._resolve_infer_config(dataset)
    display = cfg["display"]

    print(f"[calib] Dataset: {display} — building test loader (batch_size={batch_size})")
    loader = _get_test_loader(cfg, batch_size=batch_size)

    # Build model and load weights (on CPU; calibration is offline)
    model = cfg["model"]
    state = torch.load(cfg["model_path"], map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # We'll accumulate min/max statistics for each named activation entry
    aggregated = {}

    total = len(loader.dataset)
    print(f"[calib] Running calibration forward passes over {total} test samples...")

    with torch.no_grad():
        for idx, (img, label) in enumerate(loader, 1):
            # clear previous ranges
            int8_inference.activation_ranges.clear()

            # register hooks and run one forward pass
            handles = int8_inference.register_hooks(model)
            _ = model(img)
            for h in handles:
                h.remove()

            # merge activation_ranges into aggregated
            for name, ranges in int8_inference.activation_ranges.items():
                if name not in aggregated:
                    aggregated[name] = {
                        "in_min": float(ranges["in_min"]),
                        "in_max": float(ranges["in_max"]),
                        "out_min": float(ranges["out_min"]),
                        "out_max": float(ranges["out_max"]),
                    }
                else:
                    a = aggregated[name]
                    a["in_min"] = min(a["in_min"], float(ranges["in_min"]))
                    a["in_max"] = max(a["in_max"], float(ranges["in_max"]))
                    a["out_min"] = min(a["out_min"], float(ranges["out_min"]))
                    a["out_max"] = max(a["out_max"], float(ranges["out_max"]))

            if idx % 100 == 0 or idx == total:
                print(f"[calib] Processed {idx}/{total} samples")

    # Now compute quantization params for each aggregated entry
    calib = {}
    for name, a in aggregated.items():
        out_tensor = torch.tensor([a["out_min"], a["out_max"]], dtype=torch.float32)
        in_tensor = torch.tensor([a["in_min"], a["in_max"]], dtype=torch.float32)
        out_scale, out_zp = int8_utils.get_quantization_params(out_tensor, num_bits=8)
        in_scale, in_zp = int8_utils.get_quantization_params(in_tensor, num_bits=8)
        calib[name] = {
            "in_min": a["in_min"],
            "in_max": a["in_max"],
            "in_scale": float(in_scale),
            "in_zero_point": int(in_zp),
            "out_min": a["out_min"],
            "out_max": a["out_max"],
            "out_scale": float(out_scale),
            "out_zero_point": int(out_zp),
        }

    # Default output directory is a `calibration` folder under this module
    out_dir = out_dir or os.path.join(THIS_DIR, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset.lower().replace(' ', '_')}_calibration.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": display, "layers": calib}, f, indent=2)

    print(f"[calib] Saved calibration to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--NIH-CHEST",
        dest="nih_chest",
        action="store_true",
        help="Calibrate the NIH-CHEST model using the custom test split",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to calibrate (e.g. MNIST, CIFAR10, Brain-MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST). If omitted, calibrate all supported datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use while running forward passes (default: 1)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory for the calibration JSON",
    )

    args = parser.parse_args()
    if args.nih_chest:
        args.dataset = "NIH-CHEST"

    if args.dataset is None:
        for ds in CALIBRATION_DATASETS:
            print(f"\n[calib] === Calibrating dataset: {ds} ===")
            main(ds, batch_size=args.batch_size, out_dir=args.out_dir)
    else:
        main(args.dataset, batch_size=args.batch_size, out_dir=args.out_dir)