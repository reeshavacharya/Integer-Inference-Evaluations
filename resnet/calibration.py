"""Generate dataset-wide calibration ranges for INT8 inference.

Writes a {dataset}_calibration.json containing per-layer min/max and
quantization parameters (scale, zero_point) for the activation tensors
collected across the entire deterministic 10% test split.

Usage: python3 calibration.py --dataset CHEST
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

    # Special handling for CHEST: calibrate all 15 per-label models
    if dataset.upper() == "CHEST":
        print("[calib] CHEST dataset detected - calibrating per-label models")
        
        chest_labels = train_mod.get_chest_label_names()
        print(f"[calib] Found {len(chest_labels)} CHEST labels: {chest_labels}")
        
        for label_idx, target_label in enumerate(chest_labels):
            print(f"\n[calib] [{label_idx+1}/{len(chest_labels)}] Calibrating label: {target_label}")
            
            # Get test loader for this specific label
            cfg = int8_inference._resolve_infer_config("CHEST")
            cfg["setup_fn"] = train_mod.setup_CHEST
            
            # Override to get single-label setup
            train_mod.train_loader = None
            train_mod.val_loader = None
            train_mod.test_loader = None
            train_mod.setup_CHEST(batch_size=batch_size, target_label=target_label)
            loader = train_mod.test_loader

            # Build binary model for this label
            from resnet18 import ResNet18Inference as ResNet18Model
            model = ResNet18Model(num_classes=1, in_channels=1)
            
            safe_label = target_label.lower().replace(" ", "_").replace("-", "_").replace("&", "and")
            safe_label = safe_label.replace("__", "_")
            model_path = os.path.join(THIS_DIR, f"best_resnet18_chest_{safe_label}.pth")
            
            if not os.path.exists(model_path):
                print(f"[calib] ERROR: Model not found: {model_path}")
                print(f"[calib] Please train with: python3 resnet18.py --train_data CHEST")
                continue
            
            state = torch.load(model_path, map_location="cpu")
            if list(state.keys())[0].startswith("module."):
                state = {k[7:]: v for k, v in state.items()}
            model.load_state_dict(state)
            model.eval()

            # Aggregate calibration ranges
            aggregated = {}
            total = len(loader.dataset)
            print(f"[calib] Running calibration forward passes over {total} test samples...")

            with torch.no_grad():
                for idx, (img, label) in enumerate(loader, 1):
                    int8_inference.activation_ranges.clear()
                    handles = int8_inference.register_hooks(model)
                    _ = model(img)
                    for h in handles:
                        h.remove()

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

            # Compute quantization parameters
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

            # Save calibration under chest_{label}_calibration.json
            out_dir = out_dir or os.path.join(THIS_DIR, "calibration")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"chest_{safe_label}_calibration.json")
            with open(out_path, "w") as f:
                json.dump({"dataset": f"CHEST-{target_label}", "layers": calib}, f, indent=2)

            print(f"[calib] Saved calibration to: {out_path}")
        
        return

    # Regular single-model calibration for non-CHEST datasets
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
        "--dataset",
        type=str,
        default=None,
        help="Dataset to calibrate (e.g. MNIST, CIFAR10, CHEST, Brain-Cancer, ...). If omitted, calibrate all benchmark datasets.",
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
    if args.dataset is None:
        # Load benchmark dataset list to mirror inference/benchmark behavior
        bench = _load_module(
            "resnet_benchmark", os.path.join(THIS_DIR, "benchmark.py"), THIS_DIR
        )
        datasets = getattr(bench, "BENCHMARK_DATASETS", None)
        if datasets is None:
            raise RuntimeError("Could not locate BENCHMARK_DATASETS in benchmark.py")
        for ds in datasets:
            print(f"\n[calib] === Calibrating dataset: {ds} ===")
            main(ds, batch_size=args.batch_size, out_dir=args.out_dir)
    else:
        main(args.dataset, batch_size=args.batch_size, out_dir=args.out_dir)
# This calibraiton.py should take one argument --dataset,
# The valid options are the same as the ones used in inference.py for the flag --infer
# The main purpose of this file is to calibration on the entire test split of the specified dataset
# currently, for INT8/inference.py, a random data from the test split is taken and calibration is done for that specific data only. 
# This file should generate a {dataset}_calibration.json file that contains the min, max, scale and zero point for each tensor. 
# that way, when running integer inference, we can load the calibration file and use the same quantization parameters for all the data in the test split, which should give us a more accurate evaluation of the model's performance in INT8.
# Also, this process is meant to be done offline so it doesnt impact the inference time of integer inferece.

# Very IMPORTANT:
# calibraiton should be done in the same way as done in inference.py for INT8, meaning that the same quantization scheme should be used, and the same data preprocessing should be applied to the data before feeding it to the model for calibration.
# But the one done in inference.py is done for only one data, and the one done in this file should be done for the entire test split of the specified dataset.