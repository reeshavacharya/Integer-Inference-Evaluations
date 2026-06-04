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
    "Brain_MRI",
    "NIH-CHEST",
    "OCTMNIST",
    "BloodMNIST",
    "OrganAMNIST",
    "PneumoniaMNIST",
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


def _sample_loader_dataset(loader: torch.utils.data.DataLoader, max_samples: int = 1000, seed: int = 42):
    dataset = loader.dataset
    sample_count = min(max_samples, len(dataset))
    if sample_count == len(dataset):
        return loader

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    sampled_dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        sampled_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
    )


def _get_train_loader(cfg, batch_size: int = 1):
    # Prevent stale globals in resnet18 from interfering
    import resnet18 as train_mod

    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None

    setup_result = cfg["setup_fn"](batch_size=batch_size)

    if (
        isinstance(setup_result, tuple)
        and len(setup_result) >= 3
        and hasattr(setup_result[0], "dataset")
    ):
        loader = _sample_loader_dataset(setup_result[0], max_samples=1000)
        return loader

    if train_mod.train_loader is not None:
        sampled_loader = _sample_loader_dataset(train_mod.train_loader, max_samples=1000)
        train_mod.validate_loader_preprocessing(sampled_loader, cfg["display"], stage="calibration")
        return sampled_loader

    raise RuntimeError(f"Could not resolve train loader for dataset: {cfg['display']}")


# -----------------------------------
# Local Calibration Hooks (GELU10 Aware)
# -----------------------------------
activation_ranges = {}


def _activation_label(base_name: str, activation: str) -> str:
    return f"{base_name}_{activation.lower()}"

def register_hooks(model, activation: str):
    handles = []
    activation_ranges.clear()

    def get_hook(name):
        def hook(module, input, output):
            in_tensor = input[0].detach()
            out_tensor = output.detach()
            
            # --- NVIDIA GELU10 CLIPPING FOR PTQ ---
            if isinstance(module, torch.nn.GELU):
                out_tensor = torch.clamp(out_tensor, min=-10.0, max=10.0)

            # --- LeakyReLU10 CLIPPING FOR PTQ ---
            elif isinstance(module, torch.nn.LeakyReLU):
                out_tensor = torch.clamp(out_tensor, min=-50.0, max=50.0)

            if name not in activation_ranges:
                activation_ranges[name] = {
                    "in_min": in_tensor.min().item(),
                    "in_max": in_tensor.max().item(),
                    "out_min": out_tensor.min().item(),
                    "out_max": out_tensor.max().item(),
                }
            else:
                activation_ranges[name]["in_min"] = min(activation_ranges[name]["in_min"], in_tensor.min().item())
                activation_ranges[name]["in_max"] = max(activation_ranges[name]["in_max"], in_tensor.max().item())
                activation_ranges[name]["out_min"] = min(activation_ranges[name]["out_min"], out_tensor.min().item())
                activation_ranges[name]["out_max"] = max(activation_ranges[name]["out_max"], out_tensor.max().item())
        return hook

    # 1. Hook conv1 to capture the raw input image bounds
    handles.append(model.conv1.register_forward_hook(get_hook("conv1")))

    # 2. Hook initial activation
    handles.append(model.activation.register_forward_hook(get_hook(_activation_label("conv1", activation))))

    # 3. Hook the distinct outputs of every block stage
    for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for block_idx, block in enumerate(layer):
            prefix = f"layer{layer_idx}_block{block_idx}"
            
            handles.append(block.activation1.register_forward_hook(get_hook(_activation_label(f"{prefix}_conv1", activation))))
            handles.append(block.bn2.register_forward_hook(get_hook(f"{prefix}_conv2_out")))
            handles.append(block.shortcut.register_forward_hook(get_hook(f"{prefix}_shortcut_out")))
            handles.append(block.activation2.register_forward_hook(get_hook(_activation_label(f"{prefix}_out", activation))))

    # 4. Final FC
    handles.append(model.fc.register_forward_hook(get_hook("fc")))

    return handles

def main(dataset: str, batch_size: int = 1, out_dir: Optional[str] = None, activation: str = "relu"):
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
    cfg = int8_inference._resolve_infer_config(dataset, activation)
    display = cfg["display"]

    print(f"[calib] Dataset: {display} — building test loader (batch_size={batch_size})")
    loader = _get_train_loader(cfg, batch_size=batch_size)

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
    print(f"[calib] Running calibration forward passes over {total} training samples...")

    with torch.no_grad():
        for idx, (img, label) in enumerate(loader, 1):
            # clear previous ranges
            activation_ranges.clear()

            # register hooks and run one forward pass
            handles = register_hooks(model, activation)
            _ = model(img)
            for h in handles:
                h.remove()

            # merge activation_ranges into aggregated
            for name, ranges in activation_ranges.items():
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
    dataset_slug = dataset.lower().replace(" ", "_").replace("-", "_")
    out_path = os.path.join(out_dir, f"{dataset_slug}_{activation}_calibration.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": display, "activation": activation, "layers": calib}, f, indent=2)

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
        help="Dataset to calibrate (e.g. MNIST, CIFAR10, Brain_MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST). If omitted, calibrate all supported datasets.",
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
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function the model was trained with"
    )
    args = parser.parse_args()
    if args.nih_chest:
        args.dataset = "NIH-CHEST"

    if args.dataset is None:
        for ds in CALIBRATION_DATASETS:
            print(f"\n[calib] === Calibrating dataset: {ds} ===")
            main(ds, batch_size=args.batch_size, out_dir=args.out_dir, activation=args.activation)
    else:
        main(args.dataset, batch_size=args.batch_size, out_dir=args.out_dir, activation=args.activation)