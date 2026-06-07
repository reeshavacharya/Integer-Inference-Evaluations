import argparse
import json
import os
import sys
import torch

# Ensure paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VGG_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if VGG_DIR not in sys.path:
    sys.path.insert(1, VGG_DIR)

import vgg19 as train_mod
from utils import (
    get_quantization_params, 
    get_bias_quantization_params, 
    compute_integer_multiplier, 
    quantize_tensor
)


def _normalize_dataset_key(name: str) -> str:
    key = name.strip().upper().replace("_", "-").replace(" ", "-")
    if key == "CIFR10":
        return "CIFAR10"
    if key == "BRAIN_MRI":
        return "Brain_MRI"
    if key == "MNIST":
        return "MNIST"
    if key == "CIFAR10":
        return "CIFAR10"
    if key == "OCTMNIST":
        return "OCTMNIST"
    if key == "BLOODMNIST":
        return "BloodMNIST"
    if key == "ORGANAMNIST":
        return "OrganAMNIST"
    raise ValueError(f"Unknown dataset: {name}")


def _dataset_spec(dataset: str):
    normalized = _normalize_dataset_key(dataset)
    specs = {
        "MNIST": {
            "dataset": "MNIST",
            "model_path": os.path.join(VGG_DIR, "best_vgg19_mnist.pth"),
            "in_channels": 1,
            "num_classes": 10,
        },
        "CIFAR10": {
            "dataset": "CIFAR10",
            "model_path": os.path.join(VGG_DIR, "best_vgg19_cifar10.pth"),
            "in_channels": 3,
            "num_classes": 10,
        },
        "Brain_MRI": {
            "dataset": "Brain_MRI",
            "model_path": os.path.join(VGG_DIR, "best_vgg19_brain_mri.pth"),
            "in_channels": 1,
            "num_classes": 4,
        },

        "OCTMNIST": {
            "dataset": "OCTMNIST",
            "model_path": os.path.join(VGG_DIR, "best_vgg19_octmnist.pth"),
            "in_channels": 1,
            "num_classes": 4,
        },
        "BloodMNIST": {
            "dataset": "BloodMNIST",
            "model_path": os.path.join(VGG_DIR, "best_vgg19_bloodmnist.pth"),
            "in_channels": 3,
            "num_classes": 8,
        },
        "OrganAMNIST": {
            "dataset": "OrganAMNIST",
            "model_path": os.path.join(VGG_DIR, "best_vgg19_organamnist.pth"),
            "in_channels": 1,
            "num_classes": 11,
        },
    }
    return specs[normalized]


def _infer_dataset_from_filename(model_path: str):
    filename = os.path.basename(model_path).lower()
    if "organ" in filename:
        return "OrganAMNIST"
    if "blood" in filename:
        return "BloodMNIST"
    if "oct" in filename:
        return "OCTMNIST"
    if "mnist" in filename:
        return "MNIST"
    if "cifar10" in filename or "cifr10" in filename:
        return "CIFAR10"
    if "brain_mri" in filename or "brain-mri" in filename:
        return "Brain_MRI"

    raise ValueError(
        "Could not infer dataset from checkpoint filename. "
        "Use a named checkpoint like best_vgg19_mnist.pth or pass a dataset key to --quantize."
    )


def _resolve_quantize_target(quantize_arg: str):
    candidate = quantize_arg.strip()

    # 1) Treat as explicit checkpoint path first.
    explicit_paths = [
        candidate,
        os.path.join(VGG_DIR, candidate),
        os.path.join(THIS_DIR, candidate),
    ]
    for path in explicit_paths:
        if path.lower().endswith(".pth") and os.path.exists(path):
            dataset = _infer_dataset_from_filename(path)
            return dataset, path

    # 2) Treat as dataset key and map to canonical checkpoint path.
    spec = _dataset_spec(candidate)
    model_path = spec["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing checkpoint for {spec['dataset']}: {model_path}. "
            f"Pass a valid .pth path with --quantize or place the checkpoint at this location."
        )
    return spec["dataset"], model_path

def fold_conv_bn_eval(conv, bn):
    """Mathematically folds BatchNorm parameters into Conv2d weights and biases."""
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)
    
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps
    
    multiplier = gamma / torch.sqrt(var + eps)
    w_folded = w * multiplier.view(-1, 1, 1, 1)
    b_folded = beta + (b - mean) * multiplier
    
    return w_folded, b_folded

def process_conv(conv, bn, layer_name, scale_in, activation_ranges):
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)
    
    scale_w, zp_w = get_quantization_params(w_folded, num_bits=8)
    q_w = quantize_tensor(w_folded, scale_w, zp_w, dtype=torch.uint8)

    out_range = activation_ranges[layer_name]
    scale_out = out_range["out_scale"]
    zp_out = out_range["out_zero_point"]

    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_folded, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    return {
        "q_weight": q_w.cpu(), "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0), "shift": int(shift),
        "scale_out": float(scale_out), "zp_out": int(zp_out)
    }, scale_out

def process_fc(fc, layer_name, scale_in, activation_ranges):
    weight_float = fc.weight.detach()
    scale_w, zp_w = get_quantization_params(weight_float, num_bits=8)
    q_w = quantize_tensor(weight_float, scale_w, zp_w, dtype=torch.uint8)

    out_range = activation_ranges[layer_name]
    scale_out = out_range["out_scale"]
    zp_out = out_range["out_zero_point"]

    bias_float = fc.bias.detach() if fc.bias is not None else torch.zeros(fc.out_features)
    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(bias_float, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    return {
        "q_weight": q_w.cpu(), "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0), "shift": int(shift),
        
        # Save exact input boundaries for pooling/routing
        "scale_in": float(scale_in), 
        "zp_in": int(activation_ranges[layer_name]["in_zero_point"]),
        "scale_out": float(scale_out), "zp_out": int(zp_out)
    }, scale_out

def main(quantize_arg: str):
    dataset, model_path = _resolve_quantize_target(quantize_arg)
    spec = _dataset_spec(dataset)
    in_c = spec["in_channels"]
    num_c = spec["num_classes"]

    print(f"Loading FP32 VGG19 for {dataset}...")
    model = train_mod.VGG19(num_classes=num_c, in_channels=in_c)
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Load Calibration Boundaries
    calib_name = f"{dataset.lower().replace(' ', '_').replace('-', '_')}_calibration.json"
    calib_path = os.path.join(VGG_DIR, "calibration", calib_name)
    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"Missing calibration file: {calib_path}. "
            f"Run VGGNet/calibration.py for dataset {dataset} first."
        )
    with open(calib_path, "r") as f:
        payload = json.load(f)

    activation_ranges = payload.get("layers")
    if not isinstance(activation_ranges, dict) or not activation_ranges:
        raise RuntimeError(f"Invalid calibration file format: {calib_path}")

    int8_state = {"meta": {}}

    # Extract global input scale from the very first convolutional layer
    first_conv_idx = next(i for i, m in enumerate(model.features) if isinstance(m, torch.nn.Conv2d))
    first_conv_name = f"features_{first_conv_idx}"
    scale_in = activation_ranges[first_conv_name]["in_scale"]
    int8_state["meta"]["in_scale"] = scale_in
    int8_state["meta"]["in_zp"] = activation_ranges[first_conv_name]["in_zero_point"]

    print("Compiling Features (Conv + BN Folding)...")
    for i, module in enumerate(model.features):
        if isinstance(module, torch.nn.Conv2d):
            layer_name = f"features_{i}"
            bn_module = model.features[i+1] # In VGG, BN is always exactly 1 index after Conv
            
            layer_dict, scale_in = process_conv(
                module, bn_module, layer_name, scale_in, activation_ranges
            )
            int8_state[layer_name] = layer_dict

    print("Compiling Classifier (FC Blocks)...")
    first_fc_idx = next(i for i, m in enumerate(model.classifier) if isinstance(m, torch.nn.Linear))
    first_fc_name = f"classifier_{first_fc_idx}"
    scale_in = activation_ranges[first_fc_name]["in_scale"]

    for i, module in enumerate(model.classifier):
        if isinstance(module, torch.nn.Linear):
            layer_name = f"classifier_{i}"
            layer_dict, scale_in = process_fc(
                module, layer_name, scale_in, activation_ranges
            )
            int8_state[layer_name] = layer_dict

    out_path = os.path.join(VGG_DIR, os.path.basename(model_path).replace(".pth", "_int8.pth"))
    torch.save(int8_state, out_path)
    print(f"[+] Successfully exported integer-only VGG19 model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize",
        type=str,
        required=True,
        help="Dataset key (MNIST, CIFAR10, Brain_MRI) or path to a floating-point .pth checkpoint",
    )
    args = parser.parse_args()
    main(args.quantize)