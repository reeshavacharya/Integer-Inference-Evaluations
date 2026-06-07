import argparse
import json
import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VGG_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path: sys.path.insert(0, THIS_DIR)
if VGG_DIR not in sys.path: sys.path.insert(1, VGG_DIR)

from utils import (
    get_quantization_params, 
    get_bias_quantization_params, 
    compute_integer_multiplier, 
    quantize_tensor,
    generate_layer_gelu_lut
)

WEIGHT_BITS = 8

def _activation_label(base_name: str, activation: str) -> str:
    return f"{base_name}_{activation.lower()}"

import vgg19 as train_mod
from vgg19 import VGG19

def _normalize_dataset_name(dataset_name: str) -> str:
	key = dataset_name.strip().upper().replace(" ", "-")
	if key == "MNIST":
		return "MNIST"
	if key == "CIFR10":
		return "CIFAR10"
	if key == "BRAIN_MRI":
		return "Brain_MRI"
	if key == "OCTMNIST":
		return "OCTMNIST"
	if key == "ORGANAMNIST":
		return "OrganAMNIST"
	if key == "BLOODMNIST":
		return "BloodMNIST"
	if key == "PNEUMONIAMNIST":
		return "PneumoniaMNIST"
	raise ValueError(f"Unknown dataset: {dataset_name}")

def _resolve_infer_config(infer_data: str):
    name = _normalize_dataset_name(infer_data)

    if name == "MNIST":
        return {
            "display": "MNIST",
            "setup_fn": train_mod.setup_MNIST,
            "model": VGG19(num_classes=10, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_mnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }
    if name == "CIFAR10":
        return {
            "display": "CIFAR10",
            "setup_fn": train_mod.setup_CIFAR10,
            "model": VGG19(num_classes=10, in_channels=3),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_cifar10.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }
    if name == "BRAIN_MRI":
        return {
            "display": "Brain_MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": VGG19(num_classes=4, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_brain_mri.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }
    if name == "OCTMNIST":
        return {
            "display": "OCTMNIST",
            "setup_fn": train_mod.setup_OCTMNIST,
            "model": VGG19(num_classes=4, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_octmnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }
    if name == "BLOODMNIST":
        return {
            "display": "BloodMNIST",
            "setup_fn": train_mod.setup_BloodMNIST,
            "model": VGG19(num_classes=8, in_channels=3),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_bloodmnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }
    if name == "ORGANAMNIST":
        return {
            "display": "OrganAMNIST",
            "setup_fn": train_mod.setup_OrganAMNIST,
            "model": VGG19(num_classes=11, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_organamnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }
    if name == "PNEUMONIAMNIST":
        return {
            "display": "PneumoniaMNIST",
            "setup_fn": train_mod.setup_PneumoniaMNIST,
            "model": VGG19(num_classes=2, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_pneumoniamnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    raise ValueError(f"Unknown dataset config: {name}")

def fold_conv_bn_eval(conv, bn):
    """Folds BatchNorm parameters into Conv2d weights and biases dynamically."""
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
    
    scale_w, zp_w = get_quantization_params(w_folded, num_bits=WEIGHT_BITS)
    q_w = quantize_tensor(w_folded, scale_w, zp_w, dtype=torch.uint32, num_bits=WEIGHT_BITS)

    out_range = activation_ranges[layer_name]
    scale_out = out_range["out_scale"]
    zp_out = out_range["out_zero_point"]

    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_folded, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    layer_data = {
        "q_weight": q_w.cpu(), "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0), "shift": int(shift),
        "scale_out": float(scale_out), "zp_out": int(zp_out)
    }
    return layer_data, scale_out

def process_fc(fc, layer_name, scale_in, activation_ranges):
    w_float = fc.weight.detach()
    b_float = fc.bias.detach() if fc.bias is not None else torch.zeros(fc.out_features)
    
    scale_w, zp_w = get_quantization_params(w_float, num_bits=WEIGHT_BITS)
    q_w = quantize_tensor(w_float, scale_w, zp_w, dtype=torch.uint32, num_bits=WEIGHT_BITS)

    out_range = activation_ranges[layer_name]
    scale_out = out_range["out_scale"]
    zp_out = out_range["out_zero_point"]

    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_float, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    layer_data = {
        "q_weight": q_w.cpu(), "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0), "shift": int(shift),
        "scale_in": float(scale_in), "zp_in": int(scale_in),
        "scale_out": float(scale_out), "zp_out": int(zp_out)
    }
    return layer_data, scale_out

def main(quantize_target, activation):
    cfg = _resolve_infer_config(quantize_target)
    dataset = cfg["display"]
    
    # Overwrite the FP32 model path logically to match our dynamic standard
    model_path = os.path.join(VGG_DIR, f"best_vgg19_{activation}_{dataset.lower().replace(' ', '_').replace('-', '_')}.pth")
    cfg["model_path"] = model_path
    cfg["model"].activation = activation

    print(f"Quantizing VGG19 Model: {dataset} | Activation: {activation} to INT32 ...")

    model = cfg["model"]
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    calib_file = f"{dataset.lower().replace(' ', '_').replace('-', '_')}_{activation}_calibration.json"
    calib_path = os.path.join(VGG_DIR, "calibration", calib_file)
    with open(calib_path, "r") as f:
        calib_data = json.load(f)
    activation_ranges = calib_data.get("layers", calib_data)

    int32_state = {"meta": {}}
    first_act_name = "features_0"
    scale_in = activation_ranges[first_act_name]["in_scale"]
    int32_state["meta"]["in_scale"] = scale_in
    int32_state["meta"]["in_zp"] = activation_ranges[first_act_name]["in_zero_point"]

    # --- PROCESS CONVOLUTIONS ---
    for i, module in enumerate(model.features):
        if isinstance(module, torch.nn.Conv2d):
            layer_name = f"features_{i}"
            bn_module = model.features[i+1]
            layer_dict, scale_in = process_conv(module, bn_module, layer_name, scale_in, activation_ranges)
            int32_state[layer_name] = layer_dict
            
            if activation == "gelu":
                _, _, lut = generate_layer_gelu_lut(layer_dict["scale_out"], layer_dict["zp_out"], layer_dict["scale_out"], layer_dict["zp_out"])
                int32_state[f"{layer_name}_gelu_lut"] = lut

    # --- PROCESS CLASSIFIER ---
    first_fc_idx = next(i for i, m in enumerate(model.classifier) if isinstance(m, torch.nn.Linear))
    first_fc_name = f"classifier_{first_fc_idx}"
    scale_in = activation_ranges[first_fc_name]["in_scale"]

    for i, module in enumerate(model.classifier):
        if isinstance(module, torch.nn.Linear):
            layer_name = f"classifier_{i}"
            layer_dict, scale_in = process_fc(module, layer_name, scale_in, activation_ranges)
            int32_state[layer_name] = layer_dict
            
            # The last layer doesn't have an activation after it, so we skip LUT generation for it
            if activation == "gelu" and i != len(model.classifier) - 1:
                _, _, lut = generate_layer_gelu_lut(layer_dict["scale_out"], layer_dict["zp_out"], layer_dict["scale_out"], layer_dict["zp_out"])
                int32_state[f"{layer_name}_gelu_lut"] = lut

    out_path = os.path.join(VGG_DIR, os.path.basename(model_path).replace(".pth", "_int32.pth"))
    torch.save(int32_state, out_path)
    print(f"[+] Successfully exported integer-only VGG19 model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", type=str, required=True)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "leaky_relu"])
    args = parser.parse_args()
    main(args.quantize, args.activation)