import argparse
import json
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LENET_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if LENET_DIR not in sys.path:
    sys.path.insert(1, LENET_DIR)

import lenet5 as train_mod
from utils import (
    compute_integer_multiplier,
    get_bias_quantization_params,
    get_quantization_params,
    quantize_tensor,
)


def _load_model_module():
    from inference import (
        _resolve_infer_config,
        load_calibration_ranges,
        activation_ranges,
    )

    return _resolve_infer_config, load_calibration_ranges, activation_ranges

WEIGHT_BITS = 8


_resolve_infer_config, load_calibration_ranges, activation_ranges = _load_model_module()


def _normalize_dataset_key(name: str) -> str:
    key = name.strip().upper().replace("_", "-").replace(" ", "-")
    if key == "CIFR10":
        return "CIFAR10"
    if key == "OCTMNIST":
        return "OCTMNIST"
    if key == "BLOODMNIST":
        return "BloodMNIST"
    if key == "ORGANAMNIST":
        return "OrganAMNIST"
    if key == "PNEUMONIAMNIST":
        return "PneumoniaMNIST"
    if key == "BRAIN_MRI":
        return "Brain_MRI"
    if key == "NIH-CHEST":
        return "NIH-CHEST"
    if key == "MNIST":
        return "MNIST"
    if key == "CIFAR10":
        return "CIFAR10"
    raise ValueError(f"Unknown dataset: {name}")

def _activation_label(base_name: str, activation: str) -> str:
    return f"{base_name}_{activation.lower()}"


def _infer_dataset_from_filename(model_path: str) -> str:
    filename = os.path.basename(model_path).lower()
    if "organ" in filename or "organamnist" in filename:
        return "OrganAMNIST"
    if "blood" in filename or "bloodmnist" in filename:
        return "BloodMNIST"
    if "oct" in filename or "octmnist" in filename:
        return "OCTMNIST"
    if "pneumonia" in filename or "pneumoniamnist" in filename:
        return "PneumoniaMNIST"
    if "chest" in filename or "nih" in filename:
        return "NIH-CHEST"
    if "mnist" in filename:
        return "MNIST"
    if "cifar10" in filename or "cifr10" in filename:
        return "CIFAR10"
    if "brain_mri" in filename or "brain-mri" in filename:
        return "Brain_MRI"
    if "chest" in filename or "nih" in filename:
        return "NIH-CHEST"
    raise ValueError(
        "Could not infer dataset from checkpoint filename. Pass a dataset key or a checkpoint path."
    )


def _resolve_quantize_target(quantize_arg: str, activation: str):
    candidate = quantize_arg.strip()

    explicit_paths = [
        candidate,
        os.path.join(LENET_DIR, candidate),
        os.path.join(THIS_DIR, candidate),
    ]
    for path in explicit_paths:
        if path.lower().endswith(".pth") and os.path.exists(path):
            return _infer_dataset_from_filename(path), path

    dataset = _normalize_dataset_key(candidate)
    cfg = _resolve_infer_config(dataset, activation)
    model_path = os.path.join(LENET_DIR, cfg["model_path"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing checkpoint for {cfg['display']}: {model_path}. "
            f"Train the model first or pass an explicit checkpoint path."
        )
    return cfg["display"], model_path


def fold_conv_bn_eval(conv, bn):
    w = conv.weight.detach()
    b = (
        conv.bias.detach()
        if conv.bias is not None
        else torch.zeros(conv.out_channels, device=w.device)
    )

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    multiplier = gamma / torch.sqrt(var + eps)
    w_folded = w * multiplier.view(-1, 1, 1, 1)
    b_folded = beta + (b - mean) * multiplier
    return w_folded, b_folded


def generate_layer_gelu_lut(in_scale, in_zp, out_scale, out_zp):
    """
    Generates a localized integer-to-integer LUT for GELU mapped between [-10.0, 10.0].
    """
    in_zp_32 = torch.tensor(in_zp, dtype=torch.int32)
    q_min_bound = torch.round(torch.tensor(-10.0 / in_scale, dtype=torch.float32)).to(torch.int32) + in_zp_32
    q_max_bound = torch.round(torch.tensor(10.0 / in_scale, dtype=torch.float32)).to(torch.int32) + in_zp_32
    
    q_inputs = torch.arange(q_min_bound.item(), q_max_bound.item() + 1, dtype=torch.float64)
    
    f_inputs = in_scale * (q_inputs - in_zp)
    f_act = torch.nn.functional.gelu(f_inputs)
    f_act = torch.clamp(f_act, -10.0, 10.0)
    
    q_outputs = torch.round(f_act / out_scale) + out_zp
    q_outputs = torch.clamp(q_outputs, 0, 4294967295).to(torch.uint32)
    
    return q_min_bound, q_max_bound, q_outputs

def process_conv(conv, bn, layer_name, scale_in, activation):
    if bn is not None:
        w_float, b_float = fold_conv_bn_eval(conv, bn)
    else:
        w_float = conv.weight.detach()
        b_float = (
            conv.bias.detach()
            if conv.bias is not None
            else torch.zeros(conv.out_channels, device=w_float.device)
        )

    scale_w, zp_w = get_quantization_params(w_float, num_bits=WEIGHT_BITS)
    q_w = quantize_tensor(w_float, scale_w, zp_w, dtype=torch.int32)

    out_range = activation_ranges[layer_name]
    conv_scale_out = out_range["in_scale"]
    conv_zp_out = out_range["in_zero_point"]
    act_scale_out = out_range["out_scale"]
    act_zp_out = out_range["out_zero_point"]

    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_float, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, conv_scale_out)

    export_dict = {
        "q_weight": q_w.cpu(),
        "zp_w": torch.tensor(zp_w, dtype=torch.int32),
        "q_bias": q_bias.cpu(),
        "q_M0": q_M0.to(torch.int32),
        "shift": shift.to(torch.int32),
        "conv_scale_out": float(conv_scale_out),
        "conv_zp_out": torch.tensor(conv_zp_out, dtype=torch.int32),
        "act_scale_out": float(act_scale_out),
        "act_zp_out": torch.tensor(act_zp_out, dtype=torch.int32),
    }

    if "out_scale" in out_range and activation == "gelu":
        q_min, q_max, lut = generate_layer_gelu_lut(
            in_scale=conv_scale_out, 
            in_zp=conv_zp_out, 
            out_scale=act_scale_out, 
            out_zp=act_zp_out
        )
        export_dict["gelu_q_min"] = q_min
        export_dict["gelu_q_max"] = q_max
        export_dict["gelu_lut"] = lut.cpu()

    return export_dict, act_scale_out


def process_fc(fc, layer_name, scale_in, activation):
    weight_float = fc.weight.detach()
    scale_w, zp_w = get_quantization_params(weight_float, num_bits=WEIGHT_BITS)
    q_w = quantize_tensor(weight_float, scale_w, zp_w, dtype=torch.int32)

    out_range = activation_ranges[layer_name]
    if "out_scale" in out_range:
        act_scale_out = out_range["out_scale"]
        act_zp_out = out_range["out_zero_point"]
        conv_scale_out = out_range["in_scale"]
        conv_zp_out = out_range["in_zero_point"]
    else:
        # For the final FC, there is no activation after it, so in_scale is the final scale
        act_scale_out = out_range["in_scale"]
        act_zp_out = out_range["in_zero_point"]
        conv_scale_out = out_range["in_scale"]
        conv_zp_out = out_range["in_zero_point"]

    bias_float = fc.bias.detach() if fc.bias is not None else torch.zeros(fc.out_features)
    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(bias_float, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, conv_scale_out)

    export_dict = {
        "q_weight": q_w.cpu(),
        "zp_w": torch.tensor(zp_w, dtype=torch.int32),
        "q_bias": q_bias.cpu(),
        "q_M0": q_M0.to(torch.int32),
        "shift": shift.to(torch.int32),
        "conv_scale_out": float(conv_scale_out),
        "conv_zp_out": torch.tensor(conv_zp_out, dtype=torch.int32),
        "act_scale_out": float(act_scale_out),
        "act_zp_out": torch.tensor(act_zp_out, dtype=torch.int32),
        "scale_in": float(scale_in),
        "zp_in": torch.tensor(activation_ranges[layer_name]["in_zero_point"], dtype=torch.int32),
    }

    if "out_scale" in out_range and activation == "gelu":
        q_min, q_max, lut = generate_layer_gelu_lut(
            in_scale=conv_scale_out, 
            in_zp=conv_zp_out, 
            out_scale=act_scale_out, 
            out_zp=act_zp_out
        )
        export_dict["gelu_q_min"] = q_min
        export_dict["gelu_q_max"] = q_max
        export_dict["gelu_lut"] = lut.cpu()

    return export_dict, act_scale_out


def main(quantize_arg: str, activation: str):
    dataset, model_path_template = _resolve_quantize_target(quantize_arg, activation)
    cfg = _resolve_infer_config(dataset, activation)
    model = cfg["model"]
    model_path = model_path_template

    print(f"Loading FP32 LeNet for {dataset}...")
    state = torch.load(model_path, map_location="cpu")
    if len(state) > 0 and list(state.keys())[0].startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()

    load_calibration_ranges(cfg["display"], activation)

    int32_state = {"meta": {}}
    scale_in = activation_ranges["conv1"]["in_scale"]
    int32_state["meta"]["in_scale"] = scale_in
    int32_state["meta"]["in_zp"] = torch.tensor(activation_ranges["conv1"]["in_zero_point"], dtype=torch.int32)

    if isinstance(model, train_mod.MedicalLeNet):
        conv1, bn1 = model.features[0], model.features[1]
        conv2, bn2 = model.features[4], model.features[5]
        fc1 = model.classifier[1]
        fc2 = model.classifier[4]
        fc3 = model.classifier[7]
    else:
        conv1, bn1 = model.features[0], None
        conv2, bn2 = model.features[3], None
        fc1 = model.classifier[1]
        fc2 = model.classifier[3]
        fc3 = model.classifier[5]

    int32_state["conv1"], s_out = process_conv(conv1, bn1, _activation_label("conv1", activation), scale_in, activation)
    int32_state["conv2"], s_out = process_conv(conv2, bn2, _activation_label("conv2", activation), s_out, activation)
    int32_state["fc1"], s_out = process_fc(fc1, _activation_label("fc1", activation), s_out, activation)
    int32_state["fc2"], s_out = process_fc(fc2, _activation_label("fc2", activation), s_out, activation)
    int32_state["fc3"], _ = process_fc(fc3, "fc3", s_out, activation)

    out_path = os.path.join(LENET_DIR, os.path.basename(model_path).replace(".pth", "_int32.pth"))
    torch.save(int32_state, out_path)
    print(f"[+] Successfully exported integer-only LeNet model to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize",
        type=str,
        required=True,
        help=(
            "Dataset key (MNIST, CIFAR10, Brain_MRI, NIH-CHEST, OCTMNIST, "
            "BloodMNIST, OrganAMNIST, PneumoniaMNIST) or a checkpoint path"
        ),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function to use",
    )
    args = parser.parse_args()
    main(args.quantize, args.activation)
