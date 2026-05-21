import argparse
import os
import torch
import sys

# Ensure module paths are available
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if RESNET_DIR not in sys.path:
    sys.path.insert(1, RESNET_DIR)

from inference import _resolve_infer_config, fold_conv_bn_eval, load_calibration_ranges, activation_ranges
from utils import get_quantization_params, get_bias_quantization_params, compute_integer_multiplier, quantize_tensor


def _activation_label(base_name, activation):
    return f"{base_name}_{activation.lower()}"

def process_conv(conv, bn, layer_name, scale_in):
    """Folds, scales, and quantizes a Conv+BN block."""
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)
    
    scale_w, zp_w = get_quantization_params(w_folded, num_bits=8)
    q_w = quantize_tensor(w_folded, scale_w, zp_w, dtype=torch.uint8)

    out_range = activation_ranges[layer_name]
    
    # 1. The Conv block mathematically targets the IN_SCALE of the activation
    conv_scale_out = out_range["in_scale"]
    conv_zp_out = out_range["in_zero_point"]
    
    # 2. The Activation outputs to its own OUT_SCALE
    act_scale_out = out_range["out_scale"]
    act_zp_out = out_range["out_zero_point"]

    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_folded, scale_bias, zp_bias, dtype=torch.int32)

    # Shift multiplier targets the Conv scale
    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, conv_scale_out)

    return {
        "q_weight": q_w.cpu(), "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0), "shift": int(shift),
        "conv_scale_out": float(conv_scale_out), "conv_zp_out": int(conv_zp_out),
        "act_scale_out": float(act_scale_out), "act_zp_out": int(act_zp_out),
        "stride": conv.stride[0], "padding": conv.padding[0]
    }, act_scale_out # Return the activation scale to feed the next layer!

def process_fc(fc, layer_name, scale_in):
    """Scales and quantizes the Fully Connected layer."""
    weight_float = fc.weight.detach()
    scale_w, zp_w = get_quantization_params(weight_float, num_bits=8)
    q_w = quantize_tensor(weight_float, scale_w, zp_w, dtype=torch.uint8)

    out_range = activation_ranges[layer_name]
    scale_out = out_range["out_scale"]
    zp_out = out_range["out_zero_point"]

    bias_float = fc.bias.detach()
    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(bias_float, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    return {
        "q_weight": q_w.cpu(), "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0), "shift": int(shift),
        "scale_out": float(scale_out), "zp_out": int(zp_out),
        "scale_in": float(scale_in),
        "zp_in": int(activation_ranges[layer_name]["in_zero_point"])
    }, scale_out

def main(model_path, activation="relu"):
    # 1. Infer the dataset config from the filename
    filename = os.path.basename(model_path).lower()
    # Check specific MedMNIST names first (they contain 'mnist' too)
    if "organ" in filename or "organamnist" in filename:
        dataset = "ORGANAMNIST"
    elif "blood" in filename or "bloodmnist" in filename:
        dataset = "BLOODMNIST"
    elif "oct" in filename or "octmnist" in filename:
        dataset = "OCTMNIST"
    elif "pneumonia" in filename or "pneumoniamnist" in filename:
        dataset = "PNEUMONIAMNIST"
    elif "cifar" in filename or "cifar10" in filename:
        dataset = "CIFAR10"
    elif "brain" in filename or "brain_mri" in filename:
        dataset = "BRAIN-MRI"
    elif "chest" in filename or "nih" in filename:
        dataset = "NIH-CHEST"
    elif "mnist" in filename:
        dataset = "MNIST"
    else:
        raise ValueError("Could not infer dataset from filename. Ensure dataset name is in the .pth file.")

    cfg = _resolve_infer_config(dataset, activation)
    model = cfg["model"]
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    load_calibration_ranges(cfg["display"], activation)

    # 2. Build the pure Integer State Dictionary
    int8_state = {"meta": {}}
    scale_in = activation_ranges["conv1"]["in_scale"]
    int8_state["meta"]["in_scale"] = scale_in
    int8_state["meta"]["in_zp"] = activation_ranges["conv1"]["in_zero_point"]

    print(f"Quantizing Model: {filename}...")
    
    # Process initial Conv1
    int8_state["conv1"], s_out = process_conv(model.conv1, model.bn1, _activation_label("conv1", activation), scale_in)

    # Process all Residual Blocks
    for layer_idx, stage in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for block_idx, block in enumerate(stage):
            prefix = f"layer{layer_idx}_block{block_idx}"
            block_data = {}
            
            block_data["conv1"], s_out1 = process_conv(block.conv1, block.bn1, _activation_label(f"{prefix}_conv1", activation), s_out)
            block_data["conv2"], s_out2 = process_conv(block.conv2, block.bn2, f"{prefix}_conv2_out", s_out1)
            
            if not isinstance(block.shortcut, torch.nn.Identity):
                block_data["shortcut"], _ = process_conv(block.shortcut[0], block.shortcut[1], f"{prefix}_shortcut_out", s_out)

            out_range = activation_ranges[_activation_label(f"{prefix}_out", activation)]
            block_data["add"] = {
                "conv_scale_out": out_range["in_scale"],
                "conv_zp_out": out_range["in_zero_point"],
                "act_scale_out": out_range["out_scale"],
                "act_zp_out": out_range["out_zero_point"]
            }
            int8_state[prefix] = block_data
            s_out = out_range["out_scale"]

    # Process Final FC Layer
    fc_in_scale = activation_ranges["fc"]["in_scale"]
    int8_state["fc"], _ = process_fc(model.fc, "fc", fc_in_scale)

    # 3. Save Integer Weights
    # Extract base filename without extension and build int8 version with activation
    base_filename = os.path.basename(model_path).replace(".pth", "")
    # Ensure activation is in the filename
    if f"_{activation}_" not in base_filename:
        # If activation isn't already in the name, add it before _int8
        base_filename = base_filename.replace("_int8", "").replace(".pth", "")
        out_filename = f"{base_filename}_{activation}_int8.pth"
    else:
        out_filename = f"{base_filename}_int8.pth"
    
    out_path = os.path.join(RESNET_DIR, out_filename)
    torch.save(int8_state, out_path)
    print(f"[+] Successfully exported integer-only model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", type=str, required=True, help="Path to the floating-point .pth model")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function the model was trained with"
    )
    args = parser.parse_args()
    main(args.quantize, activation=args.activation)