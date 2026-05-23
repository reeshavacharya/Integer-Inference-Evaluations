import argparse
import os
import torch
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if RESNET_DIR not in sys.path:
    sys.path.insert(1, RESNET_DIR)

from inference import (
    _resolve_infer_config,
    fold_conv_bn_eval,
    load_calibration_ranges,
    activation_ranges,
)
from utils import (
    get_quantization_params,
    get_bias_quantization_params,
    compute_integer_multiplier,
    quantize_tensor,
)

# ---------------------------------------------------------------------------
# DESIGN NOTE: Why num_bits=16 for weights?
# ---------------------------------------------------------------------------
# The INT32 pipeline uses 32-bit ACCUMULATORS, not 32-bit weights.
#
# If we quantize weights to num_bits=32, scale_w ~ 1e-9 (the full float32
# range divided by 2^32). This makes M = scale_w * scale_in / scale_out
# extremely small, so frexp returns a large negative exponent, and
# shift = -exponent ends up at 45-60. downscale_and_cast then right-shifts
# the accumulator by (31 + shift) ~ 76-91 bits, discarding almost all
# information -- worse than INT8.
#
# With num_bits=16: scale_w ~ 1e-5, shift stays in the range 10-25 (same
# healthy range as INT8), and the 32-bit accumulator retains full precision.
# Weights still live in uint32 tensors (upper 16 bits are zero), so the
# storage format and safe_integer_conv2d API are unchanged.
#
# This is also required for safe_integer_conv2d's overflow guarantee:
# with 16-bit weights and 16-bit activations, the P_hh partial product
# (shifted left 32) is at most 2^32 * num_terms ~ 2^44, safe in int64.
# With 32-bit weights the P_hh << 32 step would reach ~2^76, overflowing.
# ---------------------------------------------------------------------------

WEIGHT_BITS = 16


def _activation_label(base_name, activation):
    return f"{base_name}_{activation.lower()}"


def process_conv(conv, bn, layer_name, scale_in):
    """Folds, quantizes (16-bit weights), and exports a Conv+BN block."""
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)

    # 16-bit weight quantization: scale is ~1e-5, not ~1e-9
    scale_w, zp_w = get_quantization_params(w_folded, num_bits=WEIGHT_BITS)
    q_w = quantize_tensor(w_folded, scale_w, zp_w, dtype=torch.uint32)
    # q_w values are 0..65535, stored in uint32 (upper 16 bits always zero)

    out_range = activation_ranges[layer_name]
    conv_scale_out = out_range["in_scale"]
    conv_zp_out = out_range["in_zero_point"]
    act_scale_out = out_range["out_scale"]
    act_zp_out = out_range["out_zero_point"]

    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_folded, scale_bias, zp_bias, dtype=torch.int64)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, conv_scale_out)

    return {
        "q_weight": q_w.cpu(),
        "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0),
        "shift": int(shift),
        "conv_scale_out": float(conv_scale_out),
        "conv_zp_out": int(conv_zp_out),
        "act_scale_out": float(act_scale_out),
        "act_zp_out": int(act_zp_out),
        "stride": conv.stride[0],
        "padding": conv.padding[0],
    }, act_scale_out


def process_fc(fc, layer_name, scale_in):
    weight_float = fc.weight.detach()
    scale_w, zp_w = get_quantization_params(weight_float, num_bits=WEIGHT_BITS)
    q_w = quantize_tensor(weight_float, scale_w, zp_w, dtype=torch.uint32)

    out_range = activation_ranges[layer_name]
    scale_out = out_range["out_scale"]
    zp_out = out_range["out_zero_point"]

    bias_float = fc.bias.detach()
    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(bias_float, scale_bias, zp_bias, dtype=torch.int64)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    return {
        "q_weight": q_w.cpu(),
        "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0),
        "shift": int(shift),
        "scale_out": float(scale_out),
        "zp_out": int(zp_out),
        "scale_in": float(scale_in),
        "zp_in": int(activation_ranges[layer_name]["in_zero_point"]),
    }, scale_out


def main(model_path, activation="relu"):
    filename = os.path.basename(model_path).lower()
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
        raise ValueError("Could not infer dataset from filename.")

    cfg = _resolve_infer_config(dataset, activation)
    model = cfg["model"]
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    load_calibration_ranges(cfg["display"], activation)

    int32_state = {"meta": {}}
    scale_in = activation_ranges["conv1"]["in_scale"]
    int32_state["meta"]["in_scale"] = scale_in
    int32_state["meta"]["in_zp"] = activation_ranges["conv1"]["in_zero_point"]

    print(f"Quantizing Model to INT32 (WEIGHT_BITS={WEIGHT_BITS}): {filename}...")

    int32_state["conv1"], s_out = process_conv(
        model.conv1, model.bn1, _activation_label("conv1", activation), scale_in
    )

    for layer_idx, stage in enumerate(
        [model.layer1, model.layer2, model.layer3, model.layer4], 1
    ):
        for block_idx, block in enumerate(stage):
            prefix = f"layer{layer_idx}_block{block_idx}"
            block_data = {}

            block_data["conv1"], s_out1 = process_conv(
                block.conv1,
                block.bn1,
                _activation_label(f"{prefix}_conv1", activation),
                s_out,
            )
            block_data["conv2"], s_out2 = process_conv(
                block.conv2, block.bn2, f"{prefix}_conv2_out", s_out1
            )

            if not isinstance(block.shortcut, torch.nn.Identity):
                block_data["shortcut"], _ = process_conv(
                    block.shortcut[0],
                    block.shortcut[1],
                    f"{prefix}_shortcut_out",
                    s_out,
                )

            out_range = activation_ranges[
                _activation_label(f"{prefix}_out", activation)
            ]
            block_data["add"] = {
                "conv_scale_out": out_range["in_scale"],
                "conv_zp_out": out_range["in_zero_point"],
                "act_scale_out": out_range["out_scale"],
                "act_zp_out": out_range["out_zero_point"],
            }
            int32_state[prefix] = block_data
            s_out = out_range["out_scale"]

    fc_in_scale = activation_ranges["fc"]["in_scale"]
    int32_state["fc"], _ = process_fc(model.fc, "fc", fc_in_scale)

    base_filename = os.path.basename(model_path).replace(".pth", "")
    if f"_{activation}_" not in base_filename:
        base_filename = base_filename.replace("_int32", "").replace(".pth", "")
        out_filename = f"{base_filename}_{activation}_int32.pth"
    else:
        out_filename = f"{base_filename}_int32.pth"

    out_path = os.path.join(RESNET_DIR, out_filename)
    torch.save(int32_state, out_path)
    print(f"[+] Successfully exported int32-only model to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", type=str, required=True)
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "gelu", "leaky_relu"])
    args = parser.parse_args()
    main(args.quantize, activation=args.activation)