import argparse
import os
import json
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UNET_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if UNET_DIR not in sys.path:
    sys.path.insert(1, UNET_DIR)

from u_net import UNet
from utils import (
    get_quantization_params, 
    get_bias_quantization_params, 
    compute_integer_multiplier, 
    quantize_tensor
)

WEIGHT_BITS = 8

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
    
    q_act = torch.round(f_act / out_scale) + out_zp
    # Strictly bound to uint32 limits for unsigned activation outputs
    q_act = torch.clamp(q_act, 0, 4294967295).to(torch.uint32)
    
    return q_min_bound.item(), q_max_bound.item(), q_act

def process_conv(conv, layer_name, scale_in, activation_ranges, activation):
    w_float = conv.weight.detach()
    
    # Force 8-bit weights for safe 32-bit accumulation
    scale_w, zp_w = get_quantization_params(w_float, num_bits=WEIGHT_BITS)
    q_w = quantize_tensor(w_float, scale_w, zp_w, dtype=torch.uint32, num_bits=32)

    out_range = activation_ranges[layer_name]
    scale_out = out_range["out_scale"]
    zp_out = out_range["out_zero_point"]

    if conv.bias is not None:
        b_float = conv.bias.detach()
    else:
        b_float = torch.zeros(conv.out_channels)
        
    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_float, scale_bias, zp_bias, dtype=torch.int32) # Biases to int32

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    layer_data = {
        "q_weight": q_w.cpu(), "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0), "shift": int(shift),
        "scale_out": float(scale_out), "zp_out": int(zp_out),
        "stride": conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
        "padding": conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
    }
    
    if isinstance(conv, torch.nn.ConvTranspose2d):
        layer_data["is_transpose"] = True
        layer_data["output_padding"] = conv.output_padding[0] if isinstance(conv.output_padding, tuple) else conv.output_padding
    else:
        layer_data["is_transpose"] = False

    if activation == "gelu":
        q_min, q_max, lut = generate_layer_gelu_lut(
            in_scale=scale_out, 
            in_zp=zp_out, 
            out_scale=scale_out,  # In Unet, post-act range = pre-act range (shared activation_ranges node)
            out_zp=zp_out
        )
        layer_data["gelu_q_min"] = q_min
        layer_data["gelu_q_max"] = q_max
        layer_data["gelu_lut"] = lut.cpu()

    return layer_data, scale_out

def main(quantize_target, activation):
    dataset_target = quantize_target.strip().lower()
    
    if "skin" in dataset_target:
        dataset = "Skin-Lesion"
        model_name = f"best_unet5_{activation}_skin_lesion.pth"
    elif "flood" in dataset_target:
        dataset = "Flood"
        model_name = f"best_unet5_{activation}_flood.pth"
    elif "brain" in dataset_target:
        dataset = "Brain-MRI-Seg"
        model_name = f"best_unet5_{activation}_brain_mri_seg.pth"
    elif "busi" in dataset_target:
        dataset = "BUSI"
        model_name = f"best_unet5_{activation}_busi.pth"
    else:
        raise ValueError(f"Unknown UNet dataset target: {quantize_target}")

    model_path = os.path.join(UNET_DIR, model_name)
    model = UNet(n_class=1)
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    calib_file = f"{dataset.lower().replace(' ', '_').replace('-', '_')}_{activation}_calibration.json"
    calib_path = os.path.join(UNET_DIR, "calibration", calib_file)
    
    with open(calib_path, "r") as f:
        calib_data = json.load(f)
        
    activation_ranges = calib_data.get("layers", calib_data)

    int32_state = {"meta": {}}
    scale_in = activation_ranges["e11"]["in_scale"]
    int32_state["meta"]["in_scale"] = scale_in
    int32_state["meta"]["in_zp"] = activation_ranges["e11"]["in_zero_point"]

    print(f"Quantizing UNet Model: {model_name} to INT32 ...")

    # --- ENCODER ---
    int32_state["e11"], s = process_conv(model.e11, "e11", scale_in, activation_ranges, activation)
    int32_state["e12"], s = process_conv(model.e12, "e12", s, activation_ranges, activation)
    int32_state["e21"], s = process_conv(model.e21, "e21", s, activation_ranges, activation)
    int32_state["e22"], s = process_conv(model.e22, "e22", s, activation_ranges, activation)
    int32_state["e31"], s = process_conv(model.e31, "e31", s, activation_ranges, activation)
    int32_state["e32"], s = process_conv(model.e32, "e32", s, activation_ranges, activation)
    int32_state["e41"], s = process_conv(model.e41, "e41", s, activation_ranges, activation)
    int32_state["e42"], s = process_conv(model.e42, "e42", s, activation_ranges, activation)
    int32_state["e51"], s = process_conv(model.e51, "e51", s, activation_ranges, activation)
    int32_state["e52"], s = process_conv(model.e52, "e52", s, activation_ranges, activation)

    # --- DECODER ---
    int32_state["upconv1"], s_up = process_conv(model.upconv1, "upconv1", s, activation_ranges, "none")
    int32_state["d11"], s = process_conv(model.d11, "d11", activation_ranges["d11"]["in_scale"], activation_ranges, activation)
    int32_state["d12"], s = process_conv(model.d12, "d12", s, activation_ranges, activation)

    int32_state["upconv2"], s_up = process_conv(model.upconv2, "upconv2", s, activation_ranges, "none")
    int32_state["d21"], s = process_conv(model.d21, "d21", activation_ranges["d21"]["in_scale"], activation_ranges, activation)
    int32_state["d22"], s = process_conv(model.d22, "d22", s, activation_ranges, activation)

    int32_state["upconv3"], s_up = process_conv(model.upconv3, "upconv3", s, activation_ranges, "none")
    int32_state["d31"], s = process_conv(model.d31, "d31", activation_ranges["d31"]["in_scale"], activation_ranges, activation)
    int32_state["d32"], s = process_conv(model.d32, "d32", s, activation_ranges, activation)

    int32_state["upconv4"], s_up = process_conv(model.upconv4, "upconv4", s, activation_ranges, "none")
    int32_state["d41"], s = process_conv(model.d41, "d41", activation_ranges["d41"]["in_scale"], activation_ranges, activation)
    int32_state["d42"], s = process_conv(model.d42, "d42", s, activation_ranges, activation)

    # --- OUTPUT LAYER ---
    int32_state["outconv"], _ = process_conv(model.outconv, "outconv", s, activation_ranges, "none")

    out_path = os.path.join(UNET_DIR, os.path.basename(model_path).replace(".pth", "_int32.pth"))
    torch.save(int32_state, out_path)
    print(f"[+] Successfully exported UNet int32 model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", type=str, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu", "leaky_relu"])
    args = parser.parse_args()

    targets = [args.quantize] if args.quantize else ["Skin-Lesion", "Flood", "Brain-MRI-Seg", "BUSI"]
    activations = [args.activation] if args.activation else ["relu", "gelu", "leaky_relu"]

    for t in targets:
        for a in activations:
            try:
                main(t, a)
            except Exception as e:
                print(f"[-] Failed to export {t} ({a}): {e}")