import argparse
import os
import json
import sys
import torch

# Ensure module paths are available
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

def process_conv(conv, layer_name, scale_in, activation_ranges):
    """
    Scales and quantizes a Conv2d or ConvTranspose2d layer.
    UNet doesn't use BatchNorm, so folding is bypassed.
    """
    w_float = conv.weight.detach()
    
    # 1. Quantize Weights
    scale_w, zp_w = get_quantization_params(w_float, num_bits=8)
    q_w = quantize_tensor(w_float, scale_w, zp_w, dtype=torch.uint8)

    # 2. Retrieve Activation Ranges
    out_range = activation_ranges[layer_name]
    scale_out = out_range["out_scale"]
    zp_out = out_range["out_zero_point"]

    # 3. Quantize Biases (if present)
    if conv.bias is not None:
        b_float = conv.bias.detach()
    else:
        b_float = torch.zeros(conv.out_channels)
        
    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_float, scale_bias, zp_bias, dtype=torch.int32)

    # 4. Calculate Integer Multiplier (M0) and Shift
    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    layer_data = {
        "q_weight": q_w.cpu(), "zp_w": int(zp_w),
        "q_bias": q_bias.cpu(),
        "q_M0": int(q_M0), "shift": int(shift),
        "scale_out": float(scale_out), "zp_out": int(zp_out),
        "stride": conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
        "padding": conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
    }
    
    # Track transpose properties for the decoder
    if isinstance(conv, torch.nn.ConvTranspose2d):
        layer_data["is_transpose"] = True
        layer_data["output_padding"] = conv.output_padding[0] if isinstance(conv.output_padding, tuple) else conv.output_padding
    else:
        layer_data["is_transpose"] = False

    return layer_data, scale_out

def main(quantize_target):
    dataset_target = quantize_target.strip()
    candidate_lower = dataset_target.lower()
    
    # Resolve Model Path & Dataset Names
    if "skin" in candidate_lower:
        dataset = "Skin-Lesion"
        model_name = "best_unet5_skin_lesion.pth"
    elif "flood" in candidate_lower:
        dataset = "Flood"
        model_name = "best_unet5_flood.pth"
    elif "brain" in candidate_lower:
        dataset = "Brain-MRI-Seg"
        model_name = "best_unet5_brain_mri_seg.pth"
    elif "busi" in candidate_lower:
        dataset = "BUSI"
        model_name = "best_unet5_busi.pth"
    else:
        raise ValueError(f"Unknown UNet dataset target: {quantize_target}")

    model_path = os.path.join(UNET_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}. Train the model first.")

    # 1. Load Float Model
    model = UNet(n_class=1)
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # 2. Load Calibration Ranges Offline JSON
    calib_file = f"{dataset.lower().replace(' ', '_').replace('-', '_')}_calibration.json"
    
    # Search for calibration JSON in standard locations
    calib_paths = [
        os.path.join(UNET_DIR, "calibration", calib_file),
        os.path.join(THIS_DIR, "..", "calibration", calib_file),
        os.path.join(THIS_DIR, calib_file)
    ]
    
    calib_path = next((p for p in calib_paths if os.path.exists(p)), None)
    if not calib_path:
        raise FileNotFoundError(f"Missing calibration file: {calib_file}. Run calibration.py first.")
    
    with open(calib_path, "r") as f:
        calib_data = json.load(f)
        
    activation_ranges = calib_data.get("layers", calib_data)

    if "e11" not in activation_ranges:
        print("Warning: 'e11' not found in calibration ranges. Did you use custom names in calibration_hook?")

    # Initialize Integer State Dictionary
    int8_state = {"meta": {}}
    scale_in = activation_ranges["e11"]["in_scale"]
    int8_state["meta"]["in_scale"] = scale_in
    int8_state["meta"]["in_zp"] = activation_ranges["e11"]["in_zero_point"]

    print(f"Quantizing UNet Model: {model_name} to INT8...")

    # --- PROCESS ENCODER ---
    int8_state["e11"], s_e11 = process_conv(model.e11, "e11", activation_ranges["e11"]["in_scale"], activation_ranges)
    int8_state["e12"], s_e12 = process_conv(model.e12, "e12", s_e11, activation_ranges)

    int8_state["e21"], s_e21 = process_conv(model.e21, "e21", s_e12, activation_ranges)
    int8_state["e22"], s_e22 = process_conv(model.e22, "e22", s_e21, activation_ranges)

    int8_state["e31"], s_e31 = process_conv(model.e31, "e31", s_e22, activation_ranges)
    int8_state["e32"], s_e32 = process_conv(model.e32, "e32", s_e31, activation_ranges)

    int8_state["e41"], s_e41 = process_conv(model.e41, "e41", s_e32, activation_ranges)
    int8_state["e42"], s_e42 = process_conv(model.e42, "e42", s_e41, activation_ranges)

    int8_state["e51"], s_e51 = process_conv(model.e51, "e51", s_e42, activation_ranges)
    int8_state["e52"], s_e52 = process_conv(model.e52, "e52", s_e51, activation_ranges)

    # --- PROCESS DECODER ---
    int8_state["upconv1"], s_up1 = process_conv(model.upconv1, "upconv1", s_e52, activation_ranges)
    # Concatenation occurs here. We pull `in_scale` directly from calibration for d11 to capture the merged tensor
    int8_state["d11"], s_d11 = process_conv(model.d11, "d11", activation_ranges["d11"]["in_scale"], activation_ranges)
    int8_state["d12"], s_d12 = process_conv(model.d12, "d12", s_d11, activation_ranges)

    int8_state["upconv2"], s_up2 = process_conv(model.upconv2, "upconv2", s_d12, activation_ranges)
    int8_state["d21"], s_d21 = process_conv(model.d21, "d21", activation_ranges["d21"]["in_scale"], activation_ranges)
    int8_state["d22"], s_d22 = process_conv(model.d22, "d22", s_d21, activation_ranges)

    int8_state["upconv3"], s_up3 = process_conv(model.upconv3, "upconv3", s_d22, activation_ranges)
    int8_state["d31"], s_d31 = process_conv(model.d31, "d31", activation_ranges["d31"]["in_scale"], activation_ranges)
    int8_state["d32"], s_d32 = process_conv(model.d32, "d32", s_d31, activation_ranges)

    int8_state["upconv4"], s_up4 = process_conv(model.upconv4, "upconv4", s_d32, activation_ranges)
    int8_state["d41"], s_d41 = process_conv(model.d41, "d41", activation_ranges["d41"]["in_scale"], activation_ranges)
    int8_state["d42"], s_d42 = process_conv(model.d42, "d42", s_d41, activation_ranges)

    # --- PROCESS OUTPUT LAYER ---
    int8_state["outconv"], _ = process_conv(model.outconv, "outconv", s_d42, activation_ranges)

    # 3. Save to file
    out_path = os.path.join(UNET_DIR, os.path.basename(model_path).replace(".pth", "_int8.pth"))
    torch.save(int8_state, out_path)
    print(f"[+] Successfully exported UNet integer-only model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize",
        type=str,
        required=True,
        help="Dataset key (Skin-Lesion, Flood, Brain-MRI-Seg, BUSI) to quantize."
    )
    args = parser.parse_args()
    main(args.quantize)