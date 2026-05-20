import argparse
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageChops
import random
import os
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UNET_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if UNET_DIR not in sys.path:
    sys.path.insert(1, UNET_DIR)

from u_net import setup_data, UNet
from utils import (
    compute_requantize_multiplier,
    quantize_tensor,
    integer_conv2d,
    integer_conv_transpose2d,
    add_bias,
    downscale_and_cast,
    quantized_relu,
    requantize_tensor,
)

def get_random_sample(dataset_name: str):
    _, _, test_loader = setup_data(train_data=dataset_name, batch_size=1, image_size=256, num_workers=0)
    img_path, mask_spec = random.choice(list(test_loader.dataset.pairs))

    image = Image.open(img_path).convert("RGB")
    if isinstance(mask_spec, list):
        mask = Image.open(mask_spec[0]).convert("L")
        for extra_mask_path in mask_spec[1:]:
            extra_mask = Image.open(extra_mask_path).convert("L")
            mask = ImageChops.lighter(mask, extra_mask)
    else:
        mask = Image.open(mask_spec).convert("L")

    resize_img = transforms.Resize((256, 256), interpolation=Image.BILINEAR)
    resize_mask = transforms.Resize((256, 256), interpolation=Image.NEAREST)

    image = transforms.ToTensor()(resize_img(image)).unsqueeze(0)
    mask = transforms.ToTensor()(resize_mask(mask)).unsqueeze(0)
    mask = (mask > 0.5).float()

    return image, mask

def run_integer_layer(q_input, layer_data, zp_in, apply_relu=True):
    device = q_input.device
    q_w = layer_data["q_weight"].to(device)
    q_bias = layer_data["q_bias"].to(device)
    zp_w = layer_data["zp_w"]
    q_M0 = layer_data["q_M0"]
    shift = layer_data["shift"]
    zp_out = layer_data["zp_out"]
    scale_out = layer_data["scale_out"]
    stride = layer_data["stride"]
    padding = layer_data["padding"]
    is_transpose = layer_data.get("is_transpose", False)

    if not is_transpose:
        int_accum = integer_conv2d(q_input, q_w, zp_in, zp_w, stride=stride, padding=padding)
    else:
        out_padding = layer_data["output_padding"]
        int_accum = integer_conv_transpose2d(q_input, q_w, zp_in, zp_w, stride=stride, padding=padding, output_padding=out_padding)

    int_accum = add_bias(int_accum, q_bias)
    q_out = downscale_and_cast(int_accum, q_M0, shift, zp_out)

    if apply_relu:
        q_out = quantized_relu(q_out, zp_out)

    return q_out, scale_out, zp_out

def pool_uint32(q_tensor):
    B, C, H, W = q_tensor.shape
    windows = q_tensor.to(torch.int64).view(B, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5)).to(torch.uint32)
    return windows

def main(infer_data):
    image_tensor, mask_tensor = get_random_sample(infer_data)

    if infer_data == "Skin-Lesion": model_path = "best_unet5_skin_lesion.pth"
    elif infer_data == "Flood": model_path = "best_unet5_flood.pth"
    elif infer_data == "Brain-MRI-Seg": model_path = "best_unet5_brain_mri_seg.pth"
    elif infer_data == "BUSI": model_path = "best_unet5_busi.pth"

    int32_model_path = os.path.join(UNET_DIR, model_path.replace(".pth", "_int32.pth"))
    if not os.path.exists(int32_model_path):
        raise FileNotFoundError(f"Missing INT32 checkpoint: {int32_model_path}")
    
    int32_state = torch.load(int32_model_path, map_location=image_tensor.device)
    calib_filename = f"{infer_data.lower().replace(' ', '_').replace('-', '_')}_calibration.json"
    
    with open(os.path.join(UNET_DIR, "calibration", calib_filename), "r") as f:
        activation_ranges = json.load(f).get("layers")

    scale_in = int32_state["meta"]["in_scale"]
    zp_in = int32_state["meta"]["in_zp"]
    q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint32)

    # --- Encoder ---
    q_x, s, z = run_integer_layer(q_x, int32_state["e11"], zp_in, apply_relu=True)
    q_x, s, z = run_integer_layer(q_x, int32_state["e12"], z, apply_relu=True)
    q_e12, s_e12, z_e12 = q_x, s, z
    q_x = pool_uint32(q_x)

    q_x, s, z = run_integer_layer(q_x, int32_state["e21"], z, apply_relu=True)
    q_x, s, z = run_integer_layer(q_x, int32_state["e22"], z, apply_relu=True)
    q_e22, s_e22, z_e22 = q_x, s, z
    q_x = pool_uint32(q_x)

    q_x, s, z = run_integer_layer(q_x, int32_state["e31"], z, apply_relu=True)
    q_x, s, z = run_integer_layer(q_x, int32_state["e32"], z, apply_relu=True)
    q_e32, s_e32, z_e32 = q_x, s, z
    q_x = pool_uint32(q_x)

    q_x, s, z = run_integer_layer(q_x, int32_state["e41"], z, apply_relu=True)
    q_x, s, z = run_integer_layer(q_x, int32_state["e42"], z, apply_relu=True)
    q_e42, s_e42, z_e42 = q_x, s, z
    q_x = pool_uint32(q_x)

    # --- Bottom ---
    q_x, s, z = run_integer_layer(q_x, int32_state["e51"], z, apply_relu=True)
    q_x, s, z = run_integer_layer(q_x, int32_state["e52"], z, apply_relu=True)

    # --- Decoder (Iterating blocks) ---
    decoder_configs = [
        ("upconv1", "d11", "d12", q_e42, s_e42, z_e42),
        ("upconv2", "d21", "d22", q_e32, s_e32, z_e32),
        ("upconv3", "d31", "d32", q_e22, s_e22, z_e22),
        ("upconv4", "d41", "d42", q_e12, s_e12, z_e12)
    ]

    for up_name, d1_name, d2_name, skip_q, skip_s, skip_z in decoder_configs:
        q_x, s_up, z_up = run_integer_layer(q_x, int32_state[up_name], z, apply_relu=False)
        s_cat = activation_ranges[d1_name]["in_scale"]
        z_cat = activation_ranges[d1_name]["in_zero_point"]
        
        M0_up, shift_up = compute_requantize_multiplier(s_up, s_cat)
        M0_skip, shift_skip = compute_requantize_multiplier(skip_s, s_cat)
        
        q_x_aligned = requantize_tensor(q_x, z_up, z_cat, M0_up, shift_up)
        q_skip_aligned = requantize_tensor(skip_q, skip_z, z_cat, M0_skip, shift_skip)
        
        q_x, s, z = run_integer_layer(torch.cat([q_x_aligned, q_skip_aligned], dim=1), int32_state[d1_name], z_cat, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int32_state[d2_name], z, apply_relu=True)

    q_out, final_s, final_z = run_integer_layer(q_x, int32_state["outconv"], z, apply_relu=False)

    with torch.no_grad():
        dequantized = final_s * (q_out.to(torch.float32) - final_z)
        preds = (torch.sigmoid(dequantized) > 0.5).float()
        
        tp = (preds * mask_tensor).sum().item()
        fp = (preds * (1 - mask_tensor)).sum().item()
        fn = ((1 - preds) * mask_tensor).sum().item()
        
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
        print(f"Sample INT32 Dice Score: {dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default="Skin-Lesion")
    args = parser.parse_args()
    main(args.infer)