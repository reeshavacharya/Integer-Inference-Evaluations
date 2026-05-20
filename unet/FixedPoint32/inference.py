import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Import the 32-bit helper functions
from utils import (
    quantize_fixed_point,
    dequantize_fixed_point,
    execute_and_shift_conv2d,
    execute_and_shift_conv_transpose2d,
    execute_and_shift_linear,
    fixed_point_relu,
    add_bias,
)

# -----------------------------
# Setup and Data Extraction
# -----------------------------
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


def _get_layer_config(model):
    return {
        "conv1": model.e11, "conv2": model.e12,
        "conv3": model.e21, "conv4": model.e22,
        "conv5": model.e31, "conv6": model.e32,
        "conv7": model.e41, "conv8": model.e42,
        "conv9": model.e51, "conv10": model.e52,
        "upconv1": model.upconv1, "conv11": model.d11, "conv12": model.d12,
        "upconv2": model.upconv2, "conv13": model.d21, "conv14": model.d22,
        "upconv3": model.upconv3, "conv15": model.d31, "conv16": model.d32,
        "upconv4": model.upconv4, "conv17": model.d41, "conv18": model.d42,
        "outconv": model.outconv,
    }

def run_static_fixed_point_layer(q_input, layer, apply_relu=True):
    q_w = quantize_fixed_point(layer.weight.detach())

    if layer.bias is not None:
        q_bias = quantize_fixed_point(layer.bias.detach())
    else:
        q_bias = torch.zeros(layer.out_channels, dtype=torch.int32)

    if isinstance(layer, nn.Conv2d):
        stride = getattr(layer, "stride", (1, 1))
        padding = getattr(layer, "padding", (0, 0))
        q_accum, max_bits, max_rem = execute_and_shift_conv2d(q_input, q_w, stride=stride, padding=padding)
    elif isinstance(layer, nn.ConvTranspose2d):
        stride = getattr(layer, "stride", (1, 1))
        padding = getattr(layer, "padding", (0, 0))
        output_padding = getattr(layer, "output_padding", (0, 0))
        q_accum, max_bits, max_rem = execute_and_shift_conv_transpose2d(
            q_input, q_w, stride=stride, padding=padding, output_padding=output_padding
        )

    q_out = add_bias(q_accum, q_bias)

    if apply_relu:
        q_out = fixed_point_relu(q_out)

    return q_out, max_bits, max_rem

def pool_fixed_point(q_tensor):
    B, C, H, W = q_tensor.shape
    windows = q_tensor.view(B, C, H // 2, 2, W // 2, 2)
    return windows.amax(dim=(3, 5))


def main(infer_data, run_floating_point=True, run_fixed_point=True):
    image_tensor, mask_tensor = get_random_sample(infer_data)

    print("--- Starting Dynamic 32-bit Fixed-Point (Q15.16) Inference ---")

    if infer_data == "Skin-Lesion": model_path = "best_unet5_skin_lesion.pth"
    elif infer_data == "Flood": model_path = "best_unet5_flood.pth"
    elif infer_data == "Brain-MRI-Seg": model_path = "best_unet5_brain_mri_seg.pth"
    elif infer_data == "BUSI": model_path = "best_unet5_busi.pth"
    else: raise ValueError(f"Unknown dataset: {infer_data}")

    model = UNet(num_classes=1)
    model.load_state_dict(torch.load(os.path.join(UNET_DIR, model_path), map_location="cpu"))
    model.eval()

    eps = 1e-7

    # -----------------------------
    # [1] Float Inference
    # -----------------------------
    if run_floating_point:
        with torch.no_grad():
            float_logits = model(image_tensor)
            float_probs = torch.sigmoid(float_logits)
            float_preds = (float_probs > 0.5).float()

            tp = (float_preds * mask_tensor).sum().item()
            fp = (float_preds * (1 - mask_tensor)).sum().item()
            fn = ((1 - float_preds) * mask_tensor).sum().item()
            float_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    # -----------------------------
    # [2] Static 32-bit Fixed-Point Forward Pass
    # -----------------------------
    print("\n[2] Executing Static 32-bit Fixed-Point Forward Pass...")

    q_x = quantize_fixed_point(image_tensor)
    layer_cfg = _get_layer_config(model)

    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv1"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv2"], apply_relu=True)
    q_e12 = q_x 
    q_x = pool_fixed_point(q_x)

    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv3"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv4"], apply_relu=True)
    q_e22 = q_x 
    q_x = pool_fixed_point(q_x)

    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv5"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv6"], apply_relu=True)
    q_e32 = q_x 
    q_x = pool_fixed_point(q_x)

    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv7"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv8"], apply_relu=True)
    q_e42 = q_x 
    q_x = pool_fixed_point(q_x)

    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv9"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv10"], apply_relu=True)

    # Decoder
    decoder_configs = [
        ("upconv1", "conv11", "conv12", q_e42),
        ("upconv2", "conv13", "conv14", q_e32),
        ("upconv3", "conv15", "conv16", q_e22),
        ("upconv4", "conv17", "conv18", q_e12)
    ]

    for up_name, conv1, conv2, skip in decoder_configs:
        q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg[up_name], apply_relu=False)
        q_x = torch.cat([q_x, skip], dim=1)
        q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg[conv1], apply_relu=True)
        q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg[conv2], apply_relu=True)

    q_out, max_bits_used, max_remainder = run_static_fixed_point_layer(q_x, layer_cfg["outconv"], apply_relu=False)

    with torch.no_grad():
        dequantized_logits = dequantize_fixed_point(q_out)
        fp_preds = (torch.sigmoid(dequantized_logits) > 0.5).float()
        
        tp = (fp_preds * mask_tensor).sum().item()
        fp = (fp_preds * (1 - mask_tensor)).sum().item()
        fn = ((1 - fp_preds) * mask_tensor).sum().item()
        fp_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    print("\n" + "=" * 50)
    print(" SEGMENTATION INFERENCE SUMMARY ")
    print("=" * 50)
    print(f"Float Model       -> Dice: {float_dice:.4f}")
    print(f"Static 32-bit     -> Dice: {fp_dice:.4f}")
    print("\n--- ZK Cryptographic Fixed-Point Stats (Final Layer) ---")
    print(f"Architecture Format:         Q15.16 (Pre-Truncated)")
    print(f"Accumulator Max Bit-Length:  {max_bits_used} bits")
    
    # Capacity is evaluated against 31 bits of headroom (since 1 bit is for sign)
    headroom_used = (max_bits_used / 31.0) * 100 if max_bits_used else 0
    print(f"PyTorch Container Headroom:  {headroom_used:.1f}% Capacity Reached")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default="Skin-Lesion")
    args = parser.parse_args()
    main(args.infer)