import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import random
import os
import json

# Import the helper functions
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
# Debug trace storage
# -----------------------------
debug_trace = {"input": {}, "layers": [], "pooling": []}


# -----------------------------
# 1. Model Definition (Matches your training script exactly)
# -----------------------------
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder
        # input: 572x572x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out


# -----------------------------
# 2. Setup and Data Extraction
# -----------------------------
def _get_test_pairs(dataset_name: str):
    """Return the 10% test split pairs for the given dataset.

    Uses the same 80/10/10 split logic as in u_net.py so that
    inference is performed on the held-out test set.
    """
    if dataset_name == "Skin-Lesion":
        image_dir = "./data/Skin-Lesion/images"
        mask_dir = "./data/Skin-Lesion/masks"
    elif dataset_name == "Flood":
        image_dir = "./data/Flood/Image"
        mask_dir = "./data/Flood/Mask"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        raise RuntimeError(
            f"{dataset_name} image/mask directories not found. Make sure the dataset is downloaded and extracted correctly."
        )

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_id_to_path = {}
    for fname in image_files:
        image_id, _ = os.path.splitext(fname)
        image_id_to_path[image_id] = os.path.join(image_dir, fname)

    mask_files = [
        f
        for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    pairs = []
    for mname in mask_files:
        base, _ = os.path.splitext(mname)
        image_id = base.split("_segmentation")[0]
        if image_id in image_id_to_path:
            img_path = image_id_to_path[image_id]
            mask_path = os.path.join(mask_dir, mname)
            pairs.append((img_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No matching image/mask pairs found in {dataset_name} dataset.")

    # Disjoint split: 80% train, 10% val, 10% test (same as u_net.py)
    n = len(pairs)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=generator).tolist()

    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    # We only need the test split here
    test_pairs = [pairs[i] for i in indices[n_train + n_val:]]

    if not test_pairs:
        raise RuntimeError(f"No test pairs created for {dataset_name} dataset.")

    return test_pairs


def get_random_sample(dataset_name: str):
    """Pick a random (image, mask) pair from the 10% test split.

    Works for both Skin-Lesion and Flood datasets.
    """
    test_pairs = _get_test_pairs(dataset_name)
    img_path, mask_path = random.choice(test_pairs)

    # Load PIL images
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Resize to match training setup (256x256)
    resize_img = transforms.Resize((256, 256), interpolation=Image.BILINEAR)
    resize_mask = transforms.Resize((256, 256), interpolation=Image.NEAREST)

    image = resize_img(image)
    mask = resize_mask(mask)

    # Convert to tensors
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)  # [C, H, W], C=3
    mask_tensor = to_tensor(mask)    # [1, H, W]

    # Binarize mask to {0,1} as in training
    mask_tensor = (mask_tensor > 0.5).float()

    # Add batch dimension: [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0)

    return image_tensor, mask_tensor


# -----------------------------
# 3. Calibration Hooks
# -----------------------------

def _get_layer_config(model):
    """Return the conv/fc modules for calibration and integer inference.

    UNet architecture needs to be mapped accordingly
    """
    return {
        # Encoder convolutions
        "conv1": model.e11,
        "conv2": model.e12,
        "conv3": model.e21,
        "conv4": model.e22,
        "conv5": model.e31,
        "conv6": model.e32,
        "conv7": model.e41,
        "conv8": model.e42,
        "conv9": model.e51,
        "conv10": model.e52,

        # Decoder up-convolutions and convolutions
        "upconv1": model.upconv1,
        "conv11": model.d11,
        "conv12": model.d12,

        "upconv2": model.upconv2,
        "conv13": model.d21,
        "conv14": model.d22,

        "upconv3": model.upconv3,
        "conv15": model.d31,
        "conv16": model.d32,

        "upconv4": model.upconv4,
        "conv17": model.d41,
        "conv18": model.d42,

        # Output layer
        "outconv": model.outconv,
    }

def run_static_fixed_point_layer(q_input, layer, apply_relu=True):
    # Quantize Weights and Biases statically
    q_w = quantize_fixed_point(layer.weight.detach())

    if layer.bias is not None:
        q_bias = quantize_fixed_point(layer.bias.detach())
    else:
        q_bias = torch.zeros(layer.out_channels, dtype=torch.int64)

    # Route math based on layer type
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
    elif isinstance(layer, nn.Linear):
        q_accum, max_bits, max_rem = execute_and_shift_linear(q_input, q_w)

    q_out = add_bias(q_accum, q_bias)

    if apply_relu:
        q_out = fixed_point_relu(q_out)

    return q_out, max_bits, max_rem

def pool_fixed_point(q_tensor):
    """Pure integer 2x2 max pooling."""
    B, C, H, W = q_tensor.shape
    windows = q_tensor.view(B, C, H // 2, 2, W // 2, 2)
    pooled = windows.amax(dim=(3, 5))
    return pooled


# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data):
    image_tensor, mask_tensor = get_random_sample(infer_data)

    print("--- Starting Dynamic Fixed-Point Segmentation Inference ---")

    # Load the trained UNet model for the requested dataset
    if infer_data == "Skin-Lesion":
        model = UNet(num_classes=1)
        model_path = "best_unet5_skin_lesion.pth"
    elif infer_data == "Flood":
        model = UNet(num_classes=1)
        model_path = "best_unet5_flood.pth"
    else:
        raise ValueError(f"Unknown dataset: {infer_data}")

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # -----------------------------
    # [1] Float Inference + Calibration
    # -----------------------------
    with torch.no_grad():
        float_logits = model(image_tensor)
        float_probs = torch.sigmoid(float_logits)
        float_preds = (float_probs > 0.5).float()

        preds_flat = float_preds.view(-1)
        masks_flat = mask_tensor.view(-1)

        tp = (preds_flat * masks_flat).sum().item()
        tn = ((1 - preds_flat) * (1 - masks_flat)).sum().item()
        fp = (preds_flat * (1 - masks_flat)).sum().item()
        fn = ((1 - preds_flat) * masks_flat).sum().item()

        eps = 1e-7
        float_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        float_iou = (tp + eps) / (tp + fp + fn + eps)
        float_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        float_f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    print("[1] Float inference and calibration complete.")

    # -----------------------------
    # [2] Static 64-bit Fixed-Point Forward Pass
    # -----------------------------
    print("\n[2] Executing Static 64-bit Fixed-Point Forward Pass...")

    # Quantize Input statically to Q31.32
    q_x = quantize_fixed_point(image_tensor)
    
    layer_cfg = _get_layer_config(model)

    # Encoder block 1
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv1"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv2"], apply_relu=True)
    q_e12 = q_x  # Skip connection 1
    q_x = pool_fixed_point(q_x)

    # Encoder block 2
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv3"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv4"], apply_relu=True)
    q_e22 = q_x  # Skip connection 2
    q_x = pool_fixed_point(q_x)

    # Encoder block 3
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv5"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv6"], apply_relu=True)
    q_e32 = q_x  # Skip connection 3
    q_x = pool_fixed_point(q_x)

    # Encoder block 4
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv7"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv8"], apply_relu=True)
    q_e42 = q_x  # Skip connection 4
    q_x = pool_fixed_point(q_x)

    # Bottleneck
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv9"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv10"], apply_relu=True)

    # Decoder block 1 (Direct Concatenation without alignment shifts!)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["upconv1"], apply_relu=False)
    q_cat = torch.cat([q_x, q_e42], dim=1)
    q_x, _, _ = run_static_fixed_point_layer(q_cat, layer_cfg["conv11"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv12"], apply_relu=True)

    # Decoder block 2
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["upconv2"], apply_relu=False)
    q_cat = torch.cat([q_x, q_e32], dim=1)
    q_x, _, _ = run_static_fixed_point_layer(q_cat, layer_cfg["conv13"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv14"], apply_relu=True)

    # Decoder block 3
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["upconv3"], apply_relu=False)
    q_cat = torch.cat([q_x, q_e22], dim=1)
    q_x, _, _ = run_static_fixed_point_layer(q_cat, layer_cfg["conv15"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv16"], apply_relu=True)

    # Decoder block 4
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["upconv4"], apply_relu=False)
    q_cat = torch.cat([q_x, q_e12], dim=1)
    q_x, _, _ = run_static_fixed_point_layer(q_cat, layer_cfg["conv17"], apply_relu=True)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv18"], apply_relu=True)

    # Final output conv
    q_out, max_bits_used, max_remainder = run_static_fixed_point_layer(
        q_x, layer_cfg["outconv"], apply_relu=False
    )

    # -----------------------------
    # [3] Fixed-Point Metrics & ZK Logging
    # -----------------------------
    with torch.no_grad():
        dequantized_logits = dequantize_fixed_point(q_out)
        fp_probs = torch.sigmoid(dequantized_logits)
        fp_preds = (fp_probs > 0.5).float()

        preds_flat = fp_preds.view(-1)
        masks_flat = mask_tensor.view(-1)

        tp = (preds_flat * masks_flat).sum().item()
        tn = ((1 - preds_flat) * (1 - masks_flat)).sum().item()
        fp = (preds_flat * (1 - masks_flat)).sum().item()
        fn = ((1 - preds_flat) * masks_flat).sum().item()

        fp_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        fp_iou = (tp + eps) / (tp + fp + fn + eps)
        fp_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        fp_f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    print("\n" + "=" * 50)
    print(" SEGMENTATION INFERENCE SUMMARY ")
    print("=" * 50)
    print(
        f"Float Model       -> Dice: {float_dice:.4f}, IoU: {float_iou:.4f}, Acc: {float_acc:.4f}, F1: {float_f1:.4f}"
    )
    print(
        f"Static 64-bit     -> Dice: {fp_dice:.4f}, IoU: {fp_iou:.4f}, Acc: {fp_acc:.4f}, F1: {fp_f1:.4f}"
    )
    
    print("\n--- ZK Cryptographic Fixed-Point Stats (Final Layer) ---")
    print(f"Architecture Format:         Q31.32 (Pre-Truncated)")
    print(f"Accumulator Max Bit-Length:  {max_bits_used} bits (Safe for native PyTorch 64-bit!)")
    
    headroom_used = (max_bits_used / 63.0) * 100 if max_bits_used else 0
    print(f"PyTorch Container Headroom:  {headroom_used:.1f}% Capacity Reached")
    print(f"Max Truncation Remainder:    {max_remainder:.0f} (Precision dropped prior to multiplication)")
    print("=" * 50)

    # trace_path = "fixed_point_inference_trace.json"
    # with open(trace_path, "w") as f:
    #     json.dump(debug_trace, f, indent=2)
    # print(f"\nSaved fixed-point inference trace to {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        type=str,
        default="Skin-Lesion",
        help="Inference data to use",
    )
    args = parser.parse_args()
    main(args.infer)