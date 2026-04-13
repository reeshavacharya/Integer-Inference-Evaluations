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
    compute_integer_multiplier,
    compute_requantize_multiplier,
    get_quantization_params,
    get_bias_quantization_params,
    quantize_tensor,
    integer_conv2d,
    integer_conv_transpose2d,
    integer_linear,
    add_bias,
    downscale_and_cast,
    quantized_relu,
    requantize_tensor,
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
activation_ranges = {}


def calibration_hook(module, input, output, name):
    """Hook to capture the min and max of activations during the forward pass."""
    in_tensor = input[0].detach()
    out_tensor = output.detach()

    activation_ranges[name] = {
        "in_min": in_tensor.min().item(),
        "in_max": in_tensor.max().item(),
        "out_min": out_tensor.min().item(),
        "out_max": out_tensor.max().item(),
    }

def get_concat_quantization_params(range_dict, layer1_name, layer2_name):
    """
    Finds the absolute minimum and maximum across two branches 
    and calculates a shared 8-bit scale and zero-point.
    """
    min1, max1 = range_dict[layer1_name]["out_min"], range_dict[layer1_name]["out_max"]
    min2, max2 = range_dict[layer2_name]["out_min"], range_dict[layer2_name]["out_max"]
    
    # The shared range must encompass both tensors
    shared_min = min(min1, min2)
    shared_max = max(max1, max2)
    
    # Force the minimum to be <= 0.0 to protect padding zero-points
    shared_min = min(shared_min, 0.0)
    
    # Standard 8-bit uint8 quantization math
    scale_cat = (shared_max - shared_min) / 255.0
    zp_cat = int(round(0.0 - (shared_min / scale_cat)))
    zp_cat = max(0, min(255, zp_cat))
    
    return scale_cat, zp_cat

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


def register_hooks(model):
    handles = []
    cfg = _get_layer_config(model)

    # Register calibration hooks for each layer in the configuration
    def make_hook(layer_name):
        def hook(m, inp, out):
            calibration_hook(m, inp, out, layer_name)
        return hook

    for name, module in cfg.items():
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)

    return handles


# -----------------------------
# 4. Core Integer Inference Engine
# -----------------------------
def run_integer_layer(
    q_input, layer, layer_name, scale_in, zp_in, apply_relu=True
):
    """
    Executes a single layer entirely in integer arithmetic.
    Dynamically routes nn.Conv2d, nn.ConvTranspose2d, and nn.Linear.
    """
    # 1. Get float weights and calculate their quantization params
    weight_float = layer.weight.detach()
    scale_w, zp_w = get_quantization_params(weight_float, num_bits=8)
    q_w = quantize_tensor(weight_float, scale_w, zp_w, dtype=torch.uint8)
    
    # 2. Calculate output activation params from calibration data
    out_range = activation_ranges[layer_name]
    pseudo_out_tensor = torch.tensor([out_range["out_min"], out_range["out_max"]])
    scale_out, zp_out = get_quantization_params(pseudo_out_tensor, num_bits=8)

    # 3. Quantize Bias to int32
    bias_float = layer.bias.detach()
    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(bias_float, scale_bias, zp_bias, dtype=torch.int32)

    # 4. Calculate Downscale Multiplier (Offline Simulation)
    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    # --- Execute Integer Math (Online Simulation) ---
    
    # ROUTE 1: Standard Convolution (Encoder)
    if isinstance(layer, nn.Conv2d):
        stride = getattr(layer, "stride", (1, 1))
        padding = getattr(layer, "padding", (0, 0))
        int32_accum = integer_conv2d(q_input, q_w, zp_in, zp_w, stride=stride, padding=padding)
        layer_type_str = "conv2d"

    # ROUTE 2: Transposed Convolution (Decoder)
    elif isinstance(layer, nn.ConvTranspose2d):
        stride = getattr(layer, "stride", (1, 1))
        padding = getattr(layer, "padding", (0, 0))
        output_padding = getattr(layer, "output_padding", (0, 0))
        int32_accum = integer_conv_transpose2d(
            q_input, q_w, zp_in, zp_w, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        layer_type_str = "conv_transpose2d"

    # ROUTE 3: Linear / Dense Layer (Not used in standard UNet, but kept for compatibility)
    elif isinstance(layer, nn.Linear):
        int32_accum = integer_linear(q_input, q_w, zp_in, zp_w)
        layer_type_str = "linear"
        
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")

    # Finish the integer pipeline
    int32_accum = add_bias(int32_accum, q_bias)
    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)

    if apply_relu:
        q_out = quantized_relu(q_out, zp_out)

    # Log layer details for debugging/analysis
    layer_log = {
        "layer_name": layer_name,
        "type": layer_type_str,
        "input": {
            "scale": float(scale_in),
            "zero_point": int(zp_in),
            "tensor": q_input.cpu().numpy().tolist(),
        },
        "weights": {
            "scale": float(scale_w),
            "zero_point": int(zp_w),
            "float": weight_float.cpu().numpy().tolist(),
            "quantized": q_w.cpu().numpy().tolist(),
        },
        "bias": {
            "scale": float(scale_bias),
            "zero_point": int(zp_bias),
            "float": bias_float.cpu().numpy().tolist(),
            "quantized": q_bias.cpu().numpy().tolist(),
        },
        "multiplier": {
            "M0": int(q_M0),
            "shift": int(shift),
        },
        "accumulator": int32_accum.cpu().numpy().tolist(),
        "output": {
            "scale": float(scale_out),
            "zero_point": int(zp_out),
            "tensor": q_out.cpu().numpy().tolist(),
        },
    }

    debug_trace["layers"].append(layer_log)

    return (
        q_out,
        scale_out,
        zp_out,
        (scale_w, zp_w),
        (scale_bias, zp_bias),
        (q_M0, shift),
    )


def pool_uint8(q_tensor, name=None):
    """Pure integer 2x2 max pooling with stride 2.

    Also logs input and output activations for debugging.
    """
    # 1. No int32 casting needed! 
    # Max pooling cannot cause overflow, so we can stay in uint8.
    B, C, H, W = q_tensor.shape

    # 2. Reshape the tensor to isolate the 2x2 spatial windows
    # Shape becomes: [Batch, Channels, Height/2, 2, Width/2, 2]
    windows = q_tensor.view(B, C, H // 2, 2, W // 2, 2)

    # 3. Extract the maximum value over the 2x2 window dimensions (dim 3 and 5)
    # Shape becomes: [Batch, Channels, Height/2, Width/2]
    pooled = windows.amax(dim=(3, 5))

    # Log pooling activations
    pool_log = {
        "name": name or "maxpool",
        "kernel_size": [2, 2],
        "stride": [2, 2],
        "input_tensor": q_tensor.cpu().numpy().tolist(),
        "output_tensor": pooled.cpu().numpy().tolist(),
    }
    
    # Assuming debug_trace is a global dict as in your original function
    debug_trace["pooling"].append(pool_log)

    return pooled


# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data):
    image_tensor, mask_tensor = get_random_sample(infer_data)

    print("--- Starting Quantized Segmentation Inference Pipeline ---")

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
    handles = register_hooks(model)
    with torch.no_grad():
        float_logits = model(image_tensor)
    for h in handles:
        h.remove()

    # Binary segmentation metrics (lesion vs background)
    with torch.no_grad():
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

    print("[1] Float inference complete.")

    # -----------------------------
    # [2] Quantize Input
    # -----------------------------
    in_range = activation_ranges["conv1"]
    pseudo_in_tensor = torch.tensor([in_range["in_min"], in_range["in_max"]])
    scale_in, zp_in = get_quantization_params(pseudo_in_tensor, num_bits=8)

    q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint8)

    # Save quantized input (first channel) as an 8-bit grayscale image
    q_x_img = q_x[0, 0].cpu().numpy().astype("uint8")
    Image.fromarray(q_x_img, mode="L").save(
        f"{infer_data.lower()}_quantized_sample.png"
    )

    debug_trace["input"] = {
        "scale": float(scale_in),
        "zero_point": int(zp_in),
        "float_tensor": image_tensor.cpu().numpy().tolist(),
        "quantized_tensor": q_x.cpu().numpy().tolist(),
    }

    # -----------------------------
    # [3] Integer-Only Forward Through UNet
    # -----------------------------
    layer_cfg = _get_layer_config(model)

    # Encoder block 1
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv1"], "conv1", scale_in, zp_in, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv2"], "conv2", s, z, apply_relu=True)
    q_e12, s_e12, z_e12 = q_x, s, z  # skip connection 1 (before pool)
    q_x = pool_uint8(q_x, name="pool1")

    # Encoder block 2
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv3"], "conv3", s, z, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv4"], "conv4", s, z, apply_relu=True)
    q_e22, s_e22, z_e22 = q_x, s, z  # skip connection 2
    q_x = pool_uint8(q_x, name="pool2")

    # Encoder block 3
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv5"], "conv5", s, z, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv6"], "conv6", s, z, apply_relu=True)
    q_e32, s_e32, z_e32 = q_x, s, z  # skip connection 3
    q_x = pool_uint8(q_x, name="pool3")

    # Encoder block 4
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv7"], "conv7", s, z, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv8"], "conv8", s, z, apply_relu=True)
    q_e42, s_e42, z_e42 = q_x, s, z  # skip connection 4
    q_x = pool_uint8(q_x, name="pool4")

    # ==========================================
    # Bottom (no pooling)
    # ==========================================
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv9"], "conv9", s, z, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv10"], "conv10", s, z, apply_relu=True)

    # ==========================================
    # Decoder block 1 (Skip connection from e42 -> "conv8")
    # ==========================================
    # 1. Run Upconv
    q_x, s_up1, z_up1, _, _, _ = run_integer_layer(q_x, layer_cfg["upconv1"], "upconv1", s, z, apply_relu=False)
    
    # 2. Get shared concatenation scale using "conv8"
    s_cat1, z_cat1 = get_concat_quantization_params(activation_ranges, "upconv1", "conv8")
    
    # 3. Calculate multipliers
    M0_up1, shift_up1 = compute_requantize_multiplier(s_up1, s_cat1)
    M0_e42, shift_e42 = compute_requantize_multiplier(s_e42, s_cat1)
    
    # 4. Requantize both tensors
    q_x_aligned = requantize_tensor(q_x, z_up1, z_cat1, M0_up1, shift_up1)
    q_skip_aligned = requantize_tensor(q_e42, z_e42, z_cat1, M0_e42, shift_e42)
    
    # 5. Concatenate and run next convolutions
    q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
    q_x, s, z, _, _, _ = run_integer_layer(q_cat, layer_cfg["conv11"], "conv11", s_cat1, z_cat1, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv12"], "conv12", s, z, apply_relu=True)

    # ==========================================
    # Decoder block 2 (Skip connection from e32 -> "conv6")
    # ==========================================
    q_x, s_up2, z_up2, _, _, _ = run_integer_layer(q_x, layer_cfg["upconv2"], "upconv2", s, z, apply_relu=False)
    
    s_cat2, z_cat2 = get_concat_quantization_params(activation_ranges, "upconv2", "conv6")
    
    M0_up2, shift_up2 = compute_requantize_multiplier(s_up2, s_cat2)
    M0_e32, shift_e32 = compute_requantize_multiplier(s_e32, s_cat2)
    
    q_x_aligned = requantize_tensor(q_x, z_up2, z_cat2, M0_up2, shift_up2)
    q_skip_aligned = requantize_tensor(q_e32, z_e32, z_cat2, M0_e32, shift_e32)
    
    q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
    q_x, s, z, _, _, _ = run_integer_layer(q_cat, layer_cfg["conv13"], "conv13", s_cat2, z_cat2, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv14"], "conv14", s, z, apply_relu=True)

    # ==========================================
    # Decoder block 3 (Skip connection from e22 -> "conv4")
    # ==========================================
    q_x, s_up3, z_up3, _, _, _ = run_integer_layer(q_x, layer_cfg["upconv3"], "upconv3", s, z, apply_relu=False)
    
    s_cat3, z_cat3 = get_concat_quantization_params(activation_ranges, "upconv3", "conv4")
    
    M0_up3, shift_up3 = compute_requantize_multiplier(s_up3, s_cat3)
    M0_e22, shift_e22 = compute_requantize_multiplier(s_e22, s_cat3)
    
    q_x_aligned = requantize_tensor(q_x, z_up3, z_cat3, M0_up3, shift_up3)
    q_skip_aligned = requantize_tensor(q_e22, z_e22, z_cat3, M0_e22, shift_e22)
    
    q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
    q_x, s, z, _, _, _ = run_integer_layer(q_cat, layer_cfg["conv15"], "conv15", s_cat3, z_cat3, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv16"], "conv16", s, z, apply_relu=True)

    # ==========================================
    # Decoder block 4 (Skip connection from e12 -> "conv2")
    # ==========================================
    q_x, s_up4, z_up4, _, _, _ = run_integer_layer(q_x, layer_cfg["upconv4"], "upconv4", s, z, apply_relu=False)
    
    s_cat4, z_cat4 = get_concat_quantization_params(activation_ranges, "upconv4", "conv2")
    
    M0_up4, shift_up4 = compute_requantize_multiplier(s_up4, s_cat4)
    M0_e12, shift_e12 = compute_requantize_multiplier(s_e12, s_cat4)
    
    q_x_aligned = requantize_tensor(q_x, z_up4, z_cat4, M0_up4, shift_up4)
    q_skip_aligned = requantize_tensor(q_e12, z_e12, z_cat4, M0_e12, shift_e12)
    
    q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
    q_x, s, z, _, _, _ = run_integer_layer(q_cat, layer_cfg["conv17"], "conv17", s_cat4, z_cat4, apply_relu=True)
    q_x, s, z, _, _, _ = run_integer_layer(q_x, layer_cfg["conv18"], "conv18", s, z, apply_relu=True)

    # Final output conv (no ReLU)
    q_out, final_s, final_z, final_w, final_b, final_M = run_integer_layer(
        q_x,
        layer_cfg["outconv"],
        "outconv",
        s,
        z,
        apply_relu=False,
    )

    # -----------------------------
    # [4] Integer Metrics
    # -----------------------------
    with torch.no_grad():
        q_out_float = q_out.to(torch.float32)
        dequantized_logits = final_s * (q_out_float - final_z)

        int_probs = torch.sigmoid(dequantized_logits)
        int_preds = (int_probs > 0.5).float()

        preds_flat = int_preds.view(-1)
        masks_flat = mask_tensor.view(-1)

        tp = (preds_flat * masks_flat).sum().item()
        tn = ((1 - preds_flat) * (1 - masks_flat)).sum().item()
        fp = (preds_flat * (1 - masks_flat)).sum().item()
        fn = ((1 - preds_flat) * masks_flat).sum().item()

        int_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        int_iou = (tp + eps) / (tp + fp + fn + eps)
        int_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        int_f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    print("[2] Integer-only UNet inference complete.")

    # -----------------------------
    # [5] Summary Logging
    # -----------------------------
    print("\n" + "=" * 40)
    print(" SEGMENTATION INFERENCE SUMMARY ")
    print("=" * 40)
    print(
        f"Float   -> Dice: {float_dice:.4f}, IoU: {float_iou:.4f}, Acc: {float_acc:.4f}, F1: {float_f1:.4f}"
    )
    print(
        f"Integer -> Dice: {int_dice:.4f}, IoU: {int_iou:.4f}, Acc: {int_acc:.4f}, F1: {int_f1:.4f}"
    )

    # trace_path = "integer_inference_trace.json"
    # with open(trace_path, "w") as f:
    #     json.dump(debug_trace, f, indent=2)
    # print(f"\nSaved integer inference trace to {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        type=str,
        default="Skin-Lesion",
        help="Inference data to use",
    )
    args = parser.parse_args()
    print(f"Using inference data: {args.infer}")
    main(args.infer)