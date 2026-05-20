import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_SKIN_LESION_DIR = os.path.join(DATA_ROOT, "Skin-Lesion")
DATA_FLOOD_DIR = os.path.join(DATA_ROOT, "Flood")
DATA_BRAIN_MRI_SEG_DIR = os.path.join(DATA_ROOT, "Brain-MRI-Seg")
DATA_BUSI_DIR = os.path.join(DATA_ROOT, "BUSI")

# Reuse the same split/pairing logic from the training script.
from u_net import setup_data

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
    """Return test split pairs from the same setup logic used in u_net.py."""
    _, _, test_loader = setup_data(
        train_data=dataset_name,
        batch_size=1,
        image_size=256,
        num_workers=0,
    )

    test_pairs = list(test_loader.dataset.pairs)
    if not test_pairs:
        raise RuntimeError(f"No test pairs created for {dataset_name} dataset.")

    return test_pairs


def get_random_sample(dataset_name: str):
    """Pick a random (image, mask) pair from the 10% test split.

    Works for Skin-Lesion, Flood, Brain-MRI-Seg, and BUSI datasets.
    """
    test_pairs = _get_test_pairs(dataset_name)
    img_path, mask_spec = random.choice(test_pairs)

    # Load PIL images
    image = Image.open(img_path).convert("RGB")
    if isinstance(mask_spec, list):
        if not mask_spec:
            raise RuntimeError(f"No masks found for sampled image: {img_path}")
        mask = Image.open(mask_spec[0]).convert("L")
        for extra_mask_path in mask_spec[1:]:
            extra_mask = Image.open(extra_mask_path).convert("L")
            mask = ImageChops.lighter(mask, extra_mask)
    else:
        mask = Image.open(mask_spec).convert("L")

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


# -----------------------------
# 4. Core Integer Inference Engine
# -----------------------------
def run_integer_layer(q_input, layer_data, zp_in, apply_relu=True):
    """
    Executes a single layer using pre-compiled offline integer data.
    """
    device = q_input.device
    
    # 1. Unpack offline parameters
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

    # 2. Execute Integer Math
    if not is_transpose:
        int32_accum = integer_conv2d(q_input, q_w, zp_in, zp_w, stride=stride, padding=padding)
    else:
        out_padding = layer_data["output_padding"]
        int32_accum = integer_conv_transpose2d(
            q_input, q_w, zp_in, zp_w, 
            stride=stride, padding=padding, output_padding=out_padding
        )

    # 3. Finish Pipeline
    int32_accum = add_bias(int32_accum, q_bias)
    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)

    if apply_relu:
        q_out = quantized_relu(q_out, zp_out)

    return q_out, scale_out, zp_out


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
def main(infer_data, run_floating_point=True, run_integer=True):
    image_tensor, mask_tensor = get_random_sample(infer_data)

    print("--- Starting Quantized Segmentation Inference Pipeline ---")

    # Load the trained UNet model for the requested dataset
    if infer_data == "Skin-Lesion":
        model = UNet(num_classes=1)
        model_path = "best_unet5_skin_lesion.pth"
    elif infer_data == "Flood":
        model = UNet(num_classes=1)
        model_path = "best_unet5_flood.pth"
    elif infer_data == "Brain-MRI-Seg":
        model = UNet(num_classes=1)
        model_path = "best_unet5_brain_mri_seg.pth"
    elif infer_data == "BUSI":
        model = UNet(num_classes=1)
        model_path = "best_unet5_busi.pth"
    else:
        raise ValueError(f"Unknown dataset: {infer_data}")

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    eps = 1e-7

    float_dice = None
    float_iou = None
    float_acc = None
    float_f1 = None

    # -----------------------------
    # [1] Float Inference + Calibration
    # -----------------------------
    int8_model_path = model_path.replace(".pth", "_int8.pth")
    if run_integer and not os.path.exists(int8_model_path):
        raise FileNotFoundError(f"Missing INT8 checkpoint: {int8_model_path}. Run export_int8_model.py first.")
    
    if run_integer:
        int8_state = torch.load(int8_model_path, map_location=image_tensor.device)
        
        # Load Calibration JSON to get concatenated input scales
        calib_filename = f"{infer_data.lower().replace(' ', '_').replace('-', '_')}_calibration.json"
        calib_path = os.path.join(UNET_DIR, "calibration", calib_filename)
        if not os.path.exists(calib_path):
            calib_path = os.path.join(UNET_DIR, calib_filename) # Fallback to current dir
            
        with open(calib_path, "r") as f:
            calib_data = json.load(f)
        activation_ranges = calib_data.get("layers", calib_data)

    if run_floating_point:
        # Binary segmentation metrics (lesion vs background)
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

            float_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
            float_iou = (tp + eps) / (tp + fp + fn + eps)
            float_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            float_f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

        print("[1] Float inference complete.")

    if not run_integer:
        print("\n" + "=" * 40)
        print(" SEGMENTATION INFERENCE SUMMARY ")
        print("=" * 40)
        print(
            f"Float   -> Dice: {float_dice:.4f}, IoU: {float_iou:.4f}, Acc: {float_acc:.4f}, F1: {float_f1:.4f}"
        )
        return

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
    # -----------------------------
    # [3] Integer-Only Forward Through UNet
    # -----------------------------
    if run_integer:
        # Use offline meta parameters for input
        scale_in = int8_state["meta"]["in_scale"]
        zp_in = int8_state["meta"]["in_zp"]
        q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint8)

        # --- Encoder ---
        q_x, s, z = run_integer_layer(q_x, int8_state["e11"], zp_in, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["e12"], z, apply_relu=True)
        q_e12, s_e12, z_e12 = q_x, s, z
        q_x = pool_uint8(q_x, name="pool1")

        q_x, s, z = run_integer_layer(q_x, int8_state["e21"], z, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["e22"], z, apply_relu=True)
        q_e22, s_e22, z_e22 = q_x, s, z
        q_x = pool_uint8(q_x, name="pool2")

        q_x, s, z = run_integer_layer(q_x, int8_state["e31"], z, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["e32"], z, apply_relu=True)
        q_e32, s_e32, z_e32 = q_x, s, z
        q_x = pool_uint8(q_x, name="pool3")

        q_x, s, z = run_integer_layer(q_x, int8_state["e41"], z, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["e42"], z, apply_relu=True)
        q_e42, s_e42, z_e42 = q_x, s, z
        q_x = pool_uint8(q_x, name="pool4")

        # --- Bottom ---
        q_x, s, z = run_integer_layer(q_x, int8_state["e51"], z, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["e52"], z, apply_relu=True)

        # --- Decoder Block 1 ---
        q_x, s_up1, z_up1 = run_integer_layer(q_x, int8_state["upconv1"], z, apply_relu=False)
        s_cat1 = activation_ranges["d11"]["in_scale"]
        z_cat1 = activation_ranges["d11"]["in_zero_point"]
        
        M0_up1, shift_up1 = compute_requantize_multiplier(s_up1, s_cat1)
        M0_e42, shift_e42 = compute_requantize_multiplier(s_e42, s_cat1)
        
        q_x_aligned = requantize_tensor(q_x, z_up1, z_cat1, M0_up1, shift_up1)
        q_skip_aligned = requantize_tensor(q_e42, z_e42, z_cat1, M0_e42, shift_e42)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        
        q_x, s, z = run_integer_layer(q_cat, int8_state["d11"], z_cat1, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["d12"], z, apply_relu=True)

        # --- Decoder Block 2 ---
        q_x, s_up2, z_up2 = run_integer_layer(q_x, int8_state["upconv2"], z, apply_relu=False)
        s_cat2 = activation_ranges["d21"]["in_scale"]
        z_cat2 = activation_ranges["d21"]["in_zero_point"]
        
        M0_up2, shift_up2 = compute_requantize_multiplier(s_up2, s_cat2)
        M0_e32, shift_e32 = compute_requantize_multiplier(s_e32, s_cat2)
        
        q_x_aligned = requantize_tensor(q_x, z_up2, z_cat2, M0_up2, shift_up2)
        q_skip_aligned = requantize_tensor(q_e32, z_e32, z_cat2, M0_e32, shift_e32)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        
        q_x, s, z = run_integer_layer(q_cat, int8_state["d21"], z_cat2, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["d22"], z, apply_relu=True)

        # --- Decoder Block 3 ---
        q_x, s_up3, z_up3 = run_integer_layer(q_x, int8_state["upconv3"], z, apply_relu=False)
        s_cat3 = activation_ranges["d31"]["in_scale"]
        z_cat3 = activation_ranges["d31"]["in_zero_point"]
        
        M0_up3, shift_up3 = compute_requantize_multiplier(s_up3, s_cat3)
        M0_e22, shift_e22 = compute_requantize_multiplier(s_e22, s_cat3)
        
        q_x_aligned = requantize_tensor(q_x, z_up3, z_cat3, M0_up3, shift_up3)
        q_skip_aligned = requantize_tensor(q_e22, z_e22, z_cat3, M0_e22, shift_e22)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        
        q_x, s, z = run_integer_layer(q_cat, int8_state["d31"], z_cat3, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["d32"], z, apply_relu=True)

        # --- Decoder Block 4 ---
        q_x, s_up4, z_up4 = run_integer_layer(q_x, int8_state["upconv4"], z, apply_relu=False)
        s_cat4 = activation_ranges["d41"]["in_scale"]
        z_cat4 = activation_ranges["d41"]["in_zero_point"]
        
        M0_up4, shift_up4 = compute_requantize_multiplier(s_up4, s_cat4)
        M0_e12, shift_e12 = compute_requantize_multiplier(s_e12, s_cat4)
        
        q_x_aligned = requantize_tensor(q_x, z_up4, z_cat4, M0_up4, shift_up4)
        q_skip_aligned = requantize_tensor(q_e12, z_e12, z_cat4, M0_e12, shift_e12)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        
        q_x, s, z = run_integer_layer(q_cat, int8_state["d41"], z_cat4, apply_relu=True)
        q_x, s, z = run_integer_layer(q_x, int8_state["d42"], z, apply_relu=True)

        # Final output conv (no ReLU)
        q_out, final_s, final_z = run_integer_layer(q_x, int8_state["outconv"], z, apply_relu=False)

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
    if run_floating_point:
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
        choices=["Skin-Lesion", "Flood", "Brain-MRI-Seg", "BUSI"],
        help="Inference data to use",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--int",
        action="store_true",
        help="Run integer inference only",
    )
    mode_group.add_argument(
        "--floating-point",
        action="store_true",
        help="Run floating-point inference only",
    )
    args = parser.parse_args()

    run_floating_point = True
    run_integer = True
    if args.int:
        run_floating_point = False
        run_integer = True
    elif args.floating_point:
        run_floating_point = True
        run_integer = False

    print(f"Using inference data: {args.infer}")
    main(
        args.infer,
        run_floating_point=run_floating_point,
        run_integer=run_integer,
    )