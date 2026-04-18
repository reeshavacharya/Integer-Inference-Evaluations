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

import lenet5 as train_mod
from lenet5 import BrainMRILeNet

# Import the helper functions
from utils import (
    execute_and_shift_conv2d,
    quantize_fixed_point,
    dequantize_fixed_point,
    fixed_point_relu,
    execute_and_shift_linear,
    add_bias,
)


# -----------------------------
# Debug / logging storage
# -----------------------------
debug_trace = {"input": {}, "layers": [], "pooling": []}

# Detailed fixed-point log written when --log is enabled
fixed_point_log = {"input": {}, "layers": []}


def _decompose_fixed_point_tensor(q_tensor: torch.Tensor, f_bits: int):
    """Return sign, integer, and fractional parts for a Q-format tensor.

    For each element q in the integer tensor, we interpret it as:
        q = sign * (integer_part * 2^f_bits + fractional_part)

    and record sign bit (0/1), integer_part, and fractional_part.
    """

    q_int32 = q_tensor.to(torch.int32)
    sign = (q_int32 < 0).int()
    abs_q = q_int32.abs()

    if f_bits > 0:
        integer_part = abs_q >> f_bits
        fractional_part = abs_q & ((1 << f_bits) - 1)
    else:
        integer_part = abs_q
        fractional_part = torch.zeros_like(integer_part)

    return {
        "sign": sign.cpu().tolist(),
        "integer": integer_part.cpu().tolist(),
        "fractional": fractional_part.cpu().tolist(),
    }


def _get_layer_config(model):
    """Return the conv/fc modules for Static Fixed-Point inference."""
    if isinstance(model, BrainMRILeNet):
        return {
            "conv1": model.features[0],
            "conv2": model.features[4],
            "fc1": model.classifier[1],
            "fc2": model.classifier[4],
            "fc3": model.classifier[7],
        }
    return {
        "conv1": model.features[0],
        "conv2": model.features[3],
        "fc1": model.classifier[1],
        "fc2": model.classifier[3],
        "fc3": model.classifier[5],
    }


# -----------------------------
# 1. Model Definition (Matches your training script exactly)
# -----------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),  # index 0
            nn.ReLU(),  # index 1
            nn.AvgPool2d(kernel_size=2, stride=2),  # index 2
            nn.Conv2d(6, 16, kernel_size=5, stride=1),  # index 3
            nn.ReLU(),  # index 4
            nn.AvgPool2d(kernel_size=2, stride=2),  # index 5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # index 0
            nn.Linear(16 * 4 * 4, 120),  # index 1
            nn.ReLU(),  # index 2
            nn.Linear(120, 84),  # index 3
            nn.ReLU(),  # index 4
            nn.Linear(84, num_classes),  # index 5
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# 2. Setup and Data Extraction
# -----------------------------


def get_random_sample(dataset_name: str):
    """Return a single preprocessed sample tensor and label.

    Uses the same 10% test partition as defined in lenet5.py for
    MNIST, Brain-MRI, and CIFAR10 by calling the corresponding
    setup_* function and sampling from its test_loader.
    """

    name = dataset_name.upper()

    if name == "MNIST":
        # Recreate the deterministic 80/10/10 split and use its test subset
        train_mod.setup_MNIST(batch_size=1)
        test_dataset = train_mod.test_loader.dataset

    elif name == "BRAIN-MRI":
        train_mod.setup_Brain_MRI(batch_size=1)
        test_dataset = train_mod.test_loader.dataset

    elif name in ("CIFR10", "CIFAR10"):
        train_mod.setup_CIFAR10(batch_size=1)
        test_dataset = train_mod.test_loader.dataset

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    idx = random.randint(0, len(test_dataset) - 1)
    image_tensor, label = test_dataset[idx]
    # Add batch dimension; channel/shape is preserved (1x28x28 or 3x28x28)
    return image_tensor.unsqueeze(0), label


def run_static_fixed_point_layer(
    q_input, layer, layer_name, apply_relu=True, is_conv=False
):
    q_w = quantize_fixed_point(layer.weight.detach())

    if layer.bias is not None:
        q_bias = quantize_fixed_point(layer.bias.detach())
    else:
        q_bias = torch.zeros(layer.out_channels, dtype=torch.int64)

    # Default stats (0) for conv layers if you didn't update execute_and_shift_conv2d
    max_bits, max_rem = 0, 0

    if is_conv:
        # Assuming you only updated the linear function, we just unpack the normal way here
        q_accum = execute_and_shift_conv2d(
            q_input, q_w, stride=layer.stride[0], padding=layer.padding[0]
        )
    else:
        # Unpack the new stats from the linear execution
        q_accum, max_bits, max_rem = execute_and_shift_linear(q_input, q_w)

    q_out = add_bias(q_accum, q_bias)

    if apply_relu:
        q_out = fixed_point_relu(q_out)

    return q_out, max_bits, max_rem


def avg_pool_fixed_point(q_tensor, name=None):
    """Pure integer 2x2 average pooling with stride 2 for 64-bit."""
    q_int64 = q_tensor.to(torch.int64)
    B, C, H, W = q_int64.shape

    windows = q_int64.view(B, C, H // 2, 2, W // 2, 2)
    window_sums = windows.sum(dim=(3, 5))

    # Divide by 4 (shift by 2) with round-to-nearest
    rounded_avg = (window_sums + 2) >> 2

    return rounded_avg.to(torch.int64)


# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data, log=False):
    print("--- Starting Static 64-bit ZK Inference Pipeline ---")

    name = infer_data.upper()

    # Load the appropriate trained model for the dataset
    if name == "MNIST":
        model = LeNet5(num_classes=10, in_channels=1)
        model_path = "best_lenet5_mnist.pth"
    elif name == "BRAIN-MRI":
        model = BrainMRILeNet(num_classes=4)
        model_path = "best_lenet5_brain_mri.pth"
    elif name in ("CIFR10", "CIFAR10"):
        model = LeNet5(num_classes=10, in_channels=3)
        model_path = "best_lenet5_cifar10.pth"
    else:
        raise ValueError(f"Unknown dataset: {infer_data}")

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Draw one random sample 
    image_tensor, true_label = get_random_sample(infer_data)

    print(
        f"\n[1] Extracted random {infer_data} sample (True Label: {true_label})."
    )

    # ---------------------------------------------------------
    # [2] Float Inference (For accuracy comparison)
    # ---------------------------------------------------------
    with torch.no_grad():
        float_logits = model(image_tensor)
        float_pred = float_logits.argmax(dim=1).item()
    
    print(f"[2] Float Inference complete. Prediction: {float_pred}")

    # ---------------------------------------------------------
    # [3] Static 64-bit Fixed-Point Inference (Q31.32)
    # ---------------------------------------------------------
    print("\n[3] Executing Static 64-bit Fixed-Point Inference...")

    # Quantize Input Image directly to Q31.32
    q_x = quantize_fixed_point(image_tensor)
    
    layer_cfg = _get_layer_config(model)

    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv1"], "conv1", apply_relu=True, is_conv=True)
    q_x = avg_pool_fixed_point(q_x, name="pool_after_conv1")

    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["conv2"], "conv2", apply_relu=True, is_conv=True)
    q_x = avg_pool_fixed_point(q_x, name="pool_after_conv2")

    q_x = q_x.view(q_x.size(0), -1)  # Flatten

    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["fc1"], "fc1", apply_relu=True, is_conv=False)
    q_x, _, _ = run_static_fixed_point_layer(q_x, layer_cfg["fc2"], "fc2", apply_relu=True, is_conv=False)

    # Final Layer Execution (Capturing ZK Stats)
    q_out, max_bits_used, max_remainder = run_static_fixed_point_layer(
        q_x, layer_cfg["fc3"], "fc3", apply_relu=False, is_conv=False
    )

    # Dequantize final output to check prediction
    dequantized_logits = dequantize_fixed_point(q_out)
    int_pred = dequantized_logits.argmax(dim=1).item()

    # ---------------------------------------------------------
    # [4] Summary Logging
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print(" INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"True Label:                   {true_label}")
    print(f"Float Model Prediction:       {float_pred}")
    print(f"Static 64-bit Prediction:     {int_pred}")

    if float_pred == int_pred:
        print("\nSuccess! The 64-bit static model exactly matches the floating-point prediction.")
    else:
        print("\nNote: The predictions differ. This can occasionally happen due to 32-bit truncation loss.")

    # --- ZK LOGGING BLOCK ---
    print("\n--- ZK Cryptographic Fixed-Point Stats (Final Layer) ---")
    print(f"Architecture Format:         Q31.32 (64-bit Static Container)")
    print(f"Accumulator Max Bit-Length:  {max_bits_used} bits (Threshold: 63 for PyTorch, ~254 for ZK Field)")
    
    headroom_used = (max_bits_used / 63.0) * 100 if max_bits_used else 0
    print(f"PyTorch Container Headroom:  {headroom_used:.1f}% Capacity Reached")
    print(f"Max Truncation Remainder:    {max_remainder:.0f} (Precision dropped during >> 32 shift)")
    
    print("\n--- Final Logit Sanity Check ---")
    print(f"Raw 64-bit Integer Max:      {q_out.max().item()}")
    print(f"Dequantized Float Max:       {dequantized_logits.max().item():.4f}")
    print("=" * 56)

    if log:
        fixed_point_log.setdefault("meta", {})
        fixed_point_log["meta"] = {
            "dataset": infer_data,
            "true_label": int(true_label),
            "float_prediction": int(float_pred),
            "fixed_point_prediction": int(int_pred),
        }
        log_path = "inference.json"
        with open(log_path, "w") as f:
            json.dump(fixed_point_log, f, indent=2)
        print(f"\nSaved detailed fixed-point log to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        type=str,
        default="MNIST",
        help="Inference data to use",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log per-layer fixed-point sign, integer, and fractional parts to inference.json",
    )
    args = parser.parse_args()
    main(args.infer, log=args.log)
