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
    get_fractional_bits,
    quantize_fixed_point,
    dequantize_fixed_point,
    downscale_fixed_point,
    fixed_point_relu,
    fixed_point_conv2d,
    fixed_point_linear,
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


# -----------------------------
# 3. Calibration Hooks
# -----------------------------
activation_ranges = {}


def calibration_hook(module, input, output, name):
    """Hook to capture the absolute max of activations during the forward pass."""
    in_tensor = input[0].detach()
    out_tensor = output.detach()

    activation_ranges[name] = {
        "in_abs_max": in_tensor.abs().max().item(),
        "out_abs_max": out_tensor.abs().max().item(),
    }


def _get_layer_config(model):
    """Return the conv/fc modules for calibration and Fixed-Point inference.

    LeNet5 and BrainMRILeNet have different Sequential layouts, so we
    centralize the index mapping here.
    """

    if isinstance(model, BrainMRILeNet):
        # BrainMRILeNet.features: [Conv, BN, ReLU, AvgPool, Conv, BN, ReLU, AvgPool]
        # BrainMRILeNet.classifier: [Flatten, Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]
        return {
            "conv1": model.features[0],
            "conv2": model.features[4],
            "fc1": model.classifier[1],
            "fc2": model.classifier[4],
            "fc3": model.classifier[7],
        }

    # Default LeNet5 mapping
    return {
        "conv1": model.features[0],
        "conv2": model.features[3],
        "fc1": model.classifier[1],
        "fc2": model.classifier[3],
        "fc3": model.classifier[5],
    }


def register_hooks(model):
    handles = []
    cfg = _get_layer_config(model)

    handles.append(
        cfg["conv1"].register_forward_hook(
            lambda m, i, o: calibration_hook(m, i, o, "conv1")
        )
    )
    handles.append(
        cfg["conv2"].register_forward_hook(
            lambda m, i, o: calibration_hook(m, i, o, "conv2")
        )
    )
    handles.append(
        cfg["fc1"].register_forward_hook(
            lambda m, i, o: calibration_hook(m, i, o, "fc1")
        )
    )
    handles.append(
        cfg["fc2"].register_forward_hook(
            lambda m, i, o: calibration_hook(m, i, o, "fc2")
        )
    )
    handles.append(
        cfg["fc3"].register_forward_hook(
            lambda m, i, o: calibration_hook(m, i, o, "fc3")
        )
    )
    return handles


def run_fixed_point_layer(
    q_input, layer, layer_name, f_in, apply_relu=True, is_conv=False, log_details=False
):
    """
    Executes a single layer entirely in Dynamic Fixed-Point arithmetic.
    """
    # 1. Weights
    weight_float = layer.weight.detach()
    f_w = get_fractional_bits(weight_float, num_bits=8)
    q_w = quantize_fixed_point(weight_float, f_w, dtype=torch.int8)

    # 2. Output ranges
    out_abs_max = activation_ranges[layer_name]["out_abs_max"]
    pseudo_out = torch.tensor([out_abs_max])
    f_out = get_fractional_bits(pseudo_out, num_bits=8)

    # 3. Bias (Must match the accumulator's fractional bits exactly)
    f_accum = f_w + f_in
    if layer.bias is not None:
        bias_float = layer.bias.detach()
        q_bias = quantize_fixed_point(bias_float, f_accum, dtype=torch.int32)
    else:
        q_bias = torch.zeros(layer.out_channels, dtype=torch.int32)

    # --- Execute Integer Math ---
    if is_conv:
        int32_accum = fixed_point_conv2d(q_input, q_w, stride=layer.stride[0], padding=layer.padding[0])
    else:
        int32_accum = fixed_point_linear(q_input, q_w)
        
    int32_accum = add_bias(int32_accum, q_bias)

    # 4. Downscale and shift
    shift_amount = f_accum - f_out
    q_out = downscale_fixed_point(int32_accum, shift_amount)

    if apply_relu:
        q_out = fixed_point_relu(q_out)

    debug_trace["layers"].append({
        "layer_name": layer_name,
        "type": "conv" if is_conv else "linear",
        "f_in": f_in, "f_w": f_w, "f_out": f_out, "shift": shift_amount,
    })

    # Optional detailed per-layer fixed-point logging
    if log_details:
        global fixed_point_log
        fixed_point_log.setdefault("layers", [])
        fixed_point_log["layers"].append(
            {
                "layer_name": layer_name,
                "type": "conv" if is_conv else "linear",
                "f_in": f_in,
                "f_w": f_w,
                "f_out": f_out,
                "shift": shift_amount,
                "weights": {
                    "f": f_w,
                    "decomposition": _decompose_fixed_point_tensor(q_w, f_w),
                },
                "bias": {
                    "f": f_accum,
                    "decomposition": _decompose_fixed_point_tensor(q_bias, f_accum),
                },
                "input": {
                    "f": f_in,
                    "decomposition": _decompose_fixed_point_tensor(q_input, f_in),
                },
                "output": {
                    "f": f_out,
                    "decomposition": _decompose_fixed_point_tensor(q_out, f_out),
                },
            }
        )

    return q_out, f_out, f_w, shift_amount


def avg_pool_fixed_point(q_tensor, name=None):
    """Pure integer 2x2 average pooling with stride 2 for fixed-point int8."""
    q_int32 = q_tensor.to(torch.int32)
    B, C, H, W = q_int32.shape

    windows = q_int32.view(B, C, H // 2, 2, W // 2, 2)
    window_sums = windows.sum(dim=(3, 5))

    # Divide by 4 (shift by 2) with round-to-nearest
    rounded_avg = (window_sums + 2) >> 2

    # Cast securely back to int8
    pooled = torch.clamp(rounded_avg, -128, 127).to(torch.int8)

    debug_trace["pooling"].append({"name": name or "pool"})
    return pooled


# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data, log=False):
    print("--- Starting Quantized Inference Pipeline ---")

    name = infer_data.upper()

    # Load the appropriate trained model for the dataset
    if name == "MNIST":
        model = LeNet5(num_classes=10, in_channels=1)
        model_path = "best_lenet5_mnist.pth"
    elif name == "BRAIN-MRI":
        # Brain-MRI models are trained with the BrainMRILeNet architecture
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

    # Draw one random sample from the correct 10% test partition
    image_tensor, true_label = get_random_sample(infer_data)

    print(
        f"\n[1] Extracted random {infer_data} sample (True Label: {true_label}). Saved to '{infer_data.lower()}_sample.png'."
    )

    # Run Calibration (Float)
    handles = register_hooks(model)
    with torch.no_grad():
        float_output = model(image_tensor)
    for h in handles:
        h.remove()

    float_pred = float_output.argmax(dim=1).item()
    print(f"[2] Calibration complete. Float Model Prediction: {float_pred}")

    # Quantize Input Image
    in_abs_max = activation_ranges["conv1"]["in_abs_max"]
    pseudo_in_tensor = torch.tensor([in_abs_max])
    f_in = get_fractional_bits(pseudo_in_tensor, num_bits=8)

    q_x = quantize_fixed_point(image_tensor, f_in, dtype=torch.int8)
    
    q_x_img = torch.clamp(q_x[0, 0].to(torch.int16) + 128, 0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(q_x_img, mode="L").save(f"{infer_data.lower()}_quantized_sample.png")

    # Log the fixed-point input state for basic debug trace
    debug_trace["input"] = {
        "f_in": f_in,
        "float_tensor": image_tensor.cpu().numpy().tolist(),
        "quantized_tensor": q_x.cpu().numpy().tolist(),
    }

    # Optional detailed fixed-point decomposition for the input tensor
    if log:
        global fixed_point_log
        fixed_point_log = {"input": {}, "layers": []}
        fixed_point_log["input"] = {
            "f": f_in,
            "decomposition": _decompose_fixed_point_tensor(q_x, f_in),
        }

    print("\n[3] Executing Dynamic Fixed-Point Inference...")

    layer_cfg = _get_layer_config(model)

    q_x, f_out, f_w, shift = run_fixed_point_layer(
        q_x, layer_cfg["conv1"], "conv1", f_in, apply_relu=True, is_conv=True, log_details=log
    )
    q_x = avg_pool_fixed_point(q_x, name="pool_after_conv1")

    q_x, f_out, f_w, shift = run_fixed_point_layer(
        q_x, layer_cfg["conv2"], "conv2", f_out, apply_relu=True, is_conv=True, log_details=log
    )
    q_x = avg_pool_fixed_point(q_x, name="pool_after_conv2")

    q_x = q_x.view(q_x.size(0), -1)  # Flatten

    q_x, f_out, f_w, shift = run_fixed_point_layer(
        q_x, layer_cfg["fc1"], "fc1", f_out, apply_relu=True, is_conv=False, log_details=log
    )
    q_x, f_out, f_w, shift = run_fixed_point_layer(
        q_x, layer_cfg["fc2"], "fc2", f_out, apply_relu=True, is_conv=False, log_details=log
    )
    fc3_f_in = f_out

    # Final Layer (No ReLU)
    # Use 'final_' prefixes so we don't overwrite the previous variables
    q_out, final_f_out, final_f_w, final_shift = run_fixed_point_layer(
        q_x, layer_cfg["fc3"], "fc3", fc3_f_in, apply_relu=False, is_conv=False, log_details=log
    )

    # Dequantize final output using fixed-point math
    dequantized_logits = dequantize_fixed_point(q_out, final_f_out)
    int_pred = dequantized_logits.argmax(dim=1).item()

    # -----------------------------
    # 6. Summary Logging
    # -----------------------------
    print("\n" + "=" * 40)
    print(" INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"True Label:                   {true_label}")
    print(f"Float Model Prediction:       {float_pred}")
    print(f"Fixed Point Model Prediction: {int_pred}")

    if float_pred == int_pred:
        print(
            "\nSuccess! The Fixed-Point-quantized model matches the floating-point prediction."
        )
    else:
        print(
            "\nNote: The predictions differ. This can happen with 8-bit quantization on border cases, but usually, they match."
        )

    print("\n--- Final Layer Dynamic Fixed-Point Stats ---")
    print(f"Input Fractional Bits (f_in):         {fc3_f_in}") 
    print(f"Weight Fractional Bits (f_w):         {final_f_w}")
    print(f"Accumulator Fractional Bits:          {fc3_f_in + final_f_w}")
    print(f"Right Bit-Shift Amount (>>):          {final_shift}")
    print(f"Final Output Fractional Bits (f_out): {final_f_out}")
    print("=" * 40)

    # Optionally write detailed fixed-point log to inference.json
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
