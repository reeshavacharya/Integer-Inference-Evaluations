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
    compute_integer_multiplier,
    get_quantization_params,
    get_bias_quantization_params,
    compute_multiplier,
    quantize_tensor,
    integer_conv2d,
    integer_linear,
    add_bias,
    downscale_and_cast,
    quantized_relu,
)


# -----------------------------
# Debug trace storage
# -----------------------------
debug_trace = {"input": {}, "layers": [], "pooling": []}


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
    """Hook to capture the min and max of activations during the forward pass."""
    in_tensor = input[0].detach()
    out_tensor = output.detach()

    activation_ranges[name] = {
        "in_min": in_tensor.min().item(),
        "in_max": in_tensor.max().item(),
        "out_min": out_tensor.min().item(),
        "out_max": out_tensor.max().item(),
    }


def _get_layer_config(model):
    """Return the conv/fc modules for calibration and integer inference.

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


# -----------------------------
# 4. Core Integer Inference Engine
# -----------------------------
def run_integer_layer(
    q_input, layer, layer_name, scale_in, zp_in, apply_relu=True, is_conv=False
):
    """
    Executes a single layer entirely in integer arithmetic.
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
    # This now returns the int32 M0 and the bit-shift n
    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    # --- Execute Integer Math (Online Simulation) ---
    if is_conv:
        int32_accum = integer_conv2d(q_input, q_w, zp_in, zp_w)
    else:
        int32_accum = integer_linear(q_input, q_w, zp_in, zp_w)
    int32_accum = add_bias(int32_accum, q_bias)

    # Pass the integer multiplier and shift instead of the float M
    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)

    if apply_relu:
        q_out = quantized_relu(q_out, zp_out)

    # Log layer details for debugging/analysis
    layer_log = {
        "layer_name": layer_name,
        "type": "conv" if is_conv else "linear",
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


def avg_pool_uint8(q_tensor, name=None):
    """Pure integer 2x2 average pooling with stride 2.

    Also logs input and output activations for debugging.
    """
    # 1. Cast to int32 to prevent overflow when summing the 4 pixels
    # (Max sum of four uint8s is 1020, so int16 would also work, but int32 is safe)
    q_int32 = q_tensor.to(torch.int32)

    B, C, H, W = q_int32.shape

    # 2. Reshape the tensor to isolate the 2x2 spatial windows
    # Shape becomes: [Batch, Channels, Height/2, 2, Width/2, 2]
    windows = q_int32.view(B, C, H // 2, 2, W // 2, 2)

    # 3. Sum over the 2x2 window dimensions (dim 3 and 5)
    # Shape becomes: [Batch, Channels, Height/2, Width/2]
    window_sums = windows.sum(dim=(3, 5))

    # 4. Divide by 4 using a right bit-shift by 2.
    # We add 2 before shifting to achieve "round-to-nearest" behavior.
    rounded_avg = (window_sums + 2) >> 2

    # 5. Cast securely back to uint8
    pooled = rounded_avg.to(torch.uint8)

    # Log pooling activations
    pool_log = {
        "name": name or "pool",
        "kernel_size": [2, 2],
        "stride": [2, 2],
        "input_tensor": q_tensor.cpu().numpy().tolist(),
        "output_tensor": pooled.cpu().numpy().tolist(),
    }
    debug_trace["pooling"].append(pool_log)

    return pooled


# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data):
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
    in_range = activation_ranges["conv1"]
    pseudo_in_tensor = torch.tensor([in_range["in_min"], in_range["in_max"]])
    scale_in, zp_in = get_quantization_params(pseudo_in_tensor, num_bits=8)

    q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint8)
    # save the image from q_x for visualization (do not dequantize) save it as quantized_sample.png
    # q_x has shape [1, 1, 28, 28] and dtype uint8; save directly as an 8-bit grayscale image.
    q_x_img = q_x[0, 0].cpu().numpy().astype("uint8")
    Image.fromarray(q_x_img, mode="L").save(
        f"{infer_data.lower()}_quantized_sample.png"
    )

    # Log quantized input tensor
    debug_trace["input"] = {
        "scale": float(scale_in),
        "zero_point": int(zp_in),
        "float_tensor": image_tensor.cpu().numpy().tolist(),
        "quantized_tensor": q_x.cpu().numpy().tolist(),
    }

    print("\n[3] Executing Integer-Only Inference...")

    # Network Forward Pass (Indexing according to architecture config)
    layer_cfg = _get_layer_config(model)

    q_x, s_out, z_out, p_w, p_b, M = run_integer_layer(
        q_x,
        layer_cfg["conv1"],
        "conv1",
        scale_in,
        zp_in,
        apply_relu=True,
        is_conv=True,
    )
    print("M: ", M)
    q_x = avg_pool_uint8(q_x, name="pool_after_conv1")

    q_x, s_out, z_out, p_w, p_b, M = run_integer_layer(
        q_x,
        layer_cfg["conv2"],
        "conv2",
        s_out,
        z_out,
        apply_relu=True,
        is_conv=True,
    )
    q_x = avg_pool_uint8(q_x, name="pool_after_conv2")

    q_x = q_x.view(q_x.size(0), -1)  # Flatten

    q_x, s_out, z_out, p_w, p_b, M = run_integer_layer(
        q_x,
        layer_cfg["fc1"],
        "fc1",
        s_out,
        z_out,
        apply_relu=True,
        is_conv=False,
    )
    q_x, s_out, z_out, p_w, p_b, M = run_integer_layer(
        q_x,
        layer_cfg["fc2"],
        "fc2",
        s_out,
        z_out,
        apply_relu=True,
        is_conv=False,
    )

    # Final Layer (No ReLU)
    q_out, final_s, final_z, final_w, final_b, final_M = run_integer_layer(
        q_x,
        layer_cfg["fc3"],
        "fc3",
        s_out,
        z_out,
        apply_relu=False,
        is_conv=False,
    )

    # Dequantize final output to get logits (for comparison/analysis)
    int_logits = q_out.to(torch.float32)
    dequantized_logits = final_s * (int_logits - final_z)
    int_pred = dequantized_logits.argmax(dim=1).item()
    # -----------------------------
    # 6. Summary Logging
    # -----------------------------
    print("\n" + "=" * 40)
    print(" INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"True Label:               {true_label}")
    print(f"Float Model Prediction:   {float_pred}")
    print(f"Integer Model Prediction: {int_pred}")

    if float_pred == int_pred:
        print(
            "\nSuccess! The integer-quantized model matches the floating-point prediction."
        )
    else:
        print(
            "\nNote: The predictions differ. This can happen with 8-bit quantization on border cases, but usually, they match."
        )

    print("\n--- Final Layer Quantization Stats ---")
    print(f"Weight Scale:      {final_w[0]:.6f}  | Zero-Point: {final_w[1]}")
    print(f"Bias Scale:        {final_b[0]:.6f}  | Zero-Point: {final_b[1]}")
    print(f"Multiplier (M):    {final_M}")
    print(f"Output Scale:      {final_s:.6f}  | Zero-Point: {final_z}")
    print("=" * 40)

    # Save debug trace to JSON for offline inspection
    # trace_path = "integer_inference_trace.json"
    # with open(trace_path, "w") as f:
    #     json.dump(debug_trace, f, indent=2)
    # print(f"\nSaved integer inference trace to {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        type=str,
        default="MNIST",
        help="Inference data to use",
    )
    args = parser.parse_args()
    main(args.infer)
