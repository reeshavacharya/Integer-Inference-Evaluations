import argparse
import os
import random
import json

import torch
import torch.nn as nn

import resnet18 as train_mod
from resnet18 import ResNet18

from utils import (
    quantize_fixed_point,
    dequantize_fixed_point,
    fixed_point_relu,
    execute_and_shift_conv2d,
    execute_and_shift_linear,
    add_bias,
    fixed_point_global_avg_pool2d,
)


# -----------------------------
# Debug / trace storage
# -----------------------------
debug_trace = {"input": {}, "layers": [], "pooling": []}

# MNIST-only integer trace (no floats) for inspecting quantized path
INT_TRACE_ENABLED = False
int_trace = {"input": {}, "layers": []}


# -----------------------------
# 1. Model Definition (mirror training ResNet18)
# -----------------------------


class FloatAdd(nn.Module):
    """A dummy module to make addition visible to calibration hooks."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.add = FloatAdd()
        self.relu2 = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        skip = self.shortcut(x)
        out = self.add(out, skip)

        out = self.relu2(out)
        return out


class ResNet18Inference(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(ResNet18Inference, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# -----------------------------
# 2. Setup and Data Extraction
# -----------------------------


def get_random_sample(dataset_name: str):
    """Return a single preprocessed sample tensor and label.

    Uses the same 10% test partition as defined in resnet18.py for
    MNIST, CIFAR10, and Brain-MRI by calling the corresponding
    setup_* function and sampling from its test_loader.
    """

    name = dataset_name.upper()

    if name == "MNIST":
        train_mod.setup_MNIST(batch_size=1)
        test_dataset = train_mod.test_loader.dataset
    elif name in ("CIFR10", "CIFAR10"):
        train_mod.setup_CIFAR10(batch_size=1)
        test_dataset = train_mod.test_loader.dataset
    elif name == "BRAIN-MRI":
        train_mod.setup_Brain_MRI(batch_size=1)
        test_dataset = train_mod.test_loader.dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    idx = random.randint(0, len(test_dataset) - 1)
    image_tensor, label = test_dataset[idx]
    return image_tensor.unsqueeze(0), label


def fold_conv_bn_eval(conv, bn):
    """Folds BatchNorm parameters into Conv2d weights and biases."""
    w = conv.weight.detach()
    if conv.bias is not None:
        b = conv.bias.detach()
    else:
        b = torch.zeros(conv.out_channels, device=w.device)

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    # Calculate multiplier: gamma / sqrt(var + eps)
    multiplier = gamma / torch.sqrt(var + eps)

    # Fold into weights: W * multiplier
    w_folded = w * multiplier.view(-1, 1, 1, 1)

    # Fold into bias: beta + (b - mean) * multiplier
    b_folded = beta + (b - mean) * multiplier

    return w_folded, b_folded


# -----------------------------
# 4. Core Fixed-Point Inference Engine
# -----------------------------


def _get_layer_config(model: ResNet18Inference):
    """Return conv/fc modules for calibration and integer inference.

    We focus on top-level conv1 and the four residual stages plus the
    final fully-connected layer.
    """

    return {
        "conv1": model.conv1,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
        "fc": model.fc,
    }


def run_static_fixed_point_conv_block(q_input, conv, bn, apply_relu=True):
    # Fold BN into Conv
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)

    # Quantize Weights and Biases statically
    q_w = quantize_fixed_point(w_folded)
    q_bias = quantize_fixed_point(b_folded)

    # Math
    q_accum = execute_and_shift_conv2d(
        q_input, q_w, stride=conv.stride[0], padding=conv.padding[0]
    )
    q_out = add_bias(q_accum, q_bias)

    if apply_relu:
        q_out = fixed_point_relu(q_out)

    return q_out


def run_static_fixed_point_basic_block(q_x, block):
    # 1. Main Branch
    q_out1 = run_static_fixed_point_conv_block(
        q_x, block.conv1, block.bn1, apply_relu=True
    )
    q_out2 = run_static_fixed_point_conv_block(
        q_out1, block.conv2, block.bn2, apply_relu=False
    )

    # 2. Shortcut Branch
    if isinstance(block.shortcut, nn.Identity):
        q_short = q_x
    else:
        short_conv, short_bn = block.shortcut[0], block.shortcut[1]
        q_short = run_static_fixed_point_conv_block(
            q_x, short_conv, short_bn, apply_relu=False
        )

    # 3. Direct Addition (No alignment shifts required!)
    q_added = torch.clamp(q_out2 + q_short, -9223372036854775808, 9223372036854775807)

    # 4. Final ReLU
    return fixed_point_relu(q_added)


def run_static_fixed_point_fc(q_input, fc):
    q_w = quantize_fixed_point(fc.weight.detach())
    q_bias = quantize_fixed_point(fc.bias.detach())

    q_out, max_bits, max_rem = execute_and_shift_linear(q_input, q_w)
    q_out = add_bias(q_out, q_bias)

    return q_out, max_bits, max_rem


# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data: str):
    print("--- Starting ResNet18 Quantized Inference Pipeline ---")

    name = infer_data.upper()

    global INT_TRACE_ENABLED, int_trace
    INT_TRACE_ENABLED = name == "MNIST"
    if INT_TRACE_ENABLED:
        # reset trace for this run
        int_trace = {"input": {}, "layers": []}

    if name == "MNIST":
        model = ResNet18Inference(num_classes=10, in_channels=1)
        model_path = "best_resnet18_mnist.pth"
    elif name in ("CIFR10", "CIFAR10"):
        model = ResNet18Inference(num_classes=10, in_channels=3)
        model_path = "best_resnet18_cifar10.pth"
    elif name == "BRAIN-MRI":
        # Note: Change in_channels=3 if your MRI training script used RGB images
        model = ResNet18Inference(num_classes=4, in_channels=1)
        model_path = "best_resnet18_brain_mri.pth"
    else:
        raise ValueError(f"Unknown dataset: {infer_data}")

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return

    # Safe State Loading (Strips 'module.' prefix if saved via DataParallel)
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()

    # ... [Keep initial setup and get_random_sample logic] ...
    image_tensor, true_label = get_random_sample(infer_data)
    print(f"\n[1] Extracted random {infer_data} sample (True Label: {true_label}).")

    # Float Inference (For accuracy comparison only)
    with torch.no_grad():
        float_output = model(image_tensor)
        float_pred = float_output.argmax(dim=1).item()
    print(f"[2] Float Inference complete. Prediction: {float_pred}")

    # Quantize Input Image directly to Q31.32
    q_x = quantize_fixed_point(image_tensor)

    print("\n[3] Executing Static 64-Bit Fixed-Point Inference...")

    # Initial conv1
    q_x = run_static_fixed_point_conv_block(
        q_x, model.conv1, model.bn1, apply_relu=True
    )

    # Traverse all residual blocks
    for layer_idx, stage in enumerate(
        [model.layer1, model.layer2, model.layer3, model.layer4], 1
    ):
        for block_idx, block in enumerate(stage):
            q_x = run_static_fixed_point_basic_block(q_x, block)

    # Global Average Pooling
    q_pooled = fixed_point_global_avg_pool2d(q_x)
    q_fc_in = q_pooled.view(q_pooled.size(0), -1)

    # Run Final FC Layer
    q_out, max_bits_used, max_remainder = run_static_fixed_point_fc(q_fc_in, model.fc)

    # Dequantize final output
    dequantized_logits = dequantize_fixed_point(q_out)
    int_pred = dequantized_logits.argmax(dim=1).item()

    # Logging
    print("\n" + "=" * 40)
    print(" RESNET18 INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"True Label:                   {true_label}")
    print(f"Float Model Prediction:       {float_pred}")
    print(f"Static 64-bit Prediction:     {int_pred}")

    if float_pred == int_pred:
        print(
            "\nSuccess! The 64-bit static model exactly matches the floating-point prediction."
        )
    else:
        print(
            "\nNote: The predictions differ. This can occasionally happen due to 32-bit truncation loss."
        )

    # ZK Logging Block
    print("\n--- ZK Cryptographic Fixed-Point Stats (Final Layer) ---")
    print(f"Architecture Format:         Q31.32 (64-bit Static Container)")
    print(
        f"Accumulator Max Bit-Length:  {max_bits_used} bits (Threshold: 63 for PyTorch, ~254 for ZK Field)"
    )

    headroom_used = (max_bits_used / 63.0) * 100 if max_bits_used else 0
    print(f"PyTorch Container Headroom:  {headroom_used:.1f}% Capacity Reached")
    print(
        f"Max Truncation Remainder:    {max_remainder:.0f} (Precision dropped during >> 32 shift)"
    )

    print("\n--- Final Logit Sanity Check ---")
    print(f"Raw 64-bit Integer Max:      {q_out.max().item()}")
    print(f"Dequantized Float Max:       {dequantized_logits.max().item():.4f}")
    print("=" * 56)

    # Save MNIST integer-only layer outputs (no floats) if enabled
    # if INT_TRACE_ENABLED:
    #     trace_path = f"mnist_integer_inference_trace.json"
    #     with open(trace_path, "w") as f:
    #         json.dump(int_trace, f, indent=2)
    #     print(f"Saved MNIST integer trace to {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        type=str,
        default="CIFAR10",
        help="Inference data to use (MNIST, CIFAR10, Brain-MRI)",
    )
    args = parser.parse_args()
    main(args.infer)
