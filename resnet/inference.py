import argparse
import os
import random
import json

import torch
import torch.nn as nn

import resnet18 as train_mod
from resnet18 import ResNet18

from utils import (
    get_fractional_bits,
    quantize_fixed_point,
    dequantize_fixed_point,
    downscale_fixed_point,
    fixed_point_relu,
    fixed_point_conv2d,
    fixed_point_linear,
    add_bias,
    fixed_point_add,
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


# -----------------------------
# 3. Calibration Hooks
# -----------------------------
activation_ranges = {}


def calibration_hook(module, input, output, name):
    activation_ranges[name] = {
        "in_abs_max": input[0].detach().abs().max().item(),
        "out_abs_max": output.detach().abs().max().item(),
    }


def register_hooks(model: ResNet18Inference):
    handles = []

    # 1. Hook conv1 JUST to capture the raw input image bounds
    handles.append(
        model.conv1.register_forward_hook(
            lambda m, i, o: calibration_hook(m, i, o, "conv1")
        )
    )

    # 2. Hook initial conv block output (post-relu) for the first integer pass
    handles.append(
        model.relu.register_forward_hook(
            lambda m, i, o: calibration_hook(m, i, o, "conv1_relu")
        )
    )

    # Hook the distinct outputs of every block stage
    for layer_idx, layer in enumerate(
        [model.layer1, model.layer2, model.layer3, model.layer4], 1
    ):
        for block_idx, block in enumerate(layer):
            prefix = f"layer{layer_idx}_block{block_idx}"

            # Post-conv1 block (after ReLU)
            handles.append(
                block.relu1.register_forward_hook(
                    lambda m, i, o, p=prefix: calibration_hook(
                        m, i, o, f"{p}_conv1_relu"
                    )
                )
            )
            # Post-conv2 block (after BN, NO ReLU)
            handles.append(
                block.bn2.register_forward_hook(
                    lambda m, i, o, p=prefix: calibration_hook(
                        m, i, o, f"{p}_conv2_out"
                    )
                )
            )
            # Post-shortcut
            handles.append(
                block.shortcut.register_forward_hook(
                    lambda m, i, o, p=prefix: calibration_hook(
                        m, i, o, f"{p}_shortcut_out"
                    )
                )
            )
            # Final Block Output (after add -> relu2)
            handles.append(
                block.relu2.register_forward_hook(
                    lambda m, i, o, p=prefix: calibration_hook(m, i, o, f"{p}_out")
                )
            )

    handles.append(
        model.fc.register_forward_hook(lambda m, i, o: calibration_hook(m, i, o, "fc"))
    )

    return handles


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
def run_fixed_point_conv_block(q_input, conv, bn, layer_name, f_in, apply_relu=True):
    # Fold BN into Conv
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)
    
    # Quantize Weights
    f_w = get_fractional_bits(w_folded, num_bits=8)
    q_w = quantize_fixed_point(w_folded, f_w, dtype=torch.int8)

    # Get target output format
    out_abs_max = activation_ranges[layer_name]["out_abs_max"]
    pseudo_out = torch.tensor([out_abs_max])
    f_out = get_fractional_bits(pseudo_out, num_bits=8)

    # Accumulator bits = f_in + f_w. Bias must exactly match this!
    f_accum = f_in + f_w
    q_bias = quantize_fixed_point(b_folded, f_accum, dtype=torch.int32)

    # Math
    int32_accum = fixed_point_conv2d(q_input, q_w, stride=conv.stride[0], padding=conv.padding[0])
    int32_accum = add_bias(int32_accum, q_bias)

    # Shift back to 8-bit and apply ReLU
    shift_amount = f_accum - f_out
    q_out_32 = downscale_fixed_point(int32_accum, shift=shift_amount)
    q_out = torch.clamp(q_out_32, -128, 127).to(torch.int8)
    
    if apply_relu:
        q_out = fixed_point_relu(q_out)

    return q_out, f_out

def run_fixed_point_basic_block(q_x, block, prefix, f_in):
    # 1. Main Branch
    q_out1, f_out1 = run_fixed_point_conv_block(
        q_x, block.conv1, block.bn1, f"{prefix}_conv1_relu", f_in, apply_relu=True
    )
    q_out2, f_out2 = run_fixed_point_conv_block(
        q_out1, block.conv2, block.bn2, f"{prefix}_conv2_out", f_out1, apply_relu=False
    )

    # 2. Shortcut Branch
    if isinstance(block.shortcut, nn.Identity):
        q_short, f_short = q_x, f_in
    else:
        short_conv, short_bn = block.shortcut[0], block.shortcut[1]
        q_short, f_short = run_fixed_point_conv_block(
            q_x, short_conv, short_bn, f"{prefix}_shortcut_out", f_in, apply_relu=False
        )

    # 3. Get target f_out for the addition
    out_abs_max = activation_ranges[f"{prefix}_out"]["out_abs_max"]
    pseudo_out = torch.tensor([out_abs_max])
    f_final = get_fractional_bits(pseudo_out, num_bits=8)

    # 4. ALIGN AND ADD
    q_added = fixed_point_add(q_out2, f_out2, q_short, f_short, f_final)

    # 5. Final ReLU
    q_final = fixed_point_relu(q_added)

    return q_final, f_final

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

def run_fixed_point_fc(q_input, fc, layer_name, f_in):
    weight_float = fc.weight.detach()
    f_w = get_fractional_bits(weight_float, num_bits=8)
    q_w = quantize_fixed_point(weight_float, f_w, dtype=torch.int8)

    out_abs_max = activation_ranges[layer_name]["out_abs_max"]
    pseudo_out = torch.tensor([out_abs_max])
    f_out = get_fractional_bits(pseudo_out, num_bits=8)

    f_accum = f_in + f_w
    bias_float = fc.bias.detach()
    q_bias = quantize_fixed_point(bias_float, f_accum, dtype=torch.int32)

    int32_accum = fixed_point_linear(q_input, q_w)
    int32_accum = add_bias(int32_accum, q_bias)

    shift_amount = f_accum - f_out
    q_out_32 = downscale_fixed_point(int32_accum, shift=shift_amount)
    q_out = torch.clamp(q_out_32, -128, 127).to(torch.int8)

    # Final Stats Return for Logging
    return q_out, f_out, f_w, shift_amount

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

    image_tensor, true_label = get_random_sample(infer_data)
    print(f"\n[1] Extracted random {infer_data} sample (True Label: {true_label}).")

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

    print("\n[3] Executing Dynamic Fixed-Point Inference...")

    # Initial conv1
    q_x, f_out = run_fixed_point_conv_block(
        q_x, model.conv1, model.bn1, "conv1_relu", f_in, apply_relu=True
    )

    # Traverse all residual blocks
    for layer_idx, stage in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for block_idx, block in enumerate(stage):
            prefix = f"layer{layer_idx}_block{block_idx}"
            q_x, f_out = run_fixed_point_basic_block(q_x, block, prefix, f_out)

    # Global Average Pooling (Pooling doesn't change fractional bits!)
    q_pooled = fixed_point_global_avg_pool2d(q_x)
    q_fc_in = q_pooled.view(q_pooled.size(0), -1)

    # Run Final FC Layer
    fc_f_in = f_out
    q_out, final_f_out, final_f_w, final_shift = run_fixed_point_fc(
        q_fc_in, model.fc, "fc", fc_f_in
    )

    # Dequantize final output using fixed-point math
    dequantized_logits = dequantize_fixed_point(q_out, final_f_out)
    int_pred = dequantized_logits.argmax(dim=1).item()

    # Logging
    print("\n" + "=" * 40)
    print(" RESNET18 INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"True Label:                   {true_label}")
    print(f"Float Model Prediction:       {float_pred}")
    print(f"Fixed Point Model Prediction: {int_pred}")

    print("\n--- Final Layer Dynamic Fixed-Point Stats ---")
    print(f"Input Fractional Bits (f_in):         {fc_f_in}") 
    print(f"Weight Fractional Bits (f_w):         {final_f_w}")
    print(f"Accumulator Fractional Bits:          {fc_f_in + final_f_w}")
    print(f"Right Bit-Shift Amount (>>):          {final_shift}")
    print(f"Final Output Fractional Bits (f_out): {final_f_out}")
    print("=" * 40)

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
