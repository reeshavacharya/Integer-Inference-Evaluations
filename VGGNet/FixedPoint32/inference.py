import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VGG_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if VGG_DIR not in sys.path:
    sys.path.insert(1, VGG_DIR)

import vgg19 as train_mod
from vgg19 import VGG19

from utils import (
    quantize_fixed_point,
    dequantize_fixed_point,
    fixed_point_relu,
    execute_and_shift_conv2d,
    execute_and_shift_linear,
    add_bias,
    fixed_point_adaptive_avg_pool2d,
    fixed_point_max_pool2d,
)

# -----------------------------
# Inference Helpers & Setup
# -----------------------------
def _resolve_infer_config(infer_data: str):
    name = infer_data.upper()
    if name == "MNIST":
        return {"display": "MNIST", "setup_fn": train_mod.setup_MNIST, "model": VGG19(10, 1), "model_path": "best_vgg19_mnist.pth", "is_multilabel": False}
    if name in ("CIFR10", "CIFAR10"):
        return {"display": "CIFAR10", "setup_fn": train_mod.setup_CIFAR10, "model": VGG19(10, 3), "model_path": "best_vgg19_cifar10.pth", "is_multilabel": False}
    if name == "BRAIN_MRI":
        return {"display": "Brain_MRI", "setup_fn": train_mod.setup_Brain_MRI, "model": VGG19(4, 1), "model_path": "best_vgg19_brain_mri.pth", "is_multilabel": False}
    if name == "OCTMNIST":
        return {"display": "OCTMNIST", "setup_fn": train_mod.setup_OCTMNIST, "model": VGG19(4, 1), "model_path": "best_vgg19_octmnist.pth", "is_multilabel": False}
    if name == "BLOODMNIST":
        return {"display": "BloodMNIST", "setup_fn": train_mod.setup_BloodMNIST, "model": VGG19(8, 3), "model_path": "best_vgg19_bloodmnist.pth", "is_multilabel": False}
    if name == "ORGANAMNIST":
        return {"display": "OrganAMNIST", "setup_fn": train_mod.setup_OrganAMNIST, "model": VGG19(11, 1), "model_path": "best_vgg19_organamnist.pth", "is_multilabel": False}
    raise ValueError(f"Unknown dataset: {infer_data}")


def fold_conv_bn_eval(conv, bn):
    """Folds BatchNorm parameters into Conv2d weights and biases."""
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    multiplier = gamma / torch.sqrt(var + eps)
    w_folded = w * multiplier.view(-1, 1, 1, 1)
    b_folded = beta + (b - mean) * multiplier

    return w_folded, b_folded


# -----------------------------
# Forward Execution Engine
# -----------------------------
def run_fixed_point_conv_bn_relu(q_input, conv, bn, apply_relu=True):
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)
    q_w = quantize_fixed_point(w_folded)
    q_bias = quantize_fixed_point(b_folded)

    q_accum, max_bits, max_rem = execute_and_shift_conv2d(q_input, q_w, stride=1, padding=1)
    q_out = add_bias(q_accum, q_bias)

    if apply_relu:
        q_out = fixed_point_relu(q_out)

    return q_out


def run_fixed_point_features(q_x, features):
    layers = list(features)
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, nn.Conv2d):
            if i + 1 < len(layers) and isinstance(layers[i + 1], nn.BatchNorm2d):
                bn_layer = layers[i + 1]
                q_x = run_fixed_point_conv_bn_relu(q_x, layer, bn_layer, apply_relu=True)
                i += 3  # Skip Conv, BN, ReLU
            else:
                i += 1
        elif isinstance(layer, nn.MaxPool2d):
            q_x = fixed_point_max_pool2d(q_x, kernel_size=2, stride=2)
            i += 1
        else:
            i += 1
    return q_x


def run_fixed_point_fc(q_input, fc):
    q_w = quantize_fixed_point(fc.weight.detach())
    q_bias = quantize_fixed_point(fc.bias.detach() if fc.bias is not None else torch.zeros(fc.out_features))

    q_out, max_bits, max_rem = execute_and_shift_linear(q_input, q_w)
    q_out = add_bias(q_out, q_bias)

    return q_out, max_bits, max_rem


def run_fixed_point_classifier(q_x, classifier):
    fc1, fc2, fc3 = classifier[0], classifier[3], classifier[6]

    q_x, _, _ = run_fixed_point_fc(q_x, fc1)
    q_x = fixed_point_relu(q_x)

    q_x, _, _ = run_fixed_point_fc(q_x, fc2)
    q_x = fixed_point_relu(q_x)

    q_out, final_max_bits, _ = run_fixed_point_fc(q_x, fc3)
    return q_out, final_max_bits


# -----------------------------
# Main Execution
# -----------------------------
def main(infer_data: str):
    print("--- Starting VGG19 Static 32-Bit Fixed-Point (Q15.16) Inference ---")

    cfg = _resolve_infer_config(infer_data)
    dataset_display = cfg["display"]
    model = cfg["model"]
    model_path = os.path.join(VGG_DIR, cfg["model_path"])

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return

    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Grab a pseudo-random image for inference testing
    c = 3 if "CIFAR" in dataset_display or "BLOOD" in dataset_display else 1
    image_tensor = torch.randn(1, c, 32, 32)

    # Float baseline
    with torch.no_grad():
        float_output = model(image_tensor)
        float_pred = float_output.argmax(dim=1).item()

    # Fixed Point Execution
    q_x = quantize_fixed_point(image_tensor)
    q_x = run_fixed_point_features(q_x, model.features)
    
    q_pooled = fixed_point_adaptive_avg_pool2d(q_x, output_size=(7, 7))
    q_fc_in = q_pooled.view(q_pooled.size(0), -1)
    
    q_out, final_bits = run_fixed_point_classifier(q_fc_in, model.classifier)
    dequantized_logits = dequantize_fixed_point(q_out)
    int_pred = dequantized_logits.argmax(dim=1).item()

    print("\n" + "=" * 40)
    print(" VGG19 INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"Dataset:                      {dataset_display}")
    print(f"Float Model Prediction:       {float_pred}")
    print(f"Static 32-bit Prediction:     {int_pred}")

    print("\n--- ZK Cryptographic Fixed-Point Stats ---")
    print(f"Architecture Format:         Q15.16 (32-bit Static Container)")
    print(f"Final Layer Accumulator Max: {final_bits} bits")
    headroom_used = (final_bits / 31.0) * 100 if final_bits else 0
    print(f"PyTorch Container Headroom:  {headroom_used:.1f}% Capacity Reached")
    print("=" * 56)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default="CIFAR10")
    args = parser.parse_args()
    main(args.infer)