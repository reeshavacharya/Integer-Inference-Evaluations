import argparse
import os
import sys
import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VGG_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path: sys.path.insert(0, THIS_DIR)
if VGG_DIR not in sys.path: sys.path.insert(1, VGG_DIR)

import vgg19 as train_mod
from vgg19 import VGG19
from utils import (
    quantize_tensor, integer_conv2d, integer_linear, add_bias,
    downscale_and_cast, quantized_relu, integer_max_pool2d, integer_adaptive_avg_pool
)


def _normalize_dataset_name(dataset_name: str) -> str:
    name = dataset_name.strip().upper().replace("_", "-").replace(" ", "-")
    if name == "CIFR10":
        return "CIFAR10"
    return name


def _resolve_infer_config(infer_data: str):
    name = _normalize_dataset_name(infer_data)

    if name == "MNIST":
        return {
            "display": "MNIST",
            "setup_fn": train_mod.setup_MNIST,
            "model": VGG19(num_classes=10, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_mnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "CIFAR10":
        return {
            "display": "CIFAR10",
            "setup_fn": train_mod.setup_CIFAR10,
            "model": VGG19(num_classes=10, in_channels=3),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_cifar10.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "BRAIN-MRI":
        return {
            "display": "Brain-MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": VGG19(num_classes=4, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_brain_mri.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "NIH-CHEST":
        return {
            "display": "NIH-CHEST",
            "setup_fn": train_mod.setup_NIH_Chest,
            "model": VGG19(num_classes=15, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_NIH_Chest_XRay.pth"),
            "is_multilabel": True,
            "eval_batch_size": 8,
        }

    if name == "OCTMNIST":
        return {
            "display": "OCTMNIST",
            "setup_fn": train_mod.setup_OCTMNIST,
            "model": VGG19(num_classes=4, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_octmnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "BLOODMNIST":
        return {
            "display": "BloodMNIST",
            "setup_fn": train_mod.setup_BloodMNIST,
            "model": VGG19(num_classes=8, in_channels=3),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_bloodmnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "ORGANAMNIST":
        return {
            "display": "OrganAMNIST",
            "setup_fn": train_mod.setup_OrganAMNIST,
            "model": VGG19(num_classes=11, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_organamnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    raise ValueError(f"Unknown dataset: {infer_data}")

# -----------------------------
# Integer Runners
# -----------------------------
def run_integer_layer(q_input, layer_data, zp_in, layer_type="conv", apply_relu=True, apply_maxpool=False):
    """Generic runner that handles both Conv and Linear VGG blocks."""
    q_w = layer_data["q_weight"].to(q_input.device)
    zp_w = layer_data["zp_w"]
    q_bias = layer_data["q_bias"].to(q_input.device)
    q_M0 = layer_data["q_M0"]
    shift = layer_data["shift"]
    zp_out = layer_data["zp_out"]

    if layer_type == "conv":
        int32_accum = integer_conv2d(q_input, q_w, zp_in, zp_w, stride=1, padding=1)
    else:
        int32_accum = integer_linear(q_input, q_w, zp_in, zp_w)

    int32_accum = add_bias(int32_accum, q_bias)
    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)
    
    if apply_relu:
        q_out = quantized_relu(q_out, zp_out)
        
    if apply_maxpool:
        q_out = integer_max_pool2d(q_out, kernel_size=2, stride=2)

    return q_out, layer_data["scale_out"], zp_out

# -----------------------------
# Main Execution
# -----------------------------
def main(infer_data: str):
    print("--- Starting VGG19 Quantized Inference Pipeline ---")
    device = torch.device("cpu") # Benchmarking pure math on CPU for safety
    
    # Resolve Paths
    dataset_key = infer_data.upper()
    model_name_map = {
        "MNIST": "best_vgg19_mnist.pth",
        "CIFAR10": "best_vgg19_cifar10.pth",
        "BRAIN-MRI": "best_vgg19_brain_mri.pth",
        "NIH-CHEST": "best_vgg19_NIH_Chest_XRay.pth"
    }
    float_path = os.path.join(VGG_DIR, model_name_map[dataset_key])
    int8_path = float_path.replace(".pth", "_int8.pth")

    if not os.path.exists(int8_path):
        print(f"\n[!] Missing offline dictionary: {int8_path}")
        print("Please run export_int8_model.py to compile the VGG19 model first.")
        return

    print(f"[0] Loading compiled integer weights from: {int8_path}")
    int8_state = torch.load(int8_path, map_location=device)
    
    # Dummy Image (Replace with dataloader for benchmark)
    print(f"\n[1] Generating random {dataset_key} test tensor for inference test...")
    if dataset_key == "NIH-CHEST":
        image_tensor = torch.randn(1, 1, 224, 224)
    else:
        c = 3 if dataset_key == "CIFAR10" else 1
        image_tensor = torch.randn(1, c, 32, 32)

    integer_start = time.perf_counter()
    
    # 1. Quantize Input
    scale_in = int8_state["meta"]["in_scale"]
    zp_in = int8_state["meta"]["in_zp"]
    q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint8)

    print("[2] Executing TRUE Integer-Only Inference...")

    # 2. Traverse VGG Features (Convs + MaxPools)
    # These exact indices map to your [64, 64, M, 128, 128, M, 256...] configuration
    conv_indices = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49]
    maxpool_indices = [6, 13, 26, 39, 52] 

    s_out, z_out = scale_in, zp_in
    for conv_idx in conv_indices:
        layer_name = f"features_{conv_idx}"
        
        # In this config, pooling always happens exactly 2 steps after a ReLU
        apply_pool = (conv_idx + 3) in maxpool_indices 
        
        q_x, s_out, z_out = run_integer_layer(
            q_x, int8_state[layer_name], z_out, 
            layer_type="conv", apply_relu=True, apply_maxpool=apply_pool
        )

    # 3. Adaptive Avg Pool Bridging
    fc_in_scale = int8_state["classifier_0"]["scale_in"]
    fc_in_zp = int8_state["classifier_0"]["zp_in"]
    q_pooled = integer_adaptive_avg_pool(q_x, z_out, s_out, fc_in_zp, fc_in_scale, output_size=(7,7))
    q_fc_in = torch.flatten(q_pooled, 1)

    # 4. Traverse VGG Classifier (FCs)
    s_out, z_out = fc_in_scale, fc_in_zp
    fc_indices = [0, 3, 6] # Linear layers in VGG classifier
    
    for i, fc_idx in enumerate(fc_indices):
        layer_name = f"classifier_{fc_idx}"
        is_last = (i == len(fc_indices) - 1)
        
        q_fc_in, s_out, z_out = run_integer_layer(
            q_fc_in, int8_state[layer_name], z_out, 
            layer_type="linear", apply_relu=not is_last, apply_maxpool=False
        )

    # 5. Dequantize Logits
    int_logits = q_fc_in.to(torch.float32)
    dequantized_logits = s_out * (int_logits - z_out)
    int_pred = dequantized_logits.argmax(dim=1).item()
    
    integer_inference_time = time.perf_counter() - integer_start

    print("\n" + "=" * 40)
    print(" VGG19 INTEGER INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"Integer Model Prediction: {int_pred}")
    print(f"Integer Inference Time:   {integer_inference_time:.4f}s")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default="MNIST")
    args = parser.parse_args()
    main(args.infer)