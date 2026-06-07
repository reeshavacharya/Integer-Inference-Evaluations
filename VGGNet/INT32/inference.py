import argparse
import os
import sys
import time
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VGG_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path: sys.path.insert(0, THIS_DIR)
if VGG_DIR not in sys.path: sys.path.insert(1, VGG_DIR)

import vgg19 as train_mod
from vgg19 import VGG19

def _normalize_dataset_name(dataset_name: str) -> str:
	key = dataset_name.strip().upper().replace(" ", "-")
	if key == "MNIST":
		return "MNIST"
	if key == "CIFR10":
		return "CIFAR10"
	if key == "BRAIN_MRI":
		return "Brain_MRI"
	if key == "OCTMNIST":
		return "OCTMNIST"
	if key == "ORGANAMNIST":
		return "OrganAMNIST"
	if key == "BLOODMNIST":
		return "BloodMNIST"
	if key == "PNEUMONIAMNIST":
		return "PneumoniaMNIST"
	raise ValueError(f"Unknown dataset: {dataset_name}")

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
    if name == "BRAIN_MRI":
        return {
            "display": "Brain_MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": VGG19(num_classes=4, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_brain_mri.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
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
    if name == "PNEUMONIAMNIST":
        return {
            "display": "PneumoniaMNIST",
            "setup_fn": train_mod.setup_PneumoniaMNIST,
            "model": VGG19(num_classes=2, in_channels=1),
            "model_path": os.path.join(VGG_DIR, "best_vgg19_pneumoniamnist.pth"),
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    raise ValueError(f"Unknown dataset config: {name}")

from utils import (
    quantize_tensor, add_bias,
    downscale_and_cast, quantized_relu, integer_gelu_lut
)
from strict_int_ops import (
    strict_integer_conv2d, strict_integer_linear, strict_integer_max_pool2d
)
import torch.nn.functional as F

def run_integer_layer(q_input, layer_data, zp_in, layer_type="conv", apply_relu=True, apply_maxpool=False, activation="relu"):
    q_w = layer_data["q_weight"].to(q_input.device)
    zp_w = layer_data["zp_w"]
    q_bias = layer_data["q_bias"].to(q_input.device)
    q_M0 = layer_data["q_M0"]
    shift = layer_data["shift"]
    zp_out = layer_data["zp_out"]

    if layer_type == "conv":
        int32_accum = strict_integer_conv2d(q_input, q_w, zp_in, zp_w, stride=1, padding=1)
    else:
        int32_accum = strict_integer_linear(q_input, q_w, zp_in, zp_w)

    int32_accum = add_bias(int32_accum, q_bias)
    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)
    
    if apply_relu:
        if activation == "gelu":
            q_out = integer_gelu_lut(q_out, layer_data["gelu_lut"], zp_out)
        elif activation == "leaky_relu":
            q_out = q_out
        else:
            q_out = quantized_relu(q_out, zp_out)
        
    if apply_maxpool:
        q_out = strict_integer_max_pool2d(q_out, kernel_size=2, stride=2)

    return q_out, layer_data["scale_out"], zp_out


def main(infer_data: str, activation: str):
    print(f"--- Starting VGG19 INT32 (Overflow Prone) Pipeline | Activation: {activation} ---")
    device = torch.device("cpu")
    
    cfg = _resolve_infer_config(infer_data)
    dataset_key = cfg["display"]
    int32_path = os.path.join(VGG_DIR, f"best_vgg19_{activation}_{dataset_key.lower().replace(' ', '_').replace('-', '_')}_int32.pth")

    if not os.path.exists(int32_path):
        print(f"\n[!] Missing offline dictionary: {int32_path}")
        return

    int32_state = torch.load(int32_path, map_location=device)
    
    c = 3 if dataset_key in ["CIFAR10", "BLOODMNIST"] else 1
    image_tensor = torch.randn(1, c, 32, 32)

    # 1. Quantize Input to 32-bit
    scale_in = int32_state["meta"]["in_scale"]
    zp_in = int32_state["meta"]["in_zp"]
    q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint32)

    # 2. Traverse VGG Features 
    conv_indices = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49]
    maxpool_indices = [6, 13, 26, 39, 52] 

    s_out, z_out = scale_in, zp_in
    for conv_idx in conv_indices:
        layer_name = f"features_{conv_idx}"
        apply_pool = (conv_idx + 3) in maxpool_indices 
        
        # Inject LUT into layer_data if GELU
        layer_data = int32_state[layer_name]
        if activation == "gelu":
            layer_data["gelu_lut"] = int32_state[f"{layer_name}_gelu_lut"]
        
        q_x, s_out, z_out = run_integer_layer(
            q_x, layer_data, z_out, 
            layer_type="conv", apply_relu=True, apply_maxpool=apply_pool, activation=activation
        )

    # 3. Adaptive Max Pool Bridging
    if q_x.shape[2] == 1 and q_x.shape[3] == 1:
        q_pooled = q_x.expand(-1, -1, 7, 7)
    elif q_x.shape[2] == 7 and q_x.shape[3] == 7:
        q_pooled = q_x
    else:
        raise ValueError(f"Unexpected spatial size before classifier: {q_x.shape}")
        
    q_fc_in = torch.flatten(q_pooled, 1)

    # Note: Since there's no arithmetic inside max pool, the output scale/zp remain identical to the input
    fc_in_scale = int32_state["classifier_0"]["scale_in"]
    fc_in_zp = int32_state["classifier_0"]["zp_in"]

    # 4. Traverse VGG Classifier 
    s_out, z_out = fc_in_scale, fc_in_zp
    fc_indices = [0, 3, 6] 
    
    for i, fc_idx in enumerate(fc_indices):
        layer_name = f"classifier_{fc_idx}"
        is_last = (i == len(fc_indices) - 1)
        
        layer_data = int32_state[layer_name]
        if not is_last and activation == "gelu":
            layer_data["gelu_lut"] = int32_state[f"{layer_name}_gelu_lut"]
            
        q_fc_in, s_out, z_out = run_integer_layer(
            q_fc_in, layer_data, z_out, 
            layer_type="linear", apply_relu=not is_last, apply_maxpool=False, activation=activation
        )

    # 5. Dequantize Logits
    int_logits = q_fc_in.to(torch.float32)
    dequantized_logits = s_out * (int_logits - z_out)
    int_pred = dequantized_logits.argmax(dim=1).item()
    
    print("=" * 40)
    print(f"Integer Model Prediction: {int_pred}")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default="MNIST")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "leaky_relu"])
    args = parser.parse_args()
    main(args.infer, args.activation)