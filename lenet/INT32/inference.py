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
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LENET_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if LENET_DIR not in sys.path:
    sys.path.insert(1, LENET_DIR)

import lenet5 as train_mod
from lenet5 import MedicalLeNet

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
CALIBRATION_DIR = os.path.join(LENET_DIR, "calibration")


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

def _resolve_infer_config(infer_data: str):
    name = infer_data.upper()

    if name == "MNIST":
        return {
            "display": "MNIST",
            "setup_fn": train_mod.setup_MNIST,
            "model": LeNet5(num_classes=10, in_channels=1),
            "model_path": "best_lenet5_mnist.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name in ("CIFR10", "CIFAR10"):
        return {
            "display": "CIFAR10",
            "setup_fn": train_mod.setup_CIFAR10,
            "model": LeNet5(num_classes=10, in_channels=3),
            "model_path": "best_lenet5_cifar10.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "BRAIN-MRI":
        return {
            "display": "Brain-MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": MedicalLeNet(num_classes=4, in_channels=1),
            "model_path": "best_lenet5_brain_mri.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "NIH-CHEST":
        return {
            "display": "NIH-CHEST",
            "setup_fn": train_mod.setup_NIH_Chest,
            "model": LeNet5(num_classes=15, in_channels=1),
            "model_path": "best_lenet5_NIH_Chest_XRay.pth",
            "is_multilabel": True,
            "eval_batch_size": 8,
        }

    if name == "OCTMNIST":
        return {
            "display": "OCTMNIST",
            "setup_fn": train_mod.setup_OCTMNIST,
            "model": LeNet5(num_classes=4, in_channels=1),
            "model_path": "best_lenet5_octmnist.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "BLOODMNIST":
        return {
            "display": "BloodMNIST",
            "setup_fn": train_mod.setup_BloodMNIST,
            "model": LeNet5(num_classes=8, in_channels=3),
            "model_path": "best_lenet5_bloodmnist.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "ORGANAMNIST":
        return {
            "display": "OrganAMNIST",
            "setup_fn": train_mod.setup_OrganAMNIST,
            "model": LeNet5(num_classes=11, in_channels=1),
            "model_path": "best_lenet5_organamnist.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    raise ValueError(f"Unknown dataset: {infer_data}")


def get_random_sample(dataset_name: str, setup_fn):
    """Return a random sample from the same deterministic 10% test split as training."""

    # Prevent stale loader leakage between different dataset setups.
    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None

    setup_result = setup_fn(batch_size=1)

    # Standard loaders from setup functions
    test_dataset = None
    if train_mod.test_loader is not None:
        test_dataset = train_mod.test_loader.dataset
    elif (
        isinstance(setup_result, tuple)
        and len(setup_result) >= 3
        and hasattr(setup_result[2], "dataset")
    ):
        test_dataset = setup_result[2].dataset

    if test_dataset is None:
        raise RuntimeError(
            f"Could not resolve test split dataset for inference target: {dataset_name}"
        )

    idx = random.randint(0, len(test_dataset) - 1)
    image_tensor, label = test_dataset[idx]
    train_mod.validate_preprocessed_batch(
        image_tensor.unsqueeze(0), dataset_name, stage="inference"
    )

    label_text = str(label)
    if isinstance(label, torch.Tensor):
        if label.numel() == 1:
            label_text = str(int(label.item()))
        else:
            label_text = str(label.detach().cpu().view(-1).tolist())

    return image_tensor.unsqueeze(0), label, label_text


# -----------------------------
# 3. Calibration Hooks
# -----------------------------
activation_ranges = {}


def _calibration_file_name(dataset_display: str):
    return f"{dataset_display.lower().replace(' ', '_').replace('-', '_')}_calibration.json"


def _calibration_file_path(dataset_display: str):
    return os.path.join(CALIBRATION_DIR, _calibration_file_name(dataset_display))


def load_calibration_ranges(dataset_display: str):
    calibration_path = _calibration_file_path(dataset_display)
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(
            f"Missing calibration file for {dataset_display}: {calibration_path}. "
            f"Run lenet/calibration.py for this dataset first."
        )

    with open(calibration_path, "r") as f:
        payload = json.load(f)

    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise RuntimeError(f"Invalid calibration file format: {calibration_path}")

    activation_ranges.clear()
    activation_ranges.update(layers)
    return activation_ranges


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

    if isinstance(model, MedicalLeNet):
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
    q_input, layer_data, layer_name, zp_in, apply_relu=True, is_conv=False
):
    """
    Executes a single layer entirely using the offline-compiled integer arithmetic.
    """
    # 1. Extract purely quantized data from the offline dictionary
    q_w = layer_data["q_weight"].to(q_input.device)
    zp_w = layer_data["zp_w"]
    q_bias = layer_data["q_bias"].to(q_input.device)
    q_M0 = layer_data["q_M0"]
    shift = layer_data["shift"]
    zp_out = layer_data["zp_out"]
    scale_out = layer_data["scale_out"]

    # 2. Execute Integer Math
    if is_conv:
        int32_accum = integer_conv2d(q_input, q_w, zp_in, zp_w)
    else:
        int32_accum = integer_linear(q_input, q_w, zp_in, zp_w)
        
    int32_accum = add_bias(int32_accum, q_bias)

    # 3. Downscale back to INT8 limits
    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)

    if apply_relu:
        q_out = quantized_relu(q_out, zp_out)

    return q_out, scale_out, zp_out


def avg_pool_uint32(q_tensor, name=None):
    # Cast to int64 to prevent overflow when summing four uint32 max values
    q_int64 = q_tensor.to(torch.int64)

    B, C, H, W = q_int64.shape
    windows = q_int64.view(B, C, H // 2, 2, W // 2, 2)
    window_sums = windows.sum(dim=(3, 5))

    rounded_avg = (window_sums + 2) >> 2

    # Secure cast back to uint32
    pooled = rounded_avg.to(torch.uint32)

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
def main(infer_data, run_floating_point=True, run_integer=True):
    print("--- Starting Quantized Inference Pipeline ---")

    cfg = _resolve_infer_config(infer_data)
    model = cfg["model"]
    model_path = cfg["model_path"]
    dataset_display = cfg["display"]

    print(f"[0] Inference target: {dataset_display}")
    print(f"[0] Loading model weights from: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    if run_integer:
        load_calibration_ranges(dataset_display)

    # Draw one random sample from the correct 10% test partition
    image_tensor, true_label, true_label_text = get_random_sample(
        infer_data,
        cfg["setup_fn"],
    )

    print(
        f"\n[1] Extracted random {dataset_display} sample from test split (True Label: {true_label_text})."
    )

    float_pred = None
    float_output = None

    if run_floating_point or run_integer:
        float_output = model(image_tensor)

    if run_integer:
        pass


    if not run_integer:
        print("\n" + "=" * 40)
        print(" INFERENCE SUMMARY ")
        print("=" * 40)
        print(f"Dataset:                  {dataset_display}")
        print(f"True Label:               {true_label_text}")
        print(f"Float Model Prediction:   {float_pred}")
        print("=" * 40)
        return

    # Quantize Input Image to 32-bit
    in_range = activation_ranges["conv1"]
    pseudo_in_tensor = torch.tensor([in_range["in_min"], in_range["in_max"]])
    scale_in, zp_in = get_quantization_params(pseudo_in_tensor, num_bits=32)

    q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint32, num_bits=32)

    # save the image from q_x for visualization (do not dequantize) save it as quantized_sample.png
    # q_x has shape [1, 1, 28, 28] and dtype uint8; save directly as an 8-bit grayscale image.
    sample_path_key = dataset_display.lower().replace("-", "_").replace(" ", "_")
    if q_x.size(1) == 1:
        q_x_img = q_x[0, 0].cpu().numpy().astype("uint8")
        Image.fromarray(q_x_img, mode="L").save(
            f"{sample_path_key}_quantized_sample.png"
        )
    else:
        q_x_img = q_x[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
        Image.fromarray(q_x_img, mode="RGB").save(
            f"{sample_path_key}_quantized_sample.png"
        )

    # Log quantized input tensor
    debug_trace["input"] = {
        "scale": float(scale_in),
        "zero_point": int(zp_in),
        "float_tensor": image_tensor.cpu().numpy().tolist(),
        "quantized_tensor": q_x.cpu().numpy().tolist(),
    }

    print("\n[3] Executing Integer Inference...")

    # Network Forward Pass (Indexing according to architecture config)
    layer_cfg = _get_layer_config(model)

    q_x, s_out, z_out= run_integer_layer(
        q_x,
        layer_cfg["conv1"],
        "conv1",
        scale_in,
        zp_in,
        apply_relu=True,
        is_conv=True,
    )
    q_x = avg_pool_uint32(q_x, name="pool_after_conv1")

    q_x, s_out, z_out = run_integer_layer(
        q_x,
        layer_cfg["conv2"],
        "conv2",
        s_out,
        z_out,
        apply_relu=True,
        is_conv=True,
    )
    q_x = avg_pool_uint32(q_x, name="pool_after_conv2")

    q_x = q_x.view(q_x.size(0), -1)  # Flatten

    q_x, s_out, z_out = run_integer_layer(
        q_x,
        layer_cfg["fc1"],
        "fc1",
        s_out,
        z_out,
        apply_relu=True,
        is_conv=False,
    )
    q_x, s_out, z_out = run_integer_layer(
        q_x,
        layer_cfg["fc2"],
        "fc2",
        s_out,
        z_out,
        apply_relu=True,
        is_conv=False,
    )

    # Final Layer (No ReLU)
    q_out, final_s, final_z = run_integer_layer(
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
    print(f"Dataset:                  {dataset_display}")
    print(f"True Label:               {true_label_text}")
    if run_floating_point:
        print(f"Float Model Prediction:   {float_pred}")
    print(f"Integer Model Prediction: {int_pred}")

    if run_floating_point:
        if float_pred == int_pred:
            print(
                "\nSuccess! The integer-quantized model matches the floating-point prediction."
            )
        else:
            print(
                "\nNote: The predictions differ. This can happen with 8-bit quantization on border cases, but usually, they match."
            )

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
        help=(
            "Inference data to use: MNIST, CIFAR10, Brain-MRI, NIH-CHEST, "
            "OCTMNIST, BloodMNIST, OrganAMNIST"
        ),
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

    main(
        args.infer,
        run_floating_point=run_floating_point,
        run_integer=run_integer,
    )
