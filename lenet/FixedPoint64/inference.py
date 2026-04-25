import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LENET_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if LENET_DIR not in sys.path:
    sys.path.insert(0, LENET_DIR)

import lenet5 as train_mod
from lenet5 import MedicalLeNet
from utils import (
    add_bias,
    dequantize_fixed_point,
    execute_and_shift_conv2d,
    execute_and_shift_linear,
    fixed_point_relu,
    quantize_fixed_point,
)


# -----------------------------
# Debug / logging storage
# -----------------------------
debug_trace = {"input": {}, "layers": [], "pooling": []}
fixed_point_log = {"input": {}, "layers": []}


# -----------------------------
# 1. Model Definition
# -----------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# 2. Dataset/Model Resolution
# -----------------------------
def _multi_cancer_infer_map():
    return {
        "BRAIN-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Brain,
            "model_path": "best_lenet5_multi_brain_cancer.pth",
            "num_classes": 3,
            "in_channels": 3,
            "display": "Brain-Cancer",
        },
        "BREAST-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Breast,
            "model_path": "best_lenet5_multi_breast_cancer.pth",
            "num_classes": 2,
            "in_channels": 3,
            "display": "Breast-Cancer",
        },
        "CERVICAL-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Cervical,
            "model_path": "best_lenet5_multi_cervical_cancer.pth",
            "num_classes": 5,
            "in_channels": 3,
            "display": "Cervical-Cancer",
        },
        "KIDNEY-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Kidney,
            "model_path": "best_lenet5_multi_kidney_cancer.pth",
            "num_classes": 2,
            "in_channels": 3,
            "display": "Kidney-Cancer",
        },
        "LUNG-AND-COLON-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Lung_Colon,
            "model_path": "best_lenet5_multi_lung_and_colon_cancer.pth",
            "num_classes": 5,
            "in_channels": 3,
            "display": "Lung-And-Colon-Cancer",
        },
        "LYMPHOMA-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Lymphoma,
            "model_path": "best_lenet5_multi_lymphoma.pth",
            "num_classes": 3,
            "in_channels": 3,
            "display": "Lymphoma-Cancer",
        },
        "ORAL-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Oral,
            "model_path": "best_lenet5_multi_oral_cancer.pth",
            "num_classes": 2,
            "in_channels": 3,
            "display": "Oral-Cancer",
        },
    }


def _resolve_infer_config(infer_data: str):
    name = infer_data.upper()

    if name == "MNIST":
        return {
            "display": "MNIST",
            "setup_fn": train_mod.setup_MNIST,
            "model": LeNet5(num_classes=10, in_channels=1),
            "model_path": "best_lenet5_mnist.pth",
            "is_multilabel": False,
        }

    if name in ("CIFR10", "CIFAR10"):
        return {
            "display": "CIFAR10",
            "setup_fn": train_mod.setup_CIFAR10,
            "model": LeNet5(num_classes=10, in_channels=3),
            "model_path": "best_lenet5_cifar10.pth",
            "is_multilabel": False,
        }

    if name == "BRAIN-MRI":
        return {
            "display": "Brain-MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": MedicalLeNet(num_classes=4, in_channels=1),
            "model_path": "best_lenet5_brain_mri.pth",
            "is_multilabel": False,
        }

    if name == "CHEST":
        return {
            "display": "CHEST",
            "setup_fn": train_mod.setup_CHEST,
            "model": MedicalLeNet(num_classes=15, in_channels=1),
            "model_path": "best_lenet5_chest.pth",
            "is_multilabel": True,
        }

    multi_map = _multi_cancer_infer_map()
    if name in multi_map:
        cfg = multi_map[name]
        return {
            "display": cfg["display"],
            "setup_fn": cfg["setup_fn"],
            "model": MedicalLeNet(
                num_classes=cfg["num_classes"],
                in_channels=cfg["in_channels"],
            ),
            "model_path": cfg["model_path"],
            "is_multilabel": False,
        }

    raise ValueError(f"Unknown dataset: {infer_data}")


def get_random_sample(dataset_name: str, setup_fn):
    """Return a random sample from the deterministic 10% test split."""

    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None

    setup_result = setup_fn(batch_size=1)

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
        if label.dim() == 0:
            label_text = str(int(label.item()))
        else:
            active_idx = torch.where(label > 0.5)[0].tolist()
            if hasattr(train_mod, "chest_label_names") and train_mod.chest_label_names:
                names = [train_mod.chest_label_names[i] for i in active_idx]
                label_text = "|".join(names) if names else "No active label"
            else:
                label_text = str(active_idx)

    return image_tensor.unsqueeze(0), label, label_text


# -----------------------------
# 3. Static 64-bit Fixed-Point Helpers
# -----------------------------
def _get_layer_config(model):
    """Return the conv/fc modules for static fixed-point inference."""
    if isinstance(model, MedicalLeNet):
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


def run_static_fixed_point_layer(q_input, layer, apply_relu=True, is_conv=False):
    q_w = quantize_fixed_point(layer.weight.detach())

    if layer.bias is not None:
        q_bias = quantize_fixed_point(layer.bias.detach())
    else:
        q_bias = torch.zeros(layer.out_channels, dtype=torch.int64)

    max_bits, max_rem = 0, 0
    if is_conv:
        q_accum = execute_and_shift_conv2d(
            q_input,
            q_w,
            stride=layer.stride[0],
            padding=layer.padding[0],
        )
    else:
        q_accum, max_bits, max_rem = execute_and_shift_linear(q_input, q_w)

    q_out = add_bias(q_accum, q_bias)

    if apply_relu:
        q_out = fixed_point_relu(q_out)

    return q_out, max_bits, max_rem


def avg_pool_fixed_point(q_tensor):
    """Pure integer 2x2 average pooling with stride 2 for 64-bit."""
    q_int64 = q_tensor.to(torch.int64)
    bsz, channels, height, width = q_int64.shape

    windows = q_int64.view(bsz, channels, height // 2, 2, width // 2, 2)
    window_sums = windows.sum(dim=(3, 5))
    rounded_avg = (window_sums + 2) >> 2

    return rounded_avg.to(torch.int64)


# -----------------------------
# 4. Main Execution
# -----------------------------
def main(infer_data, run_floating_point=True, run_fixed_point=True, log=False):
    print("--- Starting Static 64-bit ZK Inference Pipeline ---")

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

    image_tensor, true_label, true_label_text = get_random_sample(
        infer_data,
        cfg["setup_fn"],
    )

    print(
        f"\n[1] Extracted random {dataset_display} sample from test split (True Label: {true_label_text})."
    )

    float_pred = None
    if run_floating_point:
        with torch.no_grad():
            float_logits = model(image_tensor)
            if cfg["is_multilabel"]:
                float_scores = torch.sigmoid(float_logits)[0]
                float_pred = (float_scores >= 0.5).nonzero(as_tuple=True)[0].tolist()
            else:
                float_pred = float_logits.argmax(dim=1).item()

        print(f"[2] Float Inference complete. Prediction: {float_pred}")

    if not run_fixed_point:
        print("\nFixed-point path disabled for this run.")
        if run_floating_point:
            print("\n" + "=" * 40)
            print(" INFERENCE SUMMARY ")
            print("=" * 40)
            print(f"Dataset:                      {dataset_display}")
            print(f"True Label:                   {true_label_text}")
            print(f"Float Model Prediction:       {float_pred}")
            print("=" * 40)
        return

    print("\n[3] Executing Static 64-bit Fixed-Point Inference...")

    q_x = quantize_fixed_point(image_tensor)
    layer_cfg = _get_layer_config(model)

    q_x, _, _ = run_static_fixed_point_layer(
        q_x,
        layer_cfg["conv1"],
        apply_relu=True,
        is_conv=True,
    )
    q_x = avg_pool_fixed_point(q_x)

    q_x, _, _ = run_static_fixed_point_layer(
        q_x,
        layer_cfg["conv2"],
        apply_relu=True,
        is_conv=True,
    )
    q_x = avg_pool_fixed_point(q_x)

    q_x = q_x.view(q_x.size(0), -1)

    q_x, _, _ = run_static_fixed_point_layer(
        q_x,
        layer_cfg["fc1"],
        apply_relu=True,
        is_conv=False,
    )
    q_x, _, _ = run_static_fixed_point_layer(
        q_x,
        layer_cfg["fc2"],
        apply_relu=True,
        is_conv=False,
    )

    q_out, max_bits_used, max_remainder = run_static_fixed_point_layer(
        q_x,
        layer_cfg["fc3"],
        apply_relu=False,
        is_conv=False,
    )

    dequantized_logits = dequantize_fixed_point(q_out)
    if cfg["is_multilabel"]:
        fixed_scores = torch.sigmoid(dequantized_logits)[0]
        int_pred = (fixed_scores >= 0.5).nonzero(as_tuple=True)[0].tolist()
    else:
        int_pred = dequantized_logits.argmax(dim=1).item()

    print("\n" + "=" * 40)
    print(" INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"Dataset:                      {dataset_display}")
    print(f"True Label:                   {true_label_text}")
    if run_floating_point:
        print(f"Float Model Prediction:       {float_pred}")
    print(f"Static 64-bit Prediction:     {int_pred}")

    if run_floating_point:
        if float_pred == int_pred:
            print("\nSuccess! The 64-bit static model exactly matches the floating-point prediction.")
        else:
            print("\nNote: Predictions differ. This can happen due to fixed-point precision effects.")

    print("\n--- ZK Cryptographic Fixed-Point Stats (Final Layer) ---")
    print("Architecture Format:         Q31.32 (64-bit Static Container)")
    print(f"Accumulator Max Bit-Length:  {max_bits_used} bits (Threshold: 63 for PyTorch, ~254 for ZK Field)")

    headroom_used = (max_bits_used / 63.0) * 100 if max_bits_used else 0
    print(f"PyTorch Container Headroom:  {headroom_used:.1f}% Capacity Reached")
    print(f"Max Truncation Remainder:    {max_remainder:.0f} (Precision dropped during truncation)")

    print("\n--- Final Logit Sanity Check ---")
    print(f"Raw 64-bit Integer Max:      {q_out.max().item()}")
    print(f"Dequantized Float Max:       {dequantized_logits.max().item():.4f}")
    print("=" * 56)

    if log:
        fixed_point_log["meta"] = {
            "dataset": infer_data,
            "true_label": true_label_text,
            "float_prediction": float_pred,
            "fixed_point_prediction": int_pred,
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
        help=(
            "Inference data to use: MNIST, CIFAR10, Brain-MRI, CHEST, "
            "Brain-Cancer, Breast-Cancer, Cervical-Cancer, Kidney-Cancer, "
            "Lung-And-Colon-Cancer, Lymphoma-Cancer, Oral-Cancer"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fixed-point",
        action="store_true",
        help="Run fixed-point inference only",
    )
    mode_group.add_argument(
        "--floating-point",
        action="store_true",
        help="Run floating-point inference only",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log fixed-point summary to inference.json",
    )
    args = parser.parse_args()

    run_floating_point = True
    run_fixed_point = True
    if args.fixed_point:
        run_floating_point = False
        run_fixed_point = True
    elif args.floating_point:
        run_floating_point = True
        run_fixed_point = False

    main(
        args.infer,
        run_floating_point=run_floating_point,
        run_fixed_point=run_fixed_point,
        log=args.log,
    )
