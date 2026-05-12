import argparse
import os
import random
import json
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
# Debug / trace storage
# -----------------------------
debug_trace = {"input": {}, "layers": [], "pooling": []}


# -----------------------------
# Helper Functions
# -----------------------------

def _resolve_infer_config(infer_data: str):
    name = infer_data.upper()

    if name == "MNIST":
        return {
            "display": "MNIST",
            "setup_fn": train_mod.setup_MNIST,
            "model": VGG19(num_classes=10, in_channels=1),
            "model_path": "best_vgg19_mnist.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name in ("CIFR10", "CIFAR10"):
        return {
            "display": "CIFAR10",
            "setup_fn": train_mod.setup_CIFAR10,
            "model": VGG19(num_classes=10, in_channels=3),
            "model_path": "best_vgg19_cifar10.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "BRAIN-MRI":
        return {
            "display": "Brain-MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": VGG19(num_classes=4, in_channels=1),
            "model_path": "best_vgg19_brain_mri.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "NIH-CHEST":
        return {
            "display": "NIH-CHEST",
            "setup_fn": train_mod.setup_NIH_Chest,
            "model": VGG19(num_classes=15, in_channels=1),
            "model_path": "best_vgg19_NIH_Chest_XRay.pth",
            "is_multilabel": True,
            "eval_batch_size": 8,
        }

    if name == "OCTMNIST":
        return {
            "display": "OCTMNIST",
            "setup_fn": train_mod.setup_OCTMNIST,
            "model": VGG19(num_classes=4, in_channels=1),
            "model_path": "best_vgg19_octmnist.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "BLOODMNIST":
        return {
            "display": "BloodMNIST",
            "setup_fn": train_mod.setup_BloodMNIST,
            "model": VGG19(num_classes=8, in_channels=3),
            "model_path": "best_vgg19_bloodmnist.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    if name == "ORGANAMNIST":
        return {
            "display": "OrganAMNIST",
            "setup_fn": train_mod.setup_OrganAMNIST,
            "model": VGG19(num_classes=11, in_channels=1),
            "model_path": "best_vgg19_organamnist.pth",
            "is_multilabel": False,
            "eval_batch_size": 64,
        }

    raise ValueError(f"Unknown dataset: {infer_data}")


def get_random_sample(dataset_name: str, setup_fn):
    """Return a random sample from the deterministic 10% test split."""

    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None

    setup_result = setup_fn(batch_size=1)

    test_dataset = None
    if (
        isinstance(setup_result, tuple)
        and len(setup_result) >= 3
        and hasattr(setup_result[2], "dataset")
    ):
        test_dataset = setup_result[2].dataset
    elif train_mod.test_loader is not None:
        test_dataset = train_mod.test_loader.dataset

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
            label_text = str(label.detach().cpu().view(-1).tolist())

    return image_tensor.unsqueeze(0), label, label_text


def _resolve_test_loader(setup_fn, batch_size: int):
    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None

    setup_result = setup_fn(batch_size=batch_size)

    if (
        isinstance(setup_result, tuple)
        and len(setup_result) >= 3
        and hasattr(setup_result[2], "dataset")
    ):
        return setup_result[2]

    if train_mod.test_loader is not None:
        return train_mod.test_loader

    raise RuntimeError("Could not resolve test loader for the selected dataset")


def _evaluate_float_mean_auroc(model, loader, dataset_name: str):
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            all_targets.append(labels.detach().cpu())
            all_outputs.append(torch.sigmoid(outputs).detach().cpu())

    targets = torch.cat(all_targets, dim=0).numpy()
    outputs = torch.cat(all_outputs, dim=0).numpy()
    score = roc_auc_score(targets, outputs, average="macro")
    print(f"[AUROC][{dataset_name}][floating-point] Mean AUROC: {score:.4f}")
    return score


def _evaluate_fixed_point_mean_auroc(model, loader, dataset_name: str):
    all_targets = []
    all_outputs = []
    
    # Identify single-label, multi-class MedMNIST datasets
    is_medmnist = "MNIST" in dataset_name.upper() and dataset_name.upper() != "MNIST"

    for images, labels in loader:
        q_x = quantize_fixed_point(images)
        q_x = run_fixed_point_features(q_x, model.features)
        q_pooled = fixed_point_adaptive_avg_pool2d(q_x, output_size=(7, 7))
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)
        q_out = run_fixed_point_classifier(q_fc_in, model.classifier)
        logits = dequantize_fixed_point(q_out)

        all_targets.append(labels.detach().cpu())
        
        # Apply the correct activation based on dataset
        if is_medmnist:
            all_outputs.append(torch.softmax(logits, dim=1).detach().cpu())
        else:
            all_outputs.append(torch.sigmoid(logits).detach().cpu())

    targets = torch.cat(all_targets, dim=0).numpy()
    outputs = torch.cat(all_outputs, dim=0).numpy()
    
    # Apply the correct AUROC calculation based on dataset
    if is_medmnist:
        score = roc_auc_score(targets, outputs, multi_class="ovr", average="macro")
    else:
        score = roc_auc_score(targets, outputs, average="macro")
        
    print(f"[AUROC][{dataset_name}][fixed-point] Mean AUROC: {score:.4f}")
    return score


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
# Fixed-Point Inference Engine for VGG
# -----------------------------

def run_fixed_point_conv_bn_relu(q_input, conv, bn, apply_relu=True):
    """Run a Conv2d layer with folded BatchNorm in fixed-point."""
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)
    q_w = quantize_fixed_point(w_folded)
    q_bias = quantize_fixed_point(b_folded)

    q_accum = execute_and_shift_conv2d(q_input, q_w, stride=1, padding=1)
    q_out = add_bias(q_accum, q_bias)

    if apply_relu:
        q_out = fixed_point_relu(q_out)

    return q_out


def run_fixed_point_features(q_x, features):
    """Process VGG19 features (sequential conv, bn, relu, maxpool layers)."""
    layers = list(features)
    i = 0
    
    while i < len(layers):
        layer = layers[i]
    
        if isinstance(layer, nn.Conv2d):
            # Next layers are BN and ReLU
            if i + 1 < len(layers) and isinstance(layers[i + 1], nn.BatchNorm2d):
                bn_layer = layers[i + 1]
                q_x = run_fixed_point_conv_bn_relu(q_x, layer, bn_layer, apply_relu=True)
                i += 3  # Skip Conv, BN, and ReLU
            else:
                i += 1
        elif isinstance(layer, nn.MaxPool2d):
            q_x = fixed_point_max_pool2d(q_x, kernel_size=2, stride=2)
            i += 1
        else:
            i += 1
    
    return q_x


def run_fixed_point_classifier(q_x, classifier):
    """Process VGG19 classifier (3 FC layers with dropout). Return output of last layer."""
    # Extract the three FC layers from classifier
    # classifier is: Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear
    fc1 = classifier[0]
    fc2 = classifier[3]
    fc3 = classifier[6]

    # First FC layer
    q_x, _, _ = run_fixed_point_fc(q_x, fc1)
    q_x = fixed_point_relu(q_x)

    # Second FC layer
    q_x, _, _ = run_fixed_point_fc(q_x, fc2)
    q_x = fixed_point_relu(q_x)

    # Third FC layer (output layer, no ReLU)
    q_out, max_bits, max_rem = run_fixed_point_fc(q_x, fc3)

    return q_out


def run_fixed_point_fc(q_input, fc):
    """Run a fully-connected layer in fixed-point."""
    q_w = quantize_fixed_point(fc.weight.detach())
    q_bias = quantize_fixed_point(fc.bias.detach())

    q_out, max_bits, max_rem = execute_and_shift_linear(q_input, q_w)
    q_out = add_bias(q_out, q_bias)

    return q_out, max_bits, max_rem


# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data: str, run_floating_point: bool = True, run_fixed_point: bool = True):
    print("--- Starting VGG19 Quantized Inference Pipeline ---")

    cfg = _resolve_infer_config(infer_data)
    name = infer_data.upper()
    dataset_display = cfg["display"]
    model = cfg["model"]
    model_path = cfg["model_path"]

    print(f"[0] Inference target: {dataset_display}")
    print(f"[0] Loading model weights from: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return

    # Safe State Loading (Strips 'module.' prefix if saved via DataParallel)
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()

    if cfg["is_multilabel"]:
        loader = _resolve_test_loader(cfg["setup_fn"], cfg["eval_batch_size"])
        train_mod.validate_loader_preprocessing(loader, dataset_display, stage="inference")

        float_auroc = None
        fixed_point_auroc = None

        if run_floating_point:
            float_auroc = _evaluate_float_mean_auroc(model, loader, dataset_display)

        if run_fixed_point:
            fixed_point_auroc = _evaluate_fixed_point_mean_auroc(
                model, loader, dataset_display
            )

        print("\n" + "=" * 40)
        print(" VGG19 INFERENCE SUMMARY ")
        print("=" * 40)
        print(f"Dataset:                  {dataset_display}")
        if run_floating_point:
            print(f"Float Mean AUROC:         {float_auroc:.4f}")
        if run_fixed_point:
            print(f"Fixed-Point Mean AUROC:   {fixed_point_auroc:.4f}")
        print("=" * 40)
        return

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
            float_output = model(image_tensor)
        float_pred = float_output.argmax(dim=1).item()
        print(f"[2] Floating-Point Inference complete. Prediction: {float_pred}")

    if not run_fixed_point:
        print("\n" + "=" * 40)
        print(" VGG19 INFERENCE SUMMARY ")
        print("=" * 40)
        print(f"Dataset:                  {dataset_display}")
        print(f"True Label:               {true_label_text}")
        print(f"Float Model Prediction:   {float_pred}")
        print("=" * 40)
        return

    # Quantize Input Image directly to Q31.32
    q_x = quantize_fixed_point(image_tensor)

    print("\n[3] Executing Static 64-Bit Fixed-Point Inference...")

    # Process VGG features (Conv, BN, ReLU, MaxPool)
    q_x = run_fixed_point_features(q_x, model.features)

    # Adaptive Average Pooling to (7, 7)
    q_pooled = fixed_point_adaptive_avg_pool2d(q_x, output_size=(7, 7))
    q_fc_in = q_pooled.view(q_pooled.size(0), -1)

    # Process classifier (3 FC layers)
    q_out = run_fixed_point_classifier(q_fc_in, model.classifier)

    # Dequantize output
    dequantized_logits = dequantize_fixed_point(q_out)
    int_pred = dequantized_logits.argmax(dim=1).item()

    # Logging
    print("\n" + "=" * 40)
    print(" VGG19 INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"Dataset:                      {dataset_display}")
    print(f"True Label:                   {true_label_text}")
    if run_floating_point:
        print(f"Float Model Prediction:       {float_pred}")
    print(f"Static 64-bit Prediction:     {int_pred}")

    if run_floating_point:
        if float_pred == int_pred:
            print(
                "\nSuccess! The 64-bit static model exactly matches the floating-point prediction."
            )
        else:
            print(
                "\nNote: The predictions differ. This can occasionally happen due to 32-bit truncation loss."
            )

    # ZK Logging Block
    print("\n--- ZK Cryptographic Fixed-Point Stats ---")
    print(f"Architecture Format:         Q31.32 (64-bit Static Container)")
    print(f"Model:                       VGG19")

    print("\n--- Final Logit Sanity Check ---")
    print(f"Raw 64-bit Integer Max:      {q_out.max().item()}")
    print(f"Dequantized Float Max:       {dequantized_logits.max().item():.4f}")
    print("=" * 56)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        type=str,
        default="CIFAR10",
        help=("Inference data to use: MNIST, CIFAR10, Brain-MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST"),
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
    )
