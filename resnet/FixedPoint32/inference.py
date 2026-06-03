import argparse
import os
import random
import json
import sys

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if RESNET_DIR not in sys.path:
    sys.path.insert(1, RESNET_DIR)

import resnet18 as train_mod
from resnet18 import ResNet18

from utils import (
    quantize_fixed_point,
    dequantize_fixed_point,
    fixed_point_relu,
    execute_and_shift_conv2d,
    execute_and_shift_linear,
    add_bias,
    fixed_point_max_pool2d,
    fixed_point_gelu_lut,
)


# -----------------------------
# Debug / trace storage
# -----------------------------
debug_trace = {"input": {}, "layers": [], "pooling": []}

# MNIST-only integer trace (no floats) for inspecting quantized path
INT_TRACE_ENABLED = False
int_trace = {"input": {}, "layers": []}


def _normalize_activation_name(activation: str) -> str:
    name = activation.strip().lower()
    if name not in {"relu", "gelu", "leaky_relu"}:
        raise ValueError(f"Unsupported activation: {activation}")
    return name


def _checkpoint_dataset_slug(dataset_name: str) -> str:
    name = dataset_name.strip().upper().replace("_", "-").replace(" ", "-")
    if name == "CIFR10":
        name = "CIFAR10"
    return name.lower().replace("-", "_")


def _build_activation(activation: str) -> nn.Module:
    activation = _normalize_activation_name(activation)
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    return nn.LeakyReLU(inplace=True, negative_slope=1.0)


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
    def __init__(self, in_channels, out_channels, stride=1, activation="relu"):
        super(BasicBlock, self).__init__()
        self.activation1 = _build_activation(activation)
        self.activation2 = _build_activation(activation)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.add = FloatAdd()

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
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        skip = self.shortcut(x)
        out = self.add(out, skip)

        out = self.activation2(out)
        return out


class ResNet18Inference(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, activation="relu"):
        super(ResNet18Inference, self).__init__()
        self.in_channels = 64
        self.activation = _build_activation(activation)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, activation=activation)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, activation=activation)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, activation=activation)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, activation=activation)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, activation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, activation=activation))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def _resolve_infer_config(infer_data: str, activation: str = "relu", batch_size: int = 64):
    name = infer_data.upper()
    activation = _normalize_activation_name(activation)

    def _model_path(dataset_name: str) -> str:
        return f"best_resnet18_{activation}_{_checkpoint_dataset_slug(dataset_name)}.pth"

    if name == "MNIST":
        return {
            "display": "MNIST",
            "setup_fn": train_mod.setup_MNIST,
            "model": ResNet18Inference(num_classes=10, in_channels=1, activation=activation),
            "model_path": _model_path("MNIST"),
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name in ("CIFR10", "CIFAR10"):
        return {
            "display": "CIFAR10",
            "setup_fn": train_mod.setup_CIFAR10,
            "model": ResNet18Inference(num_classes=10, in_channels=3, activation=activation),
            "model_path": _model_path("CIFAR10"),
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "BRAIN-MRI":
        return {
            "display": "Brain-MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": ResNet18Inference(num_classes=4, in_channels=1, activation=activation),
            "model_path": _model_path("Brain-MRI"),
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "NIH-CHEST":
        return {
            "display": "NIH-CHEST",
            "setup_fn": train_mod.setup_NIH_Chest,
            "model": ResNet18Inference(num_classes=15, in_channels=1, activation=activation),
            "model_path": _model_path("NIH_Chest_XRay"),
            "is_multilabel": True,
            "eval_batch_size": batch_size,
        }

    if name == "OCTMNIST":
        return {
            "display": "OCTMNIST",
            "setup_fn": train_mod.setup_OCTMNIST,
            "model": ResNet18Inference(num_classes=4, in_channels=1, activation=activation),
            "model_path": _model_path("OCTMNIST"),
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "BLOODMNIST":
        return {
            "display": "BloodMNIST",
            "setup_fn": train_mod.setup_BloodMNIST,
            "model": ResNet18Inference(num_classes=8, in_channels=3, activation=activation),
            "model_path": _model_path("BloodMNIST"),
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "ORGANAMNIST":
        return {
            "display": "OrganAMNIST",
            "setup_fn": train_mod.setup_OrganAMNIST,
            "model": ResNet18Inference(num_classes=11, in_channels=1, activation=activation),
            "model_path": _model_path("OrganAMNIST"),
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "PNEUMONIAMNIST":
        return {
            "display": "PneumoniaMNIST",
            "setup_fn": train_mod.setup_PneumoniaMNIST,
            "model": ResNet18Inference(num_classes=2, in_channels=1, activation=activation),
            "model_path": _model_path("PneumoniaMNIST"),
            "is_multilabel": False,
            "eval_batch_size": batch_size,
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

        q_x = run_static_fixed_point_conv_block(
            q_x, model.conv1, model.bn1, apply_relu=True
        )

        for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in stage:
                q_x = run_static_fixed_point_basic_block(q_x, block)

        q_pooled = fixed_point_max_pool2d(q_x)
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, _, _ = run_static_fixed_point_fc(q_fc_in, model.fc)
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


def run_static_fixed_point_conv_block(q_input, conv, bn, apply_relu=True, apply_act=None, act_name="relu", lut_dict=None):
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)

    q_w = quantize_fixed_point(w_folded)
    q_bias = quantize_fixed_point(b_folded)

    q_accum = execute_and_shift_conv2d(
        q_input, q_w, stride=conv.stride[0], padding=conv.padding[0]
    )
    q_out = add_bias(q_accum, q_bias)

    should_apply_act = apply_act if apply_act is not None else apply_relu
    if should_apply_act:
        if act_name == "relu":
            q_out = fixed_point_relu(q_out)
        elif act_name == "gelu":
            q_out = fixed_point_gelu_lut(q_out, lut_dict)
        elif act_name == "leaky_relu":
            q_out = q_out
    return q_out


def run_static_fixed_point_basic_block(q_x, block, act_name="relu", lut_dict=None):
    q_out1 = run_static_fixed_point_conv_block(
        q_x, block.conv1, block.bn1, apply_relu=True, act_name=act_name, lut_dict=lut_dict
    )
    q_out2 = run_static_fixed_point_conv_block(
        q_out1, block.conv2, block.bn2, apply_relu=False, act_name=act_name, lut_dict=lut_dict
    )

    if isinstance(block.shortcut, nn.Identity):
        q_short = q_x
    else:
        short_conv, short_bn = block.shortcut[0], block.shortcut[1]
        q_short = run_static_fixed_point_conv_block(
            q_x, short_conv, short_bn, apply_relu=False, act_name=act_name, lut_dict=lut_dict
        )

    # Integer addition with 64-bit intermediate then downcast to int32
    sum_int64 = q_out2.to(torch.int64) + q_short.to(torch.int64)
    q_added = (sum_int64.to(torch.int32)).to(torch.int32)

    if act_name == "relu":
        return fixed_point_relu(q_added)
    elif act_name == "gelu":
        return fixed_point_gelu_lut(q_added, lut_dict)
    elif act_name == "leaky_relu":
        return q_added
    else:
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
def main(
    infer_data: str,
    batch_size: int = 64,
    activation: str = "relu",
    run_floating_point: bool = True,
    run_fixed_point: bool = True,
):
    print("--- Starting ResNet18 Quantized Inference Pipeline ---")

    activation = _normalize_activation_name(activation)
    cfg = _resolve_infer_config(infer_data, activation=activation, batch_size=batch_size)
    name = infer_data.upper()
    dataset_display = cfg["display"]
    model = cfg["model"]
    model_path = cfg["model_path"]

    global INT_TRACE_ENABLED, int_trace
    INT_TRACE_ENABLED = name == "MNIST"
    if INT_TRACE_ENABLED:
        # reset trace for this run
        int_trace = {"input": {}, "layers": []}

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
        print(" RESNET18 INFERENCE SUMMARY ")
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

    global_gelu_lut = None
    if activation == "gelu":
        lut_path = os.path.join(THIS_DIR, "gelu_q15_16_lut.pt")
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"Missing LUT file: {lut_path}. Run lut.py first.")
        global_gelu_lut = torch.load(lut_path, map_location="cpu")

    if not run_fixed_point:
        print("\n" + "=" * 40)
        print(" RESNET18 INFERENCE SUMMARY ")
        print("=" * 40)
        print(f"Dataset:                  {dataset_display}")
        print(f"True Label:               {true_label_text}")
        print(f"Float Model Prediction:   {float_pred}")
        print("=" * 40)
        return

    # Quantize Input Image directly to Q31.32
    q_x = quantize_fixed_point(image_tensor)

    print("\n[3] Executing Static 64-Bit Fixed-Point Inference...")

    # Initial conv1
    q_x = run_static_fixed_point_conv_block(
        q_x, model.conv1, model.bn1, apply_relu=True, act_name=activation, lut_dict=global_gelu_lut
    )

    # Traverse all residual blocks
    for layer_idx, stage in enumerate(
        [model.layer1, model.layer2, model.layer3, model.layer4], 1
    ):
        for block_idx, block in enumerate(stage):
            q_x = run_static_fixed_point_basic_block(
                q_x, block, act_name=activation, lut_dict=global_gelu_lut
            )

    # Global Average Pooling
    q_pooled = fixed_point_max_pool2d(q_x)
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
    # ZK Logging Block
    print("\n--- ZK Cryptographic Fixed-Point Stats (Final Layer) ---")
    print("Architecture Format:         Q15.16 (32-bit Static Container)")
    print(f"Internal MAC Bit-Length:     {max_bits_used} bits (Hardware Threshold: 63)")

    headroom_used = (max_bits_used / 63.0) * 100 if max_bits_used else 0
    print(f"Hardware MAC Capacity:       {headroom_used:.1f}% Capacity Reached")
    print(f"Max Truncation Remainder:    {max_remainder:.0f} (Precision dropped during truncation)")

    print("\n--- Final Logit Sanity Check ---")
    print(f"Raw 32-bit Integer Max:      {q_out.max().item()}")
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
        "--NIH-CHEST",
        dest="nih_chest",
        action="store_true",
        help="Run inference on the NIH-CHEST model using the custom test split",
    )
    parser.add_argument(
        "--infer",
        type=str,
        default="CIFAR10",
        help=(
            "Inference data to use: MNIST, CIFAR10, Brain-MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--activatoin",
        "--activation",
        dest="activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function to use for checkpoint selection and inference",
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

    if args.nih_chest:
        args.infer = "NIH-CHEST"

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
        batch_size=args.batch_size,
        activation=args.activation,
        run_floating_point=run_floating_point,
        run_fixed_point=run_fixed_point,
    )
