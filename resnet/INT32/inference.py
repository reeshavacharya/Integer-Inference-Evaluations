import argparse
import os
import random
import json
import sys
import time
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
    integer_gelu_lut,
    quantize_tensor,
    # integer_conv2d,
    # integer_linear,
    add_bias,
    downscale_and_cast,
    quantized_relu,
    integer_add,
    integer_max_pool2d,
)
from strict_int_ops import strict_integer_conv2d, strict_integer_linear

# -----------------------------
# Debug / trace storage
# -----------------------------
debug_trace = {"input": {}, "layers": [], "pooling": []}

# MNIST-only integer trace (no floats) for inspecting quantized path
INT_TRACE_ENABLED = False
int_trace = {"input": {}, "layers": []}


def _as_int32_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.int32, device=device)


# -----------------------------
# 1. Model Definition (mirror training ResNet18)
# -----------------------------


class FloatAdd(nn.Module):
    """A dummy module to make addition visible to calibration hooks."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

def _build_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True, negative_slope=1.0)
    raise ValueError(f"Unsupported activation: {activation}")

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu"):
        super(BasicBlock, self).__init__()
        self.activation1 = _build_activation(activation)
        self.activation2 = _build_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.add = FloatAdd()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
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
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = _build_activation(activation)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, activation=activation)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, activation=activation)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, activation=activation)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, activation=activation)
        
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, activation):
        strides = [stride] + [1]*(num_blocks-1)
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


# -----------------------------
# 2. Setup and Data Extraction
# -----------------------------


def _resolve_infer_config(infer_data: str, activation: str = "relu", batch_size: int = 64):
    name = infer_data.upper()

    if name == "MNIST":
        return {
            "display": "MNIST",
            "setup_fn": train_mod.setup_MNIST,
            "model": ResNet18Inference(num_classes=10, in_channels=1, activation=activation),
            "model_path": f"best_resnet18_{activation}_mnist.pth",
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name in ("CIFR10", "CIFAR10"):
        return {
            "display": "CIFAR10",
            "setup_fn": train_mod.setup_CIFAR10,
            "model": ResNet18Inference(num_classes=10, in_channels=3, activation=activation),
            "model_path": f"best_resnet18_{activation}_cifar10.pth",
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "BRAIN-MRI":
        return {
            "display": "Brain-MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": ResNet18Inference(num_classes=4, in_channels=1, activation=activation),
            "model_path": f"best_resnet18_{activation}_brain_mri.pth",
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "OCTMNIST":
        return {
            "display": "OCTMNIST",
            "setup_fn": train_mod.setup_OCTMNIST,
            "model": ResNet18Inference(num_classes=4, in_channels=1, activation=activation),
            "model_path": f"best_resnet18_{activation}_octmnist.pth",
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "BLOODMNIST":
        return {
            "display": "BloodMNIST",
            "setup_fn": train_mod.setup_BloodMNIST,
            "model": ResNet18Inference(num_classes=8, in_channels=3, activation=activation),
            "model_path": f"best_resnet18_{activation}_bloodmnist.pth",
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "ORGANAMNIST":
        return {
            "display": "OrganAMNIST",
            "setup_fn": train_mod.setup_OrganAMNIST,
            "model": ResNet18Inference(num_classes=11, in_channels=1, activation=activation),
            "model_path": f"best_resnet18_{activation}_organamnist.pth",
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "PNEUMONIAMNIST":
        return {
            "display": "PneumoniaMNIST",
            "setup_fn": train_mod.setup_PneumoniaMNIST,
            "model": ResNet18Inference(num_classes=2, in_channels=1, activation=activation),
            "model_path": f"best_resnet18_{activation}_pneumoniamnist.pth",
            "is_multilabel": False,
            "eval_batch_size": batch_size,
        }

    if name == "NIH-CHEST":
        return {
            "display": "NIH-CHEST",
            "setup_fn": train_mod.setup_NIH_Chest,
            "model": ResNet18Inference(num_classes=15, in_channels=1, activation=activation),
            "model_path": f"best_resnet18_{activation}_nih_chest_xray.pth",
            "is_multilabel": True,
            "eval_batch_size": batch_size,
        }

    raise ValueError(f"Unknown dataset: {infer_data}")


def get_random_sample(dataset_name: str, setup_fn):
    """Return a random sample from the deterministic 10% test split."""

    # Prevent stale loader leakage between different dataset setups.
    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None

    setup_result = setup_fn(batch_size=1)

    # Standard loaders from setup functions
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

    idx = torch.randint(0, len(test_dataset), (1,), device="cpu").item()
    image_tensor, label = test_dataset[idx]
    train_mod.validate_preprocessed_batch(
        image_tensor.unsqueeze(0), dataset_name, stage="inference"
    )

    label_text = str(label)
    if isinstance(label, torch.Tensor):
        if label.dim() == 0:
            label_text = str(torch.as_tensor(label, dtype=torch.int32).item())
        else:
            active_idx = torch.where(label > 0.5)[0].tolist()
            label_text = str(active_idx)

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
    is_medmnist = "MNIST" in dataset_name.upper() and dataset_name.upper() != "MNIST"

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            all_targets.append(labels.detach().cpu())

            if is_medmnist:
                all_outputs.append(torch.softmax(outputs, dim=1).detach().cpu())
            else:
                all_outputs.append(torch.sigmoid(outputs).detach().cpu())

    targets = torch.cat(all_targets, dim=0).numpy()
    outputs = torch.cat(all_outputs, dim=0).numpy()

    if is_medmnist:
        score = roc_auc_score(targets, outputs, multi_class="ovr", average="macro")
    else:
        score = roc_auc_score(targets, outputs, average="macro")

    print(f"[AUROC][{dataset_name}][floating-point] Mean AUROC: {score:.4f}")
    return score


def _evaluate_integer_mean_auroc(model, loader, dataset_name: str, activation: str):
    # 1. Load the offline compiled integer dictionary
    cfg = _resolve_infer_config(dataset_name, activation)
    int32_model_path = cfg["model_path"].replace(".pth", "_int32.pth")
    if not os.path.exists(int32_model_path):
        raise FileNotFoundError(f"Missing compiled model: {int32_model_path}")

    int32_state = torch.load(int32_model_path, map_location="cpu")

    all_targets = []
    all_outputs = []
    is_medmnist = "MNIST" in dataset_name.upper() and dataset_name.upper() != "MNIST"

    scale_in = int32_state["meta"]["in_scale"]
    zp_in = int32_state["meta"]["in_zp"]

    for images, labels in loader:
        # Quantize input
        q_x = quantize_tensor(images, scale_in, zp_in, dtype=torch.uint32)

        # Traverse Conv1
        q_x, s_out, z_out = run_integer_conv_block(
            q_x, int32_state["conv1"], zp_in, apply_act=True, act_name=activation
        )

        # Traverse Residual Blocks
        for layer_idx in range(1, 5):
            for block_idx in range(2):
                prefix = f"layer{layer_idx}_block{block_idx}"
                q_x, s_out, z_out = run_integer_basic_block(
                    q_x, int32_state[prefix], z_out, s_out, act_name=activation
                )

        # Global Average Pool
        fc_in_scale = int32_state["fc"]["scale_in"]
        fc_in_zp = int32_state["fc"]["zp_in"]

        q_pooled = integer_max_pool2d(q_x)
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        # Final FC
        q_out, final_s, final_z = run_integer_fc(q_fc_in, int32_state["fc"], fc_in_zp)

        # Dequantize & format probabilities
        int_logits = q_out.to(torch.float32)
        logits = final_s * (int_logits - final_z)

        all_targets.append(labels.detach().cpu())
        if is_medmnist:
            all_outputs.append(torch.softmax(logits, dim=1).detach().cpu())
        else:
            all_outputs.append(torch.sigmoid(logits).detach().cpu())

    targets = torch.cat(all_targets, dim=0).numpy()
    outputs = torch.cat(all_outputs, dim=0).numpy()

    if is_medmnist:
        score = roc_auc_score(targets, outputs, multi_class="ovr", average="macro")
    else:
        score = roc_auc_score(targets, outputs, average="macro")

    print(f"[AUROC][{dataset_name}][int] Mean AUROC: {score:.4f}")
    return score


# -----------------------------
# 3. Calibration Hooks
# -----------------------------
activation_ranges = {}

def _activation_label(base_name: str, activation: str) -> str:
    return f"{base_name}_{activation.lower()}"

def _calibration_file_name(dataset_display: str, activation: str = "relu"):
    dataset_slug = dataset_display.lower().replace(' ', '_').replace('-', '_')
    return f"{dataset_slug}_{activation}_calibration.json"

def _calibration_file_path(dataset_display: str, activation: str = "relu"):
    return os.path.normpath(
        os.path.join(THIS_DIR, "..", "calibration", _calibration_file_name(dataset_display, activation))
    )

def load_calibration_ranges(dataset_display: str, activation: str = "relu"):
    """Load offline calibration stats for a dataset from the calibration folder."""
    calibration_path = _calibration_file_path(dataset_display, activation)
    
    # Try the new format first (with activation and underscores)
    if not os.path.exists(calibration_path):
        # For backward compatibility, also try the old format without activation suffix
        if activation == "relu":
            old_dataset_slug = dataset_display.lower().replace(' ', '_')
            old_path = os.path.normpath(
                os.path.join(THIS_DIR, "..", "calibration", f"{old_dataset_slug}_calibration.json")
            )
            if os.path.exists(old_path):
                calibration_path = old_path
        
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(
                f"Missing calibration file for {dataset_display} ({activation}): {calibration_path}. "
                f"Run resnet/calibration.py for this dataset first."
            )

    with open(calibration_path, "r") as f:
        payload = json.load(f)

    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise RuntimeError(f"Invalid calibration file format: {calibration_path}")

    activation_ranges.clear()
    for layer_name, stats in layers.items():
        activation_ranges[layer_name] = stats

    return activation_ranges


def calibration_hook(module, input, output, name):
    in_tensor = input[0].detach()
    out_tensor = output.detach()

    activation_ranges[name] = {
        "in_min": in_tensor.min().item(),
        "in_max": in_tensor.max().item(),
        "out_min": out_tensor.min().item(),
        "out_max": out_tensor.max().item(),
    }


def _get_layer_config(model: ResNet18Inference):
    return {
        "conv1": model.conv1,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
        "fc": model.fc,
    }


def register_hooks(model: ResNet18Inference):
    handles = []
    # Identify the activation type from the loaded model
    activation = model.activation.__class__.__name__.lower()
    if "leaky" in activation:
        activation = "leaky_relu"

    # 1. Hook conv1 JUST to capture the raw input image bounds
    handles.append(model.conv1.register_forward_hook(
        lambda m, i, o: calibration_hook(m, i, o, "conv1")
    ))

    # 2. Hook initial conv block output (post-activation) for the first integer pass
    handles.append(model.activation.register_forward_hook(
        lambda m, i, o: calibration_hook(m, i, o, _activation_label("conv1", activation))
    ))

    # Hook the distinct outputs of every block stage
    for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for block_idx, block in enumerate(layer):
            prefix = f"layer{layer_idx}_block{block_idx}"
            
            # Post-conv1 block (after Activation)
            handles.append(block.activation1.register_forward_hook(
                lambda m, i, o, p=prefix: calibration_hook(m, i, o, _activation_label(f"{p}_conv1", activation))
            ))
            # Post-conv2 block (after BN, NO Activation)
            handles.append(block.bn2.register_forward_hook(
                lambda m, i, o, p=prefix: calibration_hook(m, i, o, f"{p}_conv2_out")
            ))
            # Post-shortcut
            handles.append(block.shortcut.register_forward_hook(
                lambda m, i, o, p=prefix: calibration_hook(m, i, o, f"{p}_shortcut_out")
            ))
            # Final Block Output (after add -> activation2)
            handles.append(block.activation2.register_forward_hook(
                lambda m, i, o, p=prefix: calibration_hook(m, i, o, _activation_label(f"{p}_out", activation))
            ))

    handles.append(model.fc.register_forward_hook(
        lambda m, i, o: calibration_hook(m, i, o, "fc")
    ))

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


def run_integer_conv_block(q_input, layer_data, zp_in, apply_act=True, act_name="relu"):
    q_w = layer_data["q_weight"].to(q_input.device)
    zp_w = _as_int32_tensor(layer_data["zp_w"], q_input.device)
    q_bias = layer_data["q_bias"].to(q_input.device)
    q_M0 = _as_int32_tensor(layer_data["q_M0"], q_input.device)
    shift = _as_int32_tensor(layer_data["shift"], q_input.device)

    conv_s = layer_data["conv_scale_out"]
    conv_z = _as_int32_tensor(layer_data["conv_zp_out"], q_input.device)

    int32_accum = strict_integer_conv2d(
        q_input,
        q_w,
        zp_in,
        zp_w,
        stride=layer_data["stride"],
        padding=layer_data["padding"],
    )

    int32_accum = add_bias(int32_accum, q_bias)
    
    q_out = downscale_and_cast(int32_accum, q_M0, shift, conv_z)

    if not apply_act:
        return q_out, conv_s, conv_z

    act_s = layer_data["act_scale_out"]
    act_z = _as_int32_tensor(layer_data["act_zp_out"], q_out.device)

    if act_name == "relu":
        q_out = quantized_relu(q_out, act_z)
    elif act_name == "gelu":
        lut = layer_data["gelu_lut"].to(q_out.device)
        q_min = layer_data["gelu_q_min"]
        q_max = layer_data["gelu_q_max"]
        q_out = integer_gelu_lut(q_out, lut, q_min, q_max)
    elif act_name == "leaky_relu":
        # since the model was trained with negative_slope=1, the output is basically the input. f(x)=x
        q_out = q_out

    return q_out, act_s, act_z


def run_integer_fc(q_input, fc_data, zp_in):
    q_w = fc_data["q_weight"].to(q_input.device)
    zp_w = _as_int32_tensor(fc_data["zp_w"], q_input.device)
    q_bias = fc_data["q_bias"].to(q_input.device)
    q_M0 = _as_int32_tensor(fc_data["q_M0"], q_input.device)
    shift = _as_int32_tensor(fc_data["shift"], q_input.device)
    zp_out = _as_int32_tensor(fc_data["zp_out"], q_input.device)

    int32_accum = strict_integer_linear(q_input, q_w, zp_in, zp_w)

    int32_accum = add_bias(int32_accum, q_bias)
    
    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)

    return q_out, fc_data["scale_out"], zp_out


def run_integer_basic_block(q_x, block_data, zp_in, s_in, act_name="relu"):
    q_out1, s_out1, z_out1 = run_integer_conv_block(
        q_x, block_data["conv1"], zp_in, apply_act=True, act_name=act_name
    )

    q_out2, s_out2, z_out2 = run_integer_conv_block(
        q_out1, block_data["conv2"], z_out1, apply_act=False
    )

    if "shortcut" not in block_data:
        q_short, s_short, z_short = q_x, s_in, zp_in
    else:
        q_short, s_short, z_short = run_integer_conv_block(
            q_x, block_data["shortcut"], zp_in, apply_act=False
        )

    conv_s = block_data["add"]["conv_scale_out"]
    conv_z = _as_int32_tensor(block_data["add"]["conv_zp_out"], q_x.device)
    act_s = block_data["add"]["act_scale_out"]
    act_z = _as_int32_tensor(block_data["add"]["act_zp_out"], q_x.device)

    add_M0_1 = _as_int32_tensor(block_data["add"]["add_M0_1"], q_x.device)
    add_shift_1 = _as_int32_tensor(block_data["add"]["add_shift_1"], q_x.device)
    add_M0_2 = _as_int32_tensor(block_data["add"]["add_M0_2"], q_x.device)
    add_shift_2 = _as_int32_tensor(block_data["add"]["add_shift_2"], q_x.device)

    q_added = integer_add(
        q_out2, z_out2, q_short, z_short, conv_z,
        add_M0_1, add_shift_1, add_M0_2, add_shift_2,
    )

    if act_name == "relu":
        q_final = quantized_relu(q_added, act_z)
        
    elif act_name == "gelu":
        # --- NEW: Pure Integer Execution using LUT ---
        lut = block_data["add"]["gelu_lut"].to(q_added.device)
        q_min = block_data["add"]["gelu_q_min"]
        q_max = block_data["add"]["gelu_q_max"]
        
        q_final = integer_gelu_lut(q_added, lut, q_min, q_max)
        
    elif act_name == "leaky_relu":
        # since the model was trained with negative_slope=1, the output is basically the input. f(x)=x
        q_final = q_added

    return q_final, act_s, act_z


# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data: str, batch_size: int = 64, run_floating_point: bool = True, run_integer: bool = True):
    print("--- Starting ResNet18 Quantized Inference Pipeline ---")

    # Pass the dynamic activation string to resolve the correct config and weights
    cfg = _resolve_infer_config(infer_data, args.activation, batch_size=batch_size)
    name = infer_data.upper()
    dataset_display = cfg["display"]
    model = cfg["model"]
    model_path = cfg["model_path"]

    global INT_TRACE_ENABLED, int_trace
    INT_TRACE_ENABLED = name == "MNIST"
    if INT_TRACE_ENABLED:
        int_trace = {"input": {}, "layers": []}

    print(f"[0] Inference target: {dataset_display}")
    print(f"[0] Loading model weights from: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return

    # Load Float Model
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # --- NEW: Load Offline Int32 Model ---
    int32_model_path = model_path.replace(".pth", "_int32.pth")
    int32_state = None
    if run_integer:
        if not os.path.exists(int32_model_path):
            print(
                f"\n[!] Missing {int32_model_path}. Please run export_int32_model.py first."
            )
            return
        int32_state = torch.load(int32_model_path, map_location="cpu")

    if cfg["is_multilabel"]:
        loader = _resolve_test_loader(cfg["setup_fn"], cfg["eval_batch_size"])
        train_mod.validate_loader_preprocessing(
            loader, dataset_display, stage="inference"
        )

        float_auroc = None
        int_auroc = None

        if run_floating_point:
            float_auroc = _evaluate_float_mean_auroc(model, loader, dataset_display)

        if run_integer:
            int_auroc = _evaluate_integer_mean_auroc(model, loader, dataset_display, args.activation)

        print("\n" + "=" * 40)
        print(" RESNET18 INFERENCE SUMMARY ")
        print("=" * 40)
        print(f"Dataset:                  {dataset_display}")
        if run_floating_point:
            print(f"Float Mean AUROC:         {float_auroc:.4f}")
        if run_integer:
            print(f"Integer Mean AUROC:       {int_auroc:.4f}")
        print("=" * 40)
        return

    image_tensor, true_label, true_label_text = get_random_sample(
        infer_data,
        cfg["setup_fn"],
    )
    print(
        f"\n[1] Extracted random {dataset_display} sample from test split (True Label: {true_label_text})."
    )

    float_output = None
    float_pred = None
    float_inference_time = None
    integer_inference_time = None

    if run_floating_point:
        float_start = time.perf_counter()
        with torch.no_grad():
            float_output = model(image_tensor)
            float_pred = float_output.argmax(dim=1).item()
        float_inference_time = time.perf_counter() - float_start
        print(f"[2] Floating-Point Inference complete. Prediction: {float_pred}")
        print(f"[timing] Floating-point inference time: {float_inference_time:.4f}s")

    if not run_integer:
        print("\n" + "=" * 40)
        print(" RESNET18 INFERENCE SUMMARY ")
        print("=" * 40)
        print(f"Dataset:                  {dataset_display}")
        print(f"True Label:               {true_label_text}")
        print(f"Float Model Prediction:   {float_pred}")
        print("=" * 40)
        return

    # ----------------------------------------------------
    # TRUE INTEGER-ONLY INFERENCE (Offline Compiled Mode)
    # ----------------------------------------------------
    integer_start = time.perf_counter()

    # 1. Quantize Input using Offline Meta Config
    scale_in = int32_state["meta"]["in_scale"]
    zp_in = _as_int32_tensor(int32_state["meta"]["in_zp"], device=torch.device("cpu"))
    q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint32)

    if INT_TRACE_ENABLED:
        int_trace["input"] = {"tensor": q_x.cpu().numpy().tolist()}

    print("\n[3] Executing TRUE Integer-Only Inference...")

    # 2. Traverse Conv1
    q_x, s_out, z_out = run_integer_conv_block(
        q_x, int32_state["conv1"], zp_in, apply_act=True, act_name=args.activation
    )

    # 3. Traverse Residual Blocks
    for layer_idx in range(1, 5):
        for block_idx in range(2):
            prefix = f"layer{layer_idx}_block{block_idx}"
            q_x, s_out, z_out = run_integer_basic_block(
                q_x, int32_state[prefix], z_out, s_out, act_name=args.activation
            )

    # 4. Global Average Pooling (Using IN_SCALE, not OUT_SCALE)
    fc_in_scale = int32_state["fc"]["scale_in"]
    fc_in_zp = int32_state["fc"]["zp_in"]

    q_pooled = integer_max_pool2d(q_x)
    q_fc_in = q_pooled.view(q_pooled.size(0), -1)

    # 5. Final Fully Connected Layer
    q_out, final_s, final_z = run_integer_fc(q_fc_in, int32_state["fc"], fc_in_zp)

    # 6. Dequantize final logits
    int_logits = q_out.to(torch.float32)
    dequantized_logits = final_s * (int_logits - final_z)
    int_pred = dequantized_logits.argmax(dim=1).item()

    integer_inference_time = time.perf_counter() - integer_start

    print("\n" + "=" * 40)
    print(" RESNET18 INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"Dataset:                  {dataset_display}")
    print(f"True Label:               {true_label_text}")
    if run_floating_point:
        print(f"Float Model Prediction:   {float_pred}")
        print(f"Float Inference Time:     {float_inference_time:.4f}s")
    print("-" * 40)
    print(f"Integer Model Prediction: {int_pred}")
    print(f"Integer Inference Time:   {integer_inference_time:.4f}s")

    if run_floating_point:
        if float_pred == int_pred:
            print(
                "\nSuccess! The integer-quantized model matches the floating-point prediction."
            )
        else:
            print(
                "\nNote: Predictions differ slightly. Standard 32-bit precision loss observed."
            )
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
            "Inference data to use: MNIST, CIFAR10, Brain-MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST, PneumoniaMNIST"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function the model was trained with",
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

    if args.nih_chest:
        args.infer = "NIH-CHEST"

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
        batch_size=args.batch_size,
        run_floating_point=run_floating_point,
        run_integer=run_integer,
    )