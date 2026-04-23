import argparse
import os
import random
import json

import torch
import torch.nn as nn

import resnet18 as train_mod
from resnet18 import ResNet18

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
    integer_add,
    integer_global_avg_pool2d,
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True) 
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.add = FloatAdd()
        self.relu2 = nn.ReLU(inplace=True) 
        
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
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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


def _multi_cancer_infer_map():
    return {
        "BRAIN-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Brain,
            "model_path": "best_resnet18_multi_brain_cancer.pth",
            "num_classes": 3,
            "in_channels": 3,
            "display": "Brain-Cancer",
        },
        "BREAST-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Breast,
            "model_path": "best_resnet18_multi_breast_cancer.pth",
            "num_classes": 2,
            "in_channels": 3,
            "display": "Breast-Cancer",
        },
        "CERVICAL-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Cervical,
            "model_path": "best_resnet18_multi_cervical_cancer.pth",
            "num_classes": 5,
            "in_channels": 3,
            "display": "Cervical-Cancer",
        },
        "KIDNEY-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Kidney,
            "model_path": "best_resnet18_multi_kidney_cancer.pth",
            "num_classes": 2,
            "in_channels": 3,
            "display": "Kidney-Cancer",
        },
        "LUNG-AND-COLON-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Lung_Colon,
            "model_path": "best_resnet18_multi_lung_and_colon_cancer.pth",
            "num_classes": 5,
            "in_channels": 3,
            "display": "Lung-And-Colon-Cancer",
        },
        "LYMPHOMA-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Lymphoma,
            "model_path": "best_resnet18_multi_lymphoma.pth",
            "num_classes": 3,
            "in_channels": 3,
            "display": "Lymphoma-Cancer",
        },
        "ORAL-CANCER": {
            "setup_fn": train_mod.setup_Multi_Cancer_Oral,
            "model_path": "best_resnet18_multi_oral_cancer.pth",
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
            "model": ResNet18Inference(num_classes=10, in_channels=1),
            "model_path": "best_resnet18_mnist.pth",
            "is_multilabel": False,
        }

    if name in ("CIFR10", "CIFAR10"):
        return {
            "display": "CIFAR10",
            "setup_fn": train_mod.setup_CIFAR10,
            "model": ResNet18Inference(num_classes=10, in_channels=3),
            "model_path": "best_resnet18_cifar10.pth",
            "is_multilabel": False,
        }

    if name == "BRAIN-MRI":
        return {
            "display": "Brain-MRI",
            "setup_fn": train_mod.setup_Brain_MRI,
            "model": ResNet18Inference(num_classes=4, in_channels=1),
            "model_path": "best_resnet18_brain_mri.pth",
            "is_multilabel": False,
        }

    if name == "CHEST":
        return {
            "display": "CHEST",
            "setup_fn": train_mod.setup_CHEST,
            "model": ResNet18Inference(num_classes=15, in_channels=1),
            "model_path": "best_resnet18_chest.pth",
            "is_multilabel": True,
        }

    multi_map = _multi_cancer_infer_map()
    if name in multi_map:
        cfg = multi_map[name]
        return {
            "display": cfg["display"],
            "setup_fn": cfg["setup_fn"],
            "model": ResNet18Inference(
                num_classes=cfg["num_classes"],
                in_channels=cfg["in_channels"],
            ),
            "model_path": cfg["model_path"],
            "is_multilabel": False,
        }

    raise ValueError(f"Unknown dataset: {infer_data}")


def get_random_sample(dataset_name: str, setup_fn):
    """Return a random sample from the deterministic 10% test split."""

    # Prevent stale loader leakage between different dataset setups.
    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None

    setup_result = setup_fn(batch_size=1)

    # Some setup functions populate train_mod.test_loader globals, while others
    # (e.g. per-cancer Multi-Cancer helpers) return loaders directly.
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
            active_idx = torch.where(label > 0.5)[0].tolist()
            if hasattr(train_mod, "chest_label_names") and train_mod.chest_label_names:
                names = [train_mod.chest_label_names[i] for i in active_idx]
                label_text = "|".join(names) if names else "No active label"
            else:
                label_text = str(active_idx)

    return image_tensor.unsqueeze(0), label, label_text


# -----------------------------
# 3. Calibration Hooks
# -----------------------------
activation_ranges = {}


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


def register_hooks(model: ResNet18Inference):
    handles = []
    
    # 1. Hook conv1 JUST to capture the raw input image bounds
    handles.append(model.conv1.register_forward_hook(
        lambda m, i, o: calibration_hook(m, i, o, "conv1")
    ))

    # 2. Hook initial conv block output (post-relu) for the first integer pass
    handles.append(model.relu.register_forward_hook(
        lambda m, i, o: calibration_hook(m, i, o, "conv1_relu")
    ))

    # Hook the distinct outputs of every block stage
    for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for block_idx, block in enumerate(layer):
            prefix = f"layer{layer_idx}_block{block_idx}"
            
            # Post-conv1 block (after ReLU)
            handles.append(block.relu1.register_forward_hook(
                lambda m, i, o, p=prefix: calibration_hook(m, i, o, f"{p}_conv1_relu")
            ))
            # Post-conv2 block (after BN, NO ReLU)
            handles.append(block.bn2.register_forward_hook(
                lambda m, i, o, p=prefix: calibration_hook(m, i, o, f"{p}_conv2_out")
            ))
            # Post-shortcut
            handles.append(block.shortcut.register_forward_hook(
                lambda m, i, o, p=prefix: calibration_hook(m, i, o, f"{p}_shortcut_out")
            ))
            # Final Block Output (after add -> relu2)
            handles.append(block.relu2.register_forward_hook(
                lambda m, i, o, p=prefix: calibration_hook(m, i, o, f"{p}_out")
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

# -----------------------------
# 4. Core Integer Inference Engine
# -----------------------------
def run_integer_conv_block(q_input, conv, bn, layer_name, scale_in, zp_in, apply_relu=True):
    # Mathematically fold BN into Conv before quantizing
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)
    
    scale_w, zp_w = get_quantization_params(w_folded, num_bits=8)
    q_w = quantize_tensor(w_folded, scale_w, zp_w, dtype=torch.uint8)

    out_range = activation_ranges[layer_name]
    pseudo_out_tensor = torch.tensor(
        [out_range["out_min"], out_range["out_max"]], dtype=torch.float32
    )
    scale_out, zp_out = get_quantization_params(pseudo_out_tensor, num_bits=8)

    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(b_folded, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    int32_accum = integer_conv2d(
        q_input, q_w, zp_in, zp_w, stride=conv.stride[0], padding=conv.padding[0]
    )
    int32_accum = add_bias(int32_accum, q_bias)

    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)
    
    if apply_relu:
        q_out = quantized_relu(q_out, zp_out)

    debug_trace["layers"].append({
        "layer_name": layer_name,
        "type": "conv",
        "input_scale": float(scale_in),
        "input_zero_point": int(zp_in),
        "weight_scale": float(scale_w),
        "weight_zero_point": int(zp_w),
        "output_scale": float(scale_out),
        "output_zero_point": int(zp_out),
    })

    # Optional MNIST integer-only trace: store quantized output tensor
    if INT_TRACE_ENABLED:
        int_trace["layers"].append(
            {
                "layer_name": layer_name,
                "output_tensor": q_out.cpu().numpy().tolist(),
            }
        )

    return q_out, scale_out, zp_out


def run_integer_fc(q_input, fc, layer_name, scale_in, zp_in):
    weight_float = fc.weight.detach()
    scale_w, zp_w = get_quantization_params(weight_float, num_bits=8)
    q_w = quantize_tensor(weight_float, scale_w, zp_w, dtype=torch.uint8)

    out_range = activation_ranges[layer_name]
    pseudo_out_tensor = torch.tensor(
        [out_range["out_min"], out_range["out_max"]], dtype=torch.float32
    )
    scale_out, zp_out = get_quantization_params(pseudo_out_tensor, num_bits=8)

    bias_float = fc.bias.detach()
    scale_bias, zp_bias = get_bias_quantization_params(scale_w, scale_in)
    q_bias = quantize_tensor(bias_float, scale_bias, zp_bias, dtype=torch.int32)

    q_M0, shift = compute_integer_multiplier(scale_w, scale_in, scale_out)

    int32_accum = integer_linear(q_input, q_w, zp_in, zp_w)
    int32_accum = add_bias(int32_accum, q_bias)

    q_out = downscale_and_cast(int32_accum, q_M0, shift, zp_out)
    # For final FC, activation (softmax) is applied in float domain, so no ReLU here

    debug_trace["layers"].append(
        {
            "layer_name": layer_name,
            "type": "linear",
            "input_scale": float(scale_in),
            "input_zero_point": int(zp_in),
            "weight_scale": float(scale_w),
            "weight_zero_point": int(zp_w),
            "output_scale": float(scale_out),
            "output_zero_point": int(zp_out),
        }
    )

    # Optional MNIST integer-only trace: store final FC quantized output
    if INT_TRACE_ENABLED:
        int_trace["layers"].append(
            {
                "layer_name": layer_name,
                "output_tensor": q_out.cpu().numpy().tolist(),
            }
        )

    return q_out, scale_out, zp_out, (scale_w, zp_w), (scale_bias, zp_bias), (
        q_M0,
        shift,
    )

def run_integer_basic_block(q_x, block, prefix, scale_in, zp_in):
    # 1. First convolution (Fold BN + ReLU)
    q_out1, s_out1, z_out1 = run_integer_conv_block(
        q_x, block.conv1, block.bn1, f"{prefix}_conv1_relu", scale_in, zp_in, apply_relu=True
    )

    # 2. Second convolution (Fold BN + NO ReLU yet)
    q_out2, s_out2, z_out2 = run_integer_conv_block(
        q_out1, block.conv2, block.bn2, f"{prefix}_conv2_out", s_out1, z_out1, apply_relu=False
    )

    # 3. Shortcut connection
    if isinstance(block.shortcut, nn.Identity):
        # Pass-through: Just pass the quantized inputs directly!
        q_short, s_short, z_short = q_x, scale_in, zp_in
    else:
        # Downsample conv: Fold BN + NO ReLU
        short_conv = block.shortcut[0]
        short_bn = block.shortcut[1]
        q_short, s_short, z_short = run_integer_conv_block(
            q_x, short_conv, short_bn, f"{prefix}_shortcut_out", scale_in, zp_in, apply_relu=False
        )

    # 4. Pre-fetch target calibration ranges for the addition output
    out_range = activation_ranges[f"{prefix}_out"]
    pseudo_tensor = torch.tensor([out_range["out_min"], out_range["out_max"]], dtype=torch.float32)
    s_final, z_final = get_quantization_params(pseudo_tensor, num_bits=8)

    # 5. STRICT INTEGER ADDITION
    q_added = integer_add(
        q_out2, z_out2, s_out2,
        q_short, z_short, s_short,
        z_final, s_final
    )

    # 6. Apply final ReLU in integer domain
    q_final = quantized_relu(q_added, z_final)

    return q_final, s_final, z_final

# -----------------------------
# 5. Main Execution
# -----------------------------
def main(infer_data: str):
    print("--- Starting ResNet18 Quantized Inference Pipeline ---")

    cfg = _resolve_infer_config(infer_data)
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
    if list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
        
    model.load_state_dict(state)
    model.eval()

    image_tensor, true_label, true_label_text = get_random_sample(
        infer_data,
        cfg["setup_fn"],
    )
    print(
        f"\n[1] Extracted random {dataset_display} sample from test split (True Label: {true_label_text})."
    )

    # Run Calibration (Float)
    handles = register_hooks(model)
    with torch.no_grad():
        float_output = model(image_tensor)
    for h in handles:
        h.remove()

    float_pred = float_output.argmax(dim=1).item()
    print(f"[2] Calibration complete. Float Model Prediction: {float_pred}")

    # Quantize Input
    in_range = activation_ranges["conv1"]
    pseudo_in_tensor = torch.tensor(
        [in_range["in_min"], in_range["in_max"]], dtype=torch.float32
    )
    scale_in, zp_in = get_quantization_params(pseudo_in_tensor, num_bits=8)
    q_x = quantize_tensor(image_tensor, scale_in, zp_in, dtype=torch.uint8)

    # Log quantized input for MNIST-only integer trace
    if INT_TRACE_ENABLED:
        int_trace["input"] = {
            "tensor": q_x.cpu().numpy().tolist(),
        }

    print("\n[3] Executing Integer-Only Inference for ResNet18...")

    # Initial conv1
    q_x, s_out, z_out = run_integer_conv_block(
        q_x, model.conv1, model.bn1, "conv1_relu", scale_in, zp_in, apply_relu=True
    )

    # Traverse all residual blocks properly
    for layer_idx, stage in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for block_idx, block in enumerate(stage):
            prefix = f"layer{layer_idx}_block{block_idx}"
            q_x, s_out, z_out = run_integer_basic_block(
                q_x, block, prefix, s_out, z_out
            )

    # -----------------------------
    # Integer Global Average Pooling
    # -----------------------------
    
    # 1. First, fetch the target ranges for the inputs to the FC layer 
    # (Since pooling feeds directly into FC, they share the same range/scale)
    fc_in_range = activation_ranges["fc"]
    pseudo_fc_in = torch.tensor(
        [fc_in_range["in_min"], fc_in_range["in_max"]], dtype=torch.float32
    )
    s_fc_in, z_fc_in = get_quantization_params(pseudo_fc_in, num_bits=8)

    # 2. STRICT INTEGER POOLING
    q_pooled = integer_global_avg_pool2d(
        q_x, z_out, s_out, z_fc_in, s_fc_in
    )
    
    # 3. Flatten for linear layer
    q_fc_in = q_pooled.view(q_pooled.size(0), -1)

    # Run Final FC Layer
    q_out, final_s, final_z, final_w, final_b, final_M = run_integer_fc(
        q_fc_in, model.fc, "fc", s_fc_in, z_fc_in
    )

    # Dequantize final logits and get prediction
    int_logits = q_out.to(torch.float32)
    dequantized_logits = final_s * (int_logits - final_z)
    int_pred = dequantized_logits.argmax(dim=1).item()

    print("\n" + "=" * 40)
    print(" RESNET18 INFERENCE SUMMARY ")
    print("=" * 40)
    print(f"Dataset:                  {dataset_display}")
    print(f"True Label:               {true_label_text}")
    print(f"Float Model Prediction:   {float_pred}")
    print(f"Integer Model Prediction: {int_pred}")

    if cfg["is_multilabel"]:
        chest_scores = torch.sigmoid(dequantized_logits)[0]
        active = (chest_scores >= 0.5).nonzero(as_tuple=True)[0].tolist()
        if hasattr(train_mod, "chest_label_names") and train_mod.chest_label_names:
            names = [train_mod.chest_label_names[i] for i in active]
            print(f"Predicted Active Labels:  {names if names else ['None >= 0.5']}")
        else:
            print(f"Predicted Active Labels:  {active}")

    if float_pred == int_pred:
        print("\nSuccess! The integer-quantized model matches the floating-point prediction.")
    else:
        print("\nNote: Predictions differ slightly. Standard 8-bit precision loss observed.")

    print("\n--- Final Layer Quantization Stats ---")
    print(f"Weight Scale:      {final_w[0]:.6f}  | Zero-Point: {final_w[1]}")
    print(f"Bias Scale:        {final_b[0]:.6f}  | Zero-Point: {final_b[1]}")
    print(f"Multiplier (M):    {final_M}")
    print(f"Output Scale:      {final_s:.6f}  | Zero-Point: {final_z}")
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
        help=(
            "Inference data to use: MNIST, CIFAR10, Brain-MRI, CHEST, "
            "Brain-Cancer, Breast-Cancer, Cervical-Cancer, Kidney-Cancer, "
            "Lung-And-Colon-Cancer, Lymphoma-Cancer, Oral-Cancer"
        ),
	)
	args = parser.parse_args()
	main(args.infer)

