"""Quantization Error Accumulation Tracker (moved into resnet/error/)

Saves JSON to `error/data/` and graphs to `error/graphs/`.
"""

import argparse
import importlib.util
import json
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)  # resnet directory
INT8_DIR = os.path.join(ROOT_DIR, "INT8")
INT32_DIR = os.path.join(ROOT_DIR, "INT32")
for p in (ROOT_DIR, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.makedirs(os.path.join(THIS_DIR, "json"), exist_ok=True)
os.makedirs(os.path.join(THIS_DIR, "graphs"), exist_ok=True)


def _load_module(module_name: str, file_path: str, prepend_dir: str):
    saved_path = list(sys.path)
    previous_utils = sys.modules.get("utils")
    try:
        sys.path.insert(0, prepend_dir)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = saved_path
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils


import resnet18 as train_mod

int8_utils = _load_module("resnet_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR)
int8_inference = _load_module("resnet_int8_inference", os.path.join(INT8_DIR, "inference.py"), INT8_DIR)
int32_utils = _load_module("resnet_int32_utils", os.path.join(INT32_DIR, "utils.py"), INT32_DIR)
int32_inference = _load_module("resnet_int32_inference", os.path.join(INT32_DIR, "inference.py"), INT32_DIR)


# ---------------------------------------------------------
# Error Tracking Classes
# ---------------------------------------------------------

class LayerMetrics:
    """Tracks running averages and absolute maximums for a single layer."""

    def __init__(self):
        self.cos_sim_sum = 0.0
        self.sqnr_sum = 0.0
        self.mae_sum = 0.0
        self.median_ae_sum = 0.0
        self.std_err_sum = 0.0
        self.mean_shift_sum = 0.0
        self.max_abs_err = 0.0
        self.batches = 0
        self.scale = None
        self.zp = None

    def update(self, fp_tensor, dq_tensor, scale=None, zp=None):
        if self.scale is None:
            self.scale = scale
            self.zp = zp

        # FIX 1: Enforce float64 for all reductions to prevent catastrophic cancellation
        fp_flat = fp_tensor.detach().view(-1).to(torch.float64)
        dq_flat = dq_tensor.detach().view(-1).to(torch.float64)

        cos_sim = F.cosine_similarity(fp_flat, dq_flat, dim=0).item()

        signal_power = torch.sum(fp_flat**2)
        noise_power = torch.sum((fp_flat - dq_flat) ** 2)
        sqnr = (
            (10 * torch.log10(signal_power / noise_power)).item()
            if noise_power > 1e-10
            else 100.0
        )

        abs_diff = torch.abs(fp_flat - dq_flat)
        mae = abs_diff.mean().item()
        batch_median_err = torch.median(abs_diff).item()
        batch_std_err = abs_diff.std().item()
        batch_max_err = abs_diff.max().item()

        mean_shift = (fp_flat.mean() - dq_flat.mean()).item()

        self.cos_sim_sum += cos_sim
        self.sqnr_sum += sqnr
        self.mae_sum += mae
        self.median_ae_sum += batch_median_err
        self.std_err_sum += batch_std_err
        self.mean_shift_sum += mean_shift
        self.max_abs_err = max(self.max_abs_err, batch_max_err)
        self.batches += 1

    def finalize(self):
        if self.batches == 0:
            return {}
        return {
            "cosine_similarity": self.cos_sim_sum / self.batches,
            "sqnr_db": self.sqnr_sum / self.batches,
            "mean_absolute_error": self.mae_sum / self.batches,
            "median_absolute_error": self.median_ae_sum / self.batches,
            "std_absolute_error": self.std_err_sum / self.batches,
            "max_absolute_error": self.max_abs_err,
            "mean_shift": self.mean_shift_sum / self.batches,
        }


fp32_tensors = {}

def _activation_label(base_name: str, activation: str) -> str:
    return f"{base_name}_{activation.lower()}"

def get_fp32_hook(name):
    def hook(module, input, output):
        fp32_tensors[name] = output.detach().clone()
    return hook

def attach_fp32_hooks(model, activation: str):
    handles = []
    handles.append(model.activation.register_forward_hook(get_fp32_hook(_activation_label("conv1", activation))))

    for l_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for b_idx, block in enumerate(layer):
            pfx = f"layer{l_idx}_block{b_idx}"
            handles.append(block.activation1.register_forward_hook(get_fp32_hook(_activation_label(f"{pfx}_conv1", activation))))
            handles.append(block.bn2.register_forward_hook(get_fp32_hook(f"{pfx}_conv2_out")))
            handles.append(block.shortcut.register_forward_hook(get_fp32_hook(f"{pfx}_shortcut_out")))
            handles.append(block.activation2.register_forward_hook(get_fp32_hook(_activation_label(f"{pfx}_out", activation))))

    handles.append(model.fc.register_forward_hook(get_fp32_hook("fc")))
    return handles


# ---------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------

def evaluate_error(
    dataset_name: str,
    num_data: int = None,
    batch_size: int = 64,
    activation: str = "relu",
    mode: str = "int8",
):
    print(f"\n[error] Starting Layer-by-Layer Error Analysis on {dataset_name} ({activation})...")

    infer_module = int8_inference if mode == "int8" else int32_inference
    quant_utils = int8_utils if mode == "int8" else int32_utils
    target_dtype = torch.uint8 if mode == "int8" else torch.int32

    cfg = infer_module._resolve_infer_config(dataset_name, activation)
    model = cfg["model"]

    state = torch.load(cfg["model_path"], map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    model_suffix = f"_{mode}.pth"
    int_model_path = cfg["model_path"].replace(".pth", model_suffix)
    if not os.path.exists(int_model_path):
        raise FileNotFoundError(f"Missing compiled model: {int_model_path}. Run export_{mode}_model.py first.")
    int_state = torch.load(int_model_path, map_location="cpu")

    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None
    setup_result = cfg["setup_fn"](batch_size=batch_size)
    if isinstance(setup_result, tuple) and len(setup_result) >= 3 and hasattr(setup_result[2], "dataset"):
        loader = setup_result[2]
    elif train_mod.test_loader is not None:
        loader = train_mod.test_loader
    else:
        raise RuntimeError("Could not resolve test loader.")

    attach_fp32_hooks(model, activation)
    layer_trackers = {}

    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)
    processed_images = 0

    scale_in = int_state["meta"]["in_scale"]
    zp_in = int_state["meta"]["in_zp"]

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images = images[:remaining]

        with torch.no_grad():
            _ = model(images)

        q_x = quant_utils.quantize_tensor(images, scale_in, zp_in, dtype=target_dtype)

        def _evaluate_step(q_tensor, s_out, z_out, name):
            if name not in layer_trackers:
                layer_trackers[name] = LayerMetrics()

            # FIX 2: Float64 casting guarantees precision for both uint8 and uint32
            dq_tensor = s_out * (q_tensor.to(torch.float64) - z_out)
            layer_trackers[name].update(fp32_tensors[name], dq_tensor, s_out, z_out)

        # 1. Unroll Initial Conv1
        q_pre_act, conv_s, conv_z = infer_module.run_integer_conv_block(
            q_x, int_state["conv1"], zp_in, apply_act=False
        )
        act_s = int_state["conv1"]["act_scale_out"]
        act_z = int_state["conv1"]["act_zp_out"]

        # RESTORED: Pure integer clamped ReLU
        if activation == "relu":
            q_x = quant_utils.quantized_relu(q_pre_act, act_z)
        elif activation == "gelu":
            f_accum = quant_utils.dequantize_tensor(q_pre_act, conv_s, conv_z)
            f_act = F.gelu(f_accum)
            f_act = torch.clamp(f_act, min=-10.0, max=10.0)
            q_x = quant_utils.quantize_tensor(f_act, act_s, act_z, dtype=target_dtype)
        elif activation == "leaky_relu":
            f_accum = quant_utils.dequantize_tensor(q_pre_act, conv_s, conv_z)
            f_act = F.leaky_relu(f_accum, negative_slope=1.0)
            f_act = torch.clamp(f_act, min=-50.0, max=50.0)
            q_x = quant_utils.quantize_tensor(f_act, act_s, act_z, dtype=target_dtype)

        _evaluate_step(q_x, act_s, act_z, _activation_label("conv1", activation))
        s_out, z_out = act_s, act_z

        for layer_idx in range(1, 5):
            for block_idx in range(2):
                prefix = f"layer{layer_idx}_block{block_idx}"
                block_data = int_state[prefix]

                # 2. Unroll Block Conv1
                q_out1_pre, conv_s1, conv_z1 = infer_module.run_integer_conv_block(
                    q_x, block_data["conv1"], z_out, apply_act=False
                )
                act_s1 = block_data["conv1"]["act_scale_out"]
                act_z1 = block_data["conv1"]["act_zp_out"]

                if activation == "relu":
                    q_out1 = quant_utils.quantized_relu(q_out1_pre, act_z1)
                elif activation == "gelu":
                    f_accum = quant_utils.dequantize_tensor(q_out1_pre, conv_s1, conv_z1)
                    f_act = F.gelu(f_accum)
                    f_act = torch.clamp(f_act, min=-10.0, max=10.0)
                    q_out1 = quant_utils.quantize_tensor(f_act, act_s1, act_z1, dtype=target_dtype)
                elif activation == "leaky_relu":
                    f_accum = quant_utils.dequantize_tensor(q_out1_pre, conv_s1, conv_z1)
                    f_act = F.leaky_relu(f_accum, negative_slope=1.0)
                    f_act = torch.clamp(f_act, min=-50.0, max=50.0)
                    q_out1 = quant_utils.quantize_tensor(f_act, act_s1, act_z1, dtype=target_dtype)

                _evaluate_step(q_out1, act_s1, act_z1, _activation_label(f"{prefix}_conv1", activation))
                s_out1, z_out1 = act_s1, act_z1

                q_out2, s_out2, z_out2 = infer_module.run_integer_conv_block(
                    q_out1, block_data["conv2"], z_out1, apply_act=False
                )
                _evaluate_step(q_out2, s_out2, z_out2, f"{prefix}_conv2_out")

                if "shortcut" not in block_data:
                    q_short, s_short, z_short = q_x, s_out, z_out
                else:
                    q_short, s_short, z_short = infer_module.run_integer_conv_block(
                        q_x, block_data["shortcut"], z_out, apply_act=False
                    )
                    _evaluate_step(q_short, s_short, z_short, f"{prefix}_shortcut_out")

                conv_s = block_data["add"]["conv_scale_out"]
                conv_z = block_data["add"]["conv_zp_out"]
                act_s = block_data["add"]["act_scale_out"]
                act_z = block_data["add"]["act_zp_out"]

                q_added = quant_utils.integer_add(
                    q_out2, z_out2, s_out2, q_short, z_short, s_short, conv_z, conv_s
                )

                if activation == "relu":
                    q_x = quant_utils.quantized_relu(q_added, act_z)
                elif activation == "gelu":
                    f_accum = quant_utils.dequantize_tensor(q_added, conv_s, conv_z)
                    f_act = F.gelu(f_accum)
                    f_act = torch.clamp(f_act, min=-10.0, max=10.0)
                    q_x = quant_utils.quantize_tensor(f_act, act_s, act_z, dtype=target_dtype)
                elif activation == "leaky_relu":
                    f_accum = quant_utils.dequantize_tensor(q_added, conv_s, conv_z)
                    f_act = F.leaky_relu(f_accum, negative_slope=1.0)
                    f_act = torch.clamp(f_act, min=-50.0, max=50.0)
                    q_x = quant_utils.quantize_tensor(f_act, act_s, act_z, dtype=target_dtype)

                s_out, z_out = act_s, act_z
                _evaluate_step(q_x, s_out, z_out, _activation_label(f"{prefix}_out", activation))

        fc_in_scale = int_state["fc"]["scale_in"]
        fc_in_zp = int_state["fc"]["zp_in"]

        q_pooled = quant_utils.integer_global_avg_pool2d(
            q_x, z_out, s_out, fc_in_zp, fc_in_scale
        )
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, final_s, final_z = infer_module.run_integer_fc(q_fc_in, int_state["fc"], fc_in_zp)
        _evaluate_step(q_out, final_s, final_z, "fc")

        processed_images += images.size(0)
        print(f"[error] Processed {processed_images}/{target_images} images...")

    return {name: tracker.finalize() for name, tracker in layer_trackers.items()}


def plot_error_metrics(metrics_dict, dataset_name, activation: str = "relu", mode: str = "int8"):
    print(f"\n[plot] Generating quantization error graphs for {dataset_name}...")

    layers = list(metrics_dict.keys())
    valid_layers = [l for l in layers if metrics_dict[l]]
    if not valid_layers:
        print("[-] No valid metrics to plot.")
        return

    mae = [metrics_dict[l]["mean_absolute_error"] for l in valid_layers]
    median_err = [metrics_dict[l]["median_absolute_error"] for l in valid_layers]
    std_err = [metrics_dict[l]["std_absolute_error"] for l in valid_layers]
    max_err = [metrics_dict[l]["max_absolute_error"] for l in valid_layers]
    sqnr = [metrics_dict[l]["sqnr_db"] for l in valid_layers]
    cos_sim = [metrics_dict[l]["cosine_similarity"] for l in valid_layers]
    mean_shift = [metrics_dict[l]["mean_shift"] for l in valid_layers]

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"ResNet18 FP32 vs Quantization Error Progression ({dataset_name})",
        fontsize=16,
        fontweight="bold",
    )

    x = range(len(valid_layers))

    axs[0, 0].plot(x, mae, marker="o", color="blue", label="Mean Absolute Error")
    axs[0, 0].plot(x, median_err, marker="v", color="orange", label="Median Absolute Error")
    axs[0, 0].plot(x, std_err, marker="d", color="magenta", linestyle="-.", label="Std Deviation (Noise Spread)")
    axs[0, 0].plot(x, max_err, marker="x", color="red", linestyle="--", label="Max Absolute Error (Clipping)")
    axs[0, 0].set_title("Error Magnitude Growth")
    axs[0, 0].set_ylabel("Absolute Error")
    axs[0, 0].grid(True, linestyle=":", alpha=0.7)
    axs[0, 0].legend()

    axs[0, 1].plot(x, sqnr, marker="s", color="purple")
    axs[0, 1].set_title("Signal-to-Quantization-Noise Ratio (SQNR)")
    axs[0, 1].set_ylabel("dB (Higher is better)")
    axs[0, 1].axhline(y=20, color="r", linestyle="-", alpha=0.3, label="Warning Threshold (<20dB)")
    axs[0, 1].grid(True, linestyle=":", alpha=0.7)
    axs[0, 1].legend()

    axs[1, 0].plot(x, cos_sim, marker="^", color="green")
    axs[1, 0].set_title("Cosine Similarity (Structural Integrity)")
    axs[1, 0].set_ylabel("Similarity (1.0 = Perfect)")
    axs[1, 0].set_ylim(min(0.85, min(cos_sim) - 0.05), 1.01)
    axs[1, 0].grid(True, linestyle=":", alpha=0.7)

    axs[1, 1].bar(x, mean_shift, color=["red" if v < 0 else "blue" for v in mean_shift], alpha=0.6)
    axs[1, 1].axhline(y=0, color="black", linestyle="-")
    axs[1, 1].set_title("Mean Shift (Zero-Point Drift / Bias)")
    axs[1, 1].set_ylabel("Shift Direction")
    axs[1, 1].grid(True, linestyle=":", alpha=0.7, axis="y")

    for ax in axs.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(valid_layers, rotation=45, ha="right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    graphs_dir = os.path.join(THIS_DIR, "graphs")
    filename = os.path.join(graphs_dir, f"quantization_divergence_{dataset_name.lower()}_{activation}_{mode}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[+] Saved error divergence graphs to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Dataset to evaluate.")
    parser.add_argument("--num_data", type=int, default=None, help="Number of images to process.")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function the model was trained with",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["int8", "int32"],
        help="Integer inference mode used for error analysis",
    )
    args = parser.parse_args()

    metrics = evaluate_error(args.dataset, args.num_data, activation=args.activation, mode=args.mode)

    data_dir = os.path.join(THIS_DIR, "json")
    file_name = os.path.join(data_dir, f"error_accumulation_{args.dataset.lower()}_{args.activation}_{args.mode}.json")
    
    with open(file_name, "w") as f:
        json.dump({"dataset": args.dataset, "activation": args.activation, "layer_metrics": metrics}, f, indent=2)

    print(f"\n[+] Saved error accumulation log to {file_name}")
    plot_error_metrics(metrics, args.dataset, args.activation, args.mode)