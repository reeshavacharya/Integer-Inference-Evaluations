"""Quantization Error Accumulation Tracker for ResNet18

This script runs side-by-side FP32 and INT8 inference over a dataset,
capturing intermediate tensors, dequantizing the INT8 results, and
calculating exact layer-by-layer error accumulation metrics.
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
INT8_DIR = os.path.join(THIS_DIR, "INT8")
for p in (THIS_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)


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

int8_utils = _load_module(
    "resnet_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR
)
int8_inference = _load_module(
    "resnet_int8_inference", os.path.join(INT8_DIR, "inference.py"), INT8_DIR
)


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
        self.mean_shift_sum = 0.0
        self.max_abs_err = 0.0  # We want the absolute max across the entire dataset
        self.batches = 0

    def update(self, fp_tensor, dq_tensor):
        fp_flat = fp_tensor.detach().view(-1).float()
        dq_flat = dq_tensor.detach().view(-1).float()

        # 1. Cosine Similarity
        # Force 64-bit precision to prevent Cauchy-Schwarz violation on dense medical data
        fp_64 = fp_flat.to(torch.float64)
        dq_64 = dq_flat.to(torch.float64)
        cos_sim = F.cosine_similarity(fp_64, dq_64, dim=0).item()

        # 2. SQNR
        signal_power = torch.sum(fp_flat**2)
        noise_power = torch.sum((fp_flat - dq_flat) ** 2)
        sqnr = (
            (10 * torch.log10(signal_power / noise_power)).item()
            if noise_power > 1e-7
            else 100.0
        )

        # 3. MAE, Median, & Max Error
        abs_diff = torch.abs(fp_flat - dq_flat)
        mae = abs_diff.mean().item()
        batch_median_err = torch.median(abs_diff).item()  # <-- ADD THIS
        batch_max_err = abs_diff.max().item()

        # 4. Mean Shift
        mean_shift = (fp_flat.mean() - dq_flat.mean()).item()

        self.cos_sim_sum += cos_sim
        self.sqnr_sum += sqnr
        self.mae_sum += mae
        self.median_ae_sum += batch_median_err
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
            "max_absolute_error": self.max_abs_err,  # Max is NOT averaged
            "mean_shift": self.mean_shift_sum / self.batches,
        }


# Global dictionary to hold FP32 tensors during the forward pass
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

    for l_idx, layer in enumerate(
        [model.layer1, model.layer2, model.layer3, model.layer4], 1
    ):
        for b_idx, block in enumerate(layer):
            pfx = f"layer{l_idx}_block{b_idx}"
            handles.append(
                block.activation1.register_forward_hook(get_fp32_hook(_activation_label(f"{pfx}_conv1", activation)))
            )
            handles.append(
                block.bn2.register_forward_hook(get_fp32_hook(f"{pfx}_conv2_out"))
            )
            handles.append(
                block.shortcut.register_forward_hook(
                    get_fp32_hook(f"{pfx}_shortcut_out")
                )
            )
            handles.append(
                block.activation2.register_forward_hook(get_fp32_hook(_activation_label(f"{pfx}_out", activation)))
            )

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
):
    print(
        f"\n[error] Starting Layer-by-Layer Error Analysis on {dataset_name} "
        f"({activation})..."
    )

    cfg = int8_inference._resolve_infer_config(dataset_name, activation)
    model = cfg["model"]

    # 1. Load FP32 Model (for hooks)
    state = torch.load(cfg["model_path"], map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # 2. Load Offline Compiled INT8 Model
    int8_model_path = cfg["model_path"].replace(".pth", "_int8.pth")
    if not os.path.exists(int8_model_path):
        raise FileNotFoundError(
            f"Missing compiled model: {int8_model_path}. Run export_int8_model.py first."
        )
    int8_state = torch.load(int8_model_path, map_location="cpu")

    # Load dataloader
    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None
    setup_result = cfg["setup_fn"](batch_size=batch_size)
    if (
        isinstance(setup_result, tuple)
        and len(setup_result) >= 3
        and hasattr(setup_result[2], "dataset")
    ):
        loader = setup_result[2]
    elif train_mod.test_loader is not None:
        loader = train_mod.test_loader
    else:
        raise RuntimeError("Could not resolve test loader.")

    # Attach FP32 Capture Hooks
    attach_fp32_hooks(model, activation)
    layer_trackers = {}

    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)
    processed_images = 0

    scale_in = int8_state["meta"]["in_scale"]
    zp_in = int8_state["meta"]["in_zp"]

    for batch_idx, (images, labels) in enumerate(loader, 1):
        if processed_images >= target_images:
            break

        remaining = target_images - processed_images
        if images.size(0) > remaining:
            images = images[:remaining]

        # 1. Run FP32 Pass (Populates the fp32_tensors dictionary)
        with torch.no_grad():
            _ = model(images)

        # 2. Run INT8 Pass step-by-step and capture errors
        q_x = int8_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)

        def _evaluate_step(q_tensor, s_out, z_out, name):
            if name not in layer_trackers:
                layer_trackers[name] = LayerMetrics()
            dq_tensor = s_out * (q_tensor.to(torch.float32) - z_out)
            layer_trackers[name].update(fp32_tensors[name], dq_tensor)

        # Initial Conv1
        q_x, s_out, z_out = int8_inference.run_integer_conv_block(
            q_x, int8_state["conv1"], zp_in, apply_act=True, act_name=activation
        )
        _evaluate_step(q_x, s_out, z_out, _activation_label("conv1", activation))

        # Residual Blocks (Manually unrolled to capture intermediates)
        for layer_idx in range(1, 5):
            for block_idx in range(2):
                prefix = f"layer{layer_idx}_block{block_idx}"
                block_data = int8_state[prefix]

                q_out1, s_out1, z_out1 = int8_inference.run_integer_conv_block(
                    q_x,
                    block_data["conv1"],
                    z_out,
                    apply_act=True,
                    act_name=activation,
                )
                _evaluate_step(q_out1, s_out1, z_out1, _activation_label(f"{prefix}_conv1", activation))

                q_out2, s_out2, z_out2 = int8_inference.run_integer_conv_block(
                    q_out1, block_data["conv2"], z_out1, apply_act=False
                )
                _evaluate_step(q_out2, s_out2, z_out2, f"{prefix}_conv2_out")

                if "shortcut" not in block_data:
                    q_short, s_short, z_short = q_x, s_out, z_out
                else:
                    q_short, s_short, z_short = int8_inference.run_integer_conv_block(
                        q_x, block_data["shortcut"], z_out, apply_act=False
                    )
                    _evaluate_step(q_short, s_short, z_short, f"{prefix}_shortcut_out")

                conv_s = block_data["add"]["conv_scale_out"]
                conv_z = block_data["add"]["conv_zp_out"]
                act_s = block_data["add"]["act_scale_out"]
                act_z = block_data["add"]["act_zp_out"]

                q_added = int8_utils.integer_add(
                    q_out2, z_out2, s_out2, q_short, z_short, s_short, conv_z, conv_s
                )
                if activation == "relu":
                    q_x = int8_utils.quantized_relu(q_added, act_z)
                elif activation == "gelu":
                    f_accum = int8_utils.dequantize_tensor(q_added, conv_s, conv_z)
                    f_act = F.gelu(f_accum)
                    f_act = torch.clamp(f_act, min=-0.17, max=10.0)
                    q_x = int8_utils.quantize_tensor(f_act, act_s, act_z, dtype=torch.uint8)
                else:
                    raise ValueError(f"Unknown activation function: {activation}")
                s_out, z_out = act_s, act_z

                _evaluate_step(q_x, s_out, z_out, _activation_label(f"{prefix}_out", activation))

        # Global Avg Pool & FC
        fc_in_scale = int8_state["fc"]["scale_in"]
        fc_in_zp = int8_state["fc"]["zp_in"]

        q_pooled = int8_utils.integer_global_avg_pool2d(
            q_x, z_out, s_out, fc_in_zp, fc_in_scale
        )
        q_fc_in = q_pooled.view(q_pooled.size(0), -1)

        q_out, final_s, final_z = int8_inference.run_integer_fc(
            q_fc_in, int8_state["fc"], fc_in_zp
        )
        _evaluate_step(q_out, final_s, final_z, "fc")

        processed_images += images.size(0)
        print(f"[error] Processed {processed_images}/{target_images} images...")

    # Finalize Metrics
    final_metrics = {
        name: tracker.finalize() for name, tracker in layer_trackers.items()
    }
    return final_metrics


def plot_error_metrics(metrics_dict, dataset_name, activation: str = "relu"):
    """Generates a 4-panel graph tracking quantization errors across layers."""
    print(f"\n[plot] Generating quantization error graphs for {dataset_name}...")

    # Extract data, ensuring we maintain the sequential order of layers
    layers = list(metrics_dict.keys())

    # Filter out layers that didn't process data (if any)
    valid_layers = [l for l in layers if metrics_dict[l]]
    if not valid_layers:
        print("[-] No valid metrics to plot.")
        return

    # Extract specific metrics into lists
    mae = [metrics_dict[l]["mean_absolute_error"] for l in valid_layers]
    median_err = [metrics_dict[l]["median_absolute_error"] for l in valid_layers]
    max_err = [metrics_dict[l]["max_absolute_error"] for l in valid_layers]
    sqnr = [metrics_dict[l]["sqnr_db"] for l in valid_layers]
    cos_sim = [metrics_dict[l]["cosine_similarity"] for l in valid_layers]
    mean_shift = [metrics_dict[l]["mean_shift"] for l in valid_layers]

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"ResNet18 FP32 vs INT8 Quantization Error Progression ({dataset_name})",
        fontsize=16,
        fontweight="bold",
    )

    # X-axis ticks (using indices for spacing, replacing with names later)
    x = range(len(valid_layers))

    # --- Plot 1: Absolute Errors (MAE, Median, & Max) ---
    axs[0, 0].plot(x, mae, marker="o", color="blue", label="Mean Absolute Error")
    axs[0, 0].plot(x, median_err, marker="v", color="orange", label="Median Absolute Error") # <-- ADD THIS
    axs[0, 0].plot(
        x,
        max_err,
        marker="x",
        color="red",
        linestyle="--",
        label="Max Absolute Error (Clipping)",
    )
    axs[0, 0].set_title("Error Magnitude Growth")
    axs[0, 0].set_ylabel("Absolute Error")
    axs[0, 0].grid(True, linestyle=":", alpha=0.7)
    axs[0, 0].legend()

    # --- Plot 2: SQNR (Signal Quality) ---
    axs[0, 1].plot(x, sqnr, marker="s", color="purple")
    axs[0, 1].set_title("Signal-to-Quantization-Noise Ratio (SQNR)")
    axs[0, 1].set_ylabel("dB (Higher is better)")
    axs[0, 1].axhline(
        y=20, color="r", linestyle="-", alpha=0.3, label="Warning Threshold (<20dB)"
    )
    axs[0, 1].grid(True, linestyle=":", alpha=0.7)
    axs[0, 1].legend()

    # --- Plot 3: Cosine Similarity ---
    axs[1, 0].plot(x, cos_sim, marker="^", color="green")
    axs[1, 0].set_title("Cosine Similarity (Structural Integrity)")
    axs[1, 0].set_ylabel("Similarity (1.0 = Perfect)")
    axs[1, 0].set_ylim(min(0.85, min(cos_sim) - 0.05), 1.01)  # Zoom in on the top range
    axs[1, 0].grid(True, linestyle=":", alpha=0.7)

    # --- Plot 4: Mean Shift (Zero-Point Drift) ---
    axs[1, 1].bar(
        x, mean_shift, color=["red" if v < 0 else "blue" for v in mean_shift], alpha=0.6
    )
    axs[1, 1].axhline(y=0, color="black", linestyle="-")
    axs[1, 1].set_title("Mean Shift (Zero-Point Drift / Bias)")
    axs[1, 1].set_ylabel("Shift Direction")
    axs[1, 1].grid(True, linestyle=":", alpha=0.7, axis="y")

    # Formatting the X-axes for all subplots
    for ax in axs.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(valid_layers, rotation=45, ha="right", fontsize=8)

    # Adjust layout to prevent label clipping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot with a proper descriptive name
    filename = f"quantization_divergence_{dataset_name.lower()}_{activation}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[+] Saved error divergence graphs to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="Dataset to evaluate."
    )
    parser.add_argument(
        "--num_data", type=int, default=None, help="Number of images to process."
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu"],
        help="Activation function the model was trained with",
    )
    args = parser.parse_args()

    metrics = evaluate_error(args.dataset, args.num_data, activation=args.activation)

    file_name = f"error_accumulation_{args.dataset.lower()}_{args.activation}.json"
    with open(file_name, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "activation": args.activation,
                "layer_metrics": metrics,
            },
            f,
            indent=2,
        )

    print(f"\\n[+] Saved error accumulation log to {file_name}")
    plot_error_metrics(metrics, args.dataset, args.activation)
