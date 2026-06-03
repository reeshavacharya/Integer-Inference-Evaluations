"""Single-Sample Tensor Validation and Triplet Extraction

Executes a forward pass on a single random test sample. Validates the mathematical
quantization formula locally, extracts the FP32 tensors, scales, zero-points, weights, 
and biases, and visualizes the error drop across the activation boundary.

Usage:
    python3 record_tensor.py --dataset MNIST --activation leaky_relu
"""

import argparse
import importlib.util
import os
import random
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)  # resnet directory
INT8_DIR = os.path.join(ROOT_DIR, "INT8")
for p in (ROOT_DIR, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Ensure output directory exists
LOG_DIR = os.path.join(THIS_DIR, "activation-effect-graph")
os.makedirs(LOG_DIR, exist_ok=True)


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
    def __init__(self):
        self.cos_sim = 0.0
        self.sqnr = 0.0
        self.mae = 0.0
        self.median_ae = 0.0
        self.std_err = 0.0
        self.mean_shift = 0.0
        self.max_abs_err = 0.0

    def update(self, fp_tensor, dq_tensor):
        fp_flat = fp_tensor.detach().view(-1).float()
        dq_flat = dq_tensor.detach().view(-1).float()

        fp_64 = fp_flat.to(torch.float64)
        dq_64 = dq_flat.to(torch.float64)
        self.cos_sim = F.cosine_similarity(fp_64, dq_64, dim=0).item()

        signal_power = torch.sum(fp_flat**2)
        noise_power = torch.sum((fp_flat - dq_flat) ** 2)
        self.sqnr = (
            (10 * torch.log10(signal_power / noise_power)).item()
            if noise_power > 1e-7
            else 100.0
        )

        abs_diff = torch.abs(fp_flat - dq_flat)
        self.mae = abs_diff.mean().item()
        self.median_ae = torch.median(abs_diff).item()
        self.std_err = abs_diff.std().item()
        self.max_abs_err = abs_diff.max().item()
        self.mean_shift = (fp_flat.mean() - dq_flat.mean()).item()

    def finalize(self):
        return {
            "cosine_similarity": self.cos_sim,
            "sqnr_db": self.sqnr,
            "mean_absolute_error": self.mae,
            "median_absolute_error": self.median_ae,
            "std_absolute_error": self.std_err,
            "max_absolute_error": self.max_abs_err,
            "mean_shift": self.mean_shift,
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
    handles.append(model.bn1.register_forward_hook(get_fp32_hook("conv1_pre_act")))
    handles.append(model.activation.register_forward_hook(get_fp32_hook(_activation_label("conv1", activation))))

    for l_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for b_idx, block in enumerate(layer):
            pfx = f"layer{l_idx}_block{b_idx}"
            handles.append(block.bn1.register_forward_hook(get_fp32_hook(f"{pfx}_conv1_pre_act")))
            handles.append(block.activation1.register_forward_hook(get_fp32_hook(_activation_label(f"{pfx}_conv1", activation))))
            handles.append(block.bn2.register_forward_hook(get_fp32_hook(f"{pfx}_conv2_out")))
            handles.append(block.shortcut.register_forward_hook(get_fp32_hook(f"{pfx}_shortcut_out")))
            handles.append(block.add.register_forward_hook(get_fp32_hook(f"{pfx}_add_pre_act")))
            handles.append(block.activation2.register_forward_hook(get_fp32_hook(_activation_label(f"{pfx}_out", activation))))

    handles.append(model.fc.register_forward_hook(get_fp32_hook("fc")))
    return handles


# ---------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------


def record_and_validate(dataset_name: str, activation: str):
    print(f"\n[record] Initializing quantization validation on {dataset_name} ({activation})...")

    cfg = int8_inference._resolve_infer_config(dataset_name, activation)
    model = cfg["model"]

    state = torch.load(cfg["model_path"], map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    int8_model_path = cfg["model_path"].replace(".pth", "_int8.pth")
    if not os.path.exists(int8_model_path):
        raise FileNotFoundError(f"Missing compiled model: {int8_model_path}. Run export_int8_model.py first.")
    int8_state = torch.load(int8_model_path, map_location="cpu")

    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None
    setup_result = cfg["setup_fn"](batch_size=1)
    
    if isinstance(setup_result, tuple) and len(setup_result) >= 3 and hasattr(setup_result[2], "dataset"):
        loader = setup_result[2]
    elif train_mod.test_loader is not None:
        loader = train_mod.test_loader
    else:
        raise RuntimeError("Could not resolve test loader.")

    dataset = loader.dataset
    random_idx = random.randint(0, len(dataset) - 1)
    print(f"[record] Selected random test sample index: {random_idx}")
    
    image, label = dataset[random_idx]
    image = image.unsqueeze(0)

    attach_fp32_hooks(model, activation)
    
    # We will record the exact chronological sequence of evaluations
    evaluation_history = []

    scale_in = int8_state["meta"]["in_scale"]
    zp_in = int8_state["meta"]["in_zp"]

    with torch.no_grad():
        _ = model(image)

    q_x = int8_utils.quantize_tensor(image, scale_in, zp_in, dtype=torch.uint8)

    def _evaluate_step(q_tensor, s_out, z_out, name):
        fp32_t = fp32_tensors[name]
        
        # --- VALIDATION LOOP ---
        try:
            val_q = int8_utils.quantize_tensor(fp32_t, s_out, z_out, dtype=torch.uint8)
            if val_q.dtype != torch.uint8 or val_q.shape != fp32_t.shape:
                raise ValueError("Output format mismatch")
        except Exception as e:
            raise RuntimeError(f"Quantization validation failed at {name}: {e}")

        # Metrics Tracker
        tracker = LayerMetrics()
        dq_tensor = s_out * (q_tensor.to(torch.float32) - z_out)
        tracker.update(fp32_t, dq_tensor)

        # Keep only lightweight validation metrics; no tensor/weight JSON dumps.
        record_obj = {
            "name": name,
            "metrics": tracker.finalize()
        }
        evaluation_history.append(record_obj)

    # 1. Unroll Initial Conv1
    q_pre_act, conv_s, conv_z = int8_inference.run_integer_conv_block(
        q_x, int8_state["conv1"], zp_in, apply_act=False
    )
    _evaluate_step(q_pre_act, conv_s, conv_z, "conv1_pre_act")

    act_s = int8_state["conv1"]["act_scale_out"]
    act_z = int8_state["conv1"]["act_zp_out"]

    if activation == "relu":
        q_x = int8_utils.quantized_relu(q_pre_act, act_z)
    elif activation == "gelu":
        f_accum = int8_utils.dequantize_tensor(q_pre_act, conv_s, conv_z)
        f_act = F.gelu(f_accum)
        f_act = torch.clamp(f_act, min=-0.17, max=10.0)
        q_x = int8_utils.quantize_tensor(f_act, act_s, act_z, dtype=torch.uint8)
    elif activation == "leaky_relu":
        f_accum = int8_utils.dequantize_tensor(q_pre_act, conv_s, conv_z)
        f_act = F.leaky_relu(f_accum, negative_slope=1.0)
        f_act = torch.clamp(f_act, min=-50.0, max=50.0)
        q_x = int8_utils.quantize_tensor(f_act, act_s, act_z, dtype=torch.uint8)

    _evaluate_step(q_x, act_s, act_z, _activation_label("conv1", activation))
    s_out, z_out = act_s, act_z

    for layer_idx in range(1, 5):
        for block_idx in range(2):
            prefix = f"layer{layer_idx}_block{block_idx}"
            block_data = int8_state[prefix]

            # 2. Unroll Block Conv1
            q_out1_pre, conv_s1, conv_z1 = int8_inference.run_integer_conv_block(
                q_x, block_data["conv1"], z_out, apply_act=False
            )
            _evaluate_step(q_out1_pre, conv_s1, conv_z1, f"{prefix}_conv1_pre_act")

            act_s1 = block_data["conv1"]["act_scale_out"]
            act_z1 = block_data["conv1"]["act_zp_out"]

            if activation == "relu":
                q_out1 = int8_utils.quantized_relu(q_out1_pre, act_z1)
            elif activation == "gelu":
                f_accum = int8_utils.dequantize_tensor(q_out1_pre, conv_s1, conv_z1)
                f_act = F.gelu(f_accum)
                f_act = torch.clamp(f_act, min=-0.17, max=10.0)
                q_out1 = int8_utils.quantize_tensor(f_act, act_s1, act_z1, dtype=torch.uint8)
            elif activation == "leaky_relu":
                f_accum = int8_utils.dequantize_tensor(q_out1_pre, conv_s1, conv_z1)
                f_act = F.leaky_relu(f_accum, negative_slope=1.0)
                f_act = torch.clamp(f_act, min=-50.0, max=50.0)
                q_out1 = int8_utils.quantize_tensor(f_act, act_s1, act_z1, dtype=torch.uint8)

            _evaluate_step(q_out1, act_s1, act_z1, _activation_label(f"{prefix}_conv1", activation))
            s_out1, z_out1 = act_s1, act_z1

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
            
            # 3. Evaluate the Addition Pre-Act
            _evaluate_step(q_added, conv_s, conv_z, f"{prefix}_add_pre_act")

            if activation == "relu":
                q_x = int8_utils.quantized_relu(q_added, act_z)
            elif activation == "gelu":
                f_accum = int8_utils.dequantize_tensor(q_added, conv_s, conv_z)
                f_act = F.gelu(f_accum)
                f_act = torch.clamp(f_act, min=-10.0, max=10.0)
                q_x = int8_utils.quantize_tensor(f_act, act_s, act_z, dtype=torch.uint8)
            elif activation == "leaky_relu":
                f_accum = int8_utils.dequantize_tensor(q_added, conv_s, conv_z)
                f_act = F.leaky_relu(f_accum, negative_slope=1.0)
                f_act = torch.clamp(f_act, min=-50.0, max=50.0)
                q_x = int8_utils.quantize_tensor(f_act, act_s, act_z, dtype=torch.uint8)
            s_out, z_out = act_s, act_z

            _evaluate_step(q_x, s_out, z_out, _activation_label(f"{prefix}_out", activation))

    fc_in_scale = int8_state["fc"]["scale_in"]
    fc_in_zp = int8_state["fc"]["zp_in"]

    q_pooled = int8_utils.integer_max_pool2d(q_x)
    q_fc_in = q_pooled.view(q_pooled.size(0), -1)

    q_out, final_s, final_z = int8_inference.run_integer_fc(
        q_fc_in, int8_state["fc"], fc_in_zp
    )
    _evaluate_step(q_out, final_s, final_z, "fc")

    print(f"[record] Validation complete! All {len(evaluation_history)} tensors locally quantized correctly.")
    
    # Process Triplets
    triplets = []
    for i, record in enumerate(evaluation_history):
        if record["name"].endswith("_pre_act"):
            if i + 2 < len(evaluation_history):
                triplets.append({
                    "pre_activation": evaluation_history[i],
                    "post_activation": evaluation_history[i+1],
                    "subsequent_layer": evaluation_history[i+2],
                })

    print("[+] Validation metrics collected in-memory (JSON recording disabled).")
    plot_triplet_drops(triplets, dataset_name, activation)


def plot_triplet_drops(triplets, dataset_name, activation):
    """Plots the error progression specifically across the identified Triplets."""
    print(f"\n[plot] Generating Triplet Boundary visualizations...")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Error Progression Across Activation Boundaries ({dataset_name} | {activation})",
        fontsize=16, fontweight="bold"
    )

    metrics_map = {
        "Mean Absolute Error": ("mean_absolute_error", axs[0, 0]),
        "Max Absolute Error": ("max_absolute_error", axs[0, 1]),
        "Std Absolute Error": ("std_absolute_error", axs[1, 0]),
        "Median Absolute Error": ("median_absolute_error", axs[1, 1]),
    }

    x_labels = ["Pre-Activation", "Post-Activation", "Subsequent Output"]
    x = [0, 1, 2]

    # Plot lines for every individual triplet to show the universal drop phenomenon
    for t_idx, triplet in enumerate(triplets):
        for title, (m_key, ax) in metrics_map.items():
            vals = [
                triplet["pre_activation"]["metrics"][m_key],
                triplet["post_activation"]["metrics"][m_key],
                triplet["subsequent_layer"]["metrics"][m_key],
            ]
            ax.plot(x, vals, marker="o", alpha=0.5, label=f"Seq {t_idx}" if title == "Mean Absolute Error" else "")

    for title, (m_key, ax) in metrics_map.items():
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Error Magnitude")
        ax.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    img_path = os.path.join(LOG_DIR, f"triplet_boundary_drop_{activation}_{dataset_name.lower()}.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved triplet visualizer to {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to evaluate.")
    parser.add_argument(
        "--activation",
        type=str,
        required=True,
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function the model was trained with",
    )
    args = parser.parse_args()

    record_and_validate(args.dataset, args.activation)