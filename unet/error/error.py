import argparse
import importlib.util
import json
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UNET_DIR = os.path.dirname(THIS_DIR)
INT32_DIR = os.path.join(UNET_DIR, "INT32")
for p in (UNET_DIR, THIS_DIR, INT32_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Ensure output directories exist
os.makedirs(os.path.join(THIS_DIR, "json"), exist_ok=True)
os.makedirs(os.path.join(THIS_DIR, "graphs"), exist_ok=True)

from u_net import setup_data, UNet
import utils as quant_utils
import inference as mod_inf

# ---------------------------------------------------------
# Error Tracking Classes
# ---------------------------------------------------------

class LayerMetrics:
    def __init__(self):
        self.cos_sim_sum = 0.0
        self.sqnr_sum = 0.0
        self.mae_sum = 0.0
        self.median_ae_sum = 0.0
        self.std_err_sum = 0.0
        self.q_neg_pct_sum = 0.0
        self.max_abs_err = 0.0
        self.batches = 0
        self.scale = None
        self.zp = None

    def update(self, fp_tensor, dq_tensor, q_tensor, scale=None, zp=None):
        if self.scale is None:
            self.scale = scale
            self.zp = zp

        # Ensure float64 is strictly used to prevent 32-bit mantissa truncation
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

        q_flat = q_tensor.detach().view(-1).to(torch.int64)
        q_neg = (q_flat < zp).float().mean().item() * 100.0

        self.cos_sim_sum += cos_sim
        self.sqnr_sum += sqnr
        self.mae_sum += mae
        self.median_ae_sum += batch_median_err
        self.std_err_sum += batch_std_err
        self.q_neg_pct_sum += q_neg
        self.max_abs_err = max(self.max_abs_err, batch_max_err)
        self.batches += 1

    def finalize(self):
        if self.batches == 0:
            return {}
        
        scale_val = self.scale.item() if isinstance(self.scale, torch.Tensor) else self.scale
        zp_val = self.zp.item() if isinstance(self.zp, torch.Tensor) else self.zp
        
        return {
            "scale": scale_val,
            "zero_point": zp_val,
            "cosine_similarity": self.cos_sim_sum / self.batches,
            "sqnr_db": self.sqnr_sum / self.batches,
            "mean_absolute_error": self.mae_sum / self.batches,
            "median_absolute_error": self.median_ae_sum / self.batches,
            "std_absolute_error": self.std_err_sum / self.batches,
            "max_absolute_error": self.max_abs_err,
            "q_pct_negative": self.q_neg_pct_sum / self.batches,
        }

fp32_tensors = {}

def get_fp32_hook(name):
    def hook(module, input, output):
        fp32_tensors[name] = output.detach().cpu().clone()
    return hook

def attach_fp32_hooks(model):
    handles = []
    # Hook Encoder
    handles.append(model.e11.register_forward_hook(get_fp32_hook("e11")))
    handles.append(model.e12.register_forward_hook(get_fp32_hook("e12")))
    handles.append(model.e21.register_forward_hook(get_fp32_hook("e21")))
    handles.append(model.e22.register_forward_hook(get_fp32_hook("e22")))
    handles.append(model.e31.register_forward_hook(get_fp32_hook("e31")))
    handles.append(model.e32.register_forward_hook(get_fp32_hook("e32")))
    handles.append(model.e41.register_forward_hook(get_fp32_hook("e41")))
    handles.append(model.e42.register_forward_hook(get_fp32_hook("e42")))
    handles.append(model.e51.register_forward_hook(get_fp32_hook("e51")))
    handles.append(model.e52.register_forward_hook(get_fp32_hook("e52")))
    
    # Hook Decoder
    handles.append(model.upconv1.register_forward_hook(get_fp32_hook("upconv1")))
    handles.append(model.d11.register_forward_hook(get_fp32_hook("d11")))
    handles.append(model.d12.register_forward_hook(get_fp32_hook("d12")))
    
    handles.append(model.upconv2.register_forward_hook(get_fp32_hook("upconv2")))
    handles.append(model.d21.register_forward_hook(get_fp32_hook("d21")))
    handles.append(model.d22.register_forward_hook(get_fp32_hook("d22")))
    
    handles.append(model.upconv3.register_forward_hook(get_fp32_hook("upconv3")))
    handles.append(model.d31.register_forward_hook(get_fp32_hook("d31")))
    handles.append(model.d32.register_forward_hook(get_fp32_hook("d32")))
    
    handles.append(model.upconv4.register_forward_hook(get_fp32_hook("upconv4")))
    handles.append(model.d41.register_forward_hook(get_fp32_hook("d41")))
    handles.append(model.d42.register_forward_hook(get_fp32_hook("d42")))
    
    handles.append(model.outconv.register_forward_hook(get_fp32_hook("outconv")))
    return handles

# ---------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------

def evaluate_error(
    dataset_name: str,
    num_data: int = None,
    batch_size: int = 1, # Strict Ops requires smaller batches to avoid OOM
    activation: str = "relu",
    mode: str = "int32",
):
    print(f"\n[error] Starting Layer-by-Layer Error Analysis on {dataset_name} ({activation} | {mode})...")

    if mode != "int32":
        raise ValueError("Only int32 is currently supported in UNet error.py")

    target_dtype = torch.int32

    # Resolve paths
    if dataset_name == "Skin-Lesion": model_name = f"best_unet5_{activation}_skin_lesion.pth"
    elif dataset_name == "Flood": model_name = f"best_unet5_{activation}_flood.pth"
    elif dataset_name == "Brain-MRI-Seg": model_name = f"best_unet5_{activation}_brain_mri_seg.pth"
    elif dataset_name == "BUSI": model_name = f"best_unet5_{activation}_busi.pth"
    else: raise ValueError(f"Unknown UNet dataset: {dataset_name}")

    model_path = os.path.join(UNET_DIR, model_name)
    int_model_path = os.path.join(UNET_DIR, model_name.replace(".pth", "_int32.pth"))
    
    if not os.path.exists(int_model_path):
        raise FileNotFoundError(f"Missing compiled model: {int_model_path}. Run export_{mode}_model.py first.")
    int_state = torch.load(int_model_path, map_location="cpu")

    calib_filename = f"{dataset_name.lower().replace(' ', '_').replace('-', '_')}_{activation}_calibration.json"
    with open(os.path.join(UNET_DIR, "calibration", calib_filename), "r") as f:
        activation_ranges = json.load(f).get("layers")

    model = UNet(n_class=1, activation=activation)
    state = torch.load(model_path, map_location="cpu")
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    _, _, loader = setup_data(train_data=dataset_name, batch_size=batch_size, image_size=256, num_workers=0)

    attach_fp32_hooks(model)
    layer_trackers = {}

    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)
    processed_images = 0

    def _evaluate_step(q_tensor, s_out, z_out, name, fp_tensor):
        tracker = layer_trackers.setdefault(name, LayerMetrics())
        f_deq = s_out * (q_tensor.to(torch.float64) - z_out)
        tracker.update(fp_tensor, f_deq, q_tensor, s_out, z_out)

    with torch.no_grad():
        for images, masks in loader:
            if processed_images >= target_images:
                break
            
            remaining = target_images - processed_images
            if images.size(0) > remaining:
                images, masks = images[:remaining], masks[:remaining]

            # FP32 Forward Pass (populates fp32_tensors hook dict)
            images = images.to(device)
            _ = model(images)

            # INT32 Forward Pass
            scale_in = int_state["meta"]["in_scale"]
            zp_in = int_state["meta"]["in_zp"]
            q_x = quant_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.int32).to(device)

            def process_block(q_x, layer_name, zp_in):
                q_x, s_conv, z_conv = mod_inf.run_integer_layer(q_x, int_state[layer_name], zp_in, apply_activation=False, activation="none")
                fp_pre = fp32_tensors[layer_name].to(device)
                _evaluate_step(q_x, s_conv, z_conv, f"{layer_name}_pre_act", fp_pre)
                
                fp_post = model.act(fp_pre)
                if activation == "gelu":
                    q_x = quant_utils.integer_gelu_lut(q_x, int_state[layer_name]["gelu_lut"].to(q_x.device), z_conv, int_state[layer_name]["gelu_q_min"])
                elif activation == "relu":
                    q_x = quant_utils.quantized_relu(q_x, z_conv)
                elif activation == "leaky_relu":
                    q_x = q_x # identity map
                
                _evaluate_step(q_x, s_conv, z_conv, f"{layer_name}_post_act", fp_post)
                return q_x, s_conv, z_conv

            # ENCODER
            q_x, s, z = process_block(q_x, "e11", zp_in)
            q_x, s, z = process_block(q_x, "e12", z)
            q_e12, s_e12, z_e12 = q_x, s, z
            q_x = mod_inf.pool_int32(q_x)

            q_x, s, z = process_block(q_x, "e21", z)
            q_x, s, z = process_block(q_x, "e22", z)
            q_e22, s_e22, z_e22 = q_x, s, z
            q_x = mod_inf.pool_int32(q_x)

            q_x, s, z = process_block(q_x, "e31", z)
            q_x, s, z = process_block(q_x, "e32", z)
            q_e32, s_e32, z_e32 = q_x, s, z
            q_x = mod_inf.pool_int32(q_x)

            q_x, s, z = process_block(q_x, "e41", z)
            q_x, s, z = process_block(q_x, "e42", z)
            q_e42, s_e42, z_e42 = q_x, s, z
            q_x = mod_inf.pool_int32(q_x)

            q_x, s, z = process_block(q_x, "e51", z)
            q_x, s, z = process_block(q_x, "e52", z)

            # DECODER
            decoder_configs = [
                ("upconv1", "d11", "d12", q_e42, s_e42, z_e42),
                ("upconv2", "d21", "d22", q_e32, s_e32, z_e32),
                ("upconv3", "d31", "d32", q_e22, s_e22, z_e22),
                ("upconv4", "d41", "d42", q_e12, s_e12, z_e12)
            ]

            for up_name, d1_name, d2_name, skip_q, skip_s, skip_z in decoder_configs:
                q_x, s_up, z_up = mod_inf.run_integer_layer(q_x, int_state[up_name], z, apply_activation=False, activation="none")
                _evaluate_step(q_x, s_up, z_up, up_name, fp32_tensors[up_name].to(device))
                
                s_cat = activation_ranges[d1_name]["in_scale"]
                z_cat = activation_ranges[d1_name]["in_zero_point"]
                
                M0_up, shift_up = quant_utils.compute_requantize_multiplier(s_up, s_cat)
                M0_skip, shift_skip = quant_utils.compute_requantize_multiplier(skip_s, s_cat)
                
                q_x_aligned = quant_utils.requantize_tensor(q_x, z_up, z_cat, M0_up, shift_up)
                q_skip_aligned = quant_utils.requantize_tensor(skip_q, skip_z, z_cat, M0_skip, shift_skip)
                
                q_x = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
                
                q_x, s, z = process_block(q_x, d1_name, z_cat)
                q_x, s, z = process_block(q_x, d2_name, z)

            q_out, final_s, final_z = mod_inf.run_integer_layer(q_x, int_state["outconv"], z, apply_activation=False, activation="none")
            _evaluate_step(q_out, final_s, final_z, "outconv", fp32_tensors["outconv"].to(device))

            processed_images += images.size(0)
            print(f"[error] Processed {processed_images}/{target_images} images...")

    return {name: tracker.finalize() for name, tracker in layer_trackers.items()}


def plot_error_metrics(metrics_dict, dataset_name, activation: str = "relu", mode: str = "int32"):
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
    q_neg = [metrics_dict[l]["q_pct_negative"] for l in valid_layers]

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"UNet FP32 vs {mode.upper()} Quantization Error Progression ({dataset_name} | {activation})",
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

    axs[1, 1].plot(x, q_neg, marker="v", color="red", label="Quantized % Negative (q_val < zero_point)")
    axs[1, 1].set_title("Quantized Sign Distribution (% Negative)")
    axs[1, 1].set_ylabel("Percentage (%)")
    axs[1, 1].set_ylim(-5, 105)
    axs[1, 1].grid(True, linestyle=":", alpha=0.7)
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(valid_layers, rotation=90, ha="right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    ds_part = dataset_name.lower().replace(" ", "-")
    graphs_dir = os.path.join(THIS_DIR, "graphs", ds_part)
    os.makedirs(graphs_dir, exist_ok=True)
    filename = os.path.join(graphs_dir, f"quantization_divergence_{ds_part}_{activation}_{mode}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[+] Saved error divergence graphs to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to evaluate.")
    parser.add_argument("--num_data", type=int, default=256, help="Number of images to process.")
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function the model was trained with",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="int32",
        choices=["int32"],
        help="Inference mode used for error analysis",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for error analysis (default: 1)",
    )
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else ["Skin-Lesion", "Flood", "Brain-MRI-Seg", "BUSI"]
    activations = [args.activation] if args.activation else ["relu", "gelu", "leaky_relu"]

    for t in targets:
        for a in activations:
            try:
                metrics = evaluate_error(t, args.num_data, batch_size=args.batch_size, activation=a, mode=args.mode)

                ds_part = t.lower().replace(" ", "-")
                data_dir = os.path.join(THIS_DIR, "json", ds_part)
                os.makedirs(data_dir, exist_ok=True)
                file_name = os.path.join(data_dir, f"error_accumulation_{ds_part}_{a}_{args.mode}.json")

                with open(file_name, "w") as f:
                    json.dump({"dataset": t, "activation": a, "mode": args.mode, "layer_metrics": metrics}, f, indent=2)

                print(f"\n[+] Saved error accumulation log to {file_name}")
                plot_error_metrics(metrics, t, a, f"{args.mode}")
            except Exception as e:
                print(f"[-] Failed to run error.py on {t} ({a}): {e}")
