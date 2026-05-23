"""Compare ReLU, GELU, and Leaky ReLU error-accumulation JSONs and plot MAE/MaxAE/MedianAE/StdAE.

Usage:
  python3 error_comparison.py --dataset MNIST

Requires three files to exist in the current working directory:
  error_accumulation_{dataset}_relu.json
  error_accumulation_{dataset}_gelu.json
    error_accumulation_{dataset}_leaky_relu.json
    (now expects mode suffix: error_accumulation_{dataset}_{activation}_{mode}.json)

Saves: `error_comparison_{dataset}_{mode}.png`
"""
import argparse
import json
import os
import sys
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# JSON input directory (was `data` before rename)
DATA_DIR = os.path.join(THIS_DIR, "json")
os.makedirs(DATA_DIR, exist_ok=True)
# Output directory for comparison graphs
OUT_DIR = os.path.join(THIS_DIR, "comparison")
os.makedirs(OUT_DIR, exist_ok=True)


def load_metrics_normalized(path: str, activation: str):
    with open(path, "r") as f:
        data = json.load(f)
    if "layer_metrics" not in data:
        raise ValueError(f"Missing 'layer_metrics' in {path}")
    
    # Normalize keys by replacing the specific activation suffix with a generic '_act'
    raw_metrics = data["layer_metrics"]
    normalized_metrics = {}
    suffix = f"_{activation}"
    
    for key, val in raw_metrics.items():
        if key.endswith(suffix):
            # Slice off '_relu' or '_gelu' and append '_act'
            norm_key = key[:-len(suffix)] + "_act"
        else:
            norm_key = key
        normalized_metrics[norm_key] = val
        
    return normalized_metrics


def build_series(metrics, layers, key):
    vals = []
    for l in layers:
        if l in metrics and key in metrics[l]:
            vals.append(float(metrics[l][key]))
        else:
            vals.append(float("nan"))
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset slug to compare")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["int8", "int32"],
        help="Integer inference mode to read JSONs for",
    )
    args = parser.parse_args()

    ds = args.dataset
    slug = ds.lower()
    mode = args.mode
    relu_fname = os.path.join(DATA_DIR, f"error_accumulation_{slug}_relu_{mode}.json")
    gelu_fname = os.path.join(DATA_DIR, f"error_accumulation_{slug}_gelu_{mode}.json")
    leaky_relu_fname = os.path.join(DATA_DIR, f"error_accumulation_{slug}_leaky_relu_{mode}.json")

    if not os.path.exists(relu_fname):
        print(f"ERROR: Missing file: {relu_fname}")
        sys.exit(2)
    if not os.path.exists(gelu_fname):
        print(f"ERROR: Missing file: {gelu_fname}")
        sys.exit(2)
    if not os.path.exists(leaky_relu_fname):
        print(f"ERROR: Missing file: {leaky_relu_fname}")
        sys.exit(2)

    # Load and normalize the keys so they match perfectly
    relu_metrics = load_metrics_normalized(relu_fname, "relu")
    gelu_metrics = load_metrics_normalized(gelu_fname, "gelu")
    leaky_relu_metrics = load_metrics_normalized(leaky_relu_fname, "leaky_relu")

    # Now we can safely use the normalized keys as the canonical list
    if relu_metrics:
        layers = list(relu_metrics.keys())
    else:
        layers = list(gelu_metrics.keys())

    relu_mae = build_series(relu_metrics, layers, "mean_absolute_error")
    relu_max = build_series(relu_metrics, layers, "max_absolute_error")
    relu_med = build_series(relu_metrics, layers, "median_absolute_error")
    relu_std = build_series(relu_metrics, layers, "std_absolute_error")

    gelu_mae = build_series(gelu_metrics, layers, "mean_absolute_error")
    gelu_max = build_series(gelu_metrics, layers, "max_absolute_error")
    gelu_med = build_series(gelu_metrics, layers, "median_absolute_error")
    gelu_std = build_series(gelu_metrics, layers, "std_absolute_error")

    leaky_relu_mae = build_series(leaky_relu_metrics, layers, "mean_absolute_error")
    leaky_relu_max = build_series(leaky_relu_metrics, layers, "max_absolute_error")
    leaky_relu_med = build_series(leaky_relu_metrics, layers, "median_absolute_error")
    leaky_relu_std = build_series(leaky_relu_metrics, layers, "std_absolute_error")

    x = range(len(layers))

    fig, axs = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(f"Activation Comparison: ReLU vs GELU ({ds})", fontsize=14, fontweight="bold")

    axs[0].plot(x, relu_mae, marker="o", label="ReLU", color="tab:blue")
    axs[0].plot(x, gelu_mae, marker="x", label="GELU", color="tab:orange")
    axs[0].plot(x, leaky_relu_mae, marker="^", label="Leaky ReLU", color="tab:green")
    axs[0].set_ylabel("Mean Absolute Error")
    axs[0].grid(True, linestyle=":", alpha=0.7)
    axs[0].legend()

    axs[1].plot(x, relu_max, marker="o", label="ReLU", color="tab:blue")
    axs[1].plot(x, gelu_max, marker="x", label="GELU", color="tab:orange")
    axs[1].plot(x, leaky_relu_max, marker="^", label="Leaky ReLU", color="tab:green")
    axs[1].set_ylabel("Max Absolute Error")
    axs[1].grid(True, linestyle=":", alpha=0.7)
    axs[1].legend()

    axs[2].plot(x, relu_med, marker="o", label="ReLU", color="tab:blue")
    axs[2].plot(x, gelu_med, marker="x", label="GELU", color="tab:orange")
    axs[2].plot(x, leaky_relu_med, marker="^", label="Leaky ReLU", color="tab:green")
    axs[2].set_ylabel("Median Absolute Error")
    axs[2].grid(True, linestyle=":", alpha=0.7)
    axs[2].legend()

    axs[3].plot(x, relu_std, marker="o", label="ReLU", color="tab:blue")
    axs[3].plot(x, gelu_std, marker="x", label="GELU", color="tab:orange")
    axs[3].plot(x, leaky_relu_std, marker="^", label="Leaky ReLU", color="tab:green")
    axs[3].set_ylabel("Standard Deviation of Error")
    axs[3].grid(True, linestyle=":", alpha=0.7)
    axs[3].legend()

    axs[3].set_xticks(x)
    axs[3].set_xticklabels(layers, rotation=45, ha="right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(OUT_DIR, f"error_comparison_{slug}_{mode}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[+] Saved comparison plot to {out}")


if __name__ == "__main__":
    main()