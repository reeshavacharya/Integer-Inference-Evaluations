import json
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from resnet.resnet18 import (
    setup_MNIST,
    setup_CIFAR10,
    setup_Brain_MRI,
    setup_NIH_Chest,
    setup_OCTMNIST,
    setup_OrganAMNIST,
    setup_BloodMNIST,
    setup_PneumoniaMNIST,
)
from resnet import resnet18 as resnet_mod


DATASET_SETUP_FNS = {
    "MNIST":       setup_MNIST,
    "CIFAR10":     setup_CIFAR10,
    "Brain_MRI":   setup_Brain_MRI,
    # "NIH-CHEST":   setup_NIH_Chest,
    "OCTMNIST":    setup_OCTMNIST,
    "OrganAMNIST": setup_OrganAMNIST,
    "BloodMNIST":  setup_BloodMNIST,
    "PneumoniaMNIST": setup_PneumoniaMNIST,
}

OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset-stats")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "dataset_stats.json")

# Canny thresholds — fixed across all datasets so edge density is comparable.
# Low/high follow the commonly used 1:3 ratio recommended in the original
# Canny (1986) paper.  Adjust here if your images are unusually low-contrast.
CANNY_LOW  = 50
CANNY_HIGH = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gray_uint8(img_chw: np.ndarray) -> np.ndarray:
    """Convert a CHW float image (any range) to an HxW uint8 grayscale image."""
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    lo, hi  = img_hwc.min(), img_hwc.max()
    img_norm = (img_hwc - lo) / (hi - lo + 1e-8)

    if img_norm.shape[-1] == 3:
        gray = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img_norm.squeeze() * 255).astype(np.uint8)

    return gray


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Per-pixel Sobel gradient magnitude as a float32 map."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def gradient_centroid_deviation(grad_mag: np.ndarray) -> float:
    """
    Gradient energy centroid deviation.

    Treats the Sobel gradient magnitude map as a 2-D mass distribution,
    finds its energy-weighted centroid, and returns the Euclidean distance
    of that centroid from the geometric centre of the image.

    Normalised by the half-diagonal so values are resolution-independent
    and comparable across datasets.

    Interpretation
    --------------
    Low  → gradient energy is concentrated near the image centre
           (well-centred, compact ROI — e.g. Brain_MRI, BloodMNIST).
    High → energy is off-centre or spread towards the periphery
           (scattered / irregular ROI — e.g. NIH-CHEST, OCTMNIST).
    """
    H, W   = grad_mag.shape
    total  = grad_mag.sum()
    if total == 0:
        return 0.0

    ys = np.arange(H).reshape(-1, 1).astype(np.float32)
    xs = np.arange(W).reshape(1, -1).astype(np.float32)

    cy = (grad_mag * ys).sum() / total
    cx = (grad_mag * xs).sum() / total

    centre_y, centre_x = (H - 1) / 2.0, (W - 1) / 2.0
    half_diagonal      = np.sqrt(H ** 2 + W ** 2) / 2.0
    deviation          = np.sqrt((cy - centre_y) ** 2 + (cx - centre_x) ** 2)

    return float(deviation / half_diagonal)


def spatial_entropy_of_gradient(grad_mag: np.ndarray) -> float:
    """
    Spatial entropy of the gradient magnitude map.

    Normalises the gradient magnitude map into a probability distribution
    over spatial positions and computes its Shannon entropy (in bits).

    Interpretation
    --------------
    Low  → gradient energy is localised (consistent, predictable ROI).
    High → gradient energy is diffuse across the image (scattered edges,
           variable structure).  Directly predicts a wide calibration
           range and coarser INT8 quantisation steps.
    """
    total = grad_mag.sum()
    if total == 0:
        return 0.0

    p       = (grad_mag / total).flatten()
    nonzero = p[p > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def canny_edge_density(gray: np.ndarray) -> float:
    """
    Canny edge density.

    Implements the O.ED metric from Chu et al. (2025) "What Makes a
    Visualization Image Complex?" (arXiv:2510.08332), Table 2:

        ED = P_edge / P_total

    where P_edge is the number of pixels identified as edges by the Canny
    detector and P_total is the total number of pixels in the image.

    The Canny thresholds (CANNY_LOW / CANNY_HIGH) are fixed constants
    defined at the top of this file so that edge density is directly
    comparable across all datasets.

    Interpretation
    --------------
    Low  → few, clean boundaries — geometrically simple, compact shapes
           (circles, smooth blobs).  Typical of Brain_MRI and BloodMNIST.
    High → many edges — complex, fragmented, or textured boundaries.
           Typical of NIH-CHEST (overlapping anatomical structures) or
           OCTMNIST (horizontal retinal layer bands across full width).

    Relationship to the quantisation hypothesis
    -------------------------------------------
    High edge density forces VGG19's early conv layers to activate across
    a large spatial region.  Combined with high centroid deviation or high
    spatial entropy, this widens the activation distribution and degrades
    the effectiveness of a single INT8 calibration scale.
    """
    edges    = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    p_edge   = int(np.count_nonzero(edges))
    p_total  = int(edges.size)
    return float(p_edge / p_total) if p_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class DatasetGeometryAnalyzer:
    """
    Computes three gradient / edge -based spatial geometry metrics.

    Metrics reported (mean and std across num_images samples)
    ---------------------------------------------------------
    centroid_deviation
        Gradient energy centroid deviation (normalised, dimensionless).
    spatial_gradient_entropy
        Shannon entropy of the spatial gradient distribution (bits).
    canny_edge_density
        Proportion of Canny-detected edge pixels (dimensionless, 0-1).
        Defined per Chu et al. arXiv:2510.08332, Table 2 (O.ED).
    """

    def __init__(self, dataloader, num_images: int = 500):
        self.dataloader = dataloader
        self.num_images = num_images

    def analyze(self) -> dict:
        centroid_devs    = []
        spatial_entropies = []
        edge_densities   = []

        processed = 0
        for images, _ in self.dataloader:
            if processed >= self.num_images:
                break

            images_np = images.detach().cpu().numpy()

            for i in range(images_np.shape[0]):
                if processed >= self.num_images:
                    break

                gray     = _to_gray_uint8(images_np[i])
                grad_mag = _gradient_magnitude(gray)

                centroid_devs.append(gradient_centroid_deviation(grad_mag))
                spatial_entropies.append(spatial_entropy_of_gradient(grad_mag))
                edge_densities.append(canny_edge_density(gray))

                processed += 1

        return {
            "centroid_deviation_mean":         float(np.mean(centroid_devs)),
            "centroid_deviation_std":          float(np.std(centroid_devs)),
            "spatial_gradient_entropy_mean":   float(np.mean(spatial_entropies)),
            "spatial_gradient_entropy_std":    float(np.std(spatial_entropies)),
            "canny_edge_density_mean":         float(np.mean(edge_densities)),
            "canny_edge_density_std":          float(np.std(edge_densities)),
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dataset_metrics(dataset_name: str, metrics: dict, output_dir: str):
    """
    Bar chart with error bars for the three geometry metrics.
    Uses a dual y-axis: centroid deviation and edge density share the left
    axis (both dimensionless 0-1); spatial entropy uses the right axis
    (bits, different scale).
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bar_specs = [
        # (label,                          mean key,                        std key,                         axis,  color)
        ("Centroid\nDeviation",            "centroid_deviation_mean",        "centroid_deviation_std",        ax1,   "#4C72B0"),
        ("Canny Edge\nDensity",            "canny_edge_density_mean",        "canny_edge_density_std",        ax1,   "#55A868"),
        ("Spatial Gradient\nEntropy (bits)","spatial_gradient_entropy_mean", "spatial_gradient_entropy_std", ax2,   "#DD8452"),
    ]

    x      = np.arange(len(bar_specs))
    width  = 0.5
    colors = [s[4] for s in bar_specs]

    for idx, (label, mk, sk, ax, color) in enumerate(bar_specs):
        mean = metrics[mk]
        std  = metrics[sk]
        bar  = ax.bar(idx, mean, width, yerr=std, capsize=7,
                      color=color, edgecolor="black", linewidth=0.8,
                      error_kw={"elinewidth": 1.5, "ecolor": "black"},
                      label=label)
        ax.text(idx, mean + std + mean * 0.05,
                f"{mean:.4f}\n±{std:.4f}",
                ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([s[0] for s in bar_specs], fontsize=10)
    ax1.set_ylabel("Value (dimensionless)", fontsize=10)
    ax2.set_ylabel("Spatial Gradient Entropy (bits)", fontsize=10, color="#DD8452")
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    ax1.set_title(f"Spatial Geometry Metrics: {dataset_name}",
                  fontsize=13, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    path      = os.path.join(output_dir, f"{safe_name}_geometry_metrics.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[analyzer] Saved plot → {path}")


def plot_cross_dataset_comparison(all_stats: dict, output_dir: str):
    """
    Grouped bar chart comparing all datasets on all three metrics.

    Centroid deviation and Canny edge density share the left axis
    (both are dimensionless proportions).  Spatial gradient entropy
    uses its own right axis (bits).

    This is the key figure for the hypothesis: Brain_MRI and BloodMNIST
    should cluster at low centroid deviation AND low edge density,
    separate from the rest.
    """
    datasets = list(all_stats.keys())
    n        = len(datasets)
    x        = np.arange(n)
    width    = 0.25

    cd_means = [all_stats[d]["centroid_deviation_mean"]       for d in datasets]
    cd_stds  = [all_stats[d]["centroid_deviation_std"]        for d in datasets]
    ed_means = [all_stats[d]["canny_edge_density_mean"]       for d in datasets]
    ed_stds  = [all_stats[d]["canny_edge_density_std"]        for d in datasets]
    se_means = [all_stats[d]["spatial_gradient_entropy_mean"] for d in datasets]
    se_stds  = [all_stats[d]["spatial_gradient_entropy_std"]  for d in datasets]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ekw = {"elinewidth": 1.2, "ecolor": "black"}

    ax1.bar(x - width, cd_means, width, yerr=cd_stds, capsize=4,
            label="Centroid Deviation (left)",
            color="#4C72B0", alpha=0.85, edgecolor="black", error_kw=ekw)

    ax1.bar(x,          ed_means, width, yerr=ed_stds, capsize=4,
            label="Canny Edge Density (left)",
            color="#55A868", alpha=0.85, edgecolor="black", error_kw=ekw)

    ax2.bar(x + width,  se_means, width, yerr=se_stds, capsize=4,
            label="Spatial Gradient Entropy (right)",
            color="#DD8452", alpha=0.85, edgecolor="black", error_kw=ekw)

    ax1.set_xlabel("Dataset", fontsize=12)
    ax1.set_ylabel("Value (dimensionless)", fontsize=11)
    ax2.set_ylabel("Spatial Gradient Entropy (bits)", fontsize=11, color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=30, ha="right", fontsize=10)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=9)

    ax1.set_title("Cross-Dataset Spatial Geometry Comparison",
                  fontsize=14, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_dataset_geometry_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[analyzer] Saved cross-dataset comparison → {path}")


# ---------------------------------------------------------------------------
# Loader resolution
# ---------------------------------------------------------------------------

def _resolve_test_loader(dataset_name: str, batch_size: int = 64):
    setup_fn = DATASET_SETUP_FNS[dataset_name]

    resnet_mod.train_loader = None
    resnet_mod.val_loader   = None
    resnet_mod.test_loader  = None

    setup_fn(batch_size=batch_size)

    if resnet_mod.test_loader is None:
        raise RuntimeError(f"Could not resolve test loader for {dataset_name}")

    return resnet_mod.test_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = [
    "MNIST",
    "CIFAR10",
    "Brain_MRI",
    # "NIH-CHEST",
    "OCTMNIST",
    "OrganAMNIST",
    "BloodMNIST",
    "PneumoniaMNIST",
]


def analyze_all_datasets(num_images: int = 500) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_stats = {}

    for dataset_name in DATASETS:
        print(f"[analyzer] Processing {dataset_name}...")
        loader   = _resolve_test_loader(dataset_name)
        analyzer = DatasetGeometryAnalyzer(loader, num_images=num_images)
        stats    = analyzer.analyze()

        all_stats[dataset_name] = stats
        print(f"[analyzer] {dataset_name}: {stats}")

        plot_dataset_metrics(dataset_name, stats, OUTPUT_DIR)

    plot_cross_dataset_comparison(all_stats, OUTPUT_DIR)
    return all_stats


if __name__ == "__main__":
    all_stats = analyze_all_datasets(num_images=500)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"[analyzer] Saved all stats → {OUTPUT_JSON}")