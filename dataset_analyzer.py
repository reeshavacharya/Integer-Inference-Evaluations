import json
import os

import torch
import numpy as np
import cv2
from scipy.stats import kurtosis
from skimage.measure import shannon_entropy
from resnet.resnet18 import (
    setup_MNIST,
    setup_CIFAR10,
    setup_Brain_MRI,
    setup_NIH_Chest,
    setup_OCTMNIST,
    setup_OrganAMNIST,
    setup_BloodMNIST,
)
import matplotlib.pyplot as plt
from resnet import resnet18 as resnet_mod


DATASET_SETUP_FNS = {
    "MNIST": setup_MNIST,
    "CIFAR10": setup_CIFAR10,
    "Brain-MRI": setup_Brain_MRI,
    "NIH-CHEST": setup_NIH_Chest,
    "OCTMNIST": setup_OCTMNIST,
    "OrganAMNIST": setup_OrganAMNIST,
    "BloodMNIST": setup_BloodMNIST,
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset-stats")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "dataset_stats.json")

class DatasetQualityAnalyzer:
    def __init__(self, dataloader, num_images=500):
        """
        Analyzes a PyTorch DataLoader to calculate pre-inference modality variance.
        """
        self.dataloader = dataloader
        self.num_images = num_images

    def analyze(self):
        metrics = {
            "kurtosis": [],          # Outlier metric
            "shannon_entropy": [],   # Complexity metric
            "laplacian_variance": [], # Structural Variance / High-Frequency
        }
        
        processed = 0
        for images, _ in self.dataloader:
            if processed >= self.num_images:
                break
                
            # Move to CPU and convert to numpy for statistical analysis
            images_np = images.detach().cpu().numpy()
            
            for i in range(images_np.shape[0]):
                if processed >= self.num_images:
                    break
                    
                # Denormalize roughly to [0, 1] range if needed, or analyze raw standardized data
                img = images_np[i]
                
                # Handle channel dimension (C, H, W) -> (H, W, C)
                img_hwc = np.transpose(img, (1, 2, 0))
                
                # 1. Outlier Metric: Pixel Kurtosis
                # High kurtosis means heavy tails (extreme outliers). 
                # This directly predicts INT8 min-max stretching.
                metrics["kurtosis"].append(kurtosis(img.flatten(), fisher=True))
                
                # 2. Complexity Metric: Shannon Entropy
                # Measures information density. Predicts if the network requires 
                # continuous floating-point resolution to understand the image.
                # Expected: CIFAR-10 > NIH-CHEST > Brain-MRI > MNIST
                # Entropy requires positive values, so we scale to [0, 1]
                img_scaled = (img_hwc - img_hwc.min()) / (img_hwc.max() - img_hwc.min() + 1e-8)
                metrics["shannon_entropy"].append(shannon_entropy(img_scaled))
                
                # 3. Structural Variance: Laplacian Variance
                # Measures the amount of high-frequency "edges" or texture.
                # If an image is mostly smooth (MNIST), this is low.
                if img_hwc.shape[-1] == 3:
                    gray = cv2.cvtColor((img_scaled * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = (img_scaled.squeeze() * 255).astype(np.uint8)
                metrics["laplacian_variance"].append(cv2.Laplacian(gray, cv2.CV_64F).var())

                processed += 1
                
        # Aggregate and return the mean of each metric
        return {k: float(np.mean(v)) for k, v in metrics.items()}

def plot_dataset_metrics(dataset_name, metrics_dict, output_dir):
    """Generates and saves a bar chart of the 4 quality metrics."""
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a bar chart with different colors for each metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(metrics, values, color=colors[:len(metrics)])

    # Use a symmetrical log scale to handle massive variances (10,000+) alongside negatives (-1)
    ax.set_yscale('symlog')
    
    ax.set_title(f"Pre-Inference Modality Metrics: {dataset_name}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Metric Value (SymLog Scale)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the exact numerical values on top of each bar for readability
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    
    # Format the filename safely and save to the target directory
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    filename = os.path.join(output_dir, f"{safe_name}_quality_metrics.png")
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[analyzer] Saved graph to {filename}")

def _resolve_test_loader(dataset_name: str, batch_size: int = 64):
    setup_fn = DATASET_SETUP_FNS[dataset_name]

    resnet_mod.train_loader = None
    resnet_mod.val_loader = None
    resnet_mod.test_loader = None

    setup_fn(batch_size=batch_size)

    if resnet_mod.test_loader is None:
        raise RuntimeError(f"Could not resolve test loader for {dataset_name}")

    return resnet_mod.test_loader


def analyze_all_datasets(num_images: int = 500):
    dataset_stats = {}
    
    # Ensure the output directory exists before trying to save PNGs
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    for dataset_name in [
        "MNIST",
        "CIFAR10",
        "Brain-MRI",
        "NIH-CHEST",
        "OCTMNIST",
        "OrganAMNIST",
        "BloodMNIST",
    ]:
        print(f"[analyzer] Processing {dataset_name}...")
        loader = _resolve_test_loader(dataset_name)
        analyzer = DatasetQualityAnalyzer(loader, num_images=num_images)
        
        # 1. Get the stats
        stats = analyzer.analyze()
        dataset_stats[dataset_name] = stats
        print(f"[analyzer] {dataset_name}: {stats}")
        
        # 2. Add the plotting trigger here
        plot_dataset_metrics(dataset_name, stats, OUTPUT_DIR)

    return dataset_stats


if __name__ == "__main__":
    all_stats = analyze_all_datasets(num_images=500)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[analyzer] Saved dataset stats to {OUTPUT_JSON}")