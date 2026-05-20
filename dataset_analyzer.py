import json
import os

import torch
import numpy as np
import cv2
from scipy.stats import kurtosis
from skimage.measure import shannon_entropy
from skimage.morphology import skeletonize
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
            "kurtosis": [],          
            "shannon_entropy": [],   
            "laplacian_variance": [], 
            "edge_density": [],      # Visual congestion
            "compactness": [],       # Roundness / Object complexity
            "turns": [],             # Polygons / Sharp corners
            "skeleton_complexity": [] # Internal branching
        }
        
        processed = 0
        for images, _ in self.dataloader:
            if processed >= self.num_images:
                break
                
            images_np = images.detach().cpu().numpy()
            
            for i in range(images_np.shape[0]):
                if processed >= self.num_images:
                    break
                    
                img = images_np[i]
                img_hwc = np.transpose(img, (1, 2, 0))
                
                # 1. Pixel Kurtosis
                metrics["kurtosis"].append(kurtosis(img.flatten(), fisher=True))
                
                # 2. Shannon Entropy
                img_scaled = (img_hwc - img_hwc.min()) / (img_hwc.max() - img_hwc.min() + 1e-8)
                metrics["shannon_entropy"].append(shannon_entropy(img_scaled))
                
                # Convert to Grayscale for structural/shape metrics
                if img_hwc.shape[-1] == 3:
                    gray = cv2.cvtColor((img_scaled * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = (img_scaled.squeeze() * 255).astype(np.uint8)
                    
                # 3. Laplacian Variance
                metrics["laplacian_variance"].append(cv2.Laplacian(gray, cv2.CV_64F).var())

                # ----------------------------------------------------
                # SHAPE ANALYSIS METRICS
                # ----------------------------------------------------
                
                # Create a Binary Mask using Otsu's Thresholding to isolate the object
                _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 4. Edge Density
                # Proportion of edge pixels to total pixels
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.count_nonzero(edges) / max(edges.size, 1)
                metrics["edge_density"].append(edge_density)

                # Find contours on the binary mask to evaluate shape boundaries
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Isolate the largest contour (the primary object/anatomy)
                    c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    perimeter = cv2.arcLength(c, True)
                    
                    # 5. Compactness
                    # Ratio of squared perimeter to area (perfect circle is ~12.57, higher is more irregular)
                    compactness = (perimeter ** 2) / area if area > 0 else 0
                    metrics["compactness"].append(compactness)
                    
                    # 6. Turns
                    # Approximate the contour polygon and count the vertices (turns)
                    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
                    metrics["turns"].append(len(approx))
                else:
                    metrics["compactness"].append(0)
                    metrics["turns"].append(0)

                # 7. Skeletons
                # Measure branching complexity by reducing shape to a 1-pixel wide skeleton
                bool_mask = binary_mask > 0
                skeleton = skeletonize(bool_mask)
                area_pixels = np.count_nonzero(bool_mask)
                
                # Calculate the ratio of skeleton pixels to total shape pixels
                skeleton_complexity = np.count_nonzero(skeleton) / area_pixels if area_pixels > 0 else 0
                metrics["skeleton_complexity"].append(skeleton_complexity)

                processed += 1
                
        # Aggregate and return the mean of each metric
        return {k: float(np.mean(v)) for k, v in metrics.items()}

def plot_dataset_metrics(dataset_name, metrics_dict, output_dir):
    """Generates and saves a bar chart of the quality metrics."""
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use a dynamic colormap to handle the 7 different metrics
    colors = plt.cm.tab10.colors
    bars = ax.bar(metrics, values, color=colors[:len(metrics)])

    ax.set_yscale('symlog')
    
    ax.set_title(f"Pre-Inference Modality Metrics: {dataset_name}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Metric Value (SymLog Scale)", fontsize=12)
    
    # Rotate x-axis labels to prevent text overlap
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    
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
        
        stats = analyzer.analyze()
        dataset_stats[dataset_name] = stats
        print(f"[analyzer] {dataset_name}: {stats}")
        
        plot_dataset_metrics(dataset_name, stats, OUTPUT_DIR)

    return dataset_stats


if __name__ == "__main__":
    all_stats = analyze_all_datasets(num_images=500)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[analyzer] Saved dataset stats to {OUTPUT_JSON}")