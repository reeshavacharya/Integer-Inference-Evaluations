import kagglehub
import os
import subprocess
import argparse
from collections import Counter
from bisect import bisect_right

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image
from medmnist import OCTMNIST, BloodMNIST, OrganAMNIST, PneumoniaMNIST
from sklearn.metrics import roc_auc_score
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_MNIST_DIR = os.path.join(DATA_ROOT, "MNIST")
DATA_CIFAR10_DIR = os.path.join(DATA_ROOT, "CIFAR10")
DATA_BRAIN_MRI_DIR = os.path.join(DATA_ROOT, "Brain-MRI")
DATA_NIH_CHEST_XRAY_DIR = os.path.join(DATA_ROOT, "NIH-CHEST")
DATA_OCTMNIST_DIR = os.path.join(DATA_ROOT, "OCTMNIST")
DATA_BLOODMNIST_DIR = os.path.join(DATA_ROOT, "BloodMNIST")
DATA_ORGANAMNIST_DIR = os.path.join(DATA_ROOT, "OrganAMNIST")
DATA_PNEUMONIAMNIST_DIR = os.path.join(DATA_ROOT, "PneumoniaMNIST")


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def _normalize_activation_name(activation: str) -> str:
    name = activation.strip().lower()
    if name not in {"relu", "gelu", "leaky_relu"}:
        raise ValueError(f"Unsupported activation: {activation}")
    return name

def _build_activation(activation: str) -> nn.Module:
    activation = _normalize_activation_name(activation)
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    return nn.LeakyReLU(inplace=True, negative_slope=1.0)


# -----------------------------
# Original LeNet-5 for MNIST
# -----------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, activation: str = "relu"):
        super().__init__()
        self.activation = _normalize_activation_name(activation)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),  # 28x28 -> 24x24
            _build_activation(self.activation),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24x24 -> 12x12
            nn.Conv2d(6, 16, kernel_size=5, stride=1),  # 12x12 -> 8x8
            _build_activation(self.activation),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            _build_activation(self.activation),
            nn.Linear(120, 84),
            _build_activation(self.activation),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Stronger LeNet-style model for medical imaging datasets.
# Keeps the overall spirit simple, but improves regularization.
# -----------------------------
class MedicalLeNet(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 1, activation: str = "relu"):
        super().__init__()
        self.activation = _normalize_activation_name(activation)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1),  # 28 -> 24
            nn.BatchNorm2d(16),
            _build_activation(self.activation),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24 -> 12
            nn.Conv2d(16, 32, kernel_size=5, stride=1),  # 12 -> 8
            nn.BatchNorm2d(32),
            _build_activation(self.activation),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 120),
            _build_activation(self.activation),
            nn.Dropout(p=0.4),
            nn.Linear(120, 84),
            _build_activation(self.activation),
            nn.Dropout(p=0.3),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Globals populated by setup
# -----------------------------
train_loader = None
val_loader = None
test_loader = None
model = None
criterion = None
optimizer = None
scheduler = None


PREPROCESS_SPECS = {
    "MNIST": {"channels": 1, "height": 28, "width": 28},
    "CIFAR10": {"channels": 3, "height": 28, "width": 28},
    "BRAIN-MRI": {"channels": 1, "height": 28, "width": 28},
    "NIH-CHEST": {"channels": 1, "height": 28, "width": 28},
    "OCTMNIST": {"channels": 1, "height": 28, "width": 28},
    "BLOODMNIST": {"channels": 3, "height": 28, "width": 28},
    "ORGANAMNIST": {"channels": 1, "height": 28, "width": 28},
    "PNEUMONIAMNIST": {"channels": 1, "height": 28, "width": 28},
}


def _normalize_dataset_key(dataset_name: str) -> str:
    key = dataset_name.strip().upper().replace("_", "-").replace(" ", "-")
    if key == "CIFR10":
        return "CIFAR10"
    return key


def validate_preprocessed_batch(images: torch.Tensor, dataset_name: str, stage: str = "runtime"):
    key = _normalize_dataset_key(dataset_name)
    if key not in PREPROCESS_SPECS:
        return

    spec = PREPROCESS_SPECS[key]
    if images.dim() != 4:
        raise RuntimeError(
            f"[{stage}] Expected NCHW tensor for {dataset_name}, got shape {tuple(images.shape)}"
        )

    _, c, h, w = images.shape
    if c != spec["channels"] or h != spec["height"] or w != spec["width"]:
        raise RuntimeError(
            f"[{stage}] Preprocessing mismatch for {dataset_name}: "
            f"expected (C,H,W)=({spec['channels']},{spec['height']},{spec['width']}), "
            f"got ({c},{h},{w})"
        )

    if not torch.isfinite(images).all():
        raise RuntimeError(f"[{stage}] Found non-finite values after preprocessing for {dataset_name}")


def validate_loader_preprocessing(loader: DataLoader, dataset_name: str, stage: str = "runtime"):
    images, _ = next(iter(loader))
    validate_preprocessed_batch(images, dataset_name, stage=stage)


def _deterministic_split_indices(total_len: int, train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42):
    train_size = int(train_frac * total_len)
    val_size = int(val_frac * total_len)
    test_size = total_len - train_size - val_size
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_len, generator=gen).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size: train_size + val_size]
    test_idx = perm[train_size + val_size:]
    return train_idx, val_idx, test_idx


# -----------------------------
# Data
# -----------------------------
def setup_MNIST(batch_size: int):
    global train_loader, val_loader, test_loader

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset_full = datasets.MNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )

    test_dataset_full = datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    # Combine official train and test into one pool, then split 80/10/10.
    full_dataset = ConcatDataset([train_dataset_full, test_dataset_full])

    total_len = len(full_dataset)
    train_idx, val_idx, test_idx = _deterministic_split_indices(total_len, train_frac=0.8, val_frac=0.1, seed=42)

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    validate_loader_preprocessing(train_loader, "MNIST", stage="training")
    validate_loader_preprocessing(test_loader, "MNIST", stage="training")


def setup_CIFAR10(batch_size: int = 64):
    """Prepare CIFAR-10 loaders with an 80/10/10 non-overlapping split.

    Images are resized to 28x28 but kept as 3-channel so LeNet5 can
    be run with in_channels=3 while preserving the original spatial pipeline.
    """

    global train_loader, val_loader, test_loader

    # CIFAR-10 mean/std for 3 channels
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Stronger augmentation for training, clean transform for val/test
    transform_train = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_eval = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Base datasets with eval transforms for a stable split
    train_base_eval = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_eval
    )
    test_base_eval = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_eval
    )

    full_base_eval = ConcatDataset([train_base_eval, test_base_eval])

    total_len = len(full_base_eval)
    train_idx, val_idx, test_idx = _deterministic_split_indices(total_len, train_frac=0.8, val_frac=0.1, seed=42)

    # Parallel dataset with training-time augmentation
    train_base_tf = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    test_base_tf = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_train
    )
    full_base_tf = ConcatDataset([train_base_tf, test_base_tf])

    train_subset = Subset(full_base_tf, train_idx)
    val_subset = Subset(full_base_eval, val_idx)
    test_subset = Subset(full_base_eval, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    validate_loader_preprocessing(train_loader, "CIFAR10", stage="training")
    validate_loader_preprocessing(test_loader, "CIFAR10", stage="training")


def setup_OCTMNIST(batch_size: int = 64):
    """Prepare OCTMNIST loaders using MedMNIST's predefined train/val/test splits."""
    global train_loader, val_loader, test_loader

    os.makedirs(DATA_OCTMNIST_DIR, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = OCTMNIST(
        split="train",
        transform=transform,
        download=True,
        root=DATA_OCTMNIST_DIR,
    )
    val_dataset = OCTMNIST(
        split="val",
        transform=transform,
        download=True,
        root=DATA_OCTMNIST_DIR,
    )
    test_dataset = OCTMNIST(
        split="test",
        transform=transform,
        download=True,
        root=DATA_OCTMNIST_DIR,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    validate_loader_preprocessing(train_loader, "OCTMNIST", stage="training")
    validate_loader_preprocessing(test_loader, "OCTMNIST", stage="training")


def setup_BloodMNIST(batch_size: int = 64):
    """Prepare BloodMNIST loaders using MedMNIST's predefined train/val/test splits."""
    global train_loader, val_loader, test_loader

    os.makedirs(DATA_BLOODMNIST_DIR, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = BloodMNIST(
        split="train",
        transform=transform,
        download=True,
        root=DATA_BLOODMNIST_DIR,
        as_rgb=True,
    )
    val_dataset = BloodMNIST(
        split="val",
        transform=transform,
        download=True,
        root=DATA_BLOODMNIST_DIR,
        as_rgb=True,
    )
    test_dataset = BloodMNIST(
        split="test",
        transform=transform,
        download=True,
        root=DATA_BLOODMNIST_DIR,
        as_rgb=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    validate_loader_preprocessing(train_loader, "BloodMNIST", stage="training")
    validate_loader_preprocessing(test_loader, "BloodMNIST", stage="training")


def setup_OrganAMNIST(batch_size: int = 64):
    """Prepare OrganAMNIST loaders using MedMNIST's predefined train/val/test splits."""
    global train_loader, val_loader, test_loader

    os.makedirs(DATA_ORGANAMNIST_DIR, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = OrganAMNIST(
        split="train",
        transform=transform,
        download=True,
        root=DATA_ORGANAMNIST_DIR,
    )
    val_dataset = OrganAMNIST(
        split="val",
        transform=transform,
        download=True,
        root=DATA_ORGANAMNIST_DIR,
    )
    test_dataset = OrganAMNIST(
        split="test",
        transform=transform,
        download=True,
        root=DATA_ORGANAMNIST_DIR,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    validate_loader_preprocessing(train_loader, "OrganAMNIST", stage="training")
    validate_loader_preprocessing(test_loader, "OrganAMNIST", stage="training")


def _compute_class_weights_from_subset(subset: Subset, num_classes: int):
    """
    Compute inverse-frequency class weights from a torch.utils.data.Subset
    created from ImageFolder. This helps Brain-MRI if classes are imbalanced.
    """
    dataset = subset.dataset

    # Handle both plain datasets with .targets and ConcatDataset of such datasets.
    targets = []
    if hasattr(dataset, "targets"):
        base_targets = dataset.targets
        targets = [base_targets[i] for i in subset.indices]
    elif isinstance(dataset, ConcatDataset):
        cumulative_sizes = dataset.cumulative_sizes
        datasets_list = dataset.datasets

        for idx in subset.indices:
            ds_idx = bisect_right(cumulative_sizes, idx)
            sample_offset = idx if ds_idx == 0 else idx - cumulative_sizes[ds_idx - 1]
            base_ds = datasets_list[ds_idx]
            base_targets = base_ds.targets
            targets.append(base_targets[sample_offset])
    else:
        raise TypeError("Unsupported dataset type for class weight computation")
    counts = Counter(targets)

    total = len(targets)
    weights = []
    for c in range(num_classes):
        class_count = counts.get(c, 1)
        weights.append(total / (num_classes * class_count))

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights


def setup_Brain_MRI(batch_size: int = 64):
    """
    Prepare Brain-MRI loaders from the Training/Testing folder structure.

    Expected layout:
        <project_root>/data/Brain-MRI/Training/{glioma, meningioma, notumor, pituitary}/...
        <project_root>/data/Brain-MRI/Testing/{glioma, meningioma, notumor, pituitary}/...
    """

    global train_loader, val_loader, test_loader

    train_root = os.path.join(DATA_BRAIN_MRI_DIR, "Training")
    test_root = os.path.join(DATA_BRAIN_MRI_DIR, "Testing")

    # Better augmentation for training only
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Clean deterministic transforms for validation/test
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Combine Training and Testing into a single pool, then split 80/10/10.
    full_train_base = datasets.ImageFolder(
        root=train_root,
        transform=eval_transform,
    )
    full_test_base = datasets.ImageFolder(
        root=test_root,
        transform=eval_transform,
    )

    full_base_eval = ConcatDataset([full_train_base, full_test_base])

    total_len = len(full_base_eval)
    train_idx, val_idx, test_idx = _deterministic_split_indices(total_len, train_frac=0.8, val_frac=0.1, seed=42)

    # Parallel dataset with training augmentations, indexed by the same split
    full_train_tf = datasets.ImageFolder(
        root=train_root,
        transform=train_transform,
    )
    full_test_tf = datasets.ImageFolder(
        root=test_root,
        transform=train_transform,
    )
    full_base_tf = ConcatDataset([full_train_tf, full_test_tf])

    train_dataset = Subset(full_base_tf, train_idx)
    val_dataset = Subset(full_base_eval, val_idx)
    test_dataset = Subset(full_base_eval, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    validate_loader_preprocessing(train_loader, "Brain-MRI", stage="training")
    validate_loader_preprocessing(test_loader, "Brain-MRI", stage="training")

    return train_dataset


class NIHChestDataset(Dataset):
    def __init__(self, image_list, data_dir, csv_file, transform=None):
        self.image_list = image_list
        self.data_dir = data_dir
        self.transform = transform
        self.labels = {}

        with open(csv_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) > 1:
                    img_name = parts[0]
                    finding_labels = parts[1]
                    self.labels[img_name] = finding_labels

        self.all_diseases = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
            "No Finding",
        ]
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(self.all_diseases)}

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]

        preprocessed_dir = os.path.join(self.data_dir, "preprocessed_28x28")
        if os.path.exists(preprocessed_dir):
            preprocessed_path = os.path.join(preprocessed_dir, img_name)
            if os.path.exists(preprocessed_path):
                image = Image.open(preprocessed_path).convert("L")
            else:
                image_path = self._find_original_image(img_name)
                image = Image.open(image_path).convert("L")
                image = image.resize((28, 28), Image.Resampling.LANCZOS)
        else:
            image_path = self._find_original_image(img_name)
            image = Image.open(image_path).convert("L")
            image = image.resize((28, 28), Image.Resampling.LANCZOS)

        finding_str = self.labels.get(img_name, "No Finding")
        diseases = [d.strip() for d in finding_str.split("|")]

        label = torch.zeros(len(self.all_diseases), dtype=torch.float32)
        for disease in diseases:
            if disease in self.disease_to_idx:
                label[self.disease_to_idx[disease]] = 1.0

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def _find_original_image(self, img_name):
        for i in range(1, 13):
            potential_path = os.path.join(self.data_dir, f"images_{i:03d}", "images", img_name)
            if os.path.exists(potential_path):
                return potential_path
        raise FileNotFoundError(f"Image {img_name} not found in any images_xxx/images directory")


def setup_NIH_Chest(batch_size: int = 16):
    global train_loader, val_loader, test_loader

    train_val_file = os.path.join(DATA_NIH_CHEST_XRAY_DIR, "train_val_list.txt")
    test_file = os.path.join(DATA_NIH_CHEST_XRAY_DIR, "test_list.txt")
    csv_file = os.path.join(DATA_NIH_CHEST_XRAY_DIR, "Data_Entry_2017.csv")

    with open(train_val_file, "r") as f:
        train_val_images = [line.strip() for line in f.readlines()]

    with open(test_file, "r") as f:
        test_images = [line.strip() for line in f.readlines()]

    num_train_val = len(train_val_images)
    train_size = int(0.8 * num_train_val)

    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(num_train_val, generator=gen).tolist()

    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    train_images = [train_val_images[i] for i in train_indices]
    val_images = [train_val_images[i] for i in val_indices]

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = NIHChestDataset(train_images, DATA_NIH_CHEST_XRAY_DIR, csv_file, transform=train_transform)
    val_dataset = NIHChestDataset(val_images, DATA_NIH_CHEST_XRAY_DIR, csv_file, transform=eval_transform)
    test_dataset = NIHChestDataset(test_images, DATA_NIH_CHEST_XRAY_DIR, csv_file, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    validate_loader_preprocessing(train_loader, "NIH-CHEST", stage="training")
    validate_loader_preprocessing(test_loader, "NIH-CHEST", stage="training")

    return Subset(train_dataset, list(range(len(train_dataset))))


def setup_PneumoniaMNIST(batch_size: int = 64):
    """Prepare PneumoniaMNIST loaders using MedMNIST's predefined train/val/test splits."""
    global train_loader, val_loader, test_loader

    os.makedirs(DATA_PNEUMONIAMNIST_DIR, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = PneumoniaMNIST(
        split="train",
        transform=transform,
        download=True,
        root=DATA_PNEUMONIAMNIST_DIR,
    )
    val_dataset = PneumoniaMNIST(
        split="val",
        transform=transform,
        download=True,
        root=DATA_PNEUMONIAMNIST_DIR,
    )
    test_dataset = PneumoniaMNIST(
        split="test",
        transform=transform,
        download=True,
        root=DATA_PNEUMONIAMNIST_DIR,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    validate_loader_preprocessing(train_loader, "PneumoniaMNIST", stage="training")
    validate_loader_preprocessing(test_loader, "PneumoniaMNIST", stage="training")

    return train_dataset


def _compute_multilabel_pos_weight_from_subset(subset: Subset, num_classes: int):
    dataset = subset.dataset
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    total = 0.0

    if hasattr(dataset, "labels"):
        for idx in subset.indices:
            _, label = dataset[idx]
            class_counts += label.to(torch.float32)
            total += 1.0
    elif isinstance(dataset, ConcatDataset):
        for idx in subset.indices:
            cumulative_sizes = dataset.cumulative_sizes
            ds_idx = bisect_right(cumulative_sizes, idx)
            sample_offset = idx if ds_idx == 0 else idx - cumulative_sizes[ds_idx - 1]
            _, label = dataset.datasets[ds_idx][sample_offset]
            class_counts += label.to(torch.float32)
            total += 1.0
    else:
        raise TypeError("Unsupported dataset type for multi-label weight computation")

    negatives = total - class_counts
    pos_weight = negatives / torch.clamp(class_counts, min=1.0)
    return pos_weight.to(device)


def _train_current_dataset(num_epochs: int, best_model_path: str):
    """Train the globally configured model/loaders and report final metrics."""

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                labels = labels.to(torch.float32)
            else:
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)
                    if labels.dim() > 1:
                        labels = labels.argmax(dim=1)
                labels = labels.long()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            lr_msg = f" | LR: {current_lr:.6f}"
        else:
            lr_msg = ""

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            f"{lr_msg}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    return best_val_acc, test_loss, test_acc

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(model, dataloader, criterion, is_multilabel: bool = False, is_medmnist: bool = False):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if is_multilabel:
                # keep labels as multi-hot float tensors
                labels_proc = labels.to(torch.float32)
            else:
                # collapse possible extra dims for single-label datasets
                if labels.dim() > 1:
                    labels_proc = labels.squeeze(-1)
                    if labels_proc.dim() > 1:
                        labels_proc = labels_proc.argmax(dim=1)
                else:
                    labels_proc = labels
                labels_proc = labels_proc.long()

            outputs = model(images)
            loss = criterion(outputs, labels_proc)

            total_loss += loss.item() * images.size(0)

            if is_multilabel:
                all_targets.append(labels.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            elif is_medmnist:
                # For MedMNIST (single-label multi-class), collect softmax for AUC
                all_targets.append(labels.cpu().numpy())
                all_outputs.append(torch.softmax(outputs, dim=1).cpu().numpy())
                preds = outputs.argmax(dim=1)
                correct += (preds == labels_proc).sum().item()
                total += labels_proc.size(0)
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels_proc).sum().item()
                total += labels_proc.size(0)

    avg_loss = total_loss / len(dataloader.dataset)

    if is_multilabel:
        targets = np.vstack(all_targets)
        outputs = np.vstack(all_outputs)
        metric_score = roc_auc_score(targets, outputs, average="macro")
    elif is_medmnist:
        targets = np.concatenate(all_targets)
        outputs = np.vstack(all_outputs)
        if outputs.ndim == 2 and outputs.shape[1] == 2:
            y_true = targets.ravel()
            y_score = outputs[:, 1]
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = roc_auc_score(targets, outputs, multi_class="ovr", average="macro")
        acc = 100.0 * correct / total
        metric_score = (auc, acc)
    else:
        metric_score = 100.0 * correct / total

    return avg_loss, metric_score


# -----------------------------
# Training loop with best model saving
# -----------------------------
def main(args: argparse.Namespace):
    global model, criterion, optimizer, scheduler
    global train_loader, val_loader, test_loader

    best_val_acc = 0.0
    num_epochs = 10
    is_multilabel = False
    is_medmnist = False

    if args.data_dir == DATA_MNIST_DIR:
        best_model_path = f"best_lenet5_{args.activation}_mnist.pth"
        setup_MNIST(args.batch_size)

        # Leave MNIST setup unchanged in spirit
        model = LeNet5(num_classes=10, in_channels=1, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None

    elif args.data_dir == DATA_CIFAR10_DIR:
        best_model_path = f"best_lenet5_{args.activation}_cifar10.pth"
        num_epochs = 100

        setup_CIFAR10(args.batch_size)

        # CIFAR-10 has 10 classes and 3-channel input
        model = LeNet5(num_classes=10, in_channels=3, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        # SGD with momentum and weight decay tends to work better on CIFAR-10
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = None

    elif args.data_dir == DATA_BRAIN_MRI_DIR:
        best_model_path = f"best_lenet5_{args.activation}_brain_mri.pth"
        num_epochs = 100

        train_dataset = setup_Brain_MRI(args.batch_size)

        # Brain-MRI has 4 classes
        model = MedicalLeNet(num_classes=4, in_channels=1, activation=args.activation).to(device)

        class_weights = _compute_class_weights_from_subset(train_dataset, num_classes=4)
        print(f"Using class weights: {class_weights.detach().cpu().tolist()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

    elif args.data_dir == DATA_NIH_CHEST_XRAY_DIR:
        best_model_path = f"best_lenet5_{args.activation}_NIH_Chest_XRay.pth"
        num_epochs = 30

        train_dataset = setup_NIH_Chest(args.batch_size)

        model = LeNet5(num_classes=15, in_channels=1, activation=args.activation).to(device)
        pos_weight = _compute_multilabel_pos_weight_from_subset(train_dataset, num_classes=15)
        print(f"Using NIH-CHEST positive class weights: {pos_weight.detach().cpu().tolist()}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = True
        is_medmnist = False

    elif args.data_dir == DATA_OCTMNIST_DIR:
        best_model_path = f"best_lenet5_{args.activation}_octmnist.pth"
        num_epochs = 30

        setup_OCTMNIST(args.batch_size)

        model = LeNet5(num_classes=4, in_channels=1, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = True

    elif args.data_dir == DATA_BLOODMNIST_DIR:
        best_model_path = f"best_lenet5_{args.activation}_bloodmnist.pth"
        num_epochs = 30

        setup_BloodMNIST(args.batch_size)

        model = LeNet5(num_classes=8, in_channels=3, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = True

    elif args.data_dir == DATA_ORGANAMNIST_DIR:
        best_model_path = f"best_lenet5_{args.activation}_organamnist.pth"
        num_epochs = 30

        setup_OrganAMNIST(args.batch_size)

        model = LeNet5(num_classes=11, in_channels=1, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = True

    elif args.data_dir == DATA_PNEUMONIAMNIST_DIR:
        best_model_path = f"best_lenet5_{args.activation}_pneumoniamnist.pth"
        num_epochs = 30

        setup_PneumoniaMNIST(args.batch_size)

        model = MedicalLeNet(num_classes=2, in_channels=1, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = True

    else:
        print(f"Training using default data directory: {DATA_MNIST_DIR}")
        best_model_path = f"best_lenet5_{args.activation}_mnist.pth"
        setup_MNIST(args.batch_size)

        model = LeNet5(num_classes=10, in_channels=1, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                labels = labels.to(torch.float32)
            else:
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)
                    if labels.dim() > 1:
                        labels = labels.argmax(dim=1)
                labels = labels.long()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        val_loss, val_metric = evaluate(model, val_loader, criterion, is_multilabel=is_multilabel, is_medmnist=is_medmnist)

        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            lr_msg = f" | LR: {current_lr:.6f}"
        else:
            lr_msg = ""

        if is_medmnist:
            val_auc, val_acc = val_metric
            print(
                f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val ACC: {val_acc:.2f}%"
            )
            if val_auc > best_val_acc:
                best_val_acc = val_auc
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")

        elif is_multilabel:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val AUROC: {val_metric:.4f}"
            )
            if val_metric > best_val_acc:
                best_val_acc = val_metric
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")

        else:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_metric:.2f}%"
                f"{lr_msg}"
            )
            if val_metric > best_val_acc:
                best_val_acc = val_metric
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")

    # -----------------------------
    # Load best model and test
    # -----------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


def datasetDownloader(dataset_name: str):
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    if dataset_name == "MNIST":
        if not os.path.exists(DATA_MNIST_DIR):
            print("Downloading MNIST dataset...")
            datasets.MNIST(root=DATA_ROOT, train=True, download=True)
            datasets.MNIST(root=DATA_ROOT, train=False, download=True)

    if dataset_name == "Brain-MRI":
        if not os.path.exists(DATA_BRAIN_MRI_DIR):
            print("Downloading Brain-MRI dataset from Kaggle...")
            kagglehub.dataset_download(
                "masoudnickparvar/brain-tumor-mri-dataset",
                output_dir=DATA_BRAIN_MRI_DIR,
            )
            subprocess.run(f'cd "{DATA_BRAIN_MRI_DIR}" && unzip "*.zip"', shell=True)

    if dataset_name == "NIH-CHEST":
        if not os.path.exists(DATA_NIH_CHEST_XRAY_DIR):
            print("Downloading NIH Chest X-Ray dataset from Kaggle...")
            kagglehub.dataset_download(
                "nih-chest-xrays/data",
                output_dir=DATA_NIH_CHEST_XRAY_DIR,
            )

    if dataset_name == "CIFAR10":
        cifar_root = DATA_ROOT
        cifar_folder = os.path.join(cifar_root, "cifar-10-batches-py")
        if not os.path.exists(cifar_folder):
            print("Downloading CIFAR-10 dataset...")
            datasets.CIFAR10(root=cifar_root, train=True, download=True)
            datasets.CIFAR10(root=cifar_root, train=False, download=True)

    if dataset_name == "OCTMNIST":
        print("Downloading OCTMNIST dataset (MedMNIST)...")
        os.makedirs(DATA_OCTMNIST_DIR, exist_ok=True)
        OCTMNIST(split="train", download=True, root=DATA_OCTMNIST_DIR)
        OCTMNIST(split="val", download=True, root=DATA_OCTMNIST_DIR)
        OCTMNIST(split="test", download=True, root=DATA_OCTMNIST_DIR)

    if dataset_name == "BloodMNIST":
        print("Downloading BloodMNIST dataset (MedMNIST)...")
        os.makedirs(DATA_BLOODMNIST_DIR, exist_ok=True)
        BloodMNIST(split="train", download=True, root=DATA_BLOODMNIST_DIR, as_rgb=True)
        BloodMNIST(split="val", download=True, root=DATA_BLOODMNIST_DIR, as_rgb=True)
        BloodMNIST(split="test", download=True, root=DATA_BLOODMNIST_DIR, as_rgb=True)

    if dataset_name == "OrganAMNIST":
        print("Downloading OrganAMNIST dataset (MedMNIST)...")
        os.makedirs(DATA_ORGANAMNIST_DIR, exist_ok=True)
        OrganAMNIST(split="train", download=True, root=DATA_ORGANAMNIST_DIR)
        OrganAMNIST(split="val", download=True, root=DATA_ORGANAMNIST_DIR)
        OrganAMNIST(split="test", download=True, root=DATA_ORGANAMNIST_DIR)


    if dataset_name == "PneumoniaMNIST":
        print("Downloading PneumoniaMNIST dataset (MedMNIST)...")
        os.makedirs(DATA_PNEUMONIAMNIST_DIR, exist_ok=True)
        PneumoniaMNIST(split="train", download=True, root=DATA_PNEUMONIAMNIST_DIR)
        PneumoniaMNIST(split="val", download=True, root=DATA_PNEUMONIAMNIST_DIR)
        PneumoniaMNIST(split="test", download=True, root=DATA_PNEUMONIAMNIST_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="MNIST",
        help="Training data to use: MNIST, CIFAR10, Brain-MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST, PneumoniaMNIST",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function to use",
    )

    args = parser.parse_args()

    train_data_key = args.train_data.strip().upper().replace("_", "-").replace(" ", "-")

    if train_data_key == "MNIST":
        args.data_dir = DATA_MNIST_DIR
        datasetDownloader("MNIST")
    elif train_data_key == "BRAIN-MRI":
        args.data_dir = DATA_BRAIN_MRI_DIR
        datasetDownloader("Brain-MRI")
    elif train_data_key in ("CIFR10", "CIFAR10"):
        args.data_dir = DATA_CIFAR10_DIR
        datasetDownloader("CIFAR10")
    elif train_data_key == "NIH-CHEST":
        args.data_dir = DATA_NIH_CHEST_XRAY_DIR
        datasetDownloader("NIH-CHEST")
    elif train_data_key == "OCTMNIST":
        args.data_dir = DATA_OCTMNIST_DIR
        datasetDownloader("OCTMNIST")
    elif train_data_key == "BLOODMNIST":
        args.data_dir = DATA_BLOODMNIST_DIR
        datasetDownloader("BloodMNIST")
    elif train_data_key == "ORGANAMNIST":
        args.data_dir = DATA_ORGANAMNIST_DIR
        datasetDownloader("OrganAMNIST")
    elif train_data_key == "PNEUMONIAMNIST":
        args.data_dir = DATA_PNEUMONIAMNIST_DIR
        datasetDownloader("PneumoniaMNIST")
    else:
        print(f"Invalid training data specified. Using default data directory: {DATA_MNIST_DIR}")
        args.data_dir = DATA_MNIST_DIR

    main(args)