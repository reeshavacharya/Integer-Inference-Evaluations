import kagglehub
import os
import subprocess
import argparse
import csv
import shutil
from collections import Counter
from bisect import bisect_right
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, Dataset
from torchvision import datasets, transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_MNIST_DIR = os.path.join(DATA_ROOT, "MNIST")
DATA_CIFAR10_DIR = os.path.join(DATA_ROOT, "CIFAR10")
DATA_BRAIN_MRI_DIR = os.path.join(DATA_ROOT, "Brain-MRI")
DATA_CHEST_DIR = os.path.join(DATA_ROOT, "CHEST")
DATA_MULTI_CANCER_DIR = os.path.join(DATA_ROOT, "Multi-Cancer")
CHEST_IMAGE_SIZE = 28


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# Original LeNet-5 for MNIST
# -----------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),  # 28x28 -> 24x24
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 24x24 -> 12x12
            nn.Conv2d(6, 16, kernel_size=5, stride=1),  # 12x12 -> 8x8
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
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
    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1),  # 28 -> 24
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 24 -> 12
            nn.Conv2d(16, 32, kernel_size=5, stride=1),  # 12 -> 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(120, 84),
            nn.ReLU(),
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
is_multilabel = False
chest_label_names = None


PREPROCESS_SPECS = {
    "MNIST": {"channels": 1, "height": 28, "width": 28},
    "CIFAR10": {"channels": 3, "height": 28, "width": 28},
    "BRAIN-MRI": {"channels": 1, "height": 28, "width": 28},
    "CHEST": {
        "channels": 1,
        "height": CHEST_IMAGE_SIZE,
        "width": CHEST_IMAGE_SIZE,
    },
    "MULTI-CANCER": {"channels": 3, "height": 28, "width": 28},
}


def _normalize_dataset_key(dataset_name: str) -> str:
    key = dataset_name.strip().upper().replace("_", "-").replace(" ", "-")
    if key == "CIFR10":
        return "CIFAR10"
    if "CANCER" in key and key not in ("BRAIN-MRI",):
        return "MULTI-CANCER"
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
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

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
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    train_subset_base, val_subset, test_subset = random_split(
        full_base_eval,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Parallel dataset with training-time augmentation
    train_base_tf = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    test_base_tf = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_train
    )
    full_base_tf = ConcatDataset([train_base_tf, test_base_tf])

    train_subset = Subset(full_base_tf, train_subset_base.indices)

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
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    train_subset_base, val_dataset, test_dataset = random_split(
        full_base_eval,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

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

    train_dataset = Subset(full_base_tf, train_subset_base.indices)

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


def setup_CHEST(batch_size: int = 64):
    global train_loader, val_loader, test_loader, chest_label_names

    chest_root = DATA_CHEST_DIR
    images_dir = os.path.join(chest_root, "sample", "images")
    labels_csv = os.path.join(chest_root, "sample", "sample_labels.csv")

    if not os.path.exists(images_dir):
        raise RuntimeError(f"CHEST images directory not found: {images_dir}")
    if not os.path.exists(labels_csv):
        raise RuntimeError(f"CHEST labels CSV not found: {labels_csv}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((CHEST_IMAGE_SIZE, CHEST_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((CHEST_IMAGE_SIZE, CHEST_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    records = []
    label_set = set()

    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["Image Index"].strip()
            finding_labels = [
                x.strip() for x in row["Finding Labels"].split("|") if x.strip()
            ]

            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                continue

            records.append((image_path, finding_labels))
            label_set.update(finding_labels)

    if len(records) == 0:
        raise RuntimeError("No valid CHEST samples found from sample_labels.csv")

    chest_label_names = sorted(label_set)
    label_to_idx = {name: i for i, name in enumerate(chest_label_names)}
    print(f"CHEST labels ({len(chest_label_names)}): {chest_label_names}")

    class ChestMultiLabelDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            image_path, labels = self.samples[idx]
            image = Image.open(image_path).convert("L")

            if self.transform is not None:
                image = self.transform(image)

            target = torch.zeros(len(label_to_idx), dtype=torch.float32)
            for label_name in labels:
                target[label_to_idx[label_name]] = 1.0

            return image, target

    full_eval_dataset = ChestMultiLabelDataset(records, transform=eval_transform)
    full_train_dataset = ChestMultiLabelDataset(records, transform=train_transform)

    total_len = len(full_eval_dataset)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    train_subset_base, val_subset, test_subset = random_split(
        full_eval_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_subset = Subset(full_train_dataset, train_subset_base.indices)

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

    validate_loader_preprocessing(train_loader, "CHEST", stage="training")
    validate_loader_preprocessing(test_loader, "CHEST", stage="training")

    return train_loader, val_loader, test_loader


def _setup_multi_cancer_folder(cancer_folder: str, batch_size: int = 64):
    """Build 80/10/10 loaders for one Multi-Cancer subgroup."""

    base_root = os.path.join(DATA_MULTI_CANCER_DIR, "Multi Cancer", "Multi Cancer")
    cancer_root = os.path.join(base_root, cancer_folder)

    if not os.path.exists(cancer_root):
        raise RuntimeError(f"Multi-Cancer folder not found: {cancer_root}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    eval_dataset = datasets.ImageFolder(root=cancer_root, transform=eval_transform)
    train_dataset = datasets.ImageFolder(root=cancer_root, transform=train_transform)

    total_len = len(eval_dataset)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    train_subset_base, val_subset, test_subset = random_split(
        eval_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_subset = Subset(train_dataset, train_subset_base.indices)

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

    validate_loader_preprocessing(train_loader, "Multi-Cancer", stage="training")
    validate_loader_preprocessing(test_loader, "Multi-Cancer", stage="training")

    return train_loader, val_loader, test_loader, len(eval_dataset.classes), 3, eval_dataset.classes


def setup_Multi_Cancer_Brain(batch_size: int = 64):
    return _setup_multi_cancer_folder("Brain Cancer", batch_size)


def setup_Multi_Cancer_Breast(batch_size: int = 64):
    return _setup_multi_cancer_folder("Breast Cancer", batch_size)


def setup_Multi_Cancer_Cervical(batch_size: int = 64):
    return _setup_multi_cancer_folder("Cervical Cancer", batch_size)


def setup_Multi_Cancer_Kidney(batch_size: int = 64):
    return _setup_multi_cancer_folder("Kidney Cancer", batch_size)


def setup_Multi_Cancer_Lung_Colon(batch_size: int = 64):
    return _setup_multi_cancer_folder("Lung and Colon Cancer", batch_size)


def setup_Multi_Cancer_Lymphoma(batch_size: int = 64):
    return _setup_multi_cancer_folder("Lymphoma", batch_size)


def setup_Multi_Cancer_Oral(batch_size: int = 64):
    return _setup_multi_cancer_folder("Oral Cancer", batch_size)


def setup_Multi_Cancer(batch_size: int = 64):
    """Prepare every Multi-Cancer subgroup, excluding ALL."""

    cancer_setups = [
        ("Brain Cancer", setup_Multi_Cancer_Brain),
        ("Breast Cancer", setup_Multi_Cancer_Breast),
        ("Cervical Cancer", setup_Multi_Cancer_Cervical),
        ("Kidney Cancer", setup_Multi_Cancer_Kidney),
        ("Lung and Colon Cancer", setup_Multi_Cancer_Lung_Colon),
        ("Lymphoma", setup_Multi_Cancer_Lymphoma),
        ("Oral Cancer", setup_Multi_Cancer_Oral),
    ]

    prepared = []
    for cancer_name, setup_fn in cancer_setups:
        prepared.append((cancer_name, setup_fn(batch_size)))

    return prepared


def _resolve_multi_cancer_target(train_data: str):
    """Map CLI train_data to a single Multi-Cancer setup target."""

    key = train_data.strip().upper().replace("_", "-").replace(" ", "-")
    target_map = {
        "BRAIN-CANCER": ("Brain Cancer", setup_Multi_Cancer_Brain),
        "BREAST-CANCER": ("Breast Cancer", setup_Multi_Cancer_Breast),
        "CERVICAL-CANCER": ("Cervical Cancer", setup_Multi_Cancer_Cervical),
        "KIDNEY-CANCER": ("Kidney Cancer", setup_Multi_Cancer_Kidney),
        "LUNG-AND-COLON-CANCER": (
            "Lung and Colon Cancer",
            setup_Multi_Cancer_Lung_Colon,
        ),
        "LYMPHOMA-CANCER": ("Lymphoma", setup_Multi_Cancer_Lymphoma),
        "ORAL-CANCER": ("Oral Cancer", setup_Multi_Cancer_Oral),
    }
    return target_map.get(key)


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
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            if labels.dim() > 1:
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
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            if labels.dim() > 1:
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# -----------------------------
# Training loop with best model saving
# -----------------------------
def main(args: argparse.Namespace):
    global model, criterion, optimizer, scheduler, is_multilabel
    global train_loader, val_loader, test_loader

    best_val_acc = 0.0
    num_epochs = 10
    is_multilabel = False

    if args.data_dir == DATA_MNIST_DIR:
        best_model_path = "best_lenet5_mnist.pth"
        setup_MNIST(args.batch_size)

        # Leave MNIST setup unchanged in spirit
        model = LeNet5(num_classes=10, in_channels=1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None

    elif args.data_dir == DATA_CIFAR10_DIR:
        best_model_path = "best_lenet5_cifar10.pth"
        num_epochs = 100

        setup_CIFAR10(args.batch_size)

        # CIFAR-10 has 10 classes and 3-channel input
        model = LeNet5(num_classes=10, in_channels=3).to(device)
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
        best_model_path = "best_lenet5_brain_mri.pth"
        num_epochs = 100

        train_dataset = setup_Brain_MRI(args.batch_size)

        # Brain-MRI has 4 classes
        model = MedicalLeNet(num_classes=4, in_channels=1).to(device)

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
    elif args.data_dir == DATA_CHEST_DIR:
        best_model_path = "best_lenet5_chest.pth"
        num_epochs = 100

        setup_CHEST(args.batch_size)

        if chest_label_names is None:
            raise RuntimeError("CHEST labels were not initialized by setup_CHEST")

        # Chest X-rays are multi-label, so use independent sigmoid outputs.
        model = MedicalLeNet(num_classes=len(chest_label_names), in_channels=1).to(
            device
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        is_multilabel = True

    elif args.data_dir == DATA_MULTI_CANCER_DIR:
        num_epochs = 100

        single_multi_target = getattr(args, "multi_cancer_target", None)
        if single_multi_target:
            resolved = _resolve_multi_cancer_target(single_multi_target)
            if resolved is None:
                raise ValueError(
                    f"Unknown Multi-Cancer target: {single_multi_target}"
                )

            cancer_name, setup_fn = resolved
            print(f"\n=== Training Multi-Cancer dataset: {cancer_name} ===")

            (
                train_loader,
                val_loader,
                test_loader,
                num_classes,
                in_channels,
                class_names,
            ) = setup_fn(args.batch_size)

            print(f"Classes ({num_classes}): {class_names}")

            model = MedicalLeNet(num_classes=num_classes, in_channels=in_channels).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )

            safe_name = cancer_name.lower().replace(" ", "_").replace("&", "and")
            safe_name = safe_name.replace("__", "_")
            best_model_path = f"best_lenet5_multi_{safe_name}.pth"

            _train_current_dataset(num_epochs, best_model_path)
            return

        multi_cancer_runs = setup_Multi_Cancer(args.batch_size)

        for cancer_name, cancer_setup in multi_cancer_runs:
            print(f"\n=== Training Multi-Cancer dataset: {cancer_name} ===")

            (
                train_loader,
                val_loader,
                test_loader,
                num_classes,
                in_channels,
                class_names,
            ) = cancer_setup

            print(f"Classes ({num_classes}): {class_names}")

            model = MedicalLeNet(num_classes=num_classes, in_channels=in_channels).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )

            safe_name = cancer_name.lower().replace(" ", "_").replace("&", "and")
            safe_name = safe_name.replace("__", "_")
            best_model_path = f"best_lenet5_multi_{safe_name}.pth"

            _train_current_dataset(num_epochs, best_model_path)

        return

    else:
        print(f"Training using default data directory: {DATA_MNIST_DIR}")
        best_model_path = "best_lenet5_mnist.pth"
        setup_MNIST(args.batch_size)

        model = LeNet5(num_classes=10, in_channels=1).to(device)
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
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            if labels.dim() > 1:
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

    # -----------------------------
    # Load best model and test
    # -----------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


def datasetDownloader(dataset_name: str):
    def cleanup_chest_duplicates(chest_root: str):
        primary_images = os.path.join(chest_root, "sample", "images")
        duplicate_root = os.path.join(chest_root, "sample", "sample")
        duplicate_images = os.path.join(duplicate_root, "images")

        if os.path.exists(primary_images) and os.path.exists(duplicate_images):
            print(f"Removing duplicated CHEST folder: {duplicate_root}")
            shutil.rmtree(duplicate_root)

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

    if dataset_name == "CIFAR10":
        cifar_root = DATA_ROOT
        cifar_folder = os.path.join(cifar_root, "cifar-10-batches-py")
        if not os.path.exists(cifar_folder):
            print("Downloading CIFAR-10 dataset...")
            datasets.CIFAR10(root=cifar_root, train=True, download=True)
            datasets.CIFAR10(root=cifar_root, train=False, download=True)

    if dataset_name == "CHEST":
        chest_root = DATA_CHEST_DIR
        if not os.path.exists(chest_root):
            print("Downloading NIH Chest X-ray dataset...")
            kagglehub.dataset_download(
                "nih-chest-xrays/sample", output_dir=DATA_CHEST_DIR
            )
            subprocess.run(f'cd "{DATA_CHEST_DIR}" && unzip "*.zip"', shell=True)
        cleanup_chest_duplicates(chest_root)

    if dataset_name == "Multi-Cancer":
        if not os.path.exists(DATA_MULTI_CANCER_DIR):
            print("Downloading Multi-Cancer Histology dataset...")
            kagglehub.dataset_download(
                "obulisainaren/multi-cancer", output_dir=DATA_MULTI_CANCER_DIR
            )


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
        help=(
            "Training data to use: MNIST, CIFAR10, Brain-MRI, CHEST, Multi-Cancer, "
            "Brain-Cancer, Breast-Cancer, Cervical-Cancer, Kidney-Cancer, "
            "Lung-And-Colon-Cancer, Lymphoma-Cancer, Oral-Cancer"
        ),
    )

    args = parser.parse_args()
    args.multi_cancer_target = None

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
    elif train_data_key == "CHEST":
        args.data_dir = DATA_CHEST_DIR
        datasetDownloader("CHEST")
    elif train_data_key == "MULTI-CANCER":
        args.data_dir = DATA_MULTI_CANCER_DIR
        datasetDownloader("Multi-Cancer")
    elif _resolve_multi_cancer_target(args.train_data) is not None:
        args.data_dir = DATA_MULTI_CANCER_DIR
        args.multi_cancer_target = args.train_data
        datasetDownloader("Multi-Cancer")
    else:
        print(f"Invalid training data specified. Using default data directory: {DATA_MNIST_DIR}")
        args.data_dir = DATA_MNIST_DIR

    main(args)
