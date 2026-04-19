import kagglehub
import os
import argparse
import csv
import subprocess
import shutil
from collections import Counter
from bisect import bisect_right
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, Dataset
from torchvision import datasets, transforms


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# ResNet-18
# -----------------------------
class FloatAdd(nn.Module):
    """A dummy module to make addition visible to calibration hooks."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return x + y

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True) # Explicit ReLU 1
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # The Addition Wrapper
        self.add = FloatAdd()
        self.relu2 = nn.ReLU(inplace=True) # Explicit ReLU 2
        
        # The Shortcut Fix
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Replaces the empty Sequential so the hook can record the pass-through scale
            self.shortcut = nn.Identity() 

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # The addition is now a visible module
        skip = self.shortcut(x)
        out = self.add(out, skip)
        
        out = self.relu2(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # Use the provided number of input channels instead of hardcoding 3
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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


# -----------------------------
# Data
# -----------------------------
def setup_MNIST(batch_size: int):
    global train_loader, val_loader, test_loader

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset_full = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset_full = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Combine official train and test to create a single pool,
    # then split 80/10/10 with no overlap.
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


def setup_CIFAR10(batch_size: int = 64):
    """Prepare CIFAR-10 loaders with a train/val split.

    Uses standard CIFAR-10 normalization and simple augmentations for training.
    """

    global train_loader, val_loader, test_loader

    # Standard CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Base datasets with evaluation transform for consistent splitting
    train_base_eval = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_eval
    )

    test_base_eval = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_eval
    )

    full_base_eval = ConcatDataset([train_base_eval, test_base_eval])

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
    train_base_tf = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_base_tf = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_train
    )
    full_base_tf = ConcatDataset([train_base_tf, test_base_tf])

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
        ./data/Brain-MRI/Training/{glioma, meningioma, notumor, pituitary}/...
        ./data/Brain-MRI/Testing/{glioma, meningioma, notumor, pituitary}/...
    """

    global train_loader, val_loader, test_loader

    train_root = "./data/Brain-MRI/Training"
    test_root = "./data/Brain-MRI/Testing"

    # Better augmentation for training only
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

    return train_dataset


class EyeRetinopathyDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _load_eye_samples(image_dir: str, label_csv: str, id_col: str, label_col: str):
    samples = []

    with open(label_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row[id_col].strip()
            label = int(row[label_col])
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                samples.append((image_path, label))

    return samples


def setup_EYE(batch_size: int = 64):
    global train_loader, val_loader, test_loader

    eye_root = "./data/EYE"
    labels_root = os.path.join(eye_root, "labels")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    samples = []
    samples.extend(
        _load_eye_samples(
            os.path.join(eye_root, "resized train 15"),
            os.path.join(labels_root, "trainLabels15.csv"),
            id_col="image",
            label_col="level",
        )
    )
    samples.extend(
        _load_eye_samples(
            os.path.join(eye_root, "resized test 15"),
            os.path.join(labels_root, "testLabels15.csv"),
            id_col="image",
            label_col="level",
        )
    )
    samples.extend(
        _load_eye_samples(
            os.path.join(eye_root, "resized train 19"),
            os.path.join(labels_root, "trainLabels19.csv"),
            id_col="id_code",
            label_col="diagnosis",
        )
    )

    if os.path.exists(os.path.join(labels_root, "testImages19.csv")):
        with open(os.path.join(labels_root, "testImages19.csv"), "r", newline="") as f:
            unlabeled_count = sum(1 for _ in csv.DictReader(f))
        print(f"Skipping {unlabeled_count} images from resized test 19 (no labels provided).")

    if len(samples) == 0:
        raise RuntimeError("No labeled EYE samples were found. Check data/EYE paths.")

    eval_dataset = EyeRetinopathyDataset(samples, transform=eval_transform)
    train_dataset = EyeRetinopathyDataset(samples, transform=train_transform)

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

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader, 5, 3


def setup_CHEST(batch_size: int = 64):
    global train_loader, val_loader, test_loader, chest_label_names

    chest_root = "./data/CHEST"
    images_dir = os.path.join(chest_root, "sample", "images")
    labels_csv = os.path.join(chest_root, "sample", "sample_labels.csv")

    if not os.path.exists(images_dir):
        raise RuntimeError(f"CHEST images directory not found: {images_dir}")
    if not os.path.exists(labels_csv):
        raise RuntimeError(f"CHEST labels CSV not found: {labels_csv}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
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
            finding_labels = [x.strip() for x in row["Finding Labels"].split("|") if x.strip()]
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

    eval_dataset = ChestMultiLabelDataset(records, transform=eval_transform)
    train_dataset = ChestMultiLabelDataset(records, transform=train_transform)

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

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader, len(chest_label_names), 1


def _setup_multi_cancer_folder(cancer_folder: str, batch_size: int = 64):
    base_root = "./data/Multi-Cancer/Multi Cancer/Multi Cancer"
    cancer_root = os.path.join(base_root, cancer_folder)

    if not os.path.exists(cancer_root):
        raise RuntimeError(f"Multi-Cancer folder not found: {cancer_root}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
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

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

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
    return [
        ("Brain Cancer", setup_Multi_Cancer_Brain(batch_size)),
        ("Breast Cancer", setup_Multi_Cancer_Breast(batch_size)),
        ("Cervical Cancer", setup_Multi_Cancer_Cervical(batch_size)),
        ("Kidney Cancer", setup_Multi_Cancer_Kidney(batch_size)),
        ("Lung and Colon Cancer", setup_Multi_Cancer_Lung_Colon(batch_size)),
        ("Lymphoma", setup_Multi_Cancer_Lymphoma(batch_size)),
        ("Oral Cancer", setup_Multi_Cancer_Oral(batch_size)),
    ]


def _train_current_dataset(num_epochs: int, best_model_path: str):
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
    global model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, is_multilabel

    best_val_acc = 0.0
    num_epochs = 10
    is_multilabel = False

    if args.data_dir == "./data/MNIST/":
        best_model_path = "best_resnet18_mnist.pth"
        setup_MNIST(args.batch_size)

        # Leave MNIST setup unchanged in spirit
        model = ResNet18(num_classes=10, in_channels=args.in_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None

    elif args.data_dir == "./data/CIFAR10/":
        best_model_path = "best_resnet18_cifar10.pth"
        num_epochs = 50
        setup_CIFAR10(args.batch_size)

        # CIFAR-10 has 10 classes, RGB input (in_channels should be 3)
        model = ResNet18(num_classes=10, in_channels=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None

    elif args.data_dir == "./data/Brain-MRI/":
        best_model_path = "best_resnet18_brain_mri.pth"
        num_epochs = 50

        train_dataset = setup_Brain_MRI(args.batch_size)

        # Brain-MRI has 4 classes
        model = ResNet18(num_classes=4, in_channels=args.in_channels).to(device)

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

    elif args.data_dir == "./data/EYE/":
        best_model_path = "best_resnet18_eye.pth"
        num_epochs = 50

        train_loader, val_loader, test_loader, num_classes, in_channels = setup_EYE(args.batch_size)

        model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
        criterion = nn.CrossEntropyLoss()
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

    elif args.data_dir == "./data/CHEST/":
        best_model_path = "best_resnet18_chest.pth"
        num_epochs = 50

        train_loader, val_loader, test_loader, num_classes, in_channels = setup_CHEST(args.batch_size)

        model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
        criterion = nn.BCEWithLogitsLoss()
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
        is_multilabel = True

    elif args.data_dir == "./data/Multi-Cancer/":
        num_epochs = 50
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

            model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
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
            best_model_path = f"best_resnet18_multi_{safe_name}.pth"

            _train_current_dataset(num_epochs, best_model_path)

        return

    else:
        print("Training using default data directory: ./data/MNIST/")
        best_model_path = "best_resnet18_mnist.pth"
        setup_MNIST(args.batch_size)

        model = ResNet18(num_classes=10, in_channels=args.in_channels).to(device)
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
            print("Removing duplicated CHEST folder: ./data/CHEST/sample/sample")
            shutil.rmtree(duplicate_root)

    if not os.path.exists("./data"):
        os.makedirs("./data")

    if dataset_name == "MNIST":
        if not os.path.exists("./data/MNIST"):
            print("Downloading MNIST dataset...")
            datasets.MNIST(root="./data", train=True, download=True)
            datasets.MNIST(root="./data", train=False, download=True)

    if dataset_name == "Brain-MRI":
        if not os.path.exists("./data/Brain-MRI"):
            print("Downloading Brain-MRI dataset from Kaggle...")
            kagglehub.dataset_download(
                "masoudnickparvar/brain-tumor-mri-dataset",
                output_dir="./data/Brain-MRI",
            )

    if dataset_name == "CIFAR10":
        cifar_root = "./data"
        cifar_folder = os.path.join(cifar_root, "cifar-10-batches-py")
        if not os.path.exists(cifar_folder):
            print("Downloading CIFAR-10 dataset...")
            datasets.CIFAR10(root=cifar_root, train=True, download=True)
            datasets.CIFAR10(root=cifar_root, train=False, download=True)

    if dataset_name == "EYE":
        if not os.path.exists("./data/EYE"):
            print("Downloading 2015-2019 Blindness Detection Images dataset...")
            kagglehub.dataset_download(
                "benjaminwarner/resized-2015-2019-blindness-detection-images",
                output_dir="./data/EYE",
            )
            subprocess.run('cd ./data/EYE && unzip "*.zip"', shell=True)

    if dataset_name == "CHEST":
        chest_root = "./data/CHEST"
        if not os.path.exists(chest_root):
            print("Downloading NIH Chest X-ray sample dataset...")
            kagglehub.dataset_download("nih-chest-xrays/sample", output_dir=chest_root)
            subprocess.run('cd ./data/CHEST && unzip "*.zip"', shell=True)
        cleanup_chest_duplicates(chest_root)

    if dataset_name == "Multi-Cancer":
        if not os.path.exists("./data/Multi-Cancer"):
            print("Downloading Multi-Cancer dataset...")
            kagglehub.dataset_download(
                "obulisainaren/multi-cancer",
                output_dir="./data/Multi-Cancer",
            )
            subprocess.run('cd ./data/Multi-Cancer && unzip "*.zip"', shell=True)

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
        help="Training data to use",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Number of input channels for the model (e.g., 1 for grayscale, 3 for RGB)",
    )

    args = parser.parse_args()

    if args.train_data == "MNIST":
        args.data_dir = "./data/MNIST/"
        datasetDownloader("MNIST")
    elif args.train_data == "Brain-MRI":
        args.data_dir = "./data/Brain-MRI/"
        datasetDownloader("Brain-MRI")
    elif args.train_data.upper() in ("CIFR10", "CIFAR10"):
        args.data_dir = "./data/CIFAR10/"
        datasetDownloader("CIFAR10")
    elif args.train_data == "EYE":
        args.data_dir = "./data/EYE/"
        datasetDownloader("EYE")
    elif args.train_data == "CHEST":
        args.data_dir = "./data/CHEST/"
        datasetDownloader("CHEST")
    elif args.train_data == "Multi-Cancer":
        args.data_dir = "./data/Multi-Cancer/"
        datasetDownloader("Multi-Cancer")
    else:
        print("Invalid training data specified. Using default data directory: ./data/MNIST/")
        args.data_dir = "./data/MNIST/"

    main(args)