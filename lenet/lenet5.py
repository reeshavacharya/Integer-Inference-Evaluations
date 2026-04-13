import kagglehub
import os
import subprocess
import argparse
from collections import Counter
from bisect import bisect_right

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision import datasets, transforms


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
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),   # 28x28 -> 24x24
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # 24x24 -> 12x12
            nn.Conv2d(6, 16, kernel_size=5, stride=1),  # 12x12 -> 8x8
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # 8x8 -> 4x4
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
# Stronger LeNet-style model for Brain-MRI only
# Keeps the overall spirit simple, but improves regularization.
# Input is still 1x28x28 so the pipeline remains close to your original.
# -----------------------------
class BrainMRILeNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1),   # 28 -> 24
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),       # 24 -> 12

            nn.Conv2d(16, 32, kernel_size=5, stride=1),  # 12 -> 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),       # 8 -> 4
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

    train_subset_base, val_subset, test_subset = random_split(
        full_base_eval,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Parallel dataset with training-time augmentation
    train_base_tf = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_base_tf = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_train
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
    global model, criterion, optimizer, scheduler

    best_val_acc = 0.0
    num_epochs = 10

    if args.data_dir == "./data/MNIST/":
        best_model_path = "best_lenet5_mnist.pth"
        setup_MNIST(args.batch_size)

        # Leave MNIST setup unchanged in spirit
        model = LeNet5(num_classes=10, in_channels=1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None

    elif args.data_dir == "./data/CIFAR10/":
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

    elif args.data_dir == "./data/Brain-MRI/":
        best_model_path = "best_lenet5_brain_mri.pth"
        num_epochs = 100

        train_dataset = setup_Brain_MRI(args.batch_size)

        # Brain-MRI has 4 classes
        model = BrainMRILeNet(num_classes=4).to(device)

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

    else:
        print("Training using default data directory: ./data/MNIST/")
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
            subprocess.run('cd ./data/Brain-MRI && unzip "*.zip"', shell=True)

    if dataset_name == "CIFAR10":
        cifar_root = "./data"
        cifar_folder = os.path.join(cifar_root, "cifar-10-batches-py")
        if not os.path.exists(cifar_folder):
            print("Downloading CIFAR-10 dataset...")
            datasets.CIFAR10(root=cifar_root, train=True, download=True)
            datasets.CIFAR10(root=cifar_root, train=False, download=True)

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
    else:
        print("Invalid training data specified. Using default data directory: ./data/MNIST/")
        args.data_dir = "./data/MNIST/"

    main(args)