import kagglehub
import os
import argparse
from collections import Counter
from bisect import bisect_right
from sklearn.metrics import roc_auc_score
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image
from medmnist import OCTMNIST, BloodMNIST, OrganAMNIST, PneumoniaMNIST

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_MNIST_DIR = os.path.join(DATA_ROOT, "MNIST")
DATA_CIFAR10_DIR = os.path.join(DATA_ROOT, "CIFAR10")
DATA_BRAIN_MRI_DIR = os.path.join(DATA_ROOT, "Brain_MRI")
DATA_OCTMNIST_DIR = os.path.join(DATA_ROOT, "OCTMNIST")
DATA_BLOODMNIST_DIR = os.path.join(DATA_ROOT, "BloodMNIST")
DATA_ORGANAMNIST_DIR = os.path.join(DATA_ROOT, "OrganAMNIST")
DATA_PNEUMONIAMNIST_DIR = os.path.join(DATA_ROOT, "PneumoniaMNIST")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# VGG-19 Architecture
# -----------------------------
class VGG19(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, activation="relu"):
        super(VGG19, self).__init__()
        self.activation = activation
        self.features = self._make_layers(in_channels)
        self.avgpool = nn.AdaptiveMaxPool2d((7, 7))
        
        if self.activation == "gelu":
            act_layer = nn.GELU
        elif self.activation == "leaky_relu":
            act_layer = lambda: nn.LeakyReLU(negative_slope=1.0, inplace=True)
        else:
            act_layer = lambda: nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            act_layer(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            act_layer(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, in_channels):
        layers = []
        # VGG19 Configuration: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
               512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        
        in_c = in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.activation == "gelu":
                    act_layer = nn.GELU()
                elif self.activation == "leaky_relu":
                    act_layer = nn.LeakyReLU(negative_slope=1.0, inplace=True)
                else:
                    act_layer = nn.ReLU(inplace=True)
                    
                conv2d = nn.Conv2d(in_c, v, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(v), act_layer]
                in_c = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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


# Note: MNIST and Brain_MRI resized to 32x32 to prevent spatial collapse 
# through the 5 MaxPool layers of VGG19.
PREPROCESS_SPECS = {
    "MNIST": {"channels": 1, "height": 32, "width": 32},
    "CIFAR10": {"channels": 3, "height": 32, "width": 32},
    "BRAIN_MRI": {"channels": 1, "height": 32, "width": 32},
    "OCTMNIST": {"channels": 1, "height": 32, "width": 32},
    "BLOODMNIST": {"channels": 3, "height": 32, "width": 32},
    "ORGANAMNIST": {"channels": 1, "height": 32, "width": 32},
    "PNEUMONIAMNIST": {"channels": 1, "height": 32, "width": 32},
}

def _normalize_dataset_key(dataset_name: str) -> str:
    key = dataset_name.strip().upper().replace("_", "-").replace(" ", "-")
    if key == "CIFR10":
        return "CIFAR10"
    if key == "PNEUMONIAMNIST":
        return "PneumoniaMNIST"
    return key


def _normalize_classification_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)
    return labels.long()


def validate_preprocessed_batch(images: torch.Tensor, dataset_name: str, stage: str = "runtime"):
    key = _normalize_dataset_key(dataset_name)
    if key not in PREPROCESS_SPECS:
        return

    spec = PREPROCESS_SPECS[key]
    if images.dim() != 4:
        raise RuntimeError(f"[{stage}] Expected NCHW tensor for {dataset_name}, got shape {tuple(images.shape)}")

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
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_len, generator=gen).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]
    return train_idx, val_idx, test_idx


# -----------------------------
# Data Setups
# -----------------------------
def setup_MNIST(batch_size: int):
    global train_loader, val_loader, test_loader

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resized to survive VGG pools
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset_full = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset_full = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_dataset_full, test_dataset_full])

    train_idx, val_idx, test_idx = _deterministic_split_indices(len(full_dataset), train_frac=0.8, val_frac=0.1, seed=42)
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    validate_loader_preprocessing(train_loader, "MNIST", stage="training")
    validate_loader_preprocessing(test_loader, "MNIST", stage="training")


def setup_CIFAR10(batch_size: int = 64):
    global train_loader, val_loader, test_loader

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_base_eval = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_eval)
    test_base_eval = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_eval)
    full_base_eval = ConcatDataset([train_base_eval, test_base_eval])

    total_len = len(full_base_eval)
    train_idx, val_idx, test_idx = _deterministic_split_indices(total_len, train_frac=0.8, val_frac=0.1, seed=42)

    train_base_tf = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    test_base_tf = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_train)
    full_base_tf = ConcatDataset([train_base_tf, test_base_tf])

    train_dataset = Subset(full_base_tf, train_idx)
    val_dataset = Subset(full_base_eval, val_idx)
    test_dataset = Subset(full_base_eval, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    validate_loader_preprocessing(train_loader, "CIFAR10", stage="training")
    validate_loader_preprocessing(test_loader, "CIFAR10", stage="training")


def setup_OCTMNIST(batch_size: int = 64):
    global train_loader, val_loader, test_loader

    os.makedirs(DATA_OCTMNIST_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = OCTMNIST(root=DATA_OCTMNIST_DIR, split="train", download=True, transform=transform)
    val_dataset = OCTMNIST(root=DATA_OCTMNIST_DIR, split="val", download=True, transform=transform)
    test_dataset = OCTMNIST(root=DATA_OCTMNIST_DIR, split="test", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    validate_loader_preprocessing(train_loader, "OCTMNIST", stage="training")
    validate_loader_preprocessing(test_loader, "OCTMNIST", stage="training")


def setup_BloodMNIST(batch_size: int = 64):
    global train_loader, val_loader, test_loader

    os.makedirs(DATA_BLOODMNIST_DIR, exist_ok=True)

    mean = (0.76, 0.53, 0.69)
    std = (0.14, 0.16, 0.11)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = BloodMNIST(root=DATA_BLOODMNIST_DIR, split="train", download=True, transform=transform, as_rgb=True)
    val_dataset = BloodMNIST(root=DATA_BLOODMNIST_DIR, split="val", download=True, transform=transform, as_rgb=True)
    test_dataset = BloodMNIST(root=DATA_BLOODMNIST_DIR, split="test", download=True, transform=transform, as_rgb=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    validate_loader_preprocessing(train_loader, "BloodMNIST", stage="training")
    validate_loader_preprocessing(test_loader, "BloodMNIST", stage="training")


def setup_OrganAMNIST(batch_size: int = 64):
    global train_loader, val_loader, test_loader

    os.makedirs(DATA_ORGANAMNIST_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = OrganAMNIST(root=DATA_ORGANAMNIST_DIR, split="train", download=True, transform=transform)
    val_dataset = OrganAMNIST(root=DATA_ORGANAMNIST_DIR, split="val", download=True, transform=transform)
    test_dataset = OrganAMNIST(root=DATA_ORGANAMNIST_DIR, split="test", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    validate_loader_preprocessing(train_loader, "OrganAMNIST", stage="training")
    validate_loader_preprocessing(test_loader, "OrganAMNIST", stage="training")

def setup_PneumoniaMNIST(batch_size: int = 64):
    global train_loader, val_loader, test_loader

    os.makedirs(DATA_PNEUMONIAMNIST_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = PneumoniaMNIST(root=DATA_PNEUMONIAMNIST_DIR, split="train", download=True, transform=transform)
    val_dataset = PneumoniaMNIST(root=DATA_PNEUMONIAMNIST_DIR, split="val", download=True, transform=transform)
    test_dataset = PneumoniaMNIST(root=DATA_PNEUMONIAMNIST_DIR, split="test", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    validate_loader_preprocessing(train_loader, "PneumoniaMNIST", stage="training")
    validate_loader_preprocessing(test_loader, "PneumoniaMNIST", stage="training")


def _compute_class_weights_from_subset(subset: Subset, num_classes: int):
    dataset = subset.dataset
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
            base_targets = datasets_list[ds_idx].targets
            targets.append(base_targets[sample_offset])
    counts = Counter(targets)
    total = len(targets)
    weights = []
    for c in range(num_classes):
        class_count = counts.get(c, 1)
        weights.append(total / (num_classes * class_count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def setup_Brain_MRI(batch_size: int = 64):
    global train_loader, val_loader, test_loader

    train_root = os.path.join(DATA_BRAIN_MRI_DIR, "Training")
    test_root = os.path.join(DATA_BRAIN_MRI_DIR, "Testing")

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),  # Resized to survive VGG pools
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_train_base = datasets.ImageFolder(root=train_root, transform=eval_transform)
    full_test_base = datasets.ImageFolder(root=test_root, transform=eval_transform)
    full_base_eval = ConcatDataset([full_train_base, full_test_base])

    total_len = len(full_base_eval)
    train_idx, val_idx, test_idx = _deterministic_split_indices(total_len, train_frac=0.8, val_frac=0.1, seed=42)

    full_train_tf = datasets.ImageFolder(root=train_root, transform=train_transform)
    full_test_tf = datasets.ImageFolder(root=test_root, transform=train_transform)
    full_base_tf = ConcatDataset([full_train_tf, full_test_tf])

    train_dataset = Subset(full_base_tf, train_idx)
    val_dataset = Subset(full_base_eval, val_idx)
    test_dataset = Subset(full_base_eval, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    validate_loader_preprocessing(train_loader, "Brain_MRI", stage="training")
    validate_loader_preprocessing(test_loader, "Brain_MRI", stage="training")

    return train_dataset



# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(model, dataloader, criterion, is_multilabel=False, is_medmnist=False):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_outputs = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            if not is_multilabel and not is_medmnist:
                labels = _normalize_classification_labels(labels)
            elif is_medmnist:
                if labels.dim() == 2 and labels.size(1) == 1:
                    labels = labels.squeeze(-1)
                labels = labels.long()

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            if is_multilabel:
                all_targets.append(labels.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            elif is_medmnist:
                all_targets.append(labels.cpu().numpy())
                all_outputs.append(torch.softmax(outputs, dim=1).cpu().numpy())
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    avg_loss = total_loss / len(dataloader.dataset)

    if is_multilabel:
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)
        metric_score = roc_auc_score(all_targets, all_outputs, average="macro")
    elif is_medmnist:
        all_targets = np.concatenate(all_targets)
        all_outputs = np.vstack(all_outputs)
        if all_outputs.shape[1] == 2:
            auc = roc_auc_score(all_targets, all_outputs[:, 1])
        else:
            auc = roc_auc_score(all_targets, all_outputs, multi_class="ovr", average="macro")
        acc = 100.0 * correct / max(total, 1)
        metric_score = (auc, acc)
    else:
        metric_score = 100.0 * correct / max(total, 1)

    return avg_loss, metric_score


# -----------------------------
# Training loop
# -----------------------------
def main(args: argparse.Namespace):
    global model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader

    best_val_metric = 0.0
    num_epochs = 1

    if args.data_dir == DATA_MNIST_DIR:
        best_model_path = f"best_vgg19_{args.activation}_mnist.pth"
        setup_MNIST(args.batch_size)
        model = VGG19(num_classes=10, in_channels=args.in_channels, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = False

    elif args.data_dir == DATA_CIFAR10_DIR:
        best_model_path = f"best_vgg19_{args.activation}_cifar10.pth"
        num_epochs = 50
        setup_CIFAR10(args.batch_size)
        model = VGG19(num_classes=10, in_channels=3, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = False

    elif args.data_dir == DATA_BRAIN_MRI_DIR:
        best_model_path = f"best_vgg19_{args.activation}_brain_mri.pth"
        num_epochs = 30
        train_dataset = setup_Brain_MRI(args.batch_size)
        model = VGG19(num_classes=4, in_channels=args.in_channels, activation=args.activation).to(device)
        class_weights = _compute_class_weights_from_subset(train_dataset, num_classes=4)
        print(f"Using class weights: {class_weights.detach().cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        is_multilabel = False
        is_medmnist = False

    elif args.data_dir == DATA_OCTMNIST_DIR:
        best_model_path = f"best_vgg19_{args.activation}_octmnist.pth"
        num_epochs = 20
        setup_OCTMNIST(args.batch_size)
        model = VGG19(num_classes=4, in_channels=1, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = True

    elif args.data_dir == DATA_BLOODMNIST_DIR:
        best_model_path = f"best_vgg19_{args.activation}_bloodmnist.pth"
        num_epochs = 20
        setup_BloodMNIST(args.batch_size)
        model = VGG19(num_classes=8, in_channels=3, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = True

    elif args.data_dir == DATA_ORGANAMNIST_DIR:
        best_model_path = f"best_vgg19_{args.activation}_organamnist.pth"
        num_epochs = 20
        setup_OrganAMNIST(args.batch_size)
        model = VGG19(num_classes=11, in_channels=1, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = True

    elif args.data_dir == DATA_PNEUMONIAMNIST_DIR:
        best_model_path = f"best_vgg19_{args.activation}_pneumoniamnist.pth"
        num_epochs = 20
        setup_PneumoniaMNIST(args.batch_size)
        model = VGG19(num_classes=2, in_channels=1, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False
        is_medmnist = True

    else:
        print(f"Training using default data directory: {DATA_MNIST_DIR}")
        best_model_path = f"best_vgg19_{args.activation}_mnist.pth"
        setup_MNIST(args.batch_size)
        model = VGG19(num_classes=10, in_channels=args.in_channels, activation=args.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
        is_multilabel = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        all_targets = []
        all_outputs = []
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if not is_multilabel:
                labels = _normalize_classification_labels(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if is_multilabel:
                all_targets.append(labels.detach().cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).detach().cpu().numpy())
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        if is_multilabel:
            all_targets = np.vstack(all_targets)
            all_outputs = np.vstack(all_outputs)
            train_metric = roc_auc_score(all_targets, all_outputs, average="macro")
            metric_name = "Mean AUROC"
        else:
            train_metric = 100.0 * correct / total
            metric_name = "Acc"

        val_loss, val_metric = evaluate(model, val_loader, criterion, is_multilabel=is_multilabel, is_medmnist=is_medmnist)

        if scheduler is not None:
            scheduler.step(val_loss)

        if is_medmnist:
            val_auc, val_acc = val_metric
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val ACC: {val_acc:.2f}%")
            if val_auc > best_val_metric:
                best_val_metric = val_auc
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")
        elif is_multilabel:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Train AUROC: {train_metric:.4f} | Val Loss: {val_loss:.4f}, Val AUROC: {val_metric:.4f}")
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_metric:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_metric:.2f}%")
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")

    # -----------------------------
    # Load best model and test
    # -----------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_loss, test_metric = evaluate(model, test_loader, criterion, is_multilabel=is_multilabel, is_medmnist=is_medmnist)

    if is_medmnist:
        test_auc, test_acc = test_metric
        print(f"Best Validation AUC: {best_val_metric:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.2f}%")
    elif is_multilabel:
        print(f"Best Validation {metric_name}: {best_val_metric:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test {metric_name}: {test_metric:.4f}")
    else:
        print(f"Best Validation {metric_name}: {best_val_metric:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test {metric_name}: {test_metric:.2f}%")


def datasetDownloader(dataset_name: str):
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    if dataset_name == "MNIST":
        if not os.path.exists(DATA_MNIST_DIR):
            print("Downloading MNIST dataset...")
            datasets.MNIST(root=DATA_ROOT, train=True, download=True)
            datasets.MNIST(root=DATA_ROOT, train=False, download=True)

    if dataset_name == "Brain_MRI":
        if not os.path.exists(DATA_BRAIN_MRI_DIR):
            print("Downloading Brain_MRI dataset from Kaggle...")
            kagglehub.dataset_download(
                "masoudnickparvar/brain-tumor-mri-dataset",
                output_dir=DATA_BRAIN_MRI_DIR,
            )

    if dataset_name == "CIFAR10":
        cifar_root = DATA_ROOT
        cifar_folder = os.path.join(cifar_root, "cifar-10-batches-py")
        if not os.path.exists(cifar_folder):
            print("Downloading CIFAR-10 dataset...")
            datasets.CIFAR10(root=cifar_root, train=True, download=True)
            datasets.CIFAR10(root=cifar_root, train=False, download=True)

    if dataset_name == "OCTMNIST":
        os.makedirs(DATA_OCTMNIST_DIR, exist_ok=True)
        print("Downloading OCTMNIST dataset...")
        OCTMNIST(root=DATA_OCTMNIST_DIR, split="train", download=True)
        OCTMNIST(root=DATA_OCTMNIST_DIR, split="val", download=True)
        OCTMNIST(root=DATA_OCTMNIST_DIR, split="test", download=True)

    if dataset_name == "BloodMNIST":
        os.makedirs(DATA_BLOODMNIST_DIR, exist_ok=True)
        print("Downloading BloodMNIST dataset...")
        BloodMNIST(root=DATA_BLOODMNIST_DIR, split="train", download=True, as_rgb=True)
        BloodMNIST(root=DATA_BLOODMNIST_DIR, split="val", download=True, as_rgb=True)
        BloodMNIST(root=DATA_BLOODMNIST_DIR, split="test", download=True, as_rgb=True)

    if dataset_name == "OrganAMNIST":
        os.makedirs(DATA_ORGANAMNIST_DIR, exist_ok=True)
        print("Downloading OrganAMNIST dataset...")
        OrganAMNIST(root=DATA_ORGANAMNIST_DIR, split="train", download=True)
        OrganAMNIST(root=DATA_ORGANAMNIST_DIR, split="val", download=True)
        OrganAMNIST(root=DATA_ORGANAMNIST_DIR, split="test", download=True)

    if dataset_name == "PneumoniaMNIST":
        os.makedirs(DATA_PNEUMONIAMNIST_DIR, exist_ok=True)
        print("Downloading PneumoniaMNIST dataset...")
        PneumoniaMNIST(root=DATA_PNEUMONIAMNIST_DIR, split="train", download=True)
        PneumoniaMNIST(root=DATA_PNEUMONIAMNIST_DIR, split="val", download=True)
        PneumoniaMNIST(root=DATA_PNEUMONIAMNIST_DIR, split="test", download=True)

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
        default=1e-4, # Lower default learning rate for VGG compared to ResNet
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="MNIST",
        help="Training data to use: MNIST, CIFAR10, Brain_MRI, OCTMNIST, BloodMNIST, OrganAMNIST, PneumoniaMNIST",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function to use",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Number of input channels for the model (e.g., 1 for grayscale, 3 for RGB)",
    )

    args = parser.parse_args()

    train_data_key = args.train_data.strip().upper().replace("_", "-").replace(" ", "-")

    if train_data_key == "MNIST":
        args.data_dir = DATA_MNIST_DIR
        datasetDownloader("MNIST")
    elif train_data_key == "BRAIN_MRI":
        args.data_dir = DATA_BRAIN_MRI_DIR
        datasetDownloader("Brain_MRI")
    elif train_data_key in ("CIFR10", "CIFAR10"):
        args.data_dir = DATA_CIFAR10_DIR
        datasetDownloader("CIFAR10")

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