import argparse
import os
import cv2
from torch import nn, relu
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
import torchvision.transforms.functional as TF
import kagglehub

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

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


# -----------------------------
# Data
# -----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, pairs, image_size: int = 256):
        # pairs is a list of (image_path, mask_path)
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize to a fixed size compatible with the UNet
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Binarize mask to {0,1}
        mask = (mask > 0.5).float()

        return image, mask


def setup_data(
    train_data: str, batch_size: int = 64, image_size: int = 256, num_workers: int = 4
):
    global train_loader, val_loader, test_loader

    if train_data == "Skin-Lesion":
        image_dir = "./data/Skin-Lesion/images"
        mask_dir = "./data/Skin-Lesion/masks"
    elif train_data == "Flood":
        image_dir = "./data/Flood/Image"
        mask_dir = "./data/Flood/Mask"
    else:
        raise ValueError(f"Unknown training data: {train_data}")

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_id_to_path = {}
    for fname in image_files:
        image_id, _ = os.path.splitext(fname)
        image_id_to_path[image_id] = os.path.join(image_dir, fname)

    mask_files = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    pairs = []
    for mname in mask_files:
        base, _ = os.path.splitext(mname)
        image_id = base.split("_segmentation")[0]
        if image_id in image_id_to_path:
            img_path = image_id_to_path[image_id]
            mask_path = os.path.join(mask_dir, mname)
            pairs.append((img_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No matching image/mask pairs found in {train_data} dataset.")

    # Disjoint split: 80% train, 10% val, 10% test
    n = len(pairs)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=generator).tolist()

    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train_pairs = [pairs[i] for i in indices[:n_train]]
    val_pairs = [pairs[i] for i in indices[n_train:n_train + n_val]]
    test_pairs = [pairs[i] for i in indices[n_train + n_val:]]

    train_dataset = SegmentationDataset(train_pairs, image_size=image_size)
    val_dataset = SegmentationDataset(val_pairs, image_size=image_size)
    test_dataset = SegmentationDataset(test_pairs, image_size=image_size)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(model, dataloader, criterion):
    """Evaluate a segmentation model on a dataloader.

    Assumes binary segmentation with masks in {0,1} and a single-channel
    output (logits) from the model when using BCEWithLogitsLoss.

    Returns:
        dice (float), iou (float), accuracy (float), f1_score (float)
    """

    model.eval()

    total_loss = 0.0
    total_pixels = 0
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)

            # Convert logits to probabilities then to binary predictions
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:
                # Fallback for multi-class: take argmax over channels and treat
                # the foreground (class 1) as positive.
                preds = torch.argmax(outputs, dim=1, keepdim=True).float()

            preds = preds.view(-1)
            masks_flat = masks.view(-1)

            tp += (preds * masks_flat).sum().item()
            tn += ((1 - preds) * (1 - masks_flat)).sum().item()
            fp += (preds * (1 - masks_flat)).sum().item()
            fn += ((1 - preds) * masks_flat).sum().item()

            total_pixels += masks_flat.numel()

    eps = 1e-7

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    accuracy = (tp + tn + eps) / (total_pixels + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1_score = (2.0 * precision * recall + eps) / (precision + recall + eps)

    return dice, iou, accuracy, f1_score


# -----------------------------
# Training loop with best model saving
# -----------------------------
def main(args: argparse.Namespace):
    global model, criterion, optimizer, scheduler

    best_val_dice = 0.0
    num_epochs = 50

    if args.data_dir == "./data/Skin-Lesion":
        best_model_path = "best_unet5_skin_lesion.pth"
        setup_data(
            train_data="Skin-Lesion",
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        model = UNet(n_class=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    elif args.data_dir == "./data/Flood":
        best_model_path = "best_unet5_flood.pth"
        setup_data(
            train_data="Flood",
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        model = UNet(n_class=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

    else:
        raise ValueError(f"Unknown dataset: {args.data_dir}")

    for epoch in range(num_epochs):
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        val_dice, val_iou, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )

        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val Dice: {best_val_dice:.4f}")

    # -----------------------------
    # Load best model and test
    # -----------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_dice, test_iou, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    print(
        f"Test Dice: {test_dice:.4f}, Test IoU: {test_iou:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}"
    )


def dataset_downloader(dataset_name: str):
    if not os.path.exists("./data"):
        os.makedirs("./data")

    if dataset_name == "Skin-Lesion":
        if not os.path.exists("./data/Skin-Lesion"):
            print("Downloading Skin-Lesion dataset from Kaggle...")
            kagglehub.dataset_download(
                "surajghuwalewala/ham1000-segmentation-and-classification",
                output_dir="./data/Skin-Lesion",
            )
            print("Download complete.")
    elif dataset_name == "Flood":
        if not os.path.exists("./data/Flood"):
            print("Downloading Flood dataset from Kaggle...")
            kagglehub.dataset_download(
                "faizalkarim/flood-area-segmentation", output_dir="./data/Flood"
            )
            print("Download complete.")
    else:
        print(f"Unknown dataset: {dataset_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Input image size (images will be resized to image_size x image_size)",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="Skin-Lesion",
        help="Training data to use",
    )

    args = parser.parse_args()

    if args.train_data == "Skin-Lesion":
        args.data_dir = "./data/Skin-Lesion"
        dataset_downloader("Skin-Lesion")
    elif args.train_data == "Flood":
        args.data_dir = "./data/Flood"
        dataset_downloader("Flood")
    else:
        print(
            "Invalid training data specified. Using default data directory: ./data/Skin-Lesion/"
        )
        args.data_dir = "./data/Skin-Lesion/"
    main(args)
