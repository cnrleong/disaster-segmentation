from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import XBDDataset
from model import create_unet_resnet18
from utils import segmentation_metrics, set_seed


# ----------------------------
# Dice Loss
# ----------------------------

def dice_loss(logits, targets, num_classes):
    probs = torch.softmax(logits, dim=1)

    targets_one_hot = F.one_hot(targets, num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    intersection = (probs * targets_one_hot).sum((2, 3))
    union = probs.sum((2, 3)) + targets_one_hot.sum((2, 3))

    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()


# ----------------------------
# Transforms
# ----------------------------

def get_transforms(img_size: Tuple[int, int], is_train: bool):
    if is_train:
        return A.Compose([
            A.Resize(*img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(*img_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])


# ----------------------------
# Dataloaders
# ----------------------------

def create_dataloaders(data_dir, batch_size, num_workers, img_size):

    train_ds = XBDDataset(
        data_dir/"train"/"images",
        data_dir/"train"/"masks",
        transforms=get_transforms(img_size, True),
    )

    val_ds = XBDDataset(
        data_dir/"val"/"images",
        data_dir/"val"/"masks",
        transforms=get_transforms(img_size, False),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ----------------------------
# Training
# ----------------------------

def train_one_epoch(model, loader, optimizer, ce_loss, device, num_classes):

    model.train()

    running_loss = 0

    pbar = tqdm(loader)

    for batch in pbar:

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = ce_loss(logits, masks) + dice_loss(logits, masks, num_classes)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        metrics = segmentation_metrics(
            logits.detach().cpu(),
            masks.detach().cpu(),
            num_classes,
        )

        pbar.set_postfix(
            loss=running_loss/len(loader),
            miou=metrics["miou"],
        )

    return running_loss/len(loader)


# ----------------------------
# Validation
# ----------------------------

def validate_one_epoch(model, loader, ce_loss, device, num_classes):

    model.eval()

    all_logits = []
    all_targets = []

    loss_total = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)

            loss = ce_loss(logits, masks) + dice_loss(logits, masks, num_classes)

            loss_total += loss.item()

            all_logits.append(logits.cpu())
            all_targets.append(masks.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    metrics = segmentation_metrics(logits, targets, num_classes)

    metrics["loss"] = loss_total/len(loader)

    return metrics


# ----------------------------
# Main
# ----------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", nargs=2, type=int, default=[512,512])

    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)

    train_loader, val_loader = create_dataloaders(
        data_dir,
        args.batch_size,
        args.num_workers,
        args.img_size
    )

    model = create_unet_resnet18(num_classes=3)
    model.to(device)

    # improved class weights
    class_weights = torch.tensor([0.3, 1.5, 4.0], device=device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_iou = 0

    for epoch in range(args.epochs):

        print("\nEpoch", epoch+1)

        train_loss = train_one_epoch(model, train_loader, optimizer, ce_loss, device, 3)

        val_stats = validate_one_epoch(model, val_loader, ce_loss, device, 3)

        print(
            "Train loss:", train_loss,
            "Val IoU:", val_stats["miou"],
            "Val Dice:", val_stats["dice"]
        )

        if val_stats["miou"] > best_iou:

            best_iou = val_stats["miou"]

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metrics": val_stats,
                },
                "checkpoints/best_model.pt",
            )

            print("Saved new best model")


if __name__ == "__main__":
    main()