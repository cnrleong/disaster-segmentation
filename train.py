from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import XBDDataset
from model import create_unet_resnet18
from utils import segmentation_metrics, set_seed


def get_transforms(img_size: Tuple[int, int] = (512, 512), is_train: bool = True):
    """
    Albumentations augmentation pipeline.

    Keep this relatively simple and xBD-agnostic; you can later extend
    with more aggressive augmentations or dataset-specific choices.
    """
    if is_train:
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    img_size: Tuple[int, int],
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_transforms = get_transforms(img_size=img_size, is_train=True)
    val_transforms = get_transforms(img_size=img_size, is_train=False)

    train_images = data_dir / "train" / "images"
    train_masks = data_dir / "train" / "masks"
    val_images = data_dir / "val" / "images"
    val_masks = data_dir / "val" / "masks"

    train_ds = XBDDataset(
        images_dir=train_images,
        masks_dir=train_masks,
        transforms=train_transforms,
    )
    val_ds = XBDDataset(
        images_dir=val_images,
        masks_dir=val_masks,
        transforms=val_transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

        # Move logits to CPU for metrics
        batch_metrics = segmentation_metrics(
            logits=logits.detach().cpu(),
            targets=masks.detach().cpu(),
            num_classes=num_classes,
        )

        # Store running mean IoU / Dice only for logging
        pbar.set_postfix(
            {
                "loss": f"{running_loss / total_batches:.4f}",
                "miou": f"{batch_metrics['miou']:.3f}",
                "dice": f"{batch_metrics['dice']:.3f}",
            }
        )

    epoch_loss = running_loss / max(total_batches, 1)
    # Metrics are aggregated per-batch above; for simplicity we return only the loss here.
    return {"loss": epoch_loss}


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    total_batches = 0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item()
            total_batches += 1

            all_logits.append(logits.cpu())
            all_targets.append(masks.cpu())

    epoch_loss = running_loss / max(total_batches, 1)

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = segmentation_metrics(
        logits=logits_cat,
        targets=targets_cat,
        num_classes=num_classes,
    )
    metrics["loss"] = epoch_loss
    return metrics


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train U-Net (ResNet18 encoder) for disaster damage segmentation."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root data directory containing train/ and val/ subdirs (relative or absolute).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet18",
        help="Encoder backbone for U-Net.",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Encoder pretrained weights (e.g. imagenet, None).",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=3,
        help="Number of input channels (3 for RGB, 6 for pre+post).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of segmentation classes.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="Resize images to (H, W).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (relative or absolute).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    num_workers = args.num_workers
    if device.type == "cpu" and num_workers > 0:
        num_workers = 0  # avoid multiprocessing overhead when no GPU
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
        img_size=(args.img_size[0], args.img_size[1]),
        pin_memory=(device.type == "cuda"),
    )

    model = create_unet_resnet18(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights if args.encoder_weights != "None" else None,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_miou = -1.0
    best_ckpt_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
        )
        print(f"Train loss: {train_stats['loss']:.4f}")

        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
        )
        print(
            f"Val loss: {val_stats['loss']:.4f} | "
            f"IoU: {val_stats['miou']:.4f} | "
            f"Dice: {val_stats['dice']:.4f}"
        )

        # Save best checkpoint by mIoU
        if val_stats["miou"] > best_miou:
            best_miou = val_stats["miou"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_stats,
                path=best_ckpt_path,
            )
            print(f"Saved new best model to {best_ckpt_path}")

    print("Training finished.")
    print(f"Best validation mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()

