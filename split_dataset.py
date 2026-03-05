"""
Split data/images and data/masks into train/val (default 80/20) with aligned pairs.
Creates data/train/images, data/train/masks, data/val/images, data/val/masks
and moves files accordingly.
"""

import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split image-mask pairs into train/val.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory containing images/ and masks/ (relative or absolute).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of pairs for training (default 0.8).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise SystemExit(f"Masks directory not found: {masks_dir}")

    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in suffixes:
            continue
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"Warning: no mask for {img_path.name}, skipping.")
            continue
        pairs.append((img_path, mask_path))

    if not pairs:
        raise SystemExit("No image-mask pairs found.")

    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(args.train_ratio * n)
    n_train = max(1, min(n_train, n - 1))  # ensure both splits non-empty
    n_val = n - n_train
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    train_images = data_root / "train" / "images"
    train_masks = data_root / "train" / "masks"
    val_images = data_root / "val" / "images"
    val_masks = data_root / "val" / "masks"
    for d in (train_images, train_masks, val_images, val_masks):
        d.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in train_pairs:
        shutil.move(str(img_path), str(train_images / img_path.name))
        shutil.move(str(mask_path), str(train_masks / mask_path.name))

    for img_path, mask_path in val_pairs:
        shutil.move(str(img_path), str(val_images / img_path.name))
        shutil.move(str(mask_path), str(val_masks / mask_path.name))

    print(f"Train samples: {n_train}")
    print(f"Val samples:   {n_val}")
    print("Done.")


if __name__ == "__main__":
    main()
