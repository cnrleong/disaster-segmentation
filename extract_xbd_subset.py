"""
Extract a subset of xBD (post-disaster images + labels) into a flat layout.

Supports two xBD layouts:
  A) <xbd-root>/train/images/ and train/labels/ (flat)
  B) <xbd-root>/<disaster>/images/ (or images/post_disaster/) and .../labels/
See DATA_LAYOUT.md for details.
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def collect_pairs_flat(images_dir: Path, labels_dir: Path, disaster: str) -> list[tuple[Path, Path]]:
    """Layout A: train/images and train/labels with matching filenames."""
    pairs = []
    for img_path in images_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if "_post_disaster" not in img_path.name and "_post-disaster" not in img_path.name:
            continue
        if disaster not in img_path.name:
            continue
        label_path = labels_dir / (img_path.stem + ".json")
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def collect_pairs_per_disaster(xbd_root: Path, disaster: str) -> list[tuple[Path, Path]]:
    """Layout B: xbd_root/<disaster>/images/ (or images/post_disaster/) and labels/."""
    disaster_dir = xbd_root / disaster
    labels_dir = disaster_dir / "labels"
    if not labels_dir.is_dir():
        return []
    images_base = disaster_dir / "images"
    if not images_base.is_dir():
        return []
    # Prefer post_disaster subdir if present
    images_dirs = [images_base / "post_disaster", images_base / "post-disaster", images_base]
    pairs = []
    for images_dir in images_dirs:
        if not images_dir.is_dir():
            continue
        for img_path in images_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / (img_path.stem + ".json")
            if label_path.exists():
                pairs.append((img_path, label_path))
        if pairs:
            break
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract xBD subset into flat images/ and labels/ (see DATA_LAYOUT.md)."
    )
    parser.add_argument("--xbd-root", type=str, required=True)
    parser.add_argument("--disaster", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default="data_raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    xbd_root = Path(args.xbd_root).resolve()

    # Try Layout A: train/images and train/labels
    images_dir = xbd_root / "train" / "images"
    labels_dir = xbd_root / "train" / "labels"
    if images_dir.exists() and labels_dir.exists():
        pairs = collect_pairs_flat(images_dir, labels_dir, args.disaster)
    else:
        pairs = []

    # Fallback: Layout B per-disaster folders
    if not pairs:
        pairs = collect_pairs_per_disaster(xbd_root, args.disaster)

    if not pairs:
        raise SystemExit(
            f"No matching samples for disaster '{args.disaster}'.\n"
            f"Checked: {xbd_root}/train/images and train/labels, "
            f"and {xbd_root}/{args.disaster}/images (and labels/)."
        )

    if args.num_samples > len(pairs):
        print(f"Only {len(pairs)} available. Using all.")
        args.num_samples = len(pairs)

    chosen = random.sample(pairs, args.num_samples)

    out_images = Path(args.out_dir) / "images"
    out_labels = Path(args.out_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in chosen:
        shutil.copy2(img_path, out_images / img_path.name)
        shutil.copy2(label_path, out_labels / label_path.name)

    print(f"Copied {len(chosen)} samples to {args.out_dir}")


if __name__ == "__main__":
    main()