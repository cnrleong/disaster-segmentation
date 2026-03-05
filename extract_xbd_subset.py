import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract xBD subset (flat structure).")
    parser.add_argument("--xbd-root", type=str, required=True)
    parser.add_argument("--disaster", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default="data_raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    xbd_root = Path(args.xbd_root).resolve()
    images_dir = xbd_root / "train" / "images"
    labels_dir = xbd_root / "train" / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit("Could not find train/images or train/labels folders.")

    # Collect post-disaster images matching disaster name
    pairs = []
    for img_path in images_dir.iterdir():
        if (
            img_path.is_file()
            and "_post_disaster" in img_path.name
            and args.disaster in img_path.name
        ):
            label_path = labels_dir / (img_path.stem + ".json")
            if label_path.exists():
                pairs.append((img_path, label_path))

    if not pairs:
        raise SystemExit(f"No matching samples found for disaster '{args.disaster}'.")

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