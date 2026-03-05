"""
Generate segmentation masks from xBD JSON labels.

Reads data_raw/labels/*.json and writes rasterized masks to data_raw/masks/
with pixel values: 0 = background, 1 = undamaged building, 2 = damaged building.

Expects data_raw/images/ and data_raw/labels/ (same basenames). Image dimensions
are taken from the corresponding image file.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# xBD damage subtypes -> our class index
UNDAMAGED = 1   # no-damage
DAMAGED = 2     # minor, major, destroyed, or un-classified
BACKGROUND = 0

DAMAGE_TO_CLASS = {
    "no-damage": UNDAMAGED,
    "minor-damage": DAMAGED,
    "major-damage": DAMAGED,
    "destroyed": DAMAGED,
    "un-classified": DAMAGED,
}


def get_polygon_coordinates(feature: dict, height: int, width: int):
    """
    Extract polygon from xBD feature and return Nx2 numpy array in pixel coords.
    Handles both pixel coordinates and normalized 0-1 coordinates.
    """
    geom = feature.get("geometry")
    if not geom:
        return None
    coords = geom.get("coordinates")
    if not coords:
        return None
    # GeoJSON: coordinates can be [lng, lat] or nested for polygon
    ring = coords[0] if isinstance(coords[0][0], (list, tuple)) else coords
    pts = np.array(ring, dtype=np.float64)
    if pts.size == 0:
        return None
    # If values are in [0, 1], scale to image size
    if pts.max() <= 1.0 and pts.min() >= 0:
        pts[:, 0] *= width
        pts[:, 1] *= height
    pts = np.round(pts).astype(np.int32)
    return pts


def get_damage_class(feature: dict) -> int:
    """Map xBD damage subtype to our class (1 = undamaged, 2 = damaged)."""
    props = feature.get("properties") or {}
    subtype = (props.get("subtype") or props.get("damage") or "un-classified").strip().lower()
    return DAMAGE_TO_CLASS.get(subtype, DAMAGED)


def rasterize_label_file(
    label_path: Path,
    images_dir: Path,
    masks_dir: Path,
    image_suffixes: tuple = (".png", ".jpg", ".jpeg", ".tif"),
) -> bool:
    """
    Load one JSON, rasterize all building polygons, save mask PNG.
    Returns True on success.
    """
    stem = label_path.stem
    # Find corresponding image to get dimensions
    img_path = None
    for ext in image_suffixes:
        p = images_dir / (stem + ext)
        if p.exists():
            img_path = p
            break
    if not img_path:
        return False
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    height, width = img.shape[:2]

    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)

    if isinstance(data, list):
        features = data
    else:
        features = data.get("features") or []

    for feat in features:
        if not isinstance(feat, dict):
            continue
        pts = get_polygon_coordinates(feat, height, width)
        if pts is None:
            continue
        cls = get_damage_class(feat)
        cv2.fillPoly(mask, [pts], cls)

    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path = masks_dir / (stem + ".png")
    cv2.imwrite(str(out_path), mask)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PNG masks from xBD JSON labels (0=bg, 1=undamaged, 2=damaged)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_raw",
        help="Directory containing images/ and labels/ (masks/ will be created; relative or absolute).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    masks_dir = data_dir / "masks"

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        raise SystemExit(f"No JSON files in {labels_dir}")

    ok = 0
    for label_path in label_files:
        if rasterize_label_file(label_path, images_dir, masks_dir):
            ok += 1
        else:
            print(f"Warning: skipped (no image or invalid JSON): {label_path.name}")

    print(f"Wrote {ok}/{len(label_files)} masks to {masks_dir}")


if __name__ == "__main__":
    main()
