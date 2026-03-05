"""
Generate segmentation masks from xBD dataset labels.

Reads JSON labels from <data-dir>/labels/
Reads images from <data-dir>/images/
Writes masks to <data-dir>/masks/

Mask pixel values:
0 = background
1 = undamaged building
2 = damaged building
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


BACKGROUND = 0
UNDAMAGED = 1
DAMAGED = 2


DAMAGE_TO_CLASS = {
    "no-damage": UNDAMAGED,
    "minor-damage": DAMAGED,
    "major-damage": DAMAGED,
    "destroyed": DAMAGED,
    "un-classified": DAMAGED,
}


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------
# Parse WKT polygon
# ---------------------------------------------------

def parse_wkt_polygon(wkt):
    """Parse WKT POLYGON ((x y, x y, ...)) into Nx2 float array. Handles newlines."""
    if not wkt:
        return None
    # DOTALL so . matches newlines (WKT may be multi-line)
    match = re.search(r"\(\s*\(\s*(.*?)\s*\)\s*\)", wkt, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    coords_text = match.group(1)
    points = []
    for pair in coords_text.split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
            points.append([x, y])
        except ValueError:
            continue
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.float32)


# ---------------------------------------------------
# Extract polygon from feature
# ---------------------------------------------------

def get_polygon(feature, h, w):
    """Return polygon as (N, 2) int32 array, clipped to image bounds [0,w) x [0,h)."""
    wkt = feature.get("wkt")
    if not wkt:
        return None
    pts = parse_wkt_polygon(wkt)
    if pts is None or len(pts) < 3:
        return None
    pts = np.round(pts).astype(np.int32)
    # Clip to image bounds so fillPoly draws correctly
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


# ---------------------------------------------------
# Map damage type
# ---------------------------------------------------

def get_damage_class(feature):

    props = feature.get("properties", {})

    subtype = props.get("subtype", "un-classified")

    subtype = subtype.lower()

    return DAMAGE_TO_CLASS.get(subtype, DAMAGED)


# ---------------------------------------------------
# Find matching image
# ---------------------------------------------------

def find_image(label_path, images_dir):

    stem = label_path.stem

    for ext in IMAGE_EXTENSIONS:

        p = images_dir / (stem + ext)

        if p.exists():
            return p

    return None


# ---------------------------------------------------
# Rasterize one label
# ---------------------------------------------------

def rasterize(label_path, images_dir, masks_dir):

    img_path = find_image(label_path, images_dir)

    if img_path is None:
        return False

    img = cv2.imread(str(img_path))

    if img is None:
        return False

    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    with open(label_path) as f:
        data = json.load(f)

    features = data["features"]["xy"]

    for feat in features:
        pts = get_polygon(feat, h, w)
        if pts is None:
            continue
        cls = get_damage_class(feat)
        # fillPoly expects list of polygons, each polygon (N, 2) int32
        cv2.fillPoly(mask, [pts], cls)

    masks_dir.mkdir(parents=True, exist_ok=True)

    out_path = masks_dir / (label_path.stem + ".png")

    cv2.imwrite(str(out_path), mask)

    return True


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data_raw",
        type=Path
    )

    args = parser.parse_args()

    data_dir = args.data_dir

    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    label_files = sorted(labels_dir.glob("*.json"))

    written = 0

    for label in label_files:

        ok = rasterize(label, images_dir, masks_dir)

        if ok:
            written += 1

    print("Masks written:", written)


if __name__ == "__main__":
    main()