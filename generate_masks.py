"""
Generate segmentation masks from xBD dataset labels.

Reads JSON labels from <data-dir>/labels/ and images from <data-dir>/images/,
rasterizes building polygons into masks and saves to <data-dir>/masks/ with
pixel values: 0 = background, 1 = undamaged building, 2 = damaged building.

Usage:
  python generate_masks.py --data-dir data_raw

xBD JSON structure (from xView2):
- Building list: data["features"]["xy"] (each item = one building).
- Polygon: each feature has "wkt" (Well-Known Text, e.g. "POLYGON ((x y, x y, ...))")
  with coordinates already in image pixel space.
- Damage: feature["properties"]["subtype"] or "damage_grade" (0=no-damage, 1=minor, 2=major, 3=destroyed).
"""

import argparse
import json
import re
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

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _parse_wkt_polygon(wkt_str: str) -> np.ndarray | None:
    """
    Parse a WKT POLYGON string into an Nx2 array of (x, y) coordinates.
    xBD uses format: POLYGON ((x1 y1, x2 y2, ...)); coordinates are in pixel space.
    """
    if not wkt_str or not isinstance(wkt_str, str):
        return None
    wkt_str = wkt_str.strip()
    # Match POLYGON (( ... )) - exterior ring only
    m = re.match(r"POLYGON\s*\(\s*\(\s*(.+)\s*\)\s*\)", wkt_str, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    ring_str = m.group(1).strip()
    if not ring_str:
        return None
    points = []
    for pair in ring_str.split(","):
        pair = pair.strip().split()
        if len(pair) >= 2:
            try:
                x, y = float(pair[0]), float(pair[1])
                points.append([x, y])
            except ValueError:
                continue
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.float64)


def get_features_list(data) -> list:
    """
    Return the list of building features from xBD JSON.
    xBD: data["features"]["xy"]; GeoJSON-style: data["features"] (list) or data as list.
    """
    if isinstance(data, list):
        return data
    features = data.get("features")
    if features is None:
        return []
    if isinstance(features, list):
        return features
    if isinstance(features, dict) and "xy" in features:
        return features.get("xy") or []
    return []


def get_polygon_coordinates(feature: dict, height: int, width: int) -> np.ndarray | None:
    """
    Extract polygon from xBD feature and return Nx2 numpy array in pixel coords.
    Supports: (1) xBD "wkt" (POLYGON in pixel space), (2) GeoJSON "geometry"."coordinates".
    Normalized 0-1 coordinates are scaled to image size.
    """
    # xBD: polygon in WKT (pixel coordinates)
    wkt = feature.get("wkt")
    if wkt:
        pts = _parse_wkt_polygon(wkt)
        if pts is not None and pts.size >= 6:
            return np.round(pts).astype(np.int32)
        return None

    # GeoJSON: geometry.coordinates
    geom = feature.get("geometry")
    if not geom:
        return None
    coords = geom.get("coordinates")
    if not coords:
        return None
    ring = coords[0] if isinstance(coords[0][0], (list, tuple)) else coords
    pts = np.array(ring, dtype=np.float64)
    if pts.size == 0:
        return None
    if pts.ndim == 1:
        return None
    if pts.max() <= 1.0 and pts.min() >= 0:
        pts = pts.copy()
        pts[:, 0] *= width
        pts[:, 1] *= height
    return np.round(pts).astype(np.int32)


def get_damage_class(feature: dict) -> int:
    """Map xBD damage to our class (1 = undamaged, 2 = damaged). Uses properties.subtype or damage_grade."""
    props = feature.get("properties") or {}
    # Numeric damage_grade: 0=no-damage, 1=minor, 2=major, 3=destroyed
    grade = props.get("damage_grade")
    if grade is not None:
        try:
            g = int(grade)
            return UNDAMAGED if g == 0 else DAMAGED
        except (TypeError, ValueError):
            pass
    subtype = (props.get("subtype") or props.get("damage") or "un-classified")
    if isinstance(subtype, str):
        subtype = subtype.strip().lower()
    return DAMAGE_TO_CLASS.get(subtype, DAMAGED)


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    """
    Find corresponding image for a label file.
    Tries: (1) same stem as JSON in images_dir; (2) stem + '_post_disaster';
    (3) same names under images_dir/post_disaster/ or images_dir/<disaster>/ (e.g. hurricane-florence).
    """
    stem = label_path.stem
    search_roots: list[Path] = [images_dir]
    if images_dir.exists():
        for sub in ("post_disaster", "post-disaster"):
            d = images_dir / sub
            if d.is_dir():
                search_roots.append(d)
        # e.g. hurricane-florence_00000000_post_disaster -> try images_dir/hurricane-florence/
        if "_post_disaster" in stem or "_post-disaster" in stem:
            disaster = stem.split("_post")[0].rstrip("_-")
            if disaster:
                search_roots.append(images_dir / disaster)
    for root in search_roots:
        for ext in IMAGE_EXTENSIONS:
            p = root / (stem + ext)
            if p.exists():
                return p
            p = root / (stem + "_post_disaster" + ext)
            if p.exists():
                return p
    return None


def rasterize_label_file(
    label_path: Path,
    images_dir: Path,
    masks_dir: Path,
) -> bool:
    """
    Load one JSON, find matching image, rasterize building polygons, save mask.
    Returns True if mask was successfully written.
    """
    img_path = find_image_for_label(label_path, images_dir)
    if img_path is None:
        return False

    img = cv2.imread(str(img_path))
    if img is None:
        return False
    height, width = img.shape[:2]

    try:
        with open(label_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    mask = np.zeros((height, width), dtype=np.uint8)
    features = get_features_list(data)

    for feat in features:
        if not isinstance(feat, dict):
            continue
        pts = get_polygon_coordinates(feat, height, width)
        if pts is None:
            continue
        cls = get_damage_class(feat)
        cv2.fillPoly(mask, [pts], cls)

    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path = masks_dir / (label_path.stem + ".png")
    if not cv2.imwrite(str(out_path), mask):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PNG masks from xBD JSON labels (0=bg, 1=undamaged, 2=damaged)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_raw"),
        help="Directory containing labels/ and images/; masks/ will be created here (default: data_raw).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not labels_dir.exists():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not images_dir.exists():
        raise SystemExit(
            f"Images directory not found: {images_dir}\n"
            "Put image files (same base names as the JSON labels) in that folder, or run\n"
            "  python extract_xbd_subset.py --xbd-root <path> --disaster <name> -n 300 --out-dir data_raw\n"
            "to copy both images and labels from the xBD dataset."
        )

    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        raise SystemExit(f"No JSON files in {labels_dir}")

    written = 0
    debug_limit = 5
    for i, label_path in enumerate(label_files):
        if i < debug_limit:
            img_path = find_image_for_label(label_path, images_dir)
            expected_image_filename = img_path.name if img_path else f"{label_path.stem}.png (or .jpg/.jpeg)"
            image_exists = img_path is not None
            num_polygons = 0
            try:
                with open(label_path, encoding="utf-8") as f:
                    data = json.load(f)
                features = get_features_list(data)
                num_polygons = sum(1 for feat in features if isinstance(feat, dict))
            except (json.JSONDecodeError, OSError):
                pass
            print(f"[debug {i + 1}/{debug_limit}] JSON filename: {label_path.name}")
            print(f"         expected image filename: {expected_image_filename}")
            print(f"         image exists: {image_exists}")
            print(f"         polygons found: {num_polygons}")

        if rasterize_label_file(label_path, images_dir, masks_dir):
            written += 1

    print(f"Masks successfully written: {written}")
    if written == 0 and label_files:
        print(
            "No masks written: no image files matched the label filenames in the images directory.\n"
            "Ensure <data-dir>/images/ contains PNG/JPG files whose names match the JSON stems\n"
            "(e.g. hurricane-florence_00000000_post_disaster.png for ..._post_disaster.json).\n"
            "If using raw xBD layout, run extract_xbd_subset.py to copy both images and labels into data_raw."
        )


if __name__ == "__main__":
    main()
