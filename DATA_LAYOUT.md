# How xBD is written locally

## 1. Raw xBD (after you extract the download)

The xBD archive can be structured in two ways:

**Layout A – flat train/**  
```
<xbd-root>/
  train/
    images/   ← PNGs named e.g. hurricane-florence_00000000_post_disaster.png
    labels/   ← JSONs with same stem, e.g. hurricane-florence_00000000_post_disaster.json
```

**Layout B – per-disaster folders**  
```
<xbd-root>/
  hurricane-florence/
    images/
      post_disaster/   ← or all images directly in images/
        *.png
    labels/
      *.json
  hurricane-harvey/
    images/
    labels/
  ...
```

`extract_xbd_subset.py` supports both: it looks for `train/images` and `train/labels` first, then for `<xbd-root>/<disaster>/images/` and `.../labels/` (and `images/post_disaster/` if present).

---

## 2. After `extract_xbd_subset.py` (default `--out-dir data_raw`)

Writes a **flat** layout:

```
data_raw/
  images/   ← copied PNGs, same filenames as labels
  labels/   ← copied JSONs
```

Image and label filenames match (e.g. `hurricane-florence_00000000_post_disaster.png` and `.json`).

---

## 3. After `generate_masks.py --data-dir data_raw`

Adds masks next to images and labels:

```
data_raw/
  images/
  labels/
  masks/    ← one PNG per label, same base name (e.g. ..._post_disaster.png), pixel values 0/1/2
```

---

## 4. After `split_dataset.py --data-dir data_raw`

**Moves** (does not copy) files from `data_raw/images` and `data_raw/masks` into train/val:

```
data_raw/
  train/
    images/
    masks/
  val/
    images/
    masks/
  labels/   ← unchanged (not used by training)
```

`data_raw/images/` and `data_raw/masks/` are emptied.

---

## 5. What `train.py` expects

```
<data-dir>/
  train/
    images/
    masks/
  val/
    images/
    masks/
```

So after the full pipeline, `--data-dir data_raw` points at this structure.
