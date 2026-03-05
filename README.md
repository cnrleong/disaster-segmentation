## Disaster Segmentation (xBD)

Post-disaster building damage segmentation using the xBD dataset.

### Project structure

- `dataset.py` – PyTorch `Dataset` for image–mask pairs (xBD-ready).
- `model.py` – U-Net model factory with ResNet18 encoder (via `segmentation_models_pytorch`).
- `train.py` – Training / validation loop and CLI entrypoint.
- `utils.py` – Metrics (IoU, Dice), seed helpers, and small utilities.
- `requirements.txt` – Python dependencies.
- `data/` – Expected data root (not tracked in git).
  - `train/images/` – Training images.
  - `train/masks/` – Training masks (values 0=background, 1=undamaged, 2=damaged).
  - `val/images/` – Validation images.
  - `val/masks/` – Validation masks.

### Basic usage

Install dependencies (e.g. in a fresh virtualenv or Colab notebook):

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py \
  --data-dir ./data \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --encoder resnet18 \
  --encoder-weights imagenet \
  --num-classes 3 \
  --output-dir ./checkpoints
```

All path arguments (`--data-dir`, `--output-dir`, `--xbd-root`, `--out-dir`) accept relative or absolute paths and work on Windows and Google Colab (Linux).

### Data pipeline (after downloading xBD)

1. **Extract** the xBD archive (e.g. `train_images_labels_targets`) so you have disaster folders with `images/post_disaster/` and `labels/`.
2. **Extract a subset**:  
   `python extract_xbd_subset.py --xbd-root <path_to_extracted_xbd> --disaster hurricane-harvey -n 300 --out-dir data_raw`
3. **Generate masks** from JSON labels:  
   `python generate_masks.py --data-dir data_raw`
4. **Split** into train/val:  
   `python split_dataset.py --data-dir data_raw --train-ratio 0.8 --seed 42`
5. **Train**:  
   `python train.py --data-dir data_raw --epochs 50 --batch-size 8 --output-dir checkpoints`

### Using Google Colab

1. **Upload your project** to Colab (e.g. zip the repo and upload, or clone from git, or mount Google Drive and copy the folder).
2. **Upload (or mount) your data** so the same structure exists: a folder with `train/images`, `train/masks`, `val/images`, `val/masks`. For example, zip `data_raw` after running the pipeline locally and upload it, then unzip in Colab.
3. **Install dependencies** in a cell:
   ```bash
   !pip install -r requirements.txt
   ```
4. **Enable GPU**: Runtime → Change runtime type → GPU.
5. **Run training** (adjust `--data-dir` to where your data is in Colab, e.g. `./data_raw` or `/content/data_raw`):
   ```bash
   !python train.py --data-dir ./data_raw --epochs 50 --batch-size 8 --output-dir ./checkpoints
   ```
6. **Download checkpoints**: from the Colab file browser, or zip `checkpoints/` and download.

