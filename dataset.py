from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Supported image extensions for discovery
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class XBDDataset(Dataset):
    """
    xBD-style dataset for building damage segmentation.

    Mask labels: 0 = background, 1 = undamaged building, 2 = damaged building.

    Expected folder structure:
        data/
          images/
            *.png, *.jpg, ...
          masks/
            same filenames as images

    Returns:
        image: float tensor (C, H, W) in [0, 1] or normalized
        mask: long tensor (H, W) with class indices
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transforms: Optional[Callable] = None,
        image_suffixes: Tuple[str, ...] = IMAGE_SUFFIXES,
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        self.image_suffixes = image_suffixes

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.image_paths: List[Path] = sorted(
            p
            for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.image_suffixes
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")

        missing = [self.masks_dir / p.name for p in self.image_paths if not (self.masks_dir / p.name).exists()]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} mask(s); e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = np.asarray(mask, dtype=np.int64)

        if self.transforms is not None:
            out = self.transforms(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]

        # Ensure CHW float image and (H,W) long mask
        if not torch.is_tensor(image):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        elif image.dim() == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.to(torch.long)

        return {
            "image": image,
            "mask": mask,
            "image_path": img_path.as_posix(),
            "mask_path": mask_path.as_posix(),
        }

