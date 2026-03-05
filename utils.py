from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility (within reason)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logits_to_preds(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert model logits to class predictions.

    Expects shape (B, C, H, W); returns (B, H, W) with integer class ids.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected 4D logits (B, C, H, W), got {logits.shape}")
    return torch.argmax(logits, dim=1)


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute a (num_classes, num_classes) confusion matrix on CPU.

    Rows = ground truth, columns = predictions.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch between preds {preds.shape} and targets {targets.shape}"
        )

    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)

    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    with torch.no_grad():
        indices = num_classes * targets + preds
        cm = torch.bincount(indices, minlength=num_classes**2)
        cm = cm.reshape(num_classes, num_classes).to(torch.float32)
    return cm


def iou_from_confusion_matrix(
    cm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-class and mean IoU from confusion matrix.

    Parameters
    ----------
    cm:
        Confusion matrix of shape (C, C) where rows are gt, columns are preds.

    Returns
    -------
    per_class_iou, mean_iou
    """
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = tp + fp + fn + 1e-7
    iou = tp / denom
    mean_iou = iou.mean()
    return iou, mean_iou


def dice_from_confusion_matrix(
    cm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-class and mean Dice score from confusion matrix.
    """
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = 2 * tp + fp + fn + 1e-7
    dice = 2 * tp / denom
    mean_dice = dice.mean()
    return dice, mean_dice


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute IoU and Dice metrics given logits and targets.

    Returns a flat dict with mean and per-class metrics:
        - "miou", "dice"
        - "iou_class_{k}", "dice_class_{k}" for k in [0, C-1]
    """
    preds = logits_to_preds(logits)
    cm = compute_confusion_matrix(preds, targets, num_classes=num_classes)
    per_iou, miou = iou_from_confusion_matrix(cm)
    per_dice, mdice = dice_from_confusion_matrix(cm)

    metrics: Dict[str, float] = {
        "miou": float(miou.item()),
        "dice": float(mdice.item()),
    }
    for k in range(num_classes):
        metrics[f"iou_class_{k}"] = float(per_iou[k].item())
        metrics[f"dice_class_{k}"] = float(per_dice[k].item())
    return metrics

