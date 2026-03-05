from typing import Optional

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_unet_resnet18(
    encoder_name: str = "resnet18",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    num_classes: int = 3,
) -> nn.Module:
    """
    Create a U-Net model with a ResNet encoder using segmentation_models_pytorch.

    Parameters
    ----------
    encoder_name:
        Name of the encoder backbone (e.g. "resnet18", "resnet34").
    encoder_weights:
        Pretrained weights for the encoder (e.g. "imagenet", or None for random init).
    in_channels:
        Number of input channels. For xBD, you can set this to 3 (RGB) or 6
        if you concatenate pre- and post-disaster RGB images.
    num_classes:
        Number of output segmentation classes.

    Returns
    -------
    nn.Module
        U-Net model instance.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # logits; apply softmax in loss/metrics if needed
    )
    return model

