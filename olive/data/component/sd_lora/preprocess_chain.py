# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Preprocess chain components for image training.

These are placeholder components that allow the preprocess chains to be
registered and configured. The actual chain execution is handled by
the container's pre_process method (SDLoRADataContainer, DreamBoothDataContainer).

Available chain types:
- sd_lora_preprocess: LoRA preprocessing chain (works for SD 1.5 and SDXL)
- dreambooth_preprocess: DreamBooth preprocessing chain
"""

import logging
from typing import Optional

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


def _chain_placeholder(dataset, **kwargs):
    """Placeholder - actual execution is in container.pre_process()."""
    logger.debug(
        "Preprocess chain placeholder called directly. "
        "For chain execution, use SDLoRADataContainer or DreamBoothDataContainer."
    )
    return dataset


@Registry.register_pre_process("sd_lora_preprocess")
def sd_lora_preprocess(
    dataset,
    # Resolution params
    base_resolution: int = 512,
    bucket_mode: str = "auto",
    # Chain control
    enable_steps: Optional[list[str]] = None,
    disable_steps: Optional[list[str]] = None,
    step_params: Optional[dict[str, dict]] = None,
    # Global params
    device: str = "cuda",
    overwrite: bool = False,
    **kwargs,
):
    """LoRA preprocessing chain for Stable Diffusion.

    Works for both SD 1.5 and SDXL - just change base_resolution:
    - SD 1.5: base_resolution=512 (default)
    - SDXL: base_resolution=1024, bucket_mode="sdxl"

    Default chain:
    1. image_filtering (disabled) - Filter low quality images
    2. auto_caption (disabled) - Generate captions with BLIP-2/Florence-2
    3. auto_tagging (disabled) - Generate tags with WD14 tagger
    4. caption_tag_merge (disabled) - Merge captions and tags
    5. image_resizing (disabled) - Resize images
    6. aspect_ratio_bucketing (enabled) - Group by aspect ratio

    Args:
        dataset: The dataset to process.
        base_resolution: Base resolution (512 for SD1.5, 1024 for SDXL).
        bucket_mode: Bucket mode ("auto", "sdxl", "sd15").
        enable_steps: Steps to enable (e.g., ["auto_caption", "auto_tagging"]).
        disable_steps: Steps to disable.
        step_params: Parameters for specific steps.
        device: Device for neural network operations.
        overwrite: Whether to overwrite existing files.
        **kwargs: Additional parameters.

    Returns:
        Processed dataset.
    """
    return _chain_placeholder(dataset, **kwargs)


@Registry.register_pre_process("dreambooth_preprocess")
def dreambooth_preprocess(
    dataset,
    # Resolution params
    target_resolution: int = 512,
    # Chain control
    enable_steps: Optional[list[str]] = None,
    disable_steps: Optional[list[str]] = None,
    step_params: Optional[dict[str, dict]] = None,
    # Global params
    device: str = "cuda",
    overwrite: bool = False,
    **kwargs,
):
    """DreamBooth preprocessing chain.

    Default chain:
    1. image_filtering (disabled) - Filter low quality images
    2. dreambooth_prompt (enabled) - Generate fixed prompts with identifier
    3. generate_class_images (disabled) - Generate regularization images
    4. image_resizing (enabled) - Resize to fixed size, center crop

    Args:
        dataset: The dataset to process.
        target_resolution: Target image size (default 512).
        enable_steps: Steps to enable (e.g., ["generate_class_images"]).
        disable_steps: Steps to disable.
        step_params: Parameters for specific steps.
        device: Device for neural network operations.
        overwrite: Whether to overwrite existing files.
        **kwargs: Additional parameters.

    Returns:
        Processed dataset.
    """
    return _chain_placeholder(dataset, **kwargs)
