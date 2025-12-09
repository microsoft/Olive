# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

from olive.data.component.sd_lora.utils import (
    CropPosition,
    ResampleMode,
    ResizeMode,
    get_resample_filter,
    resize_image,
)
from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_pre_process()
def image_resizing(
    dataset,
    target_resolution: int = 512,
    resize_mode: ResizeMode = ResizeMode.COVER,
    resample_mode: ResampleMode = ResampleMode.LANCZOS,
    fill_color: tuple[int, int, int] = (255, 255, 255),
    crop_position: CropPosition = CropPosition.CENTER,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
):
    """Resize images to fixed size for DreamBooth training.

    Resize modes:
    - cover: Fill target size, maintaining aspect ratio (may crop)
    - contain: Fit image within target size, maintaining aspect ratio (may have padding)

    Args:
        dataset: The dataset to process.
        target_resolution: Target resolution (width and height).
        resize_mode: How to handle aspect ratio (ResizeMode.COVER or ResizeMode.CONTAIN).
        resample_mode: Resampling filter (ResampleMode.LANCZOS, BILINEAR, BICUBIC, NEAREST).
        fill_color: RGB color for padding in "contain" mode.
        crop_position: Where to crop in "cover" mode (CropPosition.CENTER, TOP, BOTTOM, LEFT, RIGHT).
        output_dir: Directory for resized images (None to overwrite in place).
        overwrite: Whether to overwrite existing resized images.

    Returns:
        The dataset with resized images.

    """
    from PIL import Image

    # Validate resize_mode early
    resize_mode = ResizeMode(resize_mode)
    resample_filter = get_resample_filter(resample_mode)

    # Prepare output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for i in range(len(dataset)):
        item = dataset[i]
        image_path = Path(item["image_path"])

        # Determine output path
        if output_dir:
            out_path = Path(output_dir) / image_path.name
        else:
            out_path = image_path

        # Check if already processed
        if not overwrite and out_path.exists() and out_path != image_path:
            skipped_count += 1
            continue

        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")  # noqa: PLW2901

                result = resize_image(
                    img,
                    target_resolution,
                    target_resolution,
                    resize_mode=resize_mode,
                    crop_position=crop_position,
                    fill_color=fill_color,
                    resample_filter=resample_filter,
                )

                result.save(out_path, quality=95)
                processed_count += 1

                # Update dataset path if output location changed
                if out_path != image_path:
                    dataset.image_paths[i] = out_path

        except Exception as e:
            logger.warning("Failed to resize %s: %s", image_path, e)

    logger.info("Resized %d images, skipped %d", processed_count, skipped_count)

    return dataset
