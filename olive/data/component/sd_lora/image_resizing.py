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

    from olive.data.component.sd_lora.utils import calculate_cover_size

    # Validate resize_mode early
    resize_mode_enum = ResizeMode(resize_mode)
    resample_filter = get_resample_filter(resample_mode)
    crop_to_bucket = resize_mode_enum == ResizeMode.COVER

    # Prepare output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    bucket_assignments = {}
    target_bucket = (target_resolution, target_resolution)

    def _calculate_crop_coords(orig_w: int, orig_h: int) -> tuple[int, int]:
        """Calculate crop coordinates for SDXL time embeddings."""
        if not crop_to_bucket:
            return (0, 0)

        new_w, new_h = calculate_cover_size(orig_w, orig_h, target_resolution, target_resolution)
        pos = CropPosition(crop_position)
        if pos == CropPosition.CENTER:
            left = (new_w - target_resolution) // 2
            top = (new_h - target_resolution) // 2
        elif pos == CropPosition.TOP:
            left = (new_w - target_resolution) // 2
            top = 0
        elif pos == CropPosition.BOTTOM:
            left = (new_w - target_resolution) // 2
            top = new_h - target_resolution
        elif pos == CropPosition.LEFT:
            left = 0
            top = (new_h - target_resolution) // 2
        elif pos == CropPosition.RIGHT:
            left = new_w - target_resolution
            top = (new_h - target_resolution) // 2
        else:
            left = (new_w - target_resolution) // 2
            top = (new_h - target_resolution) // 2

        return (top, left)

    def _process_image(image_path: Path, out_path: Path, prefix: str = "") -> Optional[str]:
        """Process a single image and return the final path."""
        nonlocal processed_count, skipped_count

        try:
            # Get original size for bucket assignment (needed even if skipping resize)
            with Image.open(image_path) as img:
                orig_w, orig_h = img.size

            # Check if already processed
            if not overwrite and out_path.exists() and out_path != image_path:
                skipped_count += 1
                # Still need to add bucket assignment for skipped files
                crops_coords = _calculate_crop_coords(orig_w, orig_h)
                bucket_assignments[str(out_path)] = {
                    "bucket": target_bucket,
                    "original_size": (orig_w, orig_h),
                    "aspect_ratio": orig_w / orig_h,
                    "crops_coords_top_left": crops_coords,
                }
                return str(out_path)

            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")  # noqa: PLW2901

                result = resize_image(
                    img,
                    target_resolution,
                    target_resolution,
                    resize_mode=resize_mode_enum,
                    crop_position=crop_position,
                    fill_color=fill_color,
                    resample_filter=resample_filter,
                )

                result.save(out_path, quality=95)
                processed_count += 1

                # Store bucket assignment
                crops_coords = _calculate_crop_coords(orig_w, orig_h)
                bucket_assignments[str(out_path)] = {
                    "bucket": target_bucket,
                    "original_size": (orig_w, orig_h),
                    "aspect_ratio": orig_w / orig_h,
                    "crops_coords_top_left": crops_coords,
                }

                return str(out_path)

        except Exception as e:
            logger.warning("Failed to resize %s%s: %s", prefix, image_path, e)
            return None

    # Process instance images
    for i, item in enumerate(dataset):
        image_path = Path(item["image_path"])

        # Determine output path
        if output_dir:
            out_path = Path(output_dir) / image_path.name
        else:
            out_path = image_path

        final_path = _process_image(image_path, out_path)

        # Update dataset path if output location changed
        if final_path and out_path != image_path:
            if hasattr(dataset, "set_image_path"):
                dataset.set_image_path(i, out_path)
            elif hasattr(dataset, "image_paths"):
                dataset.image_paths[i] = out_path

    logger.info("Resized %d instance images, skipped %d", processed_count, skipped_count)

    # Process class images for DreamBooth (if present)
    if hasattr(dataset, "class_image_paths") and dataset.class_image_paths:
        logger.info("Processing %d class images for DreamBooth", len(dataset.class_image_paths))

        class_processed = 0
        class_output_dir = None
        if output_dir:
            class_output_dir = Path(output_dir) / "class_images"
            class_output_dir.mkdir(parents=True, exist_ok=True)

        for i, class_path in enumerate(dataset.class_image_paths):
            class_path = Path(class_path)  # noqa: PLW2901

            if class_output_dir:
                out_path = class_output_dir / f"class_{i:06d}{class_path.suffix or '.jpg'}"
            else:
                out_path = class_path

            final_path = _process_image(class_path, out_path, prefix="class image ")

            if final_path:
                class_processed += 1
                dataset.class_image_paths[i] = Path(final_path)

        logger.info("Processed %d class images", class_processed)

    # Store bucket assignments in dataset (for compatibility with aspect_ratio_bucketing)
    dataset.bucket_assignments = bucket_assignments
    dataset.buckets = [target_bucket]

    return dataset
