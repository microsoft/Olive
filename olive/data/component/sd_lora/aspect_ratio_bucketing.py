# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from olive.data.component.sd_lora.utils import (
    CropPosition,
    ResampleMode,
    ResizeMode,
    calculate_cover_size,
    get_resample_filter,
    resize_image,
)
from olive.data.registry import Registry

logger = logging.getLogger(__name__)


# Standard SD bucket resolutions (maintaining roughly 512x512 = 262144 pixels)
SDXL_BUCKETS = [
    (1024, 1024),  # 1:1
    (1152, 896),  # ~1.29:1
    (896, 1152),  # ~0.78:1
    (1216, 832),  # ~1.46:1
    (832, 1216),  # ~0.68:1
    (1344, 768),  # ~1.75:1
    (768, 1344),  # ~0.57:1
    (1536, 640),  # 2.4:1
    (640, 1536),  # ~0.42:1
]

SD15_BUCKETS = [
    (512, 512),  # 1:1
    (576, 448),  # ~1.29:1
    (448, 576),  # ~0.78:1
    (608, 416),  # ~1.46:1
    (416, 608),  # ~0.68:1
    (672, 384),  # ~1.75:1
    (384, 672),  # ~0.57:1
    (768, 320),  # 2.4:1
    (320, 768),  # ~0.42:1
]


def generate_buckets(
    base_resolution: int = 512,
    min_dim: int = 256,
    max_dim: int = 1024,
    divisor: int = 64,
    target_pixels: Optional[int] = None,
    max_aspect_ratio: float = 2.5,
) -> list[tuple[int, int]]:
    """Generate bucket resolutions for a given base resolution.

    Args:
        base_resolution: Base square resolution (e.g., 512 for SD 1.5, 1024 for SDXL).
        min_dim: Minimum dimension for any bucket.
        max_dim: Maximum dimension for any bucket.
        divisor: All dimensions must be divisible by this value (usually 64 or 8).
        target_pixels: Target total pixels. Defaults to base_resolution^2.
        max_aspect_ratio: Maximum aspect ratio (width/height or height/width).

    Returns:
        List of (width, height) tuples representing bucket sizes.

    """
    target_pixels = target_pixels or (base_resolution * base_resolution)
    buckets = set()

    # Generate buckets by iterating through possible widths
    for width in range(min_dim, max_dim + 1, divisor):
        # Calculate height to maintain target pixel count
        height = int(target_pixels / width)
        # Round to nearest divisor
        height = round(height / divisor) * divisor

        if height < min_dim or height > max_dim:
            continue

        aspect = width / height
        if aspect > max_aspect_ratio or aspect < (1 / max_aspect_ratio):
            continue

        buckets.add((width, height))

    return sorted(buckets, key=lambda x: (abs(x[0] / x[1] - 1), -x[0] * x[1]))


@Registry.register_pre_process()
def aspect_ratio_bucketing(
    dataset,
    base_resolution: int = 512,
    custom_buckets: Optional[list[tuple[int, int]]] = None,
    resize_images: bool = True,
    resize_mode: ResizeMode = ResizeMode.COVER,
    resample_mode: ResampleMode = ResampleMode.LANCZOS,
    crop_position: CropPosition = CropPosition.CENTER,
    fill_color: tuple[int, int, int] = (255, 255, 255),
    output_dir: Optional[str] = None,
    overwrite: bool = False,
):
    """Organize images into aspect ratio buckets for efficient training.

    Aspect ratio bucketing groups images with similar aspect ratios together,
    allowing for efficient batching during training without excessive padding
    or cropping.

    Args:
        dataset: The dataset to process.
        base_resolution: Base resolution (512 for SD 1.5, 1024 for SDXL/Flux).
            - 512: Uses predefined SD 1.5 buckets
            - 1024: Uses predefined SDXL buckets
            - Other values: Auto-generates buckets
        custom_buckets: Custom list of (width, height) tuples. If provided,
            overrides base_resolution.
        resize_images: Whether to resize images to fit buckets.
        resize_mode: How to fit images to buckets (ResizeMode.COVER or ResizeMode.CONTAIN).
        resample_mode: Resampling filter (ResampleMode.LANCZOS, BILINEAR, BICUBIC, NEAREST).
        crop_position: Where to crop in cover mode (CropPosition.CENTER, TOP, BOTTOM, LEFT, RIGHT).
        fill_color: Fill color for padding in contain mode.
        output_dir: Directory for processed images (None for in-place).
        overwrite: Whether to overwrite existing processed images.

    Returns:
        The dataset with bucket information added to each item.

    """
    from PIL import Image

    # Determine buckets
    if custom_buckets:
        buckets = custom_buckets
    elif base_resolution == 512:
        buckets = SD15_BUCKETS
    elif base_resolution == 1024:
        buckets = SDXL_BUCKETS
    else:
        buckets = generate_buckets(
            base_resolution=base_resolution,
            max_dim=base_resolution * 2,
        )

    logger.info("Using %d bucket sizes", len(buckets))
    for i, (w, h) in enumerate(buckets[:5]):
        logger.info("  Bucket %d: %dx%d (aspect %.2f)", i, w, h, w / h)
    if len(buckets) > 5:
        logger.info("  ... and %d more", len(buckets) - 5)

    resample_filter = get_resample_filter(resample_mode)
    crop_to_bucket = ResizeMode(resize_mode) == ResizeMode.COVER

    # Prepare output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Assign each image to a bucket
    bucket_assignments = {}
    bucket_counts = defaultdict(int)

    # Check if dataset supports file writing (local folder datasets have data_dir)
    can_write_files = hasattr(dataset, "data_dir") and dataset.data_dir is not None

    # Warn if resize is requested but no output location available
    if resize_images and not output_dir and not can_write_files:
        logger.warning(
            "resize_images=True but no output_dir specified and dataset doesn't support in-place writing "
            "(e.g., HuggingFace dataset). Images will NOT be pre-resized. "
            "Specify output_dir to enable pre-resizing, or images will be resized on-the-fly during training."
        )

    for i, item in enumerate(dataset):
        image_path = Path(item["image_path"])

        try:
            with Image.open(image_path) as img:
                orig_w, orig_h = img.size
                orig_aspect = orig_w / orig_h

                # Find best matching bucket
                best_bucket = _find_best_bucket(orig_w, orig_h, buckets)
                bucket_w, bucket_h = best_bucket

                # Calculate crop coordinates for SDXL time embeddings
                crops_coords_top_left = _calculate_crop_coords(
                    orig_w, orig_h, bucket_w, bucket_h, crop_to_bucket, crop_position
                )

                final_path = str(image_path)

                # Resize image if requested and we have a place to save
                if resize_images and (output_dir or can_write_files):
                    if output_dir:
                        # Generate unique output filename
                        # Use index to ensure uniqueness (HF cache filenames may not have extensions)
                        if image_path.suffix:
                            out_name = f"{i:06d}{image_path.suffix}"
                        else:
                            out_name = f"{i:06d}.jpg"
                        out_path = Path(output_dir) / out_name
                    else:
                        out_path = image_path

                    if not overwrite and out_path.exists() and out_path != image_path:
                        # Use existing resized file
                        final_path = str(out_path)
                        # Update dataset path to use existing resized file
                        if hasattr(dataset, "set_image_path"):
                            dataset.set_image_path(i, out_path)
                        elif hasattr(dataset, "image_paths"):
                            dataset.image_paths[i] = out_path
                    else:
                        # Convert to RGB if needed
                        if img.mode != "RGB":
                            img = img.convert("RGB")  # noqa: PLW2901

                        # Resize to fit bucket
                        resized = resize_image(
                            img,
                            bucket_w,
                            bucket_h,
                            resize_mode=resize_mode,
                            crop_position=crop_position,
                            fill_color=fill_color,
                            resample_filter=resample_filter,
                        )

                        # Save with explicit format if no extension
                        if out_path.suffix:
                            resized.save(out_path, quality=95)
                        else:
                            resized.save(out_path, format="JPEG", quality=95)
                        final_path = str(out_path)

                        # Update dataset path if changed
                        if out_path != image_path:
                            if hasattr(dataset, "set_image_path"):
                                dataset.set_image_path(i, out_path)
                            elif hasattr(dataset, "image_paths"):
                                dataset.image_paths[i] = out_path

                # Store bucket assignment with the FINAL path (after any resizing)
                bucket_assignments[final_path] = {
                    "bucket": best_bucket,
                    "original_size": (orig_w, orig_h),
                    "aspect_ratio": orig_aspect,
                    "crops_coords_top_left": crops_coords_top_left,
                }
                bucket_counts[best_bucket] += 1

        except Exception as e:
            logger.warning("Failed to process %s: %s", image_path, e)

    # Process class images for DreamBooth (if present)
    if hasattr(dataset, "class_image_paths") and dataset.class_image_paths:
        logger.info("Processing %d class images for DreamBooth", len(dataset.class_image_paths))

        # Prepare class images output directory
        class_output_dir = None
        if output_dir:
            class_output_dir = Path(output_dir) / "class_images"
            class_output_dir.mkdir(parents=True, exist_ok=True)

        for i, class_path in enumerate(dataset.class_image_paths):
            class_path = Path(class_path)  # noqa: PLW2901
            try:
                with Image.open(class_path) as img:
                    orig_w, orig_h = img.size
                    orig_aspect = orig_w / orig_h

                    # Find best matching bucket
                    best_bucket = _find_best_bucket(orig_w, orig_h, buckets)
                    bucket_w, bucket_h = best_bucket

                    # Calculate crop coordinates
                    crops_coords_top_left = _calculate_crop_coords(
                        orig_w, orig_h, bucket_w, bucket_h, crop_to_bucket, crop_position
                    )

                    final_path = str(class_path)

                    # Resize class image if requested
                    if resize_images and class_output_dir:
                        if class_path.suffix:
                            out_name = f"class_{i:06d}{class_path.suffix}"
                        else:
                            out_name = f"class_{i:06d}.jpg"
                        out_path = class_output_dir / out_name

                        if not overwrite and out_path.exists():
                            final_path = str(out_path)
                        else:
                            if img.mode != "RGB":
                                img = img.convert("RGB")  # noqa: PLW2901

                            resized = resize_image(
                                img,
                                bucket_w,
                                bucket_h,
                                resize_mode=resize_mode,
                                crop_position=crop_position,
                                fill_color=fill_color,
                                resample_filter=resample_filter,
                            )

                            if out_path.suffix:
                                resized.save(out_path, quality=95)
                            else:
                                resized.save(out_path, format="JPEG", quality=95)
                            final_path = str(out_path)

                        # Update class image path in dataset
                        dataset.class_image_paths[i] = Path(final_path)

                    # Store bucket assignment for class image
                    bucket_assignments[final_path] = {
                        "bucket": best_bucket,
                        "original_size": (orig_w, orig_h),
                        "aspect_ratio": orig_aspect,
                        "crops_coords_top_left": crops_coords_top_left,
                    }
                    bucket_counts[best_bucket] += 1

            except Exception as e:
                logger.warning("Failed to process class image %s: %s", class_path, e)

        logger.info("Processed %d class images", len(dataset.class_image_paths))

    # Log bucket distribution
    logger.info("Bucket distribution:")
    for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info("  %dx%d: %d images", bucket[0], bucket[1], count)

    # Store bucket assignments in dataset
    dataset.bucket_assignments = bucket_assignments
    dataset.buckets = buckets

    return dataset


def _find_best_bucket(orig_w: int, orig_h: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """Find the bucket that best matches the image aspect ratio."""
    orig_aspect = orig_w / orig_h
    best_bucket = buckets[0]
    best_score = float("inf")

    for bucket_w, bucket_h in buckets:
        bucket_aspect = bucket_w / bucket_h

        # Score based on aspect ratio difference and resolution match
        aspect_diff = abs(math.log(orig_aspect) - math.log(bucket_aspect))

        # Prefer buckets that don't require too much upscaling
        scale_w = bucket_w / orig_w
        scale_h = bucket_h / orig_h
        scale = max(scale_w, scale_h)
        upscale_penalty = max(0, scale - 1) * 0.5  # Penalize upscaling

        score = aspect_diff + upscale_penalty

        if score < best_score:
            best_score = score
            best_bucket = (bucket_w, bucket_h)

    return best_bucket


def _calculate_crop_coords(
    orig_w: int,
    orig_h: int,
    bucket_w: int,
    bucket_h: int,
    crop_to_bucket: bool,
    crop_position: CropPosition,
) -> tuple[int, int]:
    """Calculate crop coordinates (top, left) for SDXL time embeddings.

    This mirrors the logic in resize_image to compute where the crop would occur.
    """
    if not crop_to_bucket:
        # Contain mode: no cropping, image is padded
        return (0, 0)

    # Cover mode: calculate intermediate size after resize
    new_w, new_h = calculate_cover_size(orig_w, orig_h, bucket_w, bucket_h)

    # Calculate crop offsets
    pos = CropPosition(crop_position)
    if pos == CropPosition.CENTER:
        left = (new_w - bucket_w) // 2
        top = (new_h - bucket_h) // 2
    elif pos == CropPosition.TOP:
        left = (new_w - bucket_w) // 2
        top = 0
    elif pos == CropPosition.BOTTOM:
        left = (new_w - bucket_w) // 2
        top = new_h - bucket_h
    elif pos == CropPosition.LEFT:
        left = 0
        top = (new_h - bucket_h) // 2
    elif pos == CropPosition.RIGHT:
        left = new_w - bucket_w
        top = (new_h - bucket_h) // 2
    else:
        left = (new_w - bucket_w) // 2
        top = (new_h - bucket_h) // 2

    return (top, left)
