# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Aspect ratio bucketing component for Stable Diffusion LoRA training."""

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


# Standard SD bucket resolutions (maintaining roughly 512x512 = 262144 pixels)
SDXL_BUCKETS = [
    (1024, 1024),  # 1:1
    (1152, 896),   # ~1.29:1
    (896, 1152),   # ~0.78:1
    (1216, 832),   # ~1.46:1
    (832, 1216),   # ~0.68:1
    (1344, 768),   # ~1.75:1
    (768, 1344),   # ~0.57:1
    (1536, 640),   # 2.4:1
    (640, 1536),   # ~0.42:1
]

SD15_BUCKETS = [
    (512, 512),    # 1:1
    (576, 448),    # ~1.29:1
    (448, 576),    # ~0.78:1
    (608, 416),    # ~1.46:1
    (416, 608),    # ~0.68:1
    (672, 384),    # ~1.75:1
    (384, 672),    # ~0.57:1
    (768, 320),    # 2.4:1
    (320, 768),    # ~0.42:1
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
    bucket_mode: str = "auto",
    custom_buckets: Optional[list[tuple[int, int]]] = None,
    min_dim: int = 256,
    max_dim: int = 1024,
    divisor: int = 64,
    max_aspect_ratio: float = 2.5,
    resize_images: bool = True,
    upscale_mode: str = "lanczos",
    downscale_mode: str = "lanczos",
    crop_to_bucket: bool = True,
    crop_position: str = "center",
    fill_color: tuple[int, int, int] = (255, 255, 255),
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    save_bucket_info: bool = True,
    bucket_info_file: str = "bucket_info.json",
    **kwargs,
):
    """Organize images into aspect ratio buckets for efficient training.

    Aspect ratio bucketing groups images with similar aspect ratios together,
    allowing for efficient batching during training without excessive padding
    or cropping.

    Bucket modes:
    - auto: Generate buckets automatically based on base_resolution
    - sd15: Use standard SD 1.5 buckets (512 base)
    - sdxl: Use standard SDXL buckets (1024 base)
    - custom: Use custom_buckets parameter

    Args:
        dataset: The dataset to process.
        base_resolution: Base resolution for bucket generation.
        bucket_mode: How to determine buckets ("auto", "sd15", "sdxl", "custom").
        custom_buckets: List of (width, height) tuples for custom bucket mode.
        min_dim: Minimum dimension for auto-generated buckets.
        max_dim: Maximum dimension for auto-generated buckets.
        divisor: Dimension divisibility requirement.
        max_aspect_ratio: Maximum aspect ratio for auto-generated buckets.
        resize_images: Whether to resize images to fit buckets.
        upscale_mode: Resampling filter for upscaling.
        downscale_mode: Resampling filter for downscaling.
        crop_to_bucket: Whether to crop images to exact bucket size.
        crop_position: Where to crop ("center", "top", "bottom", "left", "right").
        fill_color: Fill color for padding if not cropping.
        output_dir: Directory for processed images (None for in-place).
        overwrite: Whether to overwrite existing processed images.
        save_bucket_info: Whether to save bucket assignment info.
        bucket_info_file: Filename for bucket info JSON.
        **kwargs: Additional arguments.

    Returns:
        The dataset with bucket information added to each item.
    """
    from PIL import Image

    # Determine buckets
    if bucket_mode == "sd15":
        buckets = SD15_BUCKETS
    elif bucket_mode == "sdxl":
        buckets = SDXL_BUCKETS
    elif bucket_mode == "custom":
        if not custom_buckets:
            raise ValueError("custom_buckets must be provided when bucket_mode='custom'")
        buckets = custom_buckets
    else:  # auto
        buckets = generate_buckets(
            base_resolution=base_resolution,
            min_dim=min_dim,
            max_dim=max_dim,
            divisor=divisor,
            max_aspect_ratio=max_aspect_ratio,
        )

    logger.info("Using %d bucket sizes", len(buckets))
    for i, (w, h) in enumerate(buckets[:5]):
        logger.info("  Bucket %d: %dx%d (aspect %.2f)", i, w, h, w / h)
    if len(buckets) > 5:
        logger.info("  ... and %d more", len(buckets) - 5)

    # Resampling filters
    resample_filters = {
        "lanczos": Image.LANCZOS,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "nearest": Image.NEAREST,
    }
    upscale_filter = resample_filters.get(upscale_mode.lower(), Image.LANCZOS)
    downscale_filter = resample_filters.get(downscale_mode.lower(), Image.LANCZOS)

    # Prepare output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Assign each image to a bucket
    bucket_assignments = {}
    bucket_counts = defaultdict(int)

    for i in range(len(dataset)):
        item = dataset[i]
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

                # Resize image if requested
                if resize_images:
                    if output_dir:
                        out_path = Path(output_dir) / image_path.name
                    else:
                        out_path = image_path

                    if not overwrite and out_path.exists() and out_path != image_path:
                        # Store assignment with the FINAL path that will be used
                        final_path = str(out_path)
                    else:
                        # Convert to RGB if needed
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        elif img.mode != "RGB":
                            img = img.convert("RGB")

                        # Resize to fit bucket
                        resized = _resize_to_bucket(
                            img,
                            bucket_w,
                            bucket_h,
                            crop_to_bucket,
                            crop_position,
                            fill_color,
                            upscale_filter,
                            downscale_filter,
                        )

                        # Save
                        resized.save(out_path, quality=95)
                        final_path = str(out_path)

                        # Update dataset path if changed
                        if out_path != image_path:
                            dataset.image_paths[i] = out_path
                else:
                    final_path = str(image_path)

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

    # Log bucket distribution
    logger.info("Bucket distribution:")
    for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info("  %dx%d: %d images", bucket[0], bucket[1], count)

    # Save bucket info
    if save_bucket_info:
        import json

        info_path = Path(output_dir or dataset.data_dir) / bucket_info_file
        bucket_info = {
            "buckets": [list(b) for b in buckets],
            "assignments": {k: {"bucket": list(v["bucket"]), **{kk: vv for kk, vv in v.items() if kk != "bucket"}}
                           for k, v in bucket_assignments.items()},
            "bucket_counts": {f"{k[0]}x{k[1]}": v for k, v in bucket_counts.items()},
        }
        with open(info_path, "w") as f:
            json.dump(bucket_info, f, indent=2)
        logger.info("Saved bucket info to %s", info_path)

    # Store bucket assignments in dataset
    dataset.bucket_assignments = bucket_assignments
    dataset.buckets = buckets

    return dataset


def _find_best_bucket(
    orig_w: int, orig_h: int, buckets: list[tuple[int, int]]
) -> tuple[int, int]:
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
    crop_position: str,
) -> tuple[int, int]:
    """Calculate crop coordinates (top, left) for SDXL time embeddings.

    This mirrors the logic in _resize_to_bucket and _crop_to_size to compute
    where the crop would occur.
    """
    if not crop_to_bucket:
        # Contain mode: no cropping, image is padded
        return (0, 0)

    # Cover mode: calculate intermediate size after resize
    orig_aspect = orig_w / orig_h
    bucket_aspect = bucket_w / bucket_h

    if orig_aspect > bucket_aspect:
        # Image is wider, resize by height, crop width
        new_h = bucket_h
        new_w = int(bucket_h * orig_aspect)
    else:
        # Image is taller, resize by width, crop height
        new_w = bucket_w
        new_h = int(bucket_w / orig_aspect)

    # Calculate crop offsets
    if crop_position == "center":
        left = (new_w - bucket_w) // 2
        top = (new_h - bucket_h) // 2
    elif crop_position == "top":
        left = (new_w - bucket_w) // 2
        top = 0
    elif crop_position == "bottom":
        left = (new_w - bucket_w) // 2
        top = new_h - bucket_h
    elif crop_position == "left":
        left = 0
        top = (new_h - bucket_h) // 2
    elif crop_position == "right":
        left = new_w - bucket_w
        top = (new_h - bucket_h) // 2
    else:
        left = (new_w - bucket_w) // 2
        top = (new_h - bucket_h) // 2

    return (top, left)


def _resize_to_bucket(
    img,
    bucket_w: int,
    bucket_h: int,
    crop_to_bucket: bool,
    crop_position: str,
    fill_color: tuple[int, int, int],
    upscale_filter,
    downscale_filter,
):
    """Resize image to fit bucket dimensions."""
    from PIL import Image

    orig_w, orig_h = img.size
    orig_aspect = orig_w / orig_h
    bucket_aspect = bucket_w / bucket_h

    if crop_to_bucket:
        # Cover mode: resize to cover bucket, then crop
        if orig_aspect > bucket_aspect:
            # Image is wider, resize by height
            new_h = bucket_h
            new_w = int(bucket_h * orig_aspect)
        else:
            # Image is taller, resize by width
            new_w = bucket_w
            new_h = int(bucket_w / orig_aspect)

        # Choose filter based on scale direction
        if new_w > orig_w or new_h > orig_h:
            resized = img.resize((new_w, new_h), upscale_filter)
        else:
            resized = img.resize((new_w, new_h), downscale_filter)

        # Crop to bucket size
        return _crop_to_size(resized, bucket_w, bucket_h, crop_position)
    else:
        # Contain mode: resize to fit within bucket, then pad
        if orig_aspect > bucket_aspect:
            # Image is wider, resize by width
            new_w = bucket_w
            new_h = int(bucket_w / orig_aspect)
        else:
            # Image is taller, resize by height
            new_h = bucket_h
            new_w = int(bucket_h * orig_aspect)

        if new_w > orig_w or new_h > orig_h:
            resized = img.resize((new_w, new_h), upscale_filter)
        else:
            resized = img.resize((new_w, new_h), downscale_filter)

        # Create canvas and center the image
        canvas = Image.new("RGB", (bucket_w, bucket_h), fill_color)
        paste_x = (bucket_w - new_w) // 2
        paste_y = (bucket_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas


def _crop_to_size(img, target_w: int, target_h: int, position: str = "center"):
    """Crop image to target size."""
    w, h = img.size

    if position == "center":
        left = (w - target_w) // 2
        top = (h - target_h) // 2
    elif position == "top":
        left = (w - target_w) // 2
        top = 0
    elif position == "bottom":
        left = (w - target_w) // 2
        top = h - target_h
    elif position == "left":
        left = 0
        top = (h - target_h) // 2
    elif position == "right":
        left = w - target_w
        top = (h - target_h) // 2
    else:
        left = (w - target_w) // 2
        top = (h - target_h) // 2

    # Ensure bounds are valid
    left = max(0, left)
    top = max(0, top)

    return img.crop((left, top, left + target_w, top + target_h))
