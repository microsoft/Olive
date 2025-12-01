# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Image resizing component for Stable Diffusion LoRA training."""

import logging
from pathlib import Path
from typing import Optional, Union

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_pre_process()
def image_resizing(
    dataset,
    target_resolution: int = 512,
    max_resolution: Optional[int] = None,
    min_resolution: Optional[int] = None,
    resize_mode: str = "contain",
    upscale_mode: str = "lanczos",
    downscale_mode: str = "lanczos",
    upscale_small_images: bool = True,
    maintain_aspect_ratio: bool = True,
    fill_color: tuple[int, int, int] = (255, 255, 255),
    crop_position: str = "center",
    output_format: Optional[str] = None,
    output_quality: int = 95,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    **kwargs,
):
    """Resize images for Stable Diffusion training.

    Resize modes:
    - contain: Fit image within target size, maintaining aspect ratio (may have padding)
    - cover: Fill target size, maintaining aspect ratio (may crop)
    - stretch: Stretch to exact target size (changes aspect ratio)
    - resize_short: Resize so shortest side equals target
    - resize_long: Resize so longest side equals target
    - bucket: Resize to nearest bucket size (for aspect ratio bucketing)

    Args:
        dataset: The dataset to process.
        target_resolution: Target resolution (width and height for square, or base for buckets).
        max_resolution: Maximum resolution for any dimension.
        min_resolution: Minimum resolution for any dimension.
        resize_mode: How to handle aspect ratio when resizing.
        upscale_mode: Resampling filter for upscaling ("lanczos", "bilinear", "bicubic").
        downscale_mode: Resampling filter for downscaling.
        upscale_small_images: Whether to upscale images smaller than target.
        maintain_aspect_ratio: Whether to maintain original aspect ratio.
        fill_color: RGB color for padding in "contain" mode.
        crop_position: Where to crop in "cover" mode ("center", "top", "bottom", "left", "right").
        output_format: Output image format (None to keep original).
        output_quality: JPEG/WebP quality (1-100).
        output_dir: Directory for resized images (None to overwrite in place).
        overwrite: Whether to overwrite existing resized images.
        **kwargs: Additional arguments.

    Returns:
        The dataset with resized images.
    """
    from PIL import Image

    # Mapping from string to PIL resampling filter
    resample_filters = {
        "lanczos": Image.LANCZOS,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "nearest": Image.NEAREST,
    }

    upscale_filter = resample_filters.get(upscale_mode.lower(), Image.LANCZOS)
    downscale_filter = resample_filters.get(downscale_mode.lower(), Image.LANCZOS)

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
            if output_format:
                out_path = Path(output_dir) / f"{image_path.stem}.{output_format}"
            else:
                out_path = Path(output_dir) / image_path.name
        else:
            if output_format:
                out_path = image_path.with_suffix(f".{output_format}")
            else:
                out_path = image_path

        # Check if already processed
        if not overwrite and out_path.exists() and out_path != image_path:
            skipped_count += 1
            continue

        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                original_width, original_height = img.size
                original_aspect = original_width / original_height

                # Calculate target dimensions based on resize mode
                if resize_mode == "contain":
                    new_width, new_height = _calculate_contain_size(
                        original_width, original_height, target_resolution, target_resolution
                    )
                    resized = _resize_with_filter(
                        img, new_width, new_height, upscale_filter, downscale_filter, upscale_small_images
                    )
                    # Create canvas and paste
                    canvas = Image.new("RGB", (target_resolution, target_resolution), fill_color)
                    paste_x = (target_resolution - new_width) // 2
                    paste_y = (target_resolution - new_height) // 2
                    canvas.paste(resized, (paste_x, paste_y))
                    result = canvas

                elif resize_mode == "cover":
                    new_width, new_height = _calculate_cover_size(
                        original_width, original_height, target_resolution, target_resolution
                    )
                    resized = _resize_with_filter(
                        img, new_width, new_height, upscale_filter, downscale_filter, upscale_small_images
                    )
                    # Crop to target
                    result = _crop_to_size(resized, target_resolution, target_resolution, crop_position)

                elif resize_mode == "stretch":
                    result = img.resize((target_resolution, target_resolution), downscale_filter)

                elif resize_mode == "resize_short":
                    if original_width < original_height:
                        new_width = target_resolution
                        new_height = int(target_resolution / original_aspect)
                    else:
                        new_height = target_resolution
                        new_width = int(target_resolution * original_aspect)
                    result = _resize_with_filter(
                        img, new_width, new_height, upscale_filter, downscale_filter, upscale_small_images
                    )

                elif resize_mode == "resize_long":
                    if original_width > original_height:
                        new_width = target_resolution
                        new_height = int(target_resolution / original_aspect)
                    else:
                        new_height = target_resolution
                        new_width = int(target_resolution * original_aspect)
                    result = _resize_with_filter(
                        img, new_width, new_height, upscale_filter, downscale_filter, upscale_small_images
                    )

                elif resize_mode == "bucket":
                    # Find nearest bucket dimensions
                    new_width, new_height = _find_bucket_size(
                        original_width, original_height, target_resolution, max_resolution, min_resolution
                    )
                    result = _resize_with_filter(
                        img, new_width, new_height, upscale_filter, downscale_filter, upscale_small_images
                    )

                else:
                    raise ValueError(f"Unknown resize_mode: {resize_mode}")

                # Apply resolution limits
                if max_resolution:
                    if result.width > max_resolution or result.height > max_resolution:
                        scale = max_resolution / max(result.width, result.height)
                        new_w = int(result.width * scale)
                        new_h = int(result.height * scale)
                        result = result.resize((new_w, new_h), downscale_filter)

                if min_resolution:
                    if result.width < min_resolution or result.height < min_resolution:
                        if upscale_small_images:
                            scale = min_resolution / min(result.width, result.height)
                            new_w = int(result.width * scale)
                            new_h = int(result.height * scale)
                            result = result.resize((new_w, new_h), upscale_filter)

                # Save result
                save_kwargs = {}
                if output_format and output_format.lower() in ("jpg", "jpeg"):
                    save_kwargs["quality"] = output_quality
                    save_kwargs["optimize"] = True
                elif output_format and output_format.lower() == "webp":
                    save_kwargs["quality"] = output_quality
                elif output_format and output_format.lower() == "png":
                    save_kwargs["optimize"] = True

                result.save(out_path, **save_kwargs)
                processed_count += 1

                # Update dataset path if output location changed
                if out_path != image_path:
                    dataset.image_paths[i] = out_path

        except Exception as e:
            logger.warning("Failed to resize %s: %s", image_path, e)

    logger.info("Resized %d images, skipped %d", processed_count, skipped_count)

    return dataset


def _calculate_contain_size(
    orig_w: int, orig_h: int, target_w: int, target_h: int
) -> tuple[int, int]:
    """Calculate dimensions to fit image within target while maintaining aspect ratio."""
    aspect = orig_w / orig_h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        # Width limited
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        # Height limited
        new_h = target_h
        new_w = int(target_h * aspect)

    return new_w, new_h


def _calculate_cover_size(
    orig_w: int, orig_h: int, target_w: int, target_h: int
) -> tuple[int, int]:
    """Calculate dimensions to fill target while maintaining aspect ratio."""
    aspect = orig_w / orig_h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        # Height limited
        new_h = target_h
        new_w = int(target_h * aspect)
    else:
        # Width limited
        new_w = target_w
        new_h = int(target_w / aspect)

    return new_w, new_h


def _resize_with_filter(
    img, new_w: int, new_h: int, upscale_filter, downscale_filter, upscale_small: bool
):
    """Resize image using appropriate filter based on scale direction."""
    from PIL import Image

    orig_w, orig_h = img.size

    if new_w > orig_w or new_h > orig_h:
        # Upscaling
        if not upscale_small:
            return img
        return img.resize((new_w, new_h), upscale_filter)
    else:
        # Downscaling
        return img.resize((new_w, new_h), downscale_filter)


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

    return img.crop((left, top, left + target_w, top + target_h))


def _find_bucket_size(
    orig_w: int,
    orig_h: int,
    base_resolution: int,
    max_resolution: Optional[int] = None,
    min_resolution: Optional[int] = None,
) -> tuple[int, int]:
    """Find the nearest bucket size that maintains aspect ratio.

    Bucket sizes are multiples of 64 that maintain approximately
    the same total pixel count as base_resolution^2.
    """
    aspect = orig_w / orig_h
    target_pixels = base_resolution * base_resolution

    # Calculate dimensions that maintain aspect ratio and target pixel count
    # w * h = target_pixels, w/h = aspect
    # w = aspect * h, so aspect * h * h = target_pixels
    # h = sqrt(target_pixels / aspect)
    import math

    new_h = int(math.sqrt(target_pixels / aspect))
    new_w = int(new_h * aspect)

    # Round to nearest multiple of 64
    new_w = round(new_w / 64) * 64
    new_h = round(new_h / 64) * 64

    # Ensure minimums
    new_w = max(new_w, 64)
    new_h = max(new_h, 64)

    # Apply limits
    if max_resolution:
        new_w = min(new_w, max_resolution)
        new_h = min(new_h, max_resolution)
    if min_resolution:
        new_w = max(new_w, min_resolution)
        new_h = max(new_h, min_resolution)

    return new_w, new_h
