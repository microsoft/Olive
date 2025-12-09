# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared utilities for SD LoRA data preprocessing."""

from olive.common.utils import StrEnumBase


class ResizeMode(StrEnumBase):
    """How to fit images to target dimensions."""

    COVER = "cover"  # Resize to cover target, then crop
    CONTAIN = "contain"  # Resize to fit within target, then pad


class ResampleMode(StrEnumBase):
    """Resampling filter for image resizing."""

    LANCZOS = "lanczos"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"


class CropPosition(StrEnumBase):
    """Where to crop images in cover mode."""

    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


def get_resample_filter(resample_mode: ResampleMode):
    """Get PIL resampling filter from ResampleMode."""
    from PIL import Image

    resample_filters = {
        ResampleMode.LANCZOS: Image.LANCZOS,
        ResampleMode.BILINEAR: Image.BILINEAR,
        ResampleMode.BICUBIC: Image.BICUBIC,
        ResampleMode.NEAREST: Image.NEAREST,
    }
    return resample_filters.get(ResampleMode(resample_mode), Image.LANCZOS)


def calculate_contain_size(orig_w: int, orig_h: int, target_w: int, target_h: int) -> tuple[int, int]:
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


def calculate_cover_size(orig_w: int, orig_h: int, target_w: int, target_h: int) -> tuple[int, int]:
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


def crop_to_size(img, target_w: int, target_h: int, position: CropPosition = CropPosition.CENTER):
    """Crop image to target size."""
    w, h = img.size
    pos = CropPosition(position)

    if pos == CropPosition.CENTER:
        left = (w - target_w) // 2
        top = (h - target_h) // 2
    elif pos == CropPosition.TOP:
        left = (w - target_w) // 2
        top = 0
    elif pos == CropPosition.BOTTOM:
        left = (w - target_w) // 2
        top = h - target_h
    elif pos == CropPosition.LEFT:
        left = 0
        top = (h - target_h) // 2
    elif pos == CropPosition.RIGHT:
        left = w - target_w
        top = (h - target_h) // 2
    else:
        left = (w - target_w) // 2
        top = (h - target_h) // 2

    # Ensure bounds are valid
    left = max(0, left)
    top = max(0, top)

    return img.crop((left, top, left + target_w, top + target_h))


def resize_image(
    img,
    target_w: int,
    target_h: int,
    resize_mode: ResizeMode = ResizeMode.COVER,
    crop_position: CropPosition = CropPosition.CENTER,
    fill_color: tuple[int, int, int] = (255, 255, 255),
    resample_filter=None,
):
    """Resize image to target dimensions.

    Args:
        img: PIL Image to resize.
        target_w: Target width.
        target_h: Target height.
        resize_mode: How to fit image (COVER or CONTAIN).
        crop_position: Where to crop in cover mode.
        fill_color: Fill color for padding in contain mode.
        resample_filter: PIL resampling filter (default: LANCZOS).

    Returns:
        Resized PIL Image.

    """
    from PIL import Image

    if resample_filter is None:
        resample_filter = Image.LANCZOS

    orig_w, orig_h = img.size

    if ResizeMode(resize_mode) == ResizeMode.COVER:
        # Cover mode: resize to cover target, then crop
        new_w, new_h = calculate_cover_size(orig_w, orig_h, target_w, target_h)
        resized = img.resize((new_w, new_h), resample_filter)
        return crop_to_size(resized, target_w, target_h, crop_position)
    else:
        # Contain mode: resize to fit within target, then pad
        new_w, new_h = calculate_contain_size(orig_w, orig_h, target_w, target_h)
        resized = img.resize((new_w, new_h), resample_filter)
        canvas = Image.new("RGB", (target_w, target_h), fill_color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas
