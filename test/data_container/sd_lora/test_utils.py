# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from PIL import Image

from olive.data.component.sd_lora.utils import (
    CropPosition,
    ResizeMode,
    calculate_contain_size,
    calculate_cover_size,
    crop_to_size,
    resize_image,
)


def test_calculate_contain_size():
    w, h = calculate_contain_size(800, 400, 512, 512)
    assert w == 512
    assert h == 256


def test_calculate_cover_size():
    w, h = calculate_cover_size(800, 400, 512, 512)
    assert h == 512
    assert w == 1024


def test_crop_to_size():
    img = Image.new("RGB", (600, 600), color="red")
    result = crop_to_size(img, 512, 512, CropPosition.CENTER)
    assert result.size == (512, 512)


def test_resize_image_cover():
    img = Image.new("RGB", (800, 600), color="blue")
    result = resize_image(img, 512, 512, resize_mode=ResizeMode.COVER)
    assert result.size == (512, 512)


def test_resize_image_contain():
    img = Image.new("RGB", (800, 400), color="blue")
    result = resize_image(img, 512, 512, resize_mode=ResizeMode.CONTAIN)
    assert result.size == (512, 512)
