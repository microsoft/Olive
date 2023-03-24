# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tarfile
import tempfile
from pathlib import Path
from urllib import request

import numpy as np
from inception_utils import get_directories
from PIL import Image
from torchvision import transforms

from olive.common.utils import run_subprocess
from olive.snpe.utils.input_list import create_input_list


def download_model():
    # directories
    _, models_dir, data_dir = get_directories()

    # temporary directory
    tempdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="olive_tmp_")
    stage_dir = Path(tempdir.name)

    # download onnx model
    inception_v3_archive_file = "inception_v3_2016_08_28_frozen.pb.tar.gz"
    inception_v3_archive_url = (
        "https://storage.googleapis.com/download.tensorflow.org/models/" + inception_v3_archive_file
    )
    inception_v3_archive_path = stage_dir / inception_v3_archive_file
    request.urlretrieve(inception_v3_archive_url, inception_v3_archive_path)

    with tarfile.open(inception_v3_archive_path) as tar_ref:
        tar_ref.extractall(stage_dir)

    model_path = models_dir / "inception_v3.pb"
    if model_path.exists():
        model_path.unlink()
    Path(stage_dir / inception_v3_archive_file.strip(".tar.gz")).rename(model_path)


def download_data():
    # directories
    _, _, data_dir = get_directories()

    # temporary directory
    tempdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="olive_tmp_")
    stage_dir = Path(tempdir.name)

    # download evaluation data
    github_source = "https://github.com/EliSchwartz/imagenet-sample-images.git"
    run_subprocess(f"git clone {github_source} {stage_dir}")

    # sort jpegs
    jpegs = list(stage_dir.glob("*.JPEG"))
    jpegs.sort()

    input_data_path = data_dir / "input"
    input_data_path.mkdir(parents=True, exist_ok=True)
    for jpeg in jpegs[:20]:
        out = preprocess_image(jpeg)
        out.tofile((input_data_path / jpeg.name).with_suffix(".raw"))

    # create input list
    create_input_list(str(data_dir), ["input"])

    # create labels file
    labels = np.arange(1, 21)
    np.save(data_dir / "labels.npy", labels)


def preprocess_image(image):
    src_img = Image.open(image)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2:
        src_img = src_img.convert(mode="RGB")
    # center crop to square
    width, height = src_img.size
    short_dim = min(height, width)

    transformations = transforms.Compose(
        [
            transforms.CenterCrop(short_dim),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),
            transforms.Normalize(128, 128),
        ]
    )
    transformed_img = transformations(src_img).numpy().astype(np.float32).transpose(1, 2, 0)
    return transformed_img


def main():
    download_model()
    download_data()


if __name__ == "__main__":
    main()
