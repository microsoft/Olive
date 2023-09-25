# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib import request

import numpy as np
from PIL import Image
from torchvision import transforms

from olive.common.utils import run_subprocess


def get_directories():
    current_dir = Path(__file__).resolve().parent

    # models directory for resnet sample
    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # data directory for resnet sample
    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, models_dir, data_dir


def download_model():
    # directories
    _, models_dir, _ = get_directories()

    # temporary directory
    tempdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="olive_tmp_")
    stage_dir = Path(tempdir.name)

    # download onnx model
    mobilenet_name = "mobilenetv2-12"
    mobilenet_archive_file = "mobilenetv2-12.tar.gz"
    mobilenet_archive_url = (
        "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/" + mobilenet_archive_file
    )
    mobilenet_archive_path = stage_dir / mobilenet_archive_file
    request.urlretrieve(mobilenet_archive_url, mobilenet_archive_path)

    with tarfile.open(mobilenet_archive_path) as tar_ref:
        tar_ref.extractall(stage_dir)

    original_model_path = stage_dir / mobilenet_name / f"{mobilenet_name}.onnx"
    model_path = models_dir / f"{mobilenet_name}.onnx"
    if model_path.exists():
        model_path.unlink()
    run_subprocess(
        f"python -m onnxruntime.tools.make_dynamic_shape_fixed {original_model_path} {model_path} --dim_param"
        " batch_size --dim_value 1"
    )


def download_eval_data():
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

    data_path = data_dir / "eval"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # preprocess images
    inputs = []
    for jpeg in jpegs[:20]:
        out = preprocess_image(jpeg)
        inputs.append(out)
    inputs = np.stack(inputs)
    np.save(data_path / "data.npy", inputs)

    # create labels file
    labels = np.arange(0, 20)
    np.save(data_path / "labels.npy", labels)


def preprocess_image(image):
    src_img = Image.open(image)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2:
        src_img = src_img.convert(mode="RGB")

    src_np = np.array(src_img)
    min_val = src_np.min()
    max_val = src_np.max()

    transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),
            transforms.Normalize(min_val, max_val - min_val),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transformations(src_img).numpy().astype(np.float32)


def create_quant_data():
    # directories
    _, _, data_dir = get_directories()

    # N X C X H X W
    input_shape = [50, 3, 224, 224]

    # create random data
    # rng
    rng = np.random.default_rng(0)
    data = rng.uniform(0.0, 1.0, input_shape).astype(np.float32)

    # normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
    normalized_data = (data - mean) / std

    # save data
    data_path = data_dir / "quant"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    np.save(data_path / "data.npy", normalized_data)


def main():
    download_model()
    download_eval_data()
    create_quant_data()


if __name__ == "__main__":
    main()
