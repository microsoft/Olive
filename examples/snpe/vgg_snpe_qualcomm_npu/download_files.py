# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from urllib import request

import numpy as np
from PIL import Image
from torchvision import transforms
from vgg_utils import get_directories


def preprocess_image(image):
    src_img = Image.open(image)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2:
        src_img = src_img.convert(mode="RGB")

    src_np = np.array(src_img)
    min = src_np.min()
    max = src_np.max()

    transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),
            transforms.Normalize(min, max - min),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transformed_img = transformations(src_img).numpy().astype(np.float32).transpose(1, 2, 0)
    return transformed_img


def main():
    # directories
    _, models_dir, data_dir = get_directories()
    # download onnx model
    onnx_model_url = "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx"
    request.urlretrieve(onnx_model_url, models_dir / "vgg.onnx")

    # download tuning data
    kitten_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    kitten_jpg = data_dir / "kitten.jpg"
    request.urlretrieve(kitten_url, kitten_jpg)

    # preprocess data
    kitten_raw = preprocess_image(kitten_jpg)
    kitten_raw.tofile(data_dir / "kitten.raw")

    # create input list
    with open(data_dir / "input_list.txt", "w") as f:
        f.write("kitten.raw\n")


if __name__ == "__main__":
    main()
