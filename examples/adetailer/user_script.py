# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from pathlib import Path
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch import from_numpy
from torch.utils.data import Dataset

from olive.data.registry import Registry

logger = getLogger(__name__)


class ImagenetDataset(Dataset):
    def __init__(self):
        img_path = r"C:\Users\fangyangci\Downloads\WIDER_val\WIDER_val\images\0--Parade\0_Parade_Parade_0_459.jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 预处理图像
        input_size = (640, 640)
        resized_img = cv2.resize(img, input_size)
        input_img = np.transpose(resized_img, (2, 0, 1))
        self.input_img = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"images": self.input_img}


# @Registry.register_post_process()
# def imagenet_post_fun(output):
#     return output.argmax(axis=1)


@Registry.register_dataset()
def face_dataset(**kwargs):
    return ImagenetDataset()
