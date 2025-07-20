#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import huggingface_hub

imagenet_classes_path = huggingface_hub.hf_hub_download("fxmarty/imagenet-classes", filename="imagenet_classes.json")

with open(imagenet_classes_path) as f:
    IMAGENET2012_CLASSES = json.load(f)
