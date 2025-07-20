#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch
import argparse


def main(args):
    model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

    model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, 640, 640])

    dummy_input = torch.rand(1, 3, 640, 640)

    torch.onnx.export(model, dummy_input, args.output_model_path, opset_version=18)

    # todo maybe need optimize the onnx model using simplifer
    print("Export onnx model finished!")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_model_path",
                        help="Specify the input model to be quantized",
                        default="yolo_nas_s.onnx",
                        required=False)
    args = parser.parse_args()

    main(args)
