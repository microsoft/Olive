#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import argparse
import numpy
from PIL import Image

import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader

from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer


def _preprocess_images(images_folder: str,
                       height: int,
                       width: int,
                       size_limit=0,
                       batch_size=100):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_path = os.listdir(images_folder)
    image_names = []
    for image_dir in image_path:
        image_name = os.listdir(os.path.join(images_folder, image_dir))
        image_names.append(os.path.join(image_dir, image_name[0]))
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    batch_data = []
    for index, image_name in enumerate(batch_filenames):
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        image_array = numpy.array(pillow_img) / 255.0
        mean = numpy.array([0.485, 0.456, 0.406])
        image_array = (image_array - mean)
        std = numpy.array([0.229, 0.224, 0.225])
        nchw_data = image_array / std
        nchw_data = nchw_data.transpose((2, 0, 1))
        nchw_data = numpy.expand_dims(nchw_data, axis=0)
        nchw_data = nchw_data.astype(numpy.float32)
        unconcatenated_batch_data.append(nchw_data)

        if (index + 1) % batch_size == 0:
            one_batch_data = numpy.concatenate(unconcatenated_batch_data,
                                               axis=0)
            unconcatenated_batch_data.clear()
            batch_data.append(one_batch_data)

    return batch_data


class ImageDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder: str, model_path: str, data_size: int, batch_size: int):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(calibration_image_folder,
                                                 height, width, data_size, batch_size)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{
                self.input_name: nhwc_data
            } for nhwc_data in self.nhwc_data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

    def reset(self):
        self.enum_data = None


def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = args.calibration_dataset_path

    # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    if calibration_dataset_path == '':
        dr = None
    else:
        dr = ImageDataReader(calibration_dataset_path, input_model_path, args.num_calib_data, args.batch_size)

    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config = Config(global_quant_config=quant_config)
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, output_model_path, dr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model_path",
                        help="Specify the input model to be quantized",
                        required=True)
    parser.add_argument("--output_model_path",
                        help="Specify the path to save the quantized model",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--calibration_dataset_path",
                        help="The path of the dataset for calibration",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--num_calib_data", help="Number of samples for calibration", type=int, default=1000)
    parser.add_argument("--batch_size", help="Batch size for calibration", type=int, default=1)
    parser.add_argument("--device", help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'", type=str, default="cpu")
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="XINT8")

    args = parser.parse_args()

    main(args)
