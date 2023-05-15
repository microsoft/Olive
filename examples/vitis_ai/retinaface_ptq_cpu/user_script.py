#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import numpy as np
import argparse
from PIL import Image
import onnxruntime
import cv2
from onnxruntime.quantization.calibrate import CalibrationDataReader
from olive.constants import Framework
from olive.evaluator.accuracy import AccuracyScore
from olive.model import OliveModel


class ResNetDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder, augmented_model_path):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(
                self.augmented_model_path, providers=['CPUExecutionProvider'])
            (_, _, height, width) = session.get_inputs()[0].shape
            nhwc_data_list = self.preprocess_func(self.image_folder,
                                                  height,
                                                  width,
                                                  size_limit=0)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{
                input_name: nhwc_data
            } for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

    def preprocess_func(self, data_dir, height, width, size_limit=0):
        '''
        Loads a batch of images and preprocess them
        parameter data_dir: path to folder storing images
        parameter height: image height in pixels
        parameter width: image width in pixels
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        '''
        image_names = os.listdir(data_dir)
        if size_limit > 0 and len(image_names) >= size_limit:
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names
        unconcatenated_batch_data = []

        for image_name in batch_filenames:
            image_filepath = data_dir + '/' + image_name
            pillow_img = Image.new("RGB", (width, height))
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            input_data = np.float32(pillow_img).astype(np.float32)
            input_data = input_data.transpose(2, 0, 1)
            input_data /= 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None,
                                                                     None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None,
                                                                    None]
            input_data = input_data - mean
            input_data /= std
            nchw_data = np.expand_dims(input_data, axis=0)
            unconcatenated_batch_data.append(nchw_data)
        batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data,
                                                   axis=0),
                                    axis=0)
        return batch_data


class RetinaFaceDataReader(ResNetDataReader):

    def preprocess_func(self, data_dir, height, width, size_limit=0):
        '''
        Loads a batch of images and preprocess them
        parameter data_dir: path to folder storing images
        parameter height: image height in pixels
        parameter width: image width in pixels
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        '''
        image_names = os.listdir(data_dir)
        if size_limit > 0 and len(image_names) >= size_limit:
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names
        unconcatenated_batch_data = []

        for image_name in batch_filenames:
            image_filepath = data_dir + '/' + image_name
            pillow_img = Image.new("RGB", (width, height))
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            input_data = np.float32(pillow_img).astype(np.float32)[:, :, ::-1]
            input_data -= (104, 117, 123)
            input_data = input_data.transpose(2, 0, 1)
            nchw_data = np.expand_dims(input_data, axis=0)
            unconcatenated_batch_data.append(nchw_data)
        batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data,
                                                   axis=0),
                                    axis=0)
        return batch_data

def resnet_calibration_reader(data_dir, model_path, batch_size=1):
    return ResNetDataReader(data_dir, model_path)

def retinaface_calibration_reader(data_dir, model_path, batch_size=1):
    return RetinaFaceDataReader(data_dir, model_path)
