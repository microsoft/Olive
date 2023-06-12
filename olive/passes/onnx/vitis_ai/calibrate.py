#!/usr/bin/env python
# coding: utf-8
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from typing import Optional, Sequence

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibraterBase, CalibrationDataCollector, CalibrationDataReader
from onnxruntime.quantization.quant_utils import clone_model_with_shape_infer

from olive.passes.onnx.vitis_ai.quant_utils import (
    PowerOfTwoMethod,
    get_bound_and_scale,
    get_pos_min_mse,
    get_pos_overflow,
)


class PowOfTwoCalibrater(CalibraterBase):
    def __init__(
        self,
        model,
        op_types_to_calibrate: Optional[Sequence[str]] = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        method=PowerOfTwoMethod.NonOverflow,
        symmetric=True,
    ):
        """
        :param model: ONNX model to calibrate. It can be a ModelProto.
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        """
        super(PowOfTwoCalibrater, self).__init__(
            model, op_types_to_calibrate, augmented_model_path, use_external_data_format
        )
        self.intermediate_outputs = []
        self.model_original_outputs = set(output.name for output in self.model.graph.output)
        self.collector = None
        self.method = method
        self.symmetric = symmetric
        self.tensors_to_calibrate = None

    def augment_graph(self):
        """
        Make all quantization_candidates op type nodes as part of the graph output.
        :return: augmented ONNX model
        """
        model = clone_model_with_shape_infer(self.model)

        self.tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(model)
        for tensor in self.tensors_to_calibrate:
            if tensor not in self.model_original_outputs:
                model.graph.output.append(value_infos[tensor])

        onnx.save(
            model,
            self.augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )
        self.augment_model = model

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):

        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [self.infer_session.get_outputs()[i].name for i in range(len(self.intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output)) for intermediate_output in self.intermediate_outputs
        ]

        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict if i in self.tensors_to_calibrate)

        if not self.collector:
            self.collector = PowOfTwoCollector(method=self.method, symmetric=self.symmetric)
        self.collector.collect(clean_merged_dict)

        self.clear_collected_data()

    def compute_range(self):
        """
        Compute the min-max range of tensor
        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError("No collector created and can't generate calibration data.")

        return self.collector.compute_collection_result()


class PowOfTwoCollector(CalibrationDataCollector):
    """
    Collecting PowOfTwoCollector quantize for each tensor. Non overflow and min error method are supported.

    """

    def __init__(self, method=PowerOfTwoMethod.NonOverflow, symmetric=True, bit_width=8):
        self.name_to_arr = {}
        self.method = method
        self.symmetric = symmetric
        self.bit_width = bit_width

    def collect(self, name_to_arr):

        self.name_to_arr = name_to_arr

        return

    def compute_collection_result(self):
        if not self.name_to_arr or len(self.name_to_arr) == 0:
            raise ValueError("PowerOfTwoMethod has not been collected. Please run collect() first.")
        print("Finding optimal threshold for each tensor using {} algorithm ...".format(self.method))

        if self.method == PowerOfTwoMethod.NonOverflow:
            return self.compute_non_overflow()
        elif self.method == PowerOfTwoMethod.MinMSE:
            return self.compute_min_mse()
        else:
            raise ValueError("Only 'NonOverflow' or 'MinMSE' method are supported")

    def compute_non_overflow(self):
        thresholds_dict = {}
        for tensor, data_arr in self.name_to_arr.items():
            pos_list = []
            for d in data_arr:
                cur_pos = get_pos_overflow(d, bit_width=self.bit_width)
                pos_list.append(int(cur_pos))
            u, indices = np.unique(pos_list, return_inverse=True)
            pos = u[np.argmax(np.bincount(indices))]
            scale, lower_bound, upper_bound = get_bound_and_scale(pos)
            thresholds_dict[tensor] = (lower_bound, upper_bound)
        return thresholds_dict

    def compute_min_mse(self):
        thresholds_dict = {}
        for tensor, data_arr in self.name_to_arr.items():
            pos_list = []
            for d in data_arr:
                cur_pos = get_pos_min_mse(d, bit_width=self.bit_width)
                pos_list.append(int(cur_pos))
            u, indices = np.unique(pos_list, return_inverse=True)
            pos = u[np.argmax(np.bincount(indices))]
            scale, lower_bound, upper_bound = get_bound_and_scale(pos)
            thresholds_dict[tensor] = (lower_bound, upper_bound)
        return thresholds_dict


def create_calibrator_power_of_two(
    model,
    op_types_to_calibrate: Optional[Sequence[str]] = None,
    augmented_model_path="augmented_model.onnx",
    method=PowerOfTwoMethod.NonOverflow,
    use_external_data_format=False,
    extra_options={},
):

    calibrator = None

    # default settings for min-max algorithm
    method = method
    symmetric = True

    calibrator = PowOfTwoCalibrater(
        model,
        op_types_to_calibrate,
        augmented_model_path,
        use_external_data_format=use_external_data_format,
        method=method,
        symmetric=symmetric,
    )

    if calibrator:
        calibrator.augment_graph()
        calibrator.create_inference_session()
        return calibrator
