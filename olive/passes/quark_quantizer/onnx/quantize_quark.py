#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from argparse import Namespace

from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.config import QConfig

from olive.passes.quark_quantizer.onnx.configuration_preparation import (
    get_algo_config,
    get_global_config,
)


def run_quark_quantization(args: Namespace) -> None:
    input_model_path = args.model_input
    output_model_path = args.model_output
    calibration_data_reader = args.calibration_data_reader

    global_config = get_global_config(args.global_config)
    algo_config = get_algo_config(args.algo_config)
    quant_config = QConfig(
        global_config=global_config,
        specific_layer_config=args.specific_layer_config,
        layer_type_config=args.layer_type_config,
        exclude=args.exclude,
        algo_config=algo_config,
        **args.extra_options,
    )

    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader)
