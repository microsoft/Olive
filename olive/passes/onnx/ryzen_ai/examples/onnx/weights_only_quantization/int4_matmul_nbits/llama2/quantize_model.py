#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import onnx
import argparse
from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer
from typing import Dict, List

def get_kv_cache_input_name(input_model_path: str) -> List[str]:
    kv_cache_input_name_list = []
    model = onnx.load(input_model_path)
    for input_tensor in model.graph.input:
        name = input_tensor.name
        if "past" in name:
            kv_cache_input_name_list.append(name)
    return kv_cache_input_name_list

def llama2_random_data_reader_input_data_range(input_model_path: str) -> Dict[str, List]:
    random_data_reader_input_data_range = {}
    random_data_reader_input_data_range['input_ids'] = [1, 32000]
    random_data_reader_input_data_range['position_ids'] = [1, 768]
    random_data_reader_input_data_range['attention_mask'] = [1, 1]
    kv_cache_input_name_list = get_kv_cache_input_name(input_model_path)
    for name in kv_cache_input_name_list:
        random_data_reader_input_data_range[name] = [-1, 1]
    return random_data_reader_input_data_range

def llama2_random_data_reader_input_shape(input_model_path: str) -> Dict[str, List]:
    random_data_reader_input_shape = {}
    random_data_reader_input_shape['input_ids'] = [1, 768]
    random_data_reader_input_shape['position_ids'] = [1, 768]
    random_data_reader_input_shape['attention_mask'] = [1, 768]
    kv_cache_input_name_list = get_kv_cache_input_name(input_model_path)
    for name in kv_cache_input_name_list:
        random_data_reader_input_shape[name] = [1, 32, 768, 128]
    return random_data_reader_input_shape

def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.use_external_data_format = True
    config_copy.extra_options['MatMulNBitsParams']['AccuracyLevel'] = 0
    if args.hqq:
        config_copy.extra_options['MatMulNBitsParams']['Algorithm'] = 'HQQ'
    config_copy.extra_options['QuantizeFP16'] = True
    config_copy.extra_options['UseFP32Scale'] = False
    config_copy.extra_options['UseRandomData'] = True
    random_data_reader_input_data_range = llama2_random_data_reader_input_data_range(input_model_path)
    config_copy.extra_options['RandomDataReaderInputDataRange'] = random_data_reader_input_data_range
    random_data_reader_input_shape = llama2_random_data_reader_input_shape(input_model_path)
    config_copy.extra_options['RandomDataReaderInputShape'] = random_data_reader_input_shape
    config = Config(global_quant_config=config_copy)
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
    parser.add_argument("--output_model_path",
                        help="Specify the path to save the quantized model",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="MATMUL_NBITS")
    parser.add_argument('--hqq', help="Whether to use HQQ algorithm", action='store_true')

    args = parser.parse_args()

    main(args)
