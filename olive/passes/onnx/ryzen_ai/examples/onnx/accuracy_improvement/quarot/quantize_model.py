#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import argparse
from transformers import AutoTokenizer
import os
import numpy as np
from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer
from data_preparation import get_calib_dataloader
import quark
import re

kv_cache_name = [
    'past_key_values.0.key', 'past_key_values.0.value',
    'past_key_values.1.key', 'past_key_values.1.value',
    'past_key_values.2.key', 'past_key_values.2.value',
    'past_key_values.3.key', 'past_key_values.3.value',
    'past_key_values.4.key', 'past_key_values.4.value',
    'past_key_values.5.key', 'past_key_values.5.value',
    'past_key_values.6.key', 'past_key_values.6.value',
    'past_key_values.7.key', 'past_key_values.7.value',
    'past_key_values.8.key', 'past_key_values.8.value',
    'past_key_values.9.key', 'past_key_values.9.value',
    'past_key_values.10.key', 'past_key_values.10.value',
    'past_key_values.11.key', 'past_key_values.11.value',
    'past_key_values.12.key', 'past_key_values.12.value',
    'past_key_values.13.key', 'past_key_values.13.value',
    'past_key_values.14.key', 'past_key_values.14.value',
    'past_key_values.15.key', 'past_key_values.15.value',
    'past_key_values.16.key', 'past_key_values.16.value',
    'past_key_values.17.key', 'past_key_values.17.value',
    'past_key_values.18.key', 'past_key_values.18.value',
    'past_key_values.19.key', 'past_key_values.19.value',
    'past_key_values.20.key', 'past_key_values.20.value',
    'past_key_values.21.key', 'past_key_values.21.value',
    'past_key_values.22.key', 'past_key_values.22.value',
    'past_key_values.23.key', 'past_key_values.23.value',
    'past_key_values.24.key', 'past_key_values.24.value',
    'past_key_values.25.key', 'past_key_values.25.value',
    'past_key_values.26.key', 'past_key_values.26.value',
    'past_key_values.27.key', 'past_key_values.27.value',
    'past_key_values.28.key', 'past_key_values.28.value',
    'past_key_values.29.key', 'past_key_values.29.value',
    'past_key_values.30.key', 'past_key_values.30.value',
    'past_key_values.31.key', 'past_key_values.31.value'
]

def is_number(s):
    return bool(re.match(r'^-?\d+(?:\.\d+)?$', s))

class CalibrationDataReader:

    def __init__(self, dataloader, hidden_size, num_head):
        super().__init__()
        self.iterator = iter(dataloader)
        self.hidden_size = hidden_size
        self.num_head = num_head

    def get_next(self) -> dict:
        try:
            inputs = next(self.iterator)[0]
            input_dict = {}
            input_dict["input_ids"] = inputs.numpy().reshape(1, -1)  # 1, seq_len
            # input_dict["position_ids"] = np.expand_dims(np.arange(inputs.shape[0], dtype=np.int64), axis=0)  # Uesd only optimum model
            input_dict["attention_mask"] = np.ones_like(inputs.numpy().reshape(
                1, -1))

            past_seq_len = 1  # For fake usage, this can be set as 1
            if self.num_head == 8:
                cache_shape = [1, 8, past_seq_len, self.hidden_size // 32]  # Shape like [batch, num_head, past_seq_len, hidd_dim // num_head]
            elif self.num_head == 32:
                cache_shape = [1, 32, past_seq_len, self.hidden_size // 32]
            else:
                raise NotImplementedError
            for name in kv_cache_name:
                input_dict[name] = np.ones(cache_shape).astype(np.float32)  # Used only oga model  # no effect
            return input_dict
        except StopIteration:
            return None

def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        os.path.dirname(args.input_model_path),
        do_lower_case=False,
        cache_dir=None,
    )
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # `dr` (Data Reader) is an instance of DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    calib_dataloader = get_calib_dataloader(dataset_name='pileval',
                                            tokenizer=tokenizer,
                                            batch_size=args.batch_size,
                                            seqlen=512,
                                            device=args.device,
                                            num_calib_data=args.num_calib_data
                                            )
    calib_dataloader = CalibrationDataReader(calib_dataloader, hidden_size=args.hidden_size, num_head=args.num_head)
    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.optimize_model = False
    config_copy.extra_options['RemoveInputInit'] = False
    config_copy.extra_options['SimplifyModel'] = False
    config_copy.extra_options['CopySharedInit'] = None
    config_copy.extra_options['OpTypesToExcludeOutputQuantization'] = ['MatMul', 'Gemm']
    config_copy.use_external_data_format = True

    config_copy.extra_options["CalibMovingAverage"] = args.use_moving_average

    config_copy.include_sq = args.include_sq
    if args.include_sq:
        config_copy.extra_options['SmoothAlpha'] = args.sq_alpha

    config_copy.include_rotation = args.include_rotation
    if args.include_rotation:
        config_copy.extra_options['RMatrixDim'] = args.hidden_size
        config_copy.extra_options["RConfigPath"] = args.r_config_path
        config_copy.extra_options["UseRandomHad"] = args.use_random_had


    config_copy.extra_options["ActivationSymmetric"] = True  # Must activate
    config_copy.extra_options["MatMulConstBOnly"] = True  # Must activate
    if "percentile" in args.calib_method:
        percentile_ratio = args.calib_method.split('+')[-1]  # get num
        assert is_number(percentile_ratio)

        percentile_ratio = float(percentile_ratio)
        config_copy.calibrate_method = quark.onnx.CalibrationMethod.Percentile  # Suggest to activate
        config_copy.extra_options["Percentile"] = percentile_ratio  # Suggest to activate
        print(f"Use percentile calibration with ratio {percentile_ratio}.")


    config = Config(global_quant_config=config_copy)
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, output_model_path, calib_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
    parser.add_argument("--output_model_path",
                        help="Specify the path to save the quantized model",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--use_moving_average", action='store_true', help="Using EMA when calibrating")
    parser.add_argument("--include_sq", action='store_true', help="Optimize the models using SmoothQuant")
    parser.add_argument("--sq_alpha", help="Define the alpha for smooth quant", type=float, default=0.5)
    parser.add_argument("--include_rotation", action='store_true', help="Optimize the models using rotation")
    parser.add_argument("--use_random_had", action='store_true', help="Activate that randomly generate Hadamard matrix")
    parser.add_argument("--r_config_path",
                        help="Specify the path to load the rotation configuration",
                        type=str,
                        default='',
                        required=False)

    parser.add_argument("--calib_method",
                        help="Specify the calibration method",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--hidden_size", help="Dim of the R1 rotation tensor", type=int, default=4096)
    parser.add_argument("--num_head", help="Number of heads in the QKV proj", type=int, default=32)
    parser.add_argument("--num_calib_data", help="Number of samples for calibration", type=int, default=1)
    parser.add_argument("--batch_size", help="Batch size for calibration", type=int, default=1)
    parser.add_argument("--workers", help="Number of worker threads used during calib data loading.", type=int, default=1)
    parser.add_argument("--device",
                        help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'",
                        type=str,
                        default="cpu")
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="INT8_TRANSFORMER_DEFAULT")

    args = parser.parse_args()

    main(args)
