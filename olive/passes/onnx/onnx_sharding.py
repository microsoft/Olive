# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import numpy
import onnx
import os
import pprint

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from olive.model import ONNXModel, DistributedOnnxModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from onnxruntime.transformers.onnx_model import OnnxModel
from typing import Any, Dict

import data_model
from onnx_distributed import shard_onnx_model, generate_sharded_model


def run(input_model_path: str,
        sharding_spec_path: str,
        hardware_spec_path: str,
        output_dirpath: str):
    # load hardware config and sharding spec
    with open(hardware_spec_path, 'r') as f:
        hardware = data_model.Hardware.from_json(json.load(f))
    with open(sharding_spec_path, 'r') as f:
        sharding_spec = json.load(f)

    model = onnx.load(input_model_path)

    sharding_metadata = shard_onnx_model(
        model, hardware, None, False, sharding_spec)
    generate_sharded_model(model, sharding_metadata,
                           hardware, output_dirpath, False)
    return [os.path.join(output_dirpath, f'shard_{i}.onnx') for i in range(hardware.device_count)]


class OnnxModelSharding(Pass):
    """
    Horizontal shard onnx model into multiple shard.
    """

    @staticmethod
    def _default_config() -> Dict[str, Dict[str, Any]]:
        return {
            "sharding_spec_path": PassConfigParam(
                type_=str,
                required=True,
                description=(
                    "Path to the sharding spec."
                ),
            ),
            "hardware_spec_path": PassConfigParam(
                type_=str,
                required=True,
                description=(
                    "Path to the hardware spec."
                ),
            ),
        }

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> DistributedOnnxModel:
        output_filepaths = run(model.model_path,
                               config['sharding_spec_path'],
                               config['hardware_spec_path'],
                               output_model_path)
        return DistributedOnnxModel(output_filepaths, model.name)
