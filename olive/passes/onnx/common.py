# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Union

import onnx

from olive.common.onnx_utils import model_proto_to_file
from olive.model import ONNXModelHandler
from olive.passes.pass_config import PassConfigParam
from olive.resource_path import LocalFile, LocalFolder

logger = logging.getLogger(__name__)


def get_external_data_config():
    return {
        "save_as_external_data": PassConfigParam(
            type_=bool,
            default_value=False,
            description=(
                "Serializes tensor data to separate files instead of directly in the ONNX file. Large models (>2GB)"
                " may be forced to save external data regardless of the value of this parameter."
            ),
        ),
        "all_tensors_to_one_file": PassConfigParam(
            type_=bool,
            default_value=True,
            description=(
                "Effective only if save_as_external_data is True. If true, save all tensors to one external file"
                " specified by 'external_data_name'. If false, save each tensor to a file named with the tensor name."
            ),
        ),
        "external_data_name": PassConfigParam(
            type_=str,
            default_value=None,
            description=(
                "Effective only if all_tensors_to_one_file is True and save_as_external_data is True. If not specified,"
                " the external data file will be named with <model_path_name>.data"
            ),
        ),
        "size_threshold": PassConfigParam(
            type_=int,
            default_value=1024,
            description=(
                "Effective only if save_as_external_data is True. Threshold for size of data. Only when tensor's data"
                " is >= the size_threshold it will be converted to external data. To convert every tensor with raw data"
                " to external data set size_threshold=0."
            ),
        ),
        "convert_attribute": PassConfigParam(
            type_=bool,
            default_value=False,
            description=(
                "Effective only if save_as_external_data is True. If true, convert all tensors to external data If"
                " false, convert only non-attribute tensors to external data"
            ),
        ),
    }


def model_proto_to_olive_model(
    model_proto: onnx.ModelProto,
    output_model_path: Union[str, Path],
    configs: dict,
    check_model: bool = False,
    enable_fast_mode: bool = False,
) -> ONNXModelHandler:
    """Save the ONNX model to the specified path and return the ONNXModelHandler.

    :param model_proto: The ONNX model to save.
    :param output_model_path: The path to save the ONNX model to.
    :param configs: The external data configuration. Must be a dictionary with keys
        "save_as_external_data", "all_tensors_to_one_file", and "external_data_name".
    :param check_model: If True, run onnx.checker.check_model on the model before returning.
    :param enable_fast_mode: If True, skip saving the model to file
        and return the ONNXModelHandler with in-memory loaded model directly.

    :return: The ONNXModelHandler.
    """
    config_keys = [
        "save_as_external_data",
        "all_tensors_to_one_file",
        "external_data_name",
        "size_threshold",
        "convert_attribute",
    ]
    external_data_config = {k: configs[k] for k in config_keys if k in configs}

    if enable_fast_mode:
        logger.debug("Enabling fast mode. Skipping saving model to file.")
        if model_proto is not None:
            return ONNXModelHandler(model=model_proto, external_data_config=external_data_config)
        logger.error("Error enabling fast mode: model_proto is None. Defaulting to saving model to file.")

    logger.debug("Fast mode is disabled. Saving model to file.")
    has_external_data = model_proto_to_file(model_proto, output_model_path, **external_data_config)
    if has_external_data:
        model_path = LocalFolder({"path": Path(output_model_path).parent})

        onnx_file_name = Path(output_model_path).name
    else:
        model_path = LocalFile({"path": output_model_path})
        onnx_file_name = None

    olive_model = ONNXModelHandler(model_path=model_path, onnx_file_name=onnx_file_name)

    if check_model:
        onnx.checker.check_model(olive_model.model_path)
    return olive_model
