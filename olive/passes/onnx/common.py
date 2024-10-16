# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union

import onnx

from olive.model import ONNXModelHandler
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import PassConfigParam
from olive.resource_path import LocalFile, LocalFolder

logger = logging.getLogger(__name__)


def get_external_data_config() -> Dict[str, PassConfigParam]:
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


def model_proto_to_file(
    model: onnx.ModelProto,
    output_path: Union[str, Path],
    save_as_external_data: Optional[bool] = False,
    all_tensors_to_one_file: Optional[bool] = True,
    external_data_name: Optional[Union[str, Path]] = None,
    size_threshold: Optional[int] = 1024,
    convert_attribute: Optional[bool] = False,
) -> bool:
    """Save the ONNX model to the specified path.

    :param model: The ONNX model to save.
    :param output_path: The path to save the ONNX model to.
    :param save_as_external_data: If True, save tensor data to separate files instead of directly in the ONNX file.
        Large models (>2GB) may be forced to save external data regardless of the value of this parameter.
    :param all_tensors_to_one_file: Effective only if save_as_external_data is True. If True, save all tensors to one
        external file specified by 'external_data_name'. If False, save each tensor to a file named with the tensor
        name.
    :param external_data_name: Effective only if all_tensors_to_one_file is True and save_as_external_data is True.
        If not specified, the external data file will be named with <model_path_name>.data

    :return: True if the model has external data, False otherwise.
    """
    output_path = Path(output_path)
    if output_path.exists():
        logger.info("Deleting existing onnx file: %s", output_path)
        output_path.unlink()

    # parent directory of .onnx file
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model_size = model.ByteSize()
    # model size for large models might be negative (overflow?) on Windows
    # see https://github.com/onnx/onnx/issues/5861
    if not save_as_external_data and (model_size <= 0 or model_size >= onnx.checker.MAXIMUM_PROTOBUF):
        save_as_external_data = True
        logger.debug(
            "Model is too large to save as a single file but 'save_as_external_data' is False. Saving tensors as"
            " external data, regardless."
        )

    if not save_as_external_data:
        # save model
        onnx.save_model(model, str(output_path))
        return False

    # location for external data
    external_data_path = output_dir / (external_data_name if external_data_name else f"{output_path.name}.data")
    location = external_data_path.name if all_tensors_to_one_file else None

    if all_tensors_to_one_file:
        if external_data_path.exists():
            # Delete the external data file. Otherwise, data will be appended to existing file.
            logger.info("Deleting existing external data file: %s", external_data_path)
            external_data_path.unlink()
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory ({output_dir}) for external data is not empty.")

    # save model
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=location,
        size_threshold=size_threshold,
        convert_attribute=convert_attribute,
    )
    return True


def model_proto_to_olive_model(
    model_proto: onnx.ModelProto,
    output_model_path: Union[str, Path],
    external_data_config: dict,
    check_model: bool = False,
    external_initializers_file_name: Optional[str] = None,
    constant_inputs_file_name: Optional[str] = None,
) -> ONNXModelHandler:
    """Save the ONNX model to the specified path and return the ONNXModelHandler.

    :param model_proto: The ONNX model to save.
    :param output_model_path: The path to save the ONNX model to.
    :param external_data_config: The external data configuration. Must be a dictionary with keys
        "save_as_external_data", "all_tensors_to_one_file", and "external_data_name".
    :param check_model: If True, run onnx.checker.check_model on the model before returning.
    :param external_initializers_file_name: The name of the external initializers file.
    :param constant_inputs_file_name: The name of the constant inputs file.

    :return: The ONNXModelHandler.
    """
    config_keys = [
        "save_as_external_data",
        "all_tensors_to_one_file",
        "external_data_name",
        "size_threshold",
        "convert_attribute",
    ]
    has_external_data = model_proto_to_file(
        model_proto, output_model_path, **{k: external_data_config[k] for k in config_keys if k in external_data_config}
    )
    if has_external_data or external_initializers_file_name or constant_inputs_file_name:
        model_path = LocalFolder({"path": Path(output_model_path).parent})

        onnx_file_name = Path(output_model_path).name
    else:
        model_path = LocalFile({"path": output_model_path})
        onnx_file_name = None

    olive_model = ONNXModelHandler(
        model_path=model_path,
        onnx_file_name=onnx_file_name,
        external_initializers_file_name=external_initializers_file_name,
        constant_inputs_file_name=constant_inputs_file_name,
    )

    if check_model:
        onnx.checker.check_model(olive_model.model_path)

    return olive_model


LORA_NAME_PATTERNS = [
    f".*[./]{name}[./]{matmul}$"
    for name in ["default_0", "default_0_1", "default", "default_1", "lora_A", "lora_B"]
    for matmul in ["MatMul", "MatMul_Q4"]
]


# TODO(jambayk): considering matching by subgraph pattern, more involved but more reliable
def model_has_adapters(model_path: Union[str, Path]) -> bool:
    """Check if the model has adapters.

    :param model_path: The path to the model.
    :return: True if the model has adapters, False otherwise.
    """
    dag = OnnxDAG(onnx.load(model_path, load_external_data=False))
    for node_name in dag.get_node_names():
        op_type = dag.get_node_op_type(node_name)
        if op_type in {"MatMul", "MatMulNBits"} and any(re.match(pattern, node_name) for pattern in LORA_NAME_PATTERNS):
            return True
    return False
