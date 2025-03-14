# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import onnx
from onnx import external_data_helper

from olive.common.utils import hardlink_copy_file
from olive.model import ONNXModelHandler
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam
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
        logger.debug("Deleting existing onnx file: %s", output_path)
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
    external_data_config: Union[Dict[str, Any], Type[BasePassConfig]],
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
    if not isinstance(external_data_config, dict):
        external_data_config = external_data_config.dict()
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


def get_external_data_file_names(model_path: Union[str, Path]) -> List[str]:
    """Get the external data file names from the model.

    :param model_path: Path to the model file.
    :return: List of external data file names.
    """
    file_names = set()
    for tensor in external_data_helper._get_all_tensors(  # pylint: disable=W0212
        onnx.load(model_path, load_external_data=False)
    ):
        if external_data_helper.uses_external_data(tensor):
            file_names.add(external_data_helper.ExternalDataInfo(tensor).location)
    return list(file_names)


def change_external_data_location(model_proto: onnx.ModelProto, new_location: str):
    """Change the external data location in the model.

    :param model_proto: The model proto to modify.
    :param new_location: The new location for the external data.
    """
    for tensor in external_data_helper._get_all_tensors(model_proto):  # pylint: disable=W0212
        if external_data_helper.uses_external_data(tensor):
            # set dummy raw_data since set_external_data expected the field
            tensor.raw_data = b""
            info = external_data_helper.ExternalDataInfo(tensor)
            external_data_helper.set_external_data(
                tensor, new_location, offset=info.offset, length=info.length, checksum=info.checksum
            )


def get_context_bin_file_names(model_path: Union[str, Path]) -> List[str]:
    """Get the context binary file names from the model.

    :param model_path: Path to the model file.
    :return: List of context binary file names.
    """
    file_names = set()
    for node in onnx.load(model_path, load_external_data=False).graph.node:
        if node.op_type == "EPContext":
            for attr in node.attribute:
                if attr.name == "ep_cache_context":
                    try:
                        file_names.add(attr.s.decode("utf-8"))
                    except UnicodeDecodeError:
                        # embedded context binary file
                        continue
    return list(file_names)


def copy_context_bin_files(
    model_path: Union[str, Path],
    model_dir: Union[str, Path],
    saved_cb_files: Optional[Dict[str, str]] = None,
) -> bool:
    """Copy the context binary files to the model directory.

    :param model_path: Path to the original model file.
    :param model_dir: Directory to save the copied context binary files.
    :param saved_cb_files: A dictionary of original file paths to new file names for context binary files.
    :return: True if the model has context binary files, False otherwise.
    """
    saved_cb_files = {} if saved_cb_files is None else saved_cb_files

    model_path = Path(model_path).resolve()
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # TODO(jambayk): consider renaming cb files
    cb_file_names = get_context_bin_file_names(model_path)
    for cb_file_name in cb_file_names:
        cb_file_path = str(model_path.parent / cb_file_name)
        if cb_file_path in saved_cb_files:
            continue
        elif cb_file_name in saved_cb_files.values():
            raise RuntimeError(
                f"Context binary file name {cb_file_name} already exists in {model_dir}. Please rename the file."
            )

        hardlink_copy_file(cb_file_path, model_dir / cb_file_name)
        saved_cb_files[cb_file_path] = cb_file_name

    return bool(cb_file_names)


def resave_model(
    model_path: Union[str, Path],
    new_model_path: Union[str, Path],
    force_external_data: bool = False,
    saved_external_files: Optional[Dict[str, str]] = None,
) -> bool:
    """Resave the model along with external data files.

    :param model_path: Path to the original model file.
    :param new_model_path: Path to the new model file.
    :param force_external_data: If True, force the model to be saved with external data.
    :param saved_external_files: A dictionary of original file paths to new file names for external data files.
        Reuse the same file name if the the original file path is already in the dictionary.
        Else, the new file name will be <new_model_path>.data and this dictionary will be updated with the new
        file name.
    :return: True if the model has external data, False otherwise.
    """
    saved_external_files = {} if saved_external_files is None else saved_external_files

    model_path = Path(model_path).resolve()
    new_model_path = Path(new_model_path).resolve()
    assert new_model_path.suffix == ".onnx", "new_model_path must be .onnx file"
    new_model_path.parent.mkdir(parents=True, exist_ok=True)

    # copy over context binary files
    has_cb_files = copy_context_bin_files(model_path, new_model_path.parent, saved_cb_files=saved_external_files)

    external_file_names = get_external_data_file_names(model_path)

    if not external_file_names:
        if force_external_data:
            # save the model with single external data file
            model_proto_to_file(onnx.load(model_path), new_model_path, {"save_as_external_data": True})
            return True

        # no external data, so we can just copy the model
        hardlink_copy_file(model_path, new_model_path)
        return has_cb_files or False

    if len(external_file_names) > 1:
        # save the model with single external data file
        model_proto_to_file(onnx.load(model_path), new_model_path, {"save_as_external_data": True})
        return True

    external_file_path = str(model_path.parent / external_file_names[0])
    if external_file_path in saved_external_files:
        # already saved, model will refer to the same file
        new_external_file_name = saved_external_files[external_file_path]
    else:
        new_external_file_name = f"{new_model_path.name}.data"
        # copy the external data file to the new location
        hardlink_copy_file(external_file_path, new_model_path.parent / new_external_file_name)
        # update the saved external files mapping
        saved_external_files[external_file_path] = new_external_file_name

    # change the external data location and save the model file
    model_proto = onnx.load(model_path, load_external_data=False)
    change_external_data_location(model_proto, new_external_file_name)
    model_proto_to_file(model_proto, new_model_path)
    return True


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


def _fix_output_shapes(model_proto: onnx.ModelProto):
    """Run shape inference on the model and update the output shapes to make them fixed."""
    from onnxruntime.tools.onnx_model_utils import is_fixed_size_tensor
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    # use the onnxruntime shape inference tool since it can handle large models as well as contrib ops
    inferred_proto = SymbolicShapeInference.infer_shapes(model_proto, auto_merge=True, guess_output_rank=True)

    for idx, o in enumerate(model_proto.graph.output):
        if not is_fixed_size_tensor(o):
            new_o = inferred_proto.graph.output[idx]
            if is_fixed_size_tensor(new_o):
                o.type.tensor_type.shape.CopyFrom(new_o.type.tensor_type.shape)


def fix_dim_params(model_proto: onnx.ModelProto, dim_params: List[str], dim_values: List[int]):
    """Fix the dimension parameters in the model.

    :param dim_params: The dimension parameters to fix.
    :param dim_values: The values to set for the dimension parameters.
    """
    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

    assert len(dim_params) == len(dim_values), "dim_params and dim_values must have the same number of elements."
    assert all(i >= 0 for i in dim_values), "dim_values must be all >= 0"

    for param, value in zip(dim_params, dim_values):
        make_dim_param_fixed(model_proto.graph, param, value)

    # update the output shapes to make them fixed
    _fix_output_shapes(model_proto)


def fix_input_shapes(model_proto: onnx.ModelProto, input_names: List[str], input_shapes: List[List[int]]):
    """Fix the input shapes in the model.

    :param input_names: The input names to fix.
    :param input_shapes: The shapes to set for the inputs.
    """
    from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

    assert len(input_names) == len(input_shapes), "input_names and input_shapes must have the same number of elements."
    assert all(all(i > 0 for i in shape) for shape in input_shapes), "input_shapes must be all > 0"

    for name, shape in zip(input_names, input_shapes):
        make_input_shape_fixed(model_proto.graph, name, shape)

    # update the output shapes to make them fixed
    _fix_output_shapes(model_proto)
