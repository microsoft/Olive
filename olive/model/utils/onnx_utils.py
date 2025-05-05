# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from olive.model.utils.path_utils import normalize_path_suffix

if TYPE_CHECKING:
    from onnx import ModelProto

logger = logging.getLogger(__name__)


def resolve_onnx_path(file_or_dir_path: str, model_filename: str = "model.onnx") -> str:
    """Get the model full path.

    The engine provides output paths to ONNX passes that do not contain .onnx extension
    (these paths are generally locations in the cache). This function will convert such
    paths to absolute file paths and also ensure the parent directories exist.
    If the input path is already an ONNX file it is simply returned. Examples:

    resolve_onnx_path("c:/foo/bar.onnx") -> c:/foo/bar.onnx

    resolve_onnx_path("c:/foo/bar") -> c:/foo/bar/model.onnx
    """
    return normalize_path_suffix(file_or_dir_path, model_filename)


def get_onnx_file_path(model_path: str, onnx_file_name: Optional[str] = None) -> str:
    """Get the path to the ONNX model file.

    If model_path is a file, it is returned as is. If model_path is a
    directory, the onnx_file_name is appended to it and the resulting path is returned. If onnx_file_name is not
    specified, it is inferred if there is only one .onnx file in the directory, else an error is raised.
    """
    assert Path(model_path).exists(), f"Model path {model_path} does not exist"

    # if model_path is a file, return it as is
    if Path(model_path).is_file():
        return model_path

    # if model_path is a directory, append onnx_file_name to it
    if onnx_file_name:
        onnx_file_path = Path(model_path) / onnx_file_name
        assert onnx_file_path.exists(), f"ONNX model file {onnx_file_path} does not exist"
        return str(onnx_file_path)

    # try to infer onnx_file_name
    logger.warning(
        "model_path is a directory but onnx_file_name is not specified. Trying to infer it. It is recommended to"
        " specify onnx_file_name explicitly."
    )
    onnx_file_names = list(Path(model_path).glob("*.onnx"))
    if len(onnx_file_names) == 1:
        return str(onnx_file_names[0])
    elif len(onnx_file_names) > 1:
        raise ValueError(
            f"Multiple .onnx model files found in the model folder {model_path}. Please specify one using the"
            " onnx_file_name argument."
        )
    else:
        raise ValueError(f"No .onnx file found in the model folder {model_path}.")


def get_additional_file_path(model_dir: str, file_name: str) -> Optional[str]:
    """Get the full path to the additional file.

    If file_name is specified, it is assumed to be a file in the model_dir and the full path
    is returned.
    """
    if file_name:
        model_dir = Path(model_dir)
        assert model_dir.is_dir(), f"Model path {model_dir} is not a directory."
        file_path = model_dir / file_name
        assert file_path.exists(), f"{file_name} does not exist in model path directory {model_dir}."
        return str(file_path)
    return None


def dump_tuning_result(session, tuning_result_path):
    assert tuning_result_path.endswith(".json")
    tuning_result = session.get_tuning_results()
    with Path(tuning_result_path).open("w") as f:
        json.dump(tuning_result, f, indent=2)


def get_io_config(model_proto: "ModelProto") -> dict:
    try:
        from onnx.helper import tensor_dtype_to_np_dtype
    except ImportError:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

        def tensor_dtype_to_np_dtype(tensor_type):
            return TENSOR_TYPE_TO_NP_TYPE[tensor_type]

    io_config = {
        "input_names": [],
        "input_shapes": [],
        "input_types": [],
        "output_names": [],
        "output_shapes": [],
        "output_types": [],
    }
    for prefix, ios in [("input", model_proto.graph.input), ("output", model_proto.graph.output)]:
        for io in ios:
            # get name, type, shape
            name = io.name
            tensor_type = io.type.tensor_type
            if tensor_type.elem_type == 0:
                # sequence type
                # TODO(jambayk): add support for different types
                # refer to https://github.com/lutzroeder/netron/blob/main/source/onnx.js#L1424
                tensor_type = io.type.sequence_type.elem_type.tensor_type
            data_type = str(tensor_dtype_to_np_dtype(tensor_type.elem_type))
            shape = [dim.dim_param if dim.dim_param else dim.dim_value for dim in tensor_type.shape.dim]

            # append to io_config
            io_config[f"{prefix}_names"].append(name)
            io_config[f"{prefix}_types"].append(data_type)
            io_config[f"{prefix}_shapes"].append(shape)

    return io_config
