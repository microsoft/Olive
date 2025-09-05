# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re


def get_io_config(model_path: str) -> dict:
    """Get the input/output configuration of the ONNX model.

    :param model_path: The path to the ONNX model file.
    :return: A dictionary containing the input/output configuration of the ONNX model.
    """
    import onnx

    try:
        from onnx.helper import tensor_dtype_to_np_dtype
    except ImportError:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

        def tensor_dtype_to_np_dtype(tensor_type):
            return TENSOR_TYPE_TO_NP_TYPE[tensor_type]

    model_proto = onnx.load(model_path, load_external_data=False)

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


def get_kv_info(io_config: dict) -> dict | None:
    """Return the kv_info dictionary containing information about past keys and values.

    :param io_config: A dictionary containing the input and output names and shapes.
    :return: A dictionary with keys "past_names", "present_to_past", "num_kv_heads", "head_size" and "dtype"
        If no kv_info is found, returns None. Only dynamic shapes are accepted currently.
    """
    # assuming batch_size, num_kv_heads, past_seq_len, head_size
    kv_options = {
        r"past_key_values.(\d+).key": {
            "past_key": "past_key_values.%d.key",
            "past_value": "past_key_values.%d.value",
            "present_key": "present.%d.key",
            "present_value": "present.%d.value",
        },
        r"past_key_(\d+)": {
            "past_key": "past_key_%d",
            "past_value": "past_value_%d",
            "present_key": "present_key_%d",
            "present_value": "present_value_%d",
        },
    }

    # Find the format of the past keys and values
    # only accept dynamic shapes for now
    kv_format = None
    for idx, i_name in enumerate(io_config["input_names"]):
        for pattern in kv_options:
            if re.match(pattern, i_name) and not isinstance(io_config["input_shapes"][idx][2], int):
                kv_format = pattern
                break
        if kv_format:
            break

    if kv_format is None:
        return None

    # find the number of layers
    num_layers = 0
    for i_name in io_config["input_names"]:
        num_layers += int(re.match(kv_format, i_name) is not None)

    past_names = []
    present_to_past = {}
    for k in ["key", "value"]:
        past_names.extend([kv_options[kv_format][f"past_{k}"] % i for i in range(num_layers)])
        present_to_past.update(
            {
                kv_options[kv_format][f"present_{k}"] % i: kv_options[kv_format][f"past_{k}"] % i
                for i in range(num_layers)
            }
        )

    past_shape = io_config["input_shapes"][io_config["input_names"].index(past_names[0])]

    return {
        "past_names": past_names,
        "present_to_past": present_to_past,
        "num_kv_heads": past_shape[1],
        "head_size": past_shape[3],
        "dtype": io_config["input_types"][io_config["input_names"].index(past_names[0])],
    }


def get_io_dtypes(io_config: dict) -> dict:
    """Return a dictionary mapping input/output names to their data types.

    :param io_config: A dictionary containing the input and output names and shapes.
    :return: A dictionary mapping input/output names to their data types.
    """
    return dict(
        zip(
            io_config["input_names"] + io_config["output_names"],
            io_config["input_types"] + io_config["output_types"],
        )
    )
