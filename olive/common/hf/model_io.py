# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Optional

from olive.common.hf.mlflow import get_pretrained_name_or_path
from olive.common.hf.peft import is_peft_model
from olive.common.hf.utils import get_model_config, get_tokenizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from optimum.exporters.onnx import OnnxConfig
    from transformers import PreTrainedModel


def get_preprocessors(model_name: str, **kwargs) -> Optional[Dict]:
    """Get the preprocessors for the model_name."""
    from optimum.utils.save_utils import maybe_load_preprocessors

    # get tokenizer separately to support mlflow models
    tokenizer = None
    try:
        tokenizer = get_tokenizer(model_name, **kwargs)
    except Exception:
        # there is no tokenizer for the model_name
        pass

    model_name = get_pretrained_name_or_path(model_name, "model")
    preprocessors = maybe_load_preprocessors(
        model_name, subfolder=kwargs.get("subfolder", ""), trust_remote_code=kwargs.get("trust_remote_code", False)
    )
    if tokenizer:
        for i, preprocessor in enumerate(preprocessors):
            if isinstance(preprocessor, type(tokenizer)):
                preprocessors[i] = tokenizer
                break
    return preprocessors


def get_export_config(model_name: str, task: str, **kwargs) -> Optional["OnnxConfig"]:
    """Get the export config for the model_name and task."""
    try:
        from optimum.exporters.tasks import TasksManager
    except ImportError:
        logger.debug("optimum is not installed. Cannot get export config")
        return None

    model_config = get_model_config(model_name, **kwargs)
    model_type = model_config.model_type.replace("_", "-")

    task = TasksManager.map_from_synonym(task)
    # use try except block since we don't want to access private class attributes like
    # TasksManager._SUPPORTED_MODEL_TYPE
    try:
        supported_tasks = TasksManager.get_supported_tasks_for_model_type(
            model_type, exporter="onnx", library_name="transformers"
        )
        if task not in supported_tasks:
            logger.debug("Task %s is not supported for model type %s", task, model_type)
            return None
    except KeyError:
        logger.debug("Model type %s is not supported", model_type)
        return None

    # TODO(jambayk): ask caller for dtype?
    dtype = getattr(model_config, "torch_dtype", "float32")
    if "bfloat16" in str(dtype):
        float_dtype = "bf16"
    elif "float16" in str(dtype):
        float_dtype = "fp16"
    else:
        float_dtype = "fp32"

    export_config_constructor = TasksManager.get_exporter_config_constructor(
        exporter="onnx", task=task, model_type=model_type, library_name="transformers"
    )
    export_config = export_config_constructor(
        model_config,
        int_dtype="int64",
        float_dtype=float_dtype,
        # TODO(jambayk): other preprocessors needed?
        preprocessors=get_preprocessors(model_name, **kwargs),
    )

    if task.startswith("text-generation"):
        # need kv cache for both input and output
        export_config = export_config.__class__(
            model_config,
            # if use_cache is False, there is no kv cache output
            # else both text-generation and text-generation-with-past have kv cache output
            use_past=kwargs.get("use_cache", True),
            # only text-generation-with-past has kv cache input
            use_past_in_inputs=export_config.use_past,
            # text-generation-with-past doesn't return position_ids
            task="text-generation",
            float_dtype=float_dtype,
            int_dtype="int64",
        )

    return export_config


def get_model_io_config(model_name: str, task: str, model: "PreTrainedModel", **kwargs) -> Optional[Dict]:
    """Get the input/output config for the model_name and task."""
    # just log a debug message if io_config is not found
    # this is not a critical error and the caller may not need the io_config
    export_config = get_export_config(model_name, task, **kwargs)
    if not export_config:
        return None

    if is_peft_model(model):
        # if pytorch_model is PeftModel, we need to get the base model
        # otherwise, the model forward has signature (*args, **kwargs)
        model = model.get_base_model()

    inputs = export_config.ordered_inputs(model)
    input_names = list(inputs.keys())
    output_names = list(export_config.outputs.keys())
    dynamic_axes = dict(chain(inputs.items(), export_config.outputs.items()))
    # optimum has the total sequence length as "past_sequence_length  + 1" but that is not always the case
    # change it to "past_sequence_length + sequence_length" if past is used
    for value in dynamic_axes.values():
        for axis, axis_name in value.items():
            if axis_name == "past_sequence_length + 1":
                value[axis] = "past_sequence_length + sequence_length"
    # dynamic_shapes should follow input order and format
    dynamic_shapes = _unflatten_past_key_values_with_check(inputs)
    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "dynamic_shapes": dynamic_shapes,
    }


def _unflatten_past_key_values_with_check(flattened_inputs: Dict[str, Any]) -> Dict[str, Any]:
    max_idx = -1
    past_key_value_count = 0  # Track number of key-value pairs

    # Find the max index for generating unflatten past_key_values later
    # and record the total number of past_key_values entries for validation
    for input_name in flattened_inputs:
        if input_name.startswith("past_key_values"):
            # From Optimum: past_key_values.0.key, past_key_values.0.value,
            #               past_key_values.1.key, past_key_values.1.value, ...
            idx = int(input_name.split(".")[1])
            max_idx = max(max_idx, idx)
            past_key_value_count += 1

    # Check if we have exactly 2 * (max_idx + 1) key-value pairs
    expected_count = 2 * (max_idx + 1)
    if past_key_value_count != expected_count or past_key_value_count % 2 != 0:
        logger.warning(
            "Expected %d past_key_values entries, but found %d from Optimum inputs."
            "Giving up generating dynamic_shapes from Optimum inputs."
            "Olive will use dynamic_axes instead.",
            expected_count,
            past_key_value_count,
        )
        return {}
    # No past_key_values found
    if max_idx == -1:
        return flattened_inputs
    # Keep all inputs except past_key_values
    unflattened = {
        input_name: dynamic_shapes
        for input_name, dynamic_shapes in flattened_inputs.items()
        if not input_name.startswith("past_key_values")
    }
    # Based on Optimum's implementation:
    # https://github.com/huggingface/optimum/blob/b755036ae12e0959d61085e597e7b96473c4b46d/optimum/exporters/onnx/base.py#L629
    # past_key_values is a list of lists, and it locates at the end of the input list/dict
    # Generate the past_key_values list using the max index
    unflattened["past_key_values"] = [
        [flattened_inputs[f"past_key_values.{idx}.key"], flattened_inputs[f"past_key_values.{idx}.value"]]
        for idx in range(max_idx + 1)
    ]
    return unflattened


def get_model_dummy_input(model_name: str, task: str, **kwargs) -> Optional[Dict]:
    """Get dummy inputs for the model_name and task."""
    export_config = get_export_config(model_name, task, **kwargs)
    if not export_config:
        return None

    from optimum.utils import DEFAULT_DUMMY_SHAPES

    dummy_inputs = export_config.generate_dummy_inputs(framework="pt", **DEFAULT_DUMMY_SHAPES)
    return export_config.rename_ambiguous_inputs(dummy_inputs)


def get_kv_info(io_config: Dict) -> Optional[Dict]:
    """Return the kv_info dictionary containing information about past keys and values.

    :param io_config: A dictionary containing the input and output names and shapes.
    :return: A dictionary with keys "past_names", "present_to_past", "num_kv_heads", and "head_size".
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
    logger.debug("Found %d layers with past keys/values", num_layers)

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

    past_shape = io_config["input_shapes"][io_config["input_names"].index(past_names[1])]
    present_shape = io_config["output_shapes"][io_config["output_names"].index(next(iter(present_to_past.keys())))]

    return {
        "past_names": past_names,
        "present_to_past": present_to_past,
        "num_kv_heads": past_shape[1],
        "head_size": past_shape[3],
        "past_seq_len": past_shape[2],
        "present_seq_len": present_shape[2],
    }
