# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from itertools import chain
from typing import TYPE_CHECKING, Dict, Optional

import torch

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
            use_past=export_config.use_past,
            use_past_in_inputs=export_config.use_past,
            # text-generation-with-past doesn't return position_ids
            task="text-generation",
            float_dtype=float_dtype,
            int_dtype="int64",
        )

    return export_config


def get_model_io_config(model_name: str, task: str, model: "PreTrainedModel", **kwargs) -> Optional[Dict]:
    """Get the input/output config for the model_name and task."""
    extended_mask_type = kwargs.pop("extended_mask_type", False)
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
    # causal mask is 4D with shape (batch_size, 1, sequence_length, total_sequence_length)
    if extended_mask_type and "attention_mask" in dynamic_axes:
        dynamic_axes["attention_mask"] = {
            0: dynamic_axes["attention_mask"][0],
            2: dynamic_axes["input_ids"][1],
            3: dynamic_axes["attention_mask"][1],
        }
    # NOTE: Due to the complexity of dynamic_shapes, we don't provide it here.
    # torch-onnx converter has a naive approach to auto-gen dynamic shapes based on input and
    # dynamic_axes, so we don't need to provide dynamic shapes here.
    return {"input_names": input_names, "output_names": output_names, "dynamic_axes": dynamic_axes}


def get_model_dummy_input(model_name: str, task: str, **kwargs) -> Optional[Dict]:
    """Get dummy inputs for the model_name and task."""
    extended_mask_type = kwargs.pop("extended_mask_type", False)
    export_config = get_export_config(model_name, task, **kwargs)
    if not export_config:
        return None

    from optimum.utils import DEFAULT_DUMMY_SHAPES

    dummy_inputs = export_config.generate_dummy_inputs(framework="pt", **DEFAULT_DUMMY_SHAPES)
    dummy_inputs = export_config.rename_ambiguous_inputs(dummy_inputs)
    if extended_mask_type:
        dummy_inputs = replace_with_extended_mask(dummy_inputs, extended_mask_type=extended_mask_type)
    return dummy_inputs


def replace_with_extended_mask(
    data: Dict[str, torch.Tensor], extended_mask_type: str, extended_mask_value: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    """Replace the attention_mask in the data with an extended mask."""
    if "attention_mask" not in data or data["attention_mask"].ndim > 2:
        return data

    attention_mask = data["attention_mask"]
    ndims = attention_mask.ndim
    # will use a 2d -> 4d mask creator
    if ndims == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if extended_mask_value is None:
        extended_mask_value = torch.finfo(torch.float32).min
    tgt_len = data["input_ids"].shape[-1]
    batch_size, src_len = attention_mask.shape

    # the inference caller should cast the mask to the right dtype
    expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).float()
    expanded_mask = 1.0 - expanded_mask
    expanded_mask = expanded_mask.masked_fill(expanded_mask.bool(), extended_mask_value)
    if extended_mask_type == "causal":
        causal_mask = torch.full((tgt_len, tgt_len), extended_mask_value)
        mask_cond = torch.arange(causal_mask.size(-1))
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
        causal_mask = causal_mask.float()
        if src_len > tgt_len:
            causal_mask = torch.cat([torch.zeros(tgt_len, src_len - tgt_len), causal_mask], dim=-1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, tgt_len, src_len)
        expanded_mask = causal_mask.masked_fill(expanded_mask.bool(), extended_mask_value)
    elif extended_mask_type != "non-causal":
        raise ValueError(f"extended_mask_type {extended_mask_type} is not supported")

    if ndims == 1:
        expanded_mask = expanded_mask.squeeze(0)

    data["attention_mask"] = expanded_mask
    return data
