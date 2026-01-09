# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import TYPE_CHECKING, Optional

from olive.common.hf.io_config import (
    DEFAULT_DUMMY_SHAPES,
    get_onnx_config,
    is_model_supported,
    is_task_supported,
    map_task_synonym,
)
from olive.common.hf.peft import is_peft_model
from olive.common.hf.utils import get_model_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from olive.common.hf.io_config import OnnxConfig


def get_export_config(model_name: str, task: str, **kwargs) -> Optional["OnnxConfig"]:
    """Get the export config for the model_name and task.

    Args:
        model_name: The model name or path.
        task: The task type (e.g., "feature-extraction", "text-generation").
        **kwargs: Additional arguments.

    Returns:
        An OnnxConfig instance configured for the model and task.

    """
    model_config = get_model_config(model_name, **kwargs)
    model_type = model_config.model_type

    # Map task synonyms
    task = map_task_synonym(task)

    if not is_model_supported(model_type):
        logger.debug("Model type %s is not supported in Olive io_config", model_type)
        return None

    if not is_task_supported(model_type, task):
        logger.debug("Task %s is not supported for model type %s in Olive", task, model_type)
        return None

    # Determine float_dtype from model config
    dtype = getattr(model_config, "torch_dtype", "float32")
    if "bfloat16" in str(dtype):
        float_dtype = "bf16"
    elif "float16" in str(dtype):
        float_dtype = "fp16"
    else:
        float_dtype = "fp32"

    use_past = False
    use_past_in_inputs = False
    actual_task = task

    if task.startswith("text-generation"):
        use_past = kwargs.get("use_cache", True)
        use_past_in_inputs = task == "text-generation-with-past"
        actual_task = "text-generation"

    return get_onnx_config(
        model_type=model_type,
        task=actual_task,
        config=model_config,
        int_dtype="int64",
        float_dtype=float_dtype,
        use_past=use_past,
        use_past_in_inputs=use_past_in_inputs,
    )


def get_model_io_config(model_name: str, task: str, model: "PreTrainedModel", **kwargs) -> Optional[dict]:
    """Get the input/output config for the model_name and task."""
    export_config = get_export_config(model_name, task, **kwargs)
    if not export_config:
        return None

    if is_peft_model(model):
        # if pytorch_model is PeftModel, we need to get the base model
        # otherwise, the model forward has signature (*args, **kwargs)
        model = model.get_base_model()

    return export_config.get_io_config(model)


def get_model_dummy_input(model_name: str, task: str, **kwargs) -> Optional[dict]:
    """Get dummy inputs for the model_name and task."""
    export_config = get_export_config(model_name, task, **kwargs)
    if not export_config:
        return None

    return export_config.generate_dummy_inputs(framework="pt", **DEFAULT_DUMMY_SHAPES)
