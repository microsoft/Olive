# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model IO utilities for HuggingFace models."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from olive.common.hf.io_config import (
    generate_dummy_inputs,
    get_io_config,
    is_task_supported,
    map_task_synonym,
)
from olive.common.hf.io_config.tasks import TaskType
from olive.common.hf.peft import is_peft_model
from olive.common.hf.utils import get_model_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def get_model_io_config(
    model_name: str,
    task: str,
    model: Optional["PreTrainedModel"] = None,
    **kwargs,
) -> Optional[dict[str, Any]]:
    """Get the input/output config for the model and task.

    Args:
        model_name: The model name or path.
        task: The task type (e.g., "text-generation", "text-classification").
        model: Optional loaded model for input signature inspection.
        **kwargs: Additional arguments including use_cache.

    Returns:
        Dict containing input_names, output_names, and dynamic_axes.

    """
    # Map task synonyms
    task = map_task_synonym(task)

    # Determine use_past settings and actual task
    use_past = False
    use_past_in_inputs = False
    actual_task = task

    if task.startswith(TaskType.TEXT_GENERATION):
        use_past = kwargs.get("use_cache", True)
        use_past_in_inputs = task == TaskType.TEXT_GENERATION_WITH_PAST
        actual_task = TaskType.TEXT_GENERATION
    elif task.startswith(TaskType.TEXT2TEXT_GENERATION):
        use_past = kwargs.get("use_cache", True)
        use_past_in_inputs = task == TaskType.TEXT2TEXT_GENERATION_WITH_PAST
        actual_task = TaskType.TEXT2TEXT_GENERATION
    elif task.startswith(TaskType.FEATURE_EXTRACTION):
        use_past = kwargs.get("use_cache", False)
        use_past_in_inputs = task == TaskType.FEATURE_EXTRACTION_WITH_PAST
        actual_task = TaskType.FEATURE_EXTRACTION

    # Check if actual task is supported
    if not is_task_supported(actual_task):
        logger.debug("Task %s is not supported in Olive io_config", actual_task)
        return None

    # Get model config
    model_config = get_model_config(model_name, **kwargs)

    # Handle PEFT models
    actual_model = model
    if model is not None and is_peft_model(model):
        actual_model = model.get_base_model()

    try:
        return get_io_config(
            model_name_or_config=model_config,
            task=actual_task,
            model=actual_model,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
        )
    except (ValueError, KeyError) as e:
        logger.debug("Failed to get IO config for %s with task %s: %s", model_name, task, e)
        return None


def get_model_dummy_input(
    model_name: str,
    task: str,
    model: Optional["PreTrainedModel"] = None,
    **kwargs,
) -> Optional[dict[str, Any]]:
    """Get dummy inputs for the model and task.

    Args:
        model_name: The model name or path.
        task: The task type.
        model: Optional loaded model for input signature inspection.
        **kwargs: Additional arguments including use_cache, batch_size, sequence_length.

    Returns:
        Dict of dummy input tensors.

    """
    # Map task synonyms
    task = map_task_synonym(task)

    # Determine use_past settings and actual task
    use_past = False
    use_past_in_inputs = False
    actual_task = task

    if task.startswith(TaskType.TEXT_GENERATION):
        use_past = kwargs.get("use_cache", True)
        use_past_in_inputs = task == TaskType.TEXT_GENERATION_WITH_PAST
        actual_task = TaskType.TEXT_GENERATION
    elif task.startswith(TaskType.TEXT2TEXT_GENERATION):
        use_past = kwargs.get("use_cache", True)
        use_past_in_inputs = task == TaskType.TEXT2TEXT_GENERATION_WITH_PAST
        actual_task = TaskType.TEXT2TEXT_GENERATION
    elif task.startswith(TaskType.FEATURE_EXTRACTION):
        use_past = kwargs.get("use_cache", False)
        use_past_in_inputs = task == TaskType.FEATURE_EXTRACTION_WITH_PAST
        actual_task = TaskType.FEATURE_EXTRACTION

    # Check if actual task is supported
    if not is_task_supported(actual_task):
        logger.debug("Task %s is not supported in Olive io_config", actual_task)
        return None

    # Get model config (handles MLflow paths)
    model_config = get_model_config(model_name, **kwargs)

    # Handle PEFT models
    actual_model = model
    if model is not None and is_peft_model(model):
        actual_model = model.get_base_model()

    try:
        return generate_dummy_inputs(
            model_name_or_config=model_config,
            task=actual_task,
            model=actual_model,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
        )
    except (ValueError, KeyError) as e:
        logger.debug("Failed to generate dummy inputs for %s with task %s: %s", model_name, task, e)
        return None
