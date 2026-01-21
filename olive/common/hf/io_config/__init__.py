# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.hf.io_config.input_generators import generate_diffusers_dummy_inputs
from olive.common.hf.io_config.io_resolver import is_task_supported
from olive.common.hf.io_config.task_config import (
    generate_dummy_inputs,
    get_diffusers_io_config,
    get_io_config,
)
from olive.common.hf.io_config.tasks import TaskType, map_task_synonym

__all__ = [
    "TaskType",
    "generate_diffusers_dummy_inputs",
    "generate_dummy_inputs",
    "get_diffusers_io_config",
    "get_io_config",
    "is_task_supported",
    "map_task_synonym",
]
