# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.data_container.constants import DataContainerType
from olive.data_container.container.base_container import BaseContainer
from olive.data_container.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER)
class HuggingfaceContainer(BaseContainer):
    from transformers.pipelines import SUPPORTED_TASKS
    _dataset: str = "huggingface_dataset"
    _pre_process: str = "huggingface_pre_process"

    supported_tasks = SUPPORTED_TASKS
    # Extra arguments auto generation for data components
    task_type: str = None
    model_name: str = None
    data_name: str = None
    subset_name: str = None
    split_name: str = None
    input_cols: str = None
    label_cols: str = None
    batch_size: int = 1
