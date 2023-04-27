# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Any

from olive.data_container.constants import DataComponentType, DataContainerType
from olive.data_container.container.base_container import BaseContainer
from olive.data_container.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER)
class HuggingfaceContainer(BaseContainer):
    from transformers.pipelines import SUPPORTED_TASKS

    default_components_type = {
        DataComponentType.DATASET.value: "huggingface_dataset",
        DataComponentType.PRE_PROCESS.value: "huggingface_pre_process",
    }
    supported_tasks = SUPPORTED_TASKS
    # Extra arguments auto generation for data components
    config: dict[str, Any] = {
        "task_type": None,
        "model_name": None,
        "data_name": None,
        "subset_name": None,
        "split_name": None,
        "input_cols": None,
        "label_cols": None,
        "batch_size": 1,
        "extra_args": None,
    }

    def create_dataloader(self):
        dataset = self.dataset(data_name=self.data_name, subset=self.subset_name, split=self.split_name)
        pre_process_dataset = self.pre_process(
            dataset=dataset,
            model_name=self.model_name,
            input_cols=self.input_cols,
            label_cols=self.label_cols,
            **self.extra_args
        )
        return self.dataloader(pre_process_dataset, self.batch_size)

    def run_post_process(self):
        return self.post_process()
