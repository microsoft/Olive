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

    _supported_tasks = SUPPORTED_TASKS
    # Extra arguments auto generation for data components
    task_type: str = None
    model_name: str = None
    data_name: str = None
    subset_name: str = None
    split_name: str = None
    input_cols: list = None
    label_cols: list = None
    batch_size: int = 1
    extra_args: dict = None

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
