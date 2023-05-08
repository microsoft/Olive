# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import ClassVar

from olive.data.constants import DataComponentType, DataContainerType
from olive.data.container.base_container import BaseContainer
from olive.data.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER)
class HuggingfaceContainer(BaseContainer):
    default_components_type: ClassVar[dict] = {
        DataComponentType.DATASET.value: "huggingface_dataset",
        DataComponentType.PRE_PROCESS.value: "huggingface_pre_process",
    }
    # Extra arguments auto generation for data components

    task_type_components_map: ClassVar[dict] = {
        # TODO user enumerate update task type
        "text-classification": {
            DataComponentType.POST_PROCESS.value: "text_classification_post_process",
        },
    }
