# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import ClassVar

from olive.data_container.constants import DataComponentType, DataContainerType
from olive.data_container.container.base_container import BaseContainer
from olive.data_container.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER)
class HuggingfaceContainer(BaseContainer):
    default_components_type: ClassVar[dict] = {
        DataComponentType.DATASET.value: "huggingface_dataset",
        DataComponentType.PRE_PROCESS.value: "huggingface_pre_process",
    }
    # Extra arguments auto generation for data components
