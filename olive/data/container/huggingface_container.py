# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import ClassVar

from olive.data.constants import DataComponentType, DataContainerType
from olive.data.container.data_container import DataContainer
from olive.data.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER)
class HuggingfaceContainer(DataContainer):
    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "huggingface_dataset",
        DataComponentType.PRE_PROCESS_DATA.value: "huggingface_pre_process",
    }
    # Extra arguments auto generation for data components

    task_type_components_map: ClassVar[dict] = {
        # TODO(trajep): use enumerate update task type
        "text-classification": {
            DataComponentType.POST_PROCESS_DATA.value: "text_classification_post_process",
        },
        "ner": {
            DataComponentType.PRE_PROCESS_DATA.value: "ner_huggingface_preprocess",
            DataComponentType.POST_PROCESS_DATA.value: "ner_post_process",
        },
        "text-generation": {
            DataComponentType.PRE_PROCESS_DATA.value: "text_generation_huggingface_pre_process",
            DataComponentType.POST_PROCESS_DATA.value: "text_generation_post_process",
        },
        "audio-classification": {
            DataComponentType.PRE_PROCESS_DATA.value: "audio_classification_pre_process",
            DataComponentType.POST_PROCESS_DATA.value: "text_classification_post_process",
        },
    }
