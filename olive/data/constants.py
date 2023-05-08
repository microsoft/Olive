# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from enum import Enum

DEFAULT_HF_DATA_CONTAINER_NAME = "_default_huggingface_dc"


class DataComponentType(Enum):
    """
    enumerate for the different types of data components
    """

    # dataset component type: to load data into memory
    DATASET = "dataset"
    # pre_process component type: to pre-process data for model inputs
    PRE_PROCESS = "pre_process"
    # post_process component type: to post-process model outputs for evaluation
    POST_PROCESS = "post_process"
    # dataloader component type: to batch/sampler data for model training/inference/optimization
    DATALOADER = "dataloader"


class DataContainerType(Enum):
    """
    enumerate for the different types of data containers
    """

    DATA_CONTAINER = "data_container"


class DefaultDataComponent(Enum):
    """
    enumerate for the default data components
    """

    DATASET = "default_dataset"
    PRE_PROCESS = "default_pre_process"
    POST_PROCESS = "default_post_process"
    DATALOADER = "default_dataloader"


class DefaultDataContainer(Enum):
    """
    enumerate for the default data containers
    """

    DATA_CONTAINER = "BaseDataContainer"
    # TODO
    DUMMY_DATA_CONTAINER = "DummyDataContainer"
