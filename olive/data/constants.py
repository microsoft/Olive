# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.common.utils import StrEnumBase

# index for targets that should be ignored when computing metrics
IGNORE_INDEX = -100


class DataComponentType(StrEnumBase):
    """enumerate for the different types of data components."""

    # dataset component type: to load data into memory
    LOAD_DATASET = "load_dataset"
    # pre_process component type: to pre-process data for model inputs
    PRE_PROCESS_DATA = "pre_process_data"
    # post_process component type: to post-process model outputs for evaluation
    POST_PROCESS_DATA = "post_process_data"
    # dataloader component type: to batch/sampler data for model training/inference/optimization
    DATALOADER = "dataloader"


class DataContainerType(StrEnumBase):
    """enumerate for the different types of data containers."""

    DATA_CONTAINER = "data_container"


class DefaultDataComponent(StrEnumBase):
    """enumerate for the default data components."""

    LOAD_DATASET = "default_load_dataset"
    PRE_PROCESS_DATA = "default_pre_process_data"
    POST_PROCESS_DATA = "default_post_process_data"
    DATALOADER = "default_dataloader"


class DefaultDataContainer(StrEnumBase):
    """enumerate for the default data containers."""

    DATA_CONTAINER = "DataContainer"
