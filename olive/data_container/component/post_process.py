# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.data_container.constants import DataComponentType, DefaultDataComponent
from olive.data_container.registry import Registry


@Registry.register(DataComponentType.POST_PROCESS, DefaultDataComponent.POST_PROCESS.value)
def post_process(data):
    """Post-process data.

    Args:
        data (object): Model output to be post-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Post-processed data.
    """
    return data
