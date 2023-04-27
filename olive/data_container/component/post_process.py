# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.data_container.registry import Registry


@Registry.register_default_post_process()
def post_process(_output_data, **kwargs):
    """Post-process data.

    Args:
        data (object): Model output to be post-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Post-processed data.
    """
    return _output_data
