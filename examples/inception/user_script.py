# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.data.registry import Registry


@Registry.register_post_process()
def inception_post_process(output):
    return output["InceptionV3/Predictions/Reshape_1:0"].argmax(axis=1)
