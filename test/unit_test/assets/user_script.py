# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.model import OliveModel


def eval_func(model: OliveModel, data_dir, batch_size, device, execution_providers):
    return 0.382715310


def metric_func(inference_output, actuals):
    return 0.382715311
