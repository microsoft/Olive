# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.model import OliveModelHandler


def eval_func(model: OliveModelHandler, device, execution_providers):
    return 0.382715310


def metric_func(inference_output, actuals):
    return 0.382715311
