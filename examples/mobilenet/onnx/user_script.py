# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import timm


def load_timm(model_name: str):
    model = timm.create_model(model_name, pretrained=True)
    return model.eval()
