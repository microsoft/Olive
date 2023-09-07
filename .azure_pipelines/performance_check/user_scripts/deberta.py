# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from transformers import DebertaForSequenceClassification


def torch_complied_model(model_path):
    model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base-mnli")
    return torch.compile(model)
