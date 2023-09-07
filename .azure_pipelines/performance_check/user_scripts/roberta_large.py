# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from transformers import RobertaForSequenceClassification


def torch_complied_model(model_path):
    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
    return torch.compile(model)
