# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from transformers import BertForSequenceClassification


def torch_complied_model(model_path):
    model = BertForSequenceClassification.from_pretrained("Intel/bert-base-uncased-mrpc")
    return torch.compile(model)
