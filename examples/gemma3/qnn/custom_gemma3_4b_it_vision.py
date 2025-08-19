# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import torch
from transformers import AutoModel


def load_gemma3_model(model_path):
    return AutoModel.from_pretrained("google/gemma-3-4b-it")


def get_dummy_inputs(model_handler):
    return {
        "input_ids": torch.full((1, 256), 262144, dtype=torch.long),  # Image token ID
        "pixel_values": torch.randn(1, 3, 896, 896, dtype=torch.float32),
        "attention_mask": torch.ones((1, 256), dtype=torch.long),
    }
