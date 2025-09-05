# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import logging

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


class Gemma3VisualEmbeddingGenerator(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        # Extract only the vision components
        self.vision_tower = full_model.vision_tower
        self.multi_modal_projector = full_model.multi_modal_projector

    def forward(self, pixel_values):
        # Process images through vision tower
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.last_hidden_state
        # Project to final embedding space
        return self.multi_modal_projector(selected_image_feature)


def load_gemma3_vision_model(model_path):
    full_model = AutoModel.from_pretrained("google/gemma-3-4b-it")
    logger.info("Loaded full model: %s", full_model)

    vision_model = Gemma3VisualEmbeddingGenerator(full_model)
    logger.info("Created vision-only model: %s", vision_model)
    return vision_model
