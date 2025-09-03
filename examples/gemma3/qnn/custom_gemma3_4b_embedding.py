# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import logging

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.embedding_layer = full_model.language_model.embed_tokens

    def forward(self, input_ids, image_features):
        image_token_index = 262144
        inputs_embeds = self.embedding_layer(input_ids)

        special_image_mask = (input_ids == image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(special_image_mask, image_features)


def load_gemma3_embedding_model(model_path):
    full_model = AutoModel.from_pretrained("google/gemma-3-4b-it")
    logger.info("Loaded full model: %s", full_model)

    embedding_layer = EmbeddingLayer(full_model)

    logger.info("Created embedding-only model: %s", embedding_layer)
    return embedding_layer
