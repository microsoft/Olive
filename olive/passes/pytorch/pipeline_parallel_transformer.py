# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Automatically distribute Orchard Transformer model using Pipeline Parallelism
# --------------------------------------------------------------------------

import logging

import torch

from olive.passes.pytorch.pipeline_parallel import PipelineParallel

logger = logging.getLogger(__name__)


class TransformerPipelineParallel(PipelineParallel):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.originals = {}

    def replace_layers(self):
        pass

    def restore_layers(self):
        pass

    def split_layers(self, model: torch.nn.Module):
        if (model.config.n_local_layer % self.world_size) != 0:
            raise ValueError(
                f"Number of stages (nstages={self.world_size}) should be a multiple "
                f"of number of layers in the model (nlayers={model.config.n_local_layer})"
            )

        model.config.n_local_layer = model.config.n_local_layer // self.world_size
        start_layer_index = model.config.n_local_layer * self.rank
        end_layer_index = start_layer_index + model.config.n_local_layer
        model.layers = model.layers[start_layer_index:end_layer_index]
