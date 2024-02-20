# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Automatically distribute Orchard Transformer model using Tensor Parallelism
# --------------------------------------------------------------------------

import logging
from typing import List, Optional

import torch

from olive.passes.pytorch.tensor_parallel import TensorParallel

logger = logging.getLogger(__name__)


class TransformerTensorParallel(TensorParallel):
    def replace_layers(self):
        pass

    def restore_layers(self):
        pass

    def _apply_linear(self, linear: torch.nn.Linear, style: str, weight_splits: Optional[List[int]] = None) -> None:
        # Linear's weight matrix is transposed, and is of shape
        # (linear.out_features, linear.in_features)
        dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
        assert style in dim_lookup
        shard_dim, size_attr = dim_lookup[style]

        # ensure we can shard evenly
        assert getattr(linear, size_attr) % self.world_size == 0

        def shard(x, dim):
            assert x.size(dim=dim) % self.world_size == 0
            return torch.tensor_split(x, self.world_size, dim=dim)[self.rank]

        def shard_qkv(qkv, dim, weight_splits):
            q, k, v = qkv.split(weight_splits, dim=dim)
            q = shard(q, dim)
            k = shard(k, dim)
            v = shard(v, dim)
            return torch.cat((q, k, v), dim=dim)

        # shard
        if weight_splits:
            # attention
            assert len(weight_splits) == 3

            sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
            if hasattr(linear, "scales") and style == "colwise":
                linear.scales = shard_qkv(linear.scales, 0, weight_splits)
        else:
            sharded_weight = shard(linear.weight, shard_dim)
            if hasattr(linear, "scales") and style == "colwise":
                linear.scales = shard(linear.scales, 0)

        # local_break()
        linear.weight = torch.nn.Parameter(sharded_weight, requires_grad=False)
        setattr(linear, size_attr, getattr(linear, size_attr) // self.world_size)

        # shape info should still be synced
        # assert linear.weight.shape == (linear.out_features, linear.in_features)

    def _apply_transformer(self, model: torch.nn.Module) -> None:
        # overwrite config before Transformer.setup_cache is called
        model.config.n_head = model.config.n_head // self.world_size
        model.config.dim = model.config.dim // self.world_size
        model.config.n_local_heads = model.config.n_local_heads // self.world_size

    def _apply_feedforward(self, mlp: torch.nn.Module) -> None:
        assert hasattr(mlp, "w1")
        assert hasattr(mlp, "w3")
        assert hasattr(mlp, "w2")

        self._apply_linear(mlp.w1, "colwise")
        self._apply_linear(mlp.w3, "colwise")
        self._apply_linear(mlp.w2, "rowwise")

    def _apply_attention(self, attn: torch.nn.Module) -> None:
        assert hasattr(attn, "wqkv")
        assert hasattr(attn, "wo")

        kv_size = attn.n_local_heads * attn.head_dim
        self._apply_linear(attn.wqkv, "colwise", [attn.dim, kv_size, kv_size])
        self._apply_linear(attn.wo, "rowwise")

        # overwrite
        attn.n_head = attn.n_head // self.world_size
        attn.dim = attn.dim // self.world_size
        attn.head_dim = attn.dim // attn.n_head
        attn.n_local_heads = attn.n_local_heads // self.world_size

    def split_weights(self, model: torch.nn.Module):
        self._apply_transformer(model)
        for block in model.layers:
            self._apply_feedforward(block.feed_forward)
            self._apply_attention(block.attention)

    def load_rank_weights(self, model: torch.nn.Module):
        pass
