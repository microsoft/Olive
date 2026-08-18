# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the ONNX-export wrapper modules in ``olive.common.hf.quant``."""

from __future__ import annotations

import pytest
import torch

from olive.common.hf.quant import QuantEmbeddingNbit, QuantLinearNbit
from olive.common.quant.tensor import QuantTensor


class TestOnnxBlockSizeValidation:
    """``MatMulNBits`` / ``GatherBlockQuantized`` require block_size to be a power of 2, >= 16.

    Without an explicit check, a non-positive ``group_size`` silently falls back to the full
    ``embedding_dim`` / ``in_features``, which for many real models (Qwen2.5: 1536,
    Phi-3-mini: 3072) is not a power of 2 — surfacing only as an opaque native ORT
    session-initialization crash.
    """

    @pytest.mark.parametrize("embedding_dim", [1536, 3072, 100])
    def test_embedding_non_power_of_two_fallback_raises(self, embedding_dim: int):
        with pytest.raises(ValueError, match="GatherBlockQuantized requires block_size"):
            QuantEmbeddingNbit(num_embeddings=8, embedding_dim=embedding_dim, group_size=-1)

    def test_embedding_fallback_below_minimum_raises(self):
        with pytest.raises(ValueError, match=r">= 16, got 8"):
            QuantEmbeddingNbit(num_embeddings=8, embedding_dim=8, group_size=0)

    def test_embedding_explicit_non_power_of_two_group_size_raises(self):
        with pytest.raises(ValueError, match="GatherBlockQuantized requires block_size"):
            QuantEmbeddingNbit(num_embeddings=8, embedding_dim=1536, group_size=96)

    @pytest.mark.parametrize("group_size", [-1, 32, 64])
    def test_embedding_valid_block_sizes_are_accepted(self, group_size: int):
        module = QuantEmbeddingNbit(num_embeddings=8, embedding_dim=64, group_size=group_size)
        assert module.group_size == (64 if group_size <= 0 else group_size)

    @pytest.mark.parametrize("in_features", [1536, 3072, 100])
    def test_linear_non_power_of_two_fallback_raises(self, in_features: int):
        with pytest.raises(ValueError, match="MatMulNBits requires block_size"):
            QuantLinearNbit(group_size=-1, in_features=in_features, out_features=8)

    @pytest.mark.parametrize("group_size", [-1, 32, 128])
    def test_linear_valid_block_sizes_are_accepted(self, group_size: int):
        module = QuantLinearNbit(group_size=group_size, in_features=128, out_features=8)
        assert module.group_size == (128 if group_size <= 0 else group_size)


class TestQuantEmbeddingNbitFromQuantTensor:
    """``from_quant_tensor`` must reshape (and therefore shape-validate) like the Linear one."""

    def test_per_tensor_quant_tensor_fails_immediately(self):
        """``group_size == 0`` (per-tensor) scales are ``(1, 1)``, not ``(num_embeddings, 1)``.

        The bare ``detach().clone()`` used to install that ``(1, 1)`` tensor over a buffer
        declared ``(num_embeddings, n_groups)`` without complaint, deferring the failure to a
        confusing ``reshape`` error somewhere in the exported graph. Mirroring
        ``QuantLinearNbit.from_quant_tensor``'s ``.reshape(...)`` makes it fail here instead.
        """
        qt = QuantTensor.from_float(torch.randn(8, 32), bits=4, symmetric=True, group_size=0)
        assert tuple(qt.scales.shape) == (1, 1)
        with pytest.raises(RuntimeError, match="invalid for input of size"):
            QuantEmbeddingNbit.from_quant_tensor(qt)

    def test_per_tensor_quant_tensor_fails_the_same_way_for_linear(self):
        """Same semantics as the Embedding path — per-tensor QuantTensors are not exportable."""
        qt = QuantTensor.from_float(torch.randn(8, 32), bits=4, symmetric=True, group_size=0)
        with pytest.raises(RuntimeError, match="invalid for input of size"):
            QuantLinearNbit.from_quant_tensor(qt)

    @pytest.mark.parametrize("group_size", [-1, 16])
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_supported_group_sizes_round_trip(self, group_size: int, symmetric: bool):
        weight = torch.randn(8, 32)
        qt = QuantTensor.from_float(weight, bits=4, symmetric=symmetric, group_size=group_size)
        module = QuantEmbeddingNbit.from_quant_tensor(qt)

        assert module.group_size == (32 if group_size <= 0 else group_size)
        assert torch.equal(module.qweight, qt.qweight)
        assert torch.equal(module.scales, qt.scales)
        if qt.qzeros is None:
            assert module.qzeros is None
        else:
            assert torch.equal(module.qzeros, qt.qzeros)
