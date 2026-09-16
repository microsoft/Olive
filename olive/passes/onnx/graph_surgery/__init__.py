# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Built-in surgeons, registered on import for the GraphSurgeries pass."""

from olive.passes.onnx.graph_surgery.activations import FuseBiasGelu, FuseGelu
from olive.passes.onnx.graph_surgery.attention import (
    AttentionToGroupQueryAttention,
    BlockDiagonalAttentionToPackedMHA,
    PackQKVForGroupQueryAttention,
    SeparateGroupQueryAttentionRoPE,
    UnpackGroupQueryAttentionQKV,
)
from olive.passes.onnx.graph_surgery.base import ProtoSurgeon, RewriteRuleSurgeon, Surgeon
from olive.passes.onnx.graph_surgery.lowering import (
    ClipToMinMax,
    DecomposeAttention,
    DecomposeOnnxRotaryEmbedding,
    Rank4RMSNormToRank3,
    StaticEmptyKV,
    TensorScatterToScatterND,
)
from olive.passes.onnx.graph_surgery.moe import FuseBlockQuantizedMoE, FuseDenseMoEToQMoE, MoEGraphSurgeryError
from olive.passes.onnx.graph_surgery.normalization import (
    FuseLayerNormalization,
    FuseSkipLayerNormalization,
    FuseSkipRMSNormalization,
)

__all__ = [
    "AttentionToGroupQueryAttention",
    "BlockDiagonalAttentionToPackedMHA",
    "ClipToMinMax",
    "DecomposeAttention",
    "DecomposeOnnxRotaryEmbedding",
    "FuseBiasGelu",
    "FuseBlockQuantizedMoE",
    "FuseDenseMoEToQMoE",
    "FuseGelu",
    "FuseLayerNormalization",
    "FuseSkipLayerNormalization",
    "FuseSkipRMSNormalization",
    "MoEGraphSurgeryError",
    "PackQKVForGroupQueryAttention",
    "ProtoSurgeon",
    "Rank4RMSNormToRank3",
    "RewriteRuleSurgeon",
    "SeparateGroupQueryAttentionRoPE",
    "StaticEmptyKV",
    "Surgeon",
    "TensorScatterToScatterND",
    "UnpackGroupQueryAttentionQKV",
]
