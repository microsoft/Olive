# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for olive.passes.pytorch.quant_utils helpers.

The full ``prepare_model`` / ``finalize`` pipeline is exercised end-to-end
by ``test_rtn.py`` against a real HF model. This file targets the helpers
that the MoE path adds — primarily ``flatten_quant_tensor_params`` — and
keeps the tests free of network access.
"""

import torch
from torch import nn

from olive.common.quant.tensor import QuantTensor
from olive.passes.pytorch.quant_utils import flatten_quant_tensor_params


class _ExpertsBlock(nn.Module):
    """Fake fused-3D experts module (gpt-oss / Qwen3-MoE style).

    Carries a single 3D ``nn.Parameter`` of shape ``(num_experts, out, in)``,
    matching the layout that ``prepare_model`` annotates with
    ``quant_info_3d`` and that ``finalize`` rewrites into a 3D
    :class:`QuantTensor` parameter.
    """

    def __init__(self, num_experts: int = 4, out_features: int = 32, in_features: int = 16):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, out_features, in_features, dtype=torch.float32),
            requires_grad=False,
        )


class TestFlattenQuantTensorParams:
    def test_flatten_3d_quant_tensor_param(self):
        block = _ExpertsBlock()
        qt = QuantTensor.from_float(block.gate_up_proj.detach(), bits=4, symmetric=True, group_size=16)
        block._parameters["gate_up_proj"] = nn.Parameter(qt, requires_grad=False)

        flatten_quant_tensor_params(block)

        # The original parameter is gone, replaced by buffers.
        assert "gate_up_proj" not in dict(block.named_parameters())
        buffer_names = dict(block.named_buffers())
        assert "gate_up_proj_qweight" in buffer_names
        assert "gate_up_proj_scales" in buffer_names
        # symmetric → no qzeros buffer
        assert "gate_up_proj_qzeros" not in buffer_names
        # metadata stays as a plain attribute (not in state_dict)
        meta = block.gate_up_proj_quant_meta
        assert meta["bits"] == 4
        assert meta["group_size"] == 16
        assert meta["symmetric"] is True
        assert tuple(meta["shape"]) == (4, 32, 16)

    def test_flatten_asymmetric_emits_qzeros(self):
        block = _ExpertsBlock()
        qt = QuantTensor.from_float(block.gate_up_proj.detach(), bits=4, symmetric=False, group_size=16)
        block._parameters["gate_up_proj"] = nn.Parameter(qt, requires_grad=False)

        flatten_quant_tensor_params(block)

        buffer_names = dict(block.named_buffers())
        assert "gate_up_proj_qzeros" in buffer_names

    def test_state_dict_serializable_after_flatten(self):
        """After flattening, the module's state_dict must be plain Tensors."""
        block = _ExpertsBlock()
        qt = QuantTensor.from_float(block.gate_up_proj.detach(), bits=4, symmetric=True, group_size=16)
        block._parameters["gate_up_proj"] = nn.Parameter(qt, requires_grad=False)

        flatten_quant_tensor_params(block)

        sd = block.state_dict()
        for key, value in sd.items():
            assert type(value) is torch.Tensor, f"{key} is not a plain Tensor: {type(value)}"

    def test_no_op_on_module_without_quant_tensor(self):
        """Plain modules pass through unchanged."""
        block = nn.Linear(8, 8)
        before = {n: id(p) for n, p in block.named_parameters()}
        flatten_quant_tensor_params(block)
        after = {n: id(p) for n, p in block.named_parameters()}
        assert before == after
