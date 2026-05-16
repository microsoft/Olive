# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for olive quant state-dict helpers.

The full ``prepare_model`` / ``finalize`` pipeline is exercised end-to-end
by ``test_rtn.py`` against a real HF model. This file targets the
``install_quant_tensor_param`` helper used by both the 2D linear/embed
path and the 3D fused-MoE path. The tests stay free of network access.
"""

import torch
from torch import nn

from olive.common.quant.state_dict import install_quant_tensor_param
from olive.common.quant.tensor import QuantTensor


class _ExpertsBlock(nn.Module):
    """Fake fused-3D experts module (gpt-oss / Qwen3-MoE style)."""

    def __init__(self, num_experts: int = 4, out_features: int = 32, in_features: int = 16):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, out_features, in_features, dtype=torch.float32),
            requires_grad=False,
        )


class TestInstallQuantTensorParam:
    def test_install_3d_quant_tensor(self):
        block = _ExpertsBlock()
        qt = QuantTensor.from_float(block.gate_up_proj.detach(), bits=4, symmetric=True, group_size=16)

        install_quant_tensor_param(block, "gate_up_proj", qt)

        # Parameter is a QuantTensor (tensor subclass parameters return
        # the subclass instance directly from nn.Parameter.__new__).
        param = block._parameters["gate_up_proj"]
        assert isinstance(param, QuantTensor)
        # Sibling buffers are registered and share storage with the QuantTensor.
        buffers = dict(block.named_buffers())
        assert "gate_up_proj_qweight" in buffers
        assert "gate_up_proj_scales" in buffers
        assert "gate_up_proj_qzeros" not in buffers  # symmetric → no qzeros
        assert buffers["gate_up_proj_qweight"] is param.qweight
        assert buffers["gate_up_proj_scales"] is param.scales

    def test_install_asymmetric_emits_qzeros(self):
        block = _ExpertsBlock()
        qt = QuantTensor.from_float(block.gate_up_proj.detach(), bits=4, symmetric=False, group_size=16)

        install_quant_tensor_param(block, "gate_up_proj", qt)

        buffers = dict(block.named_buffers())
        assert "gate_up_proj_qzeros" in buffers
        assert buffers["gate_up_proj_qzeros"] is block._parameters["gate_up_proj"].qzeros

    def test_state_dict_drops_quant_tensor_entry(self):
        """After install, state_dict must contain only plain Tensors (no QuantTensor entry)."""
        block = _ExpertsBlock()
        qt = QuantTensor.from_float(block.gate_up_proj.detach(), bits=4, symmetric=True, group_size=16)

        install_quant_tensor_param(block, "gate_up_proj", qt)

        sd = block.state_dict()
        # No QuantTensor instance should appear in the state_dict.
        for key, value in sd.items():
            assert not isinstance(value, QuantTensor), f"{key} should not be a QuantTensor"
        # The plain ``gate_up_proj`` key (the QuantTensor parameter) is dropped;
        # the buffers carry the on-disk representation.
        assert "gate_up_proj" not in sd
        assert "gate_up_proj_qweight" in sd
        assert "gate_up_proj_scales" in sd

    def test_install_on_linear_module(self):
        """Smoke test on a normal nn.Linear so that F.linear forward still works."""
        linear = nn.Linear(16, 32, bias=False)
        weight = linear.weight.detach().clone()
        qt = QuantTensor.from_float(weight, bits=4, symmetric=True, group_size=16)

        install_quant_tensor_param(linear, "weight", qt)

        assert isinstance(linear.weight, QuantTensor)
        # forward dispatches through QuantTensor.__torch_function__ and returns a plain Tensor
        x = torch.randn(2, 16)
        y = linear(x)
        assert y.shape == (2, 32)
        assert not isinstance(y, QuantTensor)


def test_module_weight_has_quant_info_only_for_marked_params():
    """Regression: discovery must not pick up LayerNorm / Conv2d weights.

    GPTQ / AutoClip discover quantizable layers via
    ``_module_weight_has_quant_info``. Modules that happen to expose a
    ``weight`` attribute but never had ``quant_info`` stamped on it must
    be left alone.
    """
    from olive.common.quant.utils import WeightQuantizer
    from olive.passes.pytorch.quant_utils import QuantInfo, _module_weight_has_quant_info

    ln = nn.LayerNorm(16)
    conv = nn.Conv2d(3, 8, kernel_size=3)
    linear_unmarked = nn.Linear(16, 32, bias=False)
    linear_marked = nn.Linear(16, 32, bias=False)
    linear_marked.weight.quant_info = QuantInfo(quantizer=WeightQuantizer(bits=4, symmetric=True, group_size=16))

    assert not _module_weight_has_quant_info(ln)
    assert not _module_weight_has_quant_info(conv)
    assert not _module_weight_has_quant_info(linear_unmarked)
    assert _module_weight_has_quant_info(linear_marked)
