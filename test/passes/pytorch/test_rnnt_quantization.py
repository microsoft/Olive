# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch

from olive.passes.pytorch.rnnt_gptq import RnntGptq
from olive.passes.pytorch.rnnt_selective_mixed_precision import RnntSelectiveMixedPrecision


class _RNNTMock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.Linear(12, 8))
        self.joint = torch.nn.Linear(8, 4)


def test_rnnt_selective_mixed_precision_metadata():
    model = _RNNTMock()
    pass_inst = RnntSelectiveMixedPrecision.__new__(RnntSelectiveMixedPrecision)
    result = pass_inst._run_for_config(
        model,
        type("Config", (), {"bits": 4, "high_bits": 8, "group_size": 64, "high_group_size": 64})(),
        "/tmp/rnnt_smp.json",
    )

    info = result.model_attributes["mixed_precision_info"]
    assert info["default"]["bits"] == 4
    assert any("joint" in name for name in info["overrides"])
    assert any("encoder" in name for name in info["overrides"])


def test_rnnt_selective_mixed_precision_preserves_qparam_strategy():
    model = _RNNTMock()
    pass_inst = RnntSelectiveMixedPrecision.__new__(RnntSelectiveMixedPrecision)
    result = pass_inst._run_for_config(
        model,
        type(
            "Config",
            (),
            {"bits": 4, "high_bits": 8, "group_size": 64, "high_group_size": 64, "qparam_strategy": "kquant"},
        )(),
        "/tmp/rnnt_smp.json",
    )

    info = result.model_attributes["mixed_precision_info"]
    assert info["default"]["qparam_strategy"] == "kquant"


def test_rnnt_gptq_quantizes_linear_weight():
    model = _RNNTMock()
    pass_inst = RnntSelectiveMixedPrecision.__new__(RnntSelectiveMixedPrecision)
    smp_result = pass_inst._run_for_config(
        model,
        type(
            "Config",
            (),
            {"bits": 4, "high_bits": 8, "group_size": 64, "high_group_size": 64, "qparam_strategy": "kquant"},
        )(),
        "/tmp/rnnt_smp.json",
    )

    gptq_inst = RnntGptq.__new__(RnntGptq)
    result = gptq_inst._run_for_config(
        smp_result,
        type("Config", (), {"qparam_strategy": "kquant"})(),
        "/tmp/rnnt_gptq.pt",
    )

    original = model.encoder[0].weight.detach().clone()
    quantized = result.model.encoder[0].weight.detach()
    assert not torch.equal(original, quantized)
