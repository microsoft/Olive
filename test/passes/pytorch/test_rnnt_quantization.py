# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

import torch

from olive.common.quant.tensor import QuantTensor
from olive.model import PyTorchModelHandler
from olive.model.config import ModelConfig
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.kquant import kquant_find_qparams
from olive.passes.pytorch.rnnt_gptq import RnntGptq
from olive.passes.pytorch.rnnt_selective_mixed_precision import RnntSelectiveMixedPrecision
from olive.systems.local import LocalSystem


class _RNNTMock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.Linear(12, 8))
        self.joint = torch.nn.Linear(8, 4)


def load_rnnt_mock(model_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return _RNNTMock()


def rnnt_mock_dummy_inputs(model):
    return {"input": torch.ones(1, model.load_model().encoder[0].in_features)}


def rnnt_mock_io_config(model):
    return {"input_names": ["input"], "output_names": ["output"]}


def test_rnnt_gptq_reloads_quantized_weights_after_local_system_serialization(tmp_path):
    original = PyTorchModelHandler(
        model_path="rnnt-original",
        model_loader="load_rnnt_mock",
        model_script=__file__,
        dummy_inputs_func="rnnt_mock_dummy_inputs",
        io_config="rnnt_mock_io_config",
        model_attributes={"custom_attribute": "preserved"},
    )
    original_weight = original.load_model().encoder[0].weight.detach().clone()
    expected = QuantTensor.from_float(original_weight, bits=4, symmetric=False, group_size=4)
    quantization_pass = create_pass_from_dict(RnntGptq, {"bits": 4, "group_size": 4}, disable_search=True)

    result = LocalSystem().run_pass(
        quantization_pass, ModelConfig.from_json(original.to_json()), str(tmp_path / "quantized")
    )
    restored = ModelConfig.from_json(json.loads(json.dumps(result.to_json()))).create_model()

    assert restored.model_path != original.model_path
    assert restored.model is None
    loaded = restored.load_model(cache_model=False)
    assert isinstance(loaded.encoder[0].weight, QuantTensor)
    torch.testing.assert_close(loaded.encoder[0].weight.to_dense(), expected.to_dense())
    assert restored.model_attributes["custom_attribute"] == "preserved"
    assert restored.io_config["input_names"] == ["input"]
    assert restored.get_dummy_inputs()["input"].shape == (1, 8)


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


def test_rnnt_gptq_kquant_matches_shared_group_qparams(tmp_path):
    model = PyTorchModelHandler(model_loader="load_rnnt_mock", model_script=__file__)
    weight = model.load_model(cache_model=False).encoder[0].weight.detach().clone()
    group_size = min(64, weight.shape[-1])
    scales, zero_points = kquant_find_qparams(weight, group_size=group_size, maxq=15, minq=0, symmetric=False)
    expected = QuantTensor.from_float(
        weight,
        bits=4,
        symmetric=False,
        group_size=group_size,
        scales=scales,
        zero_points=zero_points,
    )

    quantization_pass = create_pass_from_dict(
        RnntGptq,
        {"bits": 4, "group_size": group_size, "sym": False, "qparam_strategy": "kquant"},
        disable_search=True,
    )
    result = quantization_pass.run(model, str(tmp_path / "quantized"))

    actual = result.model.encoder[0].weight
    assert torch.allclose(actual.to_dense(), expected.to_dense(), atol=1e-4, rtol=1e-4)
