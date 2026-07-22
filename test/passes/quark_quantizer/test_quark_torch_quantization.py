# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

pytestmark = pytest.mark.amd


def test_build_algo_configs_awq_builds_awqconfig():
    """A recipe-supplied AWQ dict is converted into an AWQConfig for get_config(algo_configs=...)."""
    from quark.torch.quantization.config.config import AWQConfig

    from olive.passes.quark_quantizer.torch.quark_torch_quantization import _build_algo_configs

    result = _build_algo_configs({"awq": {"scaling_layers": [], "model_decoder_layers": "model.layers"}})
    assert isinstance(result["awq"], AWQConfig)
    assert result["awq"].model_decoder_layers == "model.layers"


def test_build_algo_configs_rejects_unsupported_algo():
    """Only 'awq' is supported today; other algorithms raise a clear ValueError."""
    from olive.passes.quark_quantizer.torch.quark_torch_quantization import _build_algo_configs

    with pytest.raises(ValueError, match=r"algo_configs for 'gptq'"):
        _build_algo_configs({"gptq": {"scaling_layers": []}})


def test_build_algo_configs_none_when_empty():
    """No algo_configs supplied → None (built-in Quark config path unchanged)."""
    from olive.passes.quark_quantizer.torch.quark_torch_quantization import _build_algo_configs

    assert _build_algo_configs(None) is None
    assert _build_algo_configs({}) is None
