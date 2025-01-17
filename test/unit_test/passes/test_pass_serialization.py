# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.olive_pass import FullPassConfig
from olive.passes.onnx.conversion import OnnxConversion


@pytest.mark.parametrize("host_device", [None, "cpu", "gpu"])
def test_pass_serialization(host_device):
    config = OnnxConversion.generate_config(DEFAULT_CPU_ACCELERATOR)
    onnx_conversion = OnnxConversion(DEFAULT_CPU_ACCELERATOR, config, host_device=host_device)
    json = onnx_conversion.to_json(True)

    cfg = FullPassConfig.from_json(json)
    p = cfg.create_pass()
    assert isinstance(p, OnnxConversion)
    assert p.accelerator_spec == DEFAULT_CPU_ACCELERATOR
    assert p.config == config
    assert p.host_device == host_device
