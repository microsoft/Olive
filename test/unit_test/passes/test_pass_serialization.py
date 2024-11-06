# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.olive_pass import FullPassConfig
from olive.passes.onnx.conversion import OnnxConversion


def test_pass_serialization():
    onnx_conversion_config = {}
    config = OnnxConversion.generate_search_space(DEFAULT_CPU_ACCELERATOR, onnx_conversion_config)
    onnx_conversion = OnnxConversion(DEFAULT_CPU_ACCELERATOR, config)
    json = onnx_conversion.to_json(True)

    cfg = FullPassConfig.from_json(json)
    p = cfg.create_pass()
    assert isinstance(p, OnnxConversion)
    assert p.accelerator_spec == DEFAULT_CPU_ACCELERATOR
    assert p.config == config
