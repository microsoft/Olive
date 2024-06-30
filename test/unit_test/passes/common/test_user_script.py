# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.onnx.perf_tuning import OrtPerfTuning


class TestUserScriptConfig:
    def test_no_config(self):
        config = {}
        config = OrtPerfTuning.generate_search_space(DEFAULT_CPU_ACCELERATOR, config, True)
        assert config
