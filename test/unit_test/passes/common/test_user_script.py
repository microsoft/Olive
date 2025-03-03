# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.onnx.session_params_tuning import OrtSessionParamsTuning


class TestUserScriptConfig:
    def test_no_config(self):
        config = OrtSessionParamsTuning.generate_config(DEFAULT_CPU_ACCELERATOR, disable_search=True)
        assert config
        assert OrtSessionParamsTuning.validate_config(config, DEFAULT_CPU_ACCELERATOR)
