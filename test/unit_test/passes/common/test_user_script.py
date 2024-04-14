# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile

import pytest

from olive.common.pydantic_v1 import ValidationError
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.onnx.perf_tuning import OrtPerfTuning


class TestUserScriptConfig:
    def test_no_config(self):
        config = {}
        config = OrtPerfTuning.generate_search_space(DEFAULT_CPU_ACCELERATOR, config, True)
        assert config

    def test_string_config(self):
        config = {"dataloader_func": "dataloader_func"}
        with pytest.raises(ValidationError):
            OrtPerfTuning.generate_search_space(DEFAULT_CPU_ACCELERATOR, config, True)

    def test_object_config(self):
        def dataloader_func():
            return

        config = {"dataloader_func": dataloader_func}
        config = OrtPerfTuning.generate_search_space(DEFAULT_CPU_ACCELERATOR, config, True)
        assert config

    def test_with_user_script(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as user_script_file:
            config = {"dataloader_func": "dataloader_func", "user_script": user_script_file.name}
            config = OrtPerfTuning.generate_search_space(DEFAULT_CPU_ACCELERATOR, config, True)
            assert config
