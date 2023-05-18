# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile

import pytest
from pydantic import ValidationError

from olive.passes.onnx import OrtPerfTuning


class TestUserScriptConfig:
    def test_no_config(self):
        config = {}
        config_class, config = OrtPerfTuning.generate_search_space(config, True)
        assert config_class and config

    def test_string_config(self):
        config = {"dataloader_func": "dataloader_func"}
        with pytest.raises(ValidationError):
            OrtPerfTuning.generate_search_space(config, True)

    def test_object_config(self):
        def dataloader_func():
            return

        config = {"dataloader_func": dataloader_func}
        config_class, config = OrtPerfTuning.generate_search_space(config, True)
        assert config_class and config

    def test_with_user_script(self):
        user_script_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        config = {"dataloader_func": "dataloader_func", "user_script": user_script_file.name}
        config_class, config = OrtPerfTuning.generate_search_space(config, True)
        assert config_class and config
