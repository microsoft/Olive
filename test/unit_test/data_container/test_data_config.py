# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from test.unit_test.utils import get_data_config

import pytest

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.data.registry import Registry

# pylint: disable=attribute-defined-outside-init


class TestDataConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dc_config = get_data_config()

    def test_default_property(self):
        dc_config = DataConfig(name="test_dc_config")
        assert dc_config.dataloader
        assert dc_config.load_dataset
        assert dc_config.pre_process
        assert dc_config.post_process

    def test_customized_property(self):
        # function name
        assert self.dc_config.load_dataset.__name__ == "_test_dataset"
        assert self.dc_config.dataloader.__name__ == "_test_dataloader"
        assert self.dc_config.pre_process.__name__ == Registry.get_default_pre_process_component().__name__
        assert self.dc_config.post_process.__name__ == Registry.get_default_post_process_component().__name__

    def test_user_script_validation(self, tmp_path):
        user_script_py = tmp_path / "user_script.py"

        dc_json = self.dc_config.to_json()
        dc_json["user_script"] = user_script_py.as_posix()

        with pytest.raises(ValueError) as e:  # noqa: PT011
            validate_config(dc_json, DataConfig)

        assert f"{user_script_py} doesn't exist (type=value_error)" in str(e.value)

        with open(user_script_py, "w") as f:
            f.write("")

        validate_config(dc_json, DataConfig)
