# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.model.handler.pytorch import PyTorchModelHandler


# TODO(jambayk): Add more tests
class TestPyTorchModel:
    def test_model_to_json(self, tmp_path):
        script_dir = tmp_path / "model"
        script_dir.mkdir(exist_ok=True)
        model = PyTorchModelHandler(model_path="test_path", script_dir=script_dir)
        model.set_resource("model_script", "model_script")
        model_json = model.to_json()
        assert model_json["config"]["model_path"] == "test_path"
        assert model_json["config"]["script_dir"] == str(script_dir)
        assert model_json["config"]["model_script"] == "model_script"
