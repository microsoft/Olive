# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest


class TestBuildCalibrationArgs:
    def test_hf_source_with_subset(self):
        from olive.cli.init.wizard import SOURCE_HF, build_calibration_args

        calib = {
            "source": SOURCE_HF,
            "data_name": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train",
            "num_samples": "128",
        }
        result = build_calibration_args(calib)
        assert result == " -d Salesforce/wikitext --subset wikitext-2-raw-v1 --split train --max_samples 128"

    def test_hf_source_without_subset(self):
        from olive.cli.init.wizard import SOURCE_HF, build_calibration_args

        calib = {
            "source": SOURCE_HF,
            "data_name": "Salesforce/wikitext",
            "subset": "",
            "split": "train",
            "num_samples": "64",
        }
        result = build_calibration_args(calib)
        assert "--subset" not in result
        assert " -d Salesforce/wikitext --split train --max_samples 64" == result

    def test_local_source(self):
        from olive.cli.init.wizard import SOURCE_LOCAL, build_calibration_args

        calib = {"source": SOURCE_LOCAL, "data_files": "/data/calib.json"}
        result = build_calibration_args(calib)
        assert result == " --data_files /data/calib.json"

    def test_unknown_source(self):
        from olive.cli.init.wizard import build_calibration_args

        result = build_calibration_args({"source": "unknown"})
        assert result == ""


class TestPromptCalibrationSource:
    @patch("olive.cli.init.wizard._ask")
    def test_default_returns_none(self, mock_ask):
        from olive.cli.init.wizard import SOURCE_DEFAULT, prompt_calibration_source

        mock_ask.return_value = SOURCE_DEFAULT
        result = prompt_calibration_source()
        assert result is None

    @patch("olive.cli.init.wizard._ask")
    def test_hf_source(self, mock_ask):
        from olive.cli.init.wizard import SOURCE_HF, prompt_calibration_source

        mock_ask.side_effect = [SOURCE_HF, "my_dataset", "my_subset", "validation", "64"]
        result = prompt_calibration_source()
        assert result == {
            "source": SOURCE_HF,
            "data_name": "my_dataset",
            "subset": "my_subset",
            "split": "validation",
            "num_samples": "64",
        }

    @patch("olive.cli.init.wizard._ask")
    def test_local_source(self, mock_ask):
        from olive.cli.init.wizard import SOURCE_LOCAL, prompt_calibration_source

        mock_ask.side_effect = [SOURCE_LOCAL, "/data/calib.json"]
        result = prompt_calibration_source()
        assert result == {"source": SOURCE_LOCAL, "data_files": "/data/calib.json"}


class TestAskHelpers:
    @patch("olive.cli.init.wizard.sys.exit")
    def test_ask_exits_on_none(self, mock_exit):
        from olive.cli.init.wizard import _ask

        question = MagicMock()
        question.ask.return_value = None
        _ask(question)
        mock_exit.assert_called_once_with(0)

    def test_ask_returns_value(self):
        from olive.cli.init.wizard import _ask

        question = MagicMock()
        question.ask.return_value = "hello"
        assert _ask(question) == "hello"

    @patch("olive.cli.init.wizard._ask")
    def test_ask_select_raises_go_back(self, mock_ask):
        from olive.cli.init.wizard import GoBackError, _ask_select

        mock_ask.return_value = "__back__"
        with pytest.raises(GoBackError):
            _ask_select("Pick one:", choices=["a", "b"])

    @patch("olive.cli.init.wizard._ask")
    def test_ask_select_returns_value(self, mock_ask):
        from olive.cli.init.wizard import _ask_select

        mock_ask.return_value = "a"
        result = _ask_select("Pick one:", choices=["a", "b"])
        assert result == "a"

    @patch("olive.cli.init.wizard._ask")
    def test_ask_select_no_back(self, mock_ask):
        from olive.cli.init.wizard import _ask_select

        mock_ask.return_value = "a"
        result = _ask_select("Pick:", choices=["a"], allow_back=False)
        assert result == "a"


class TestInitWizard:
    """Test InitWizard end-to-end.

    The wizard dispatches to onnx_flow which imports _ask/_ask_select at module
    level, so we must patch both wizard and onnx_flow references.
    """

    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_full_flow_generate_command(self, mock_select, mock_ask, mock_subprocess, mock_onnx_select):
        from olive.cli.init.onnx_flow import OP_CONVERT_PRECISION
        from olive.cli.init.wizard import ACTION_COMMAND, MODEL_ONNX, InitWizard

        mock_onnx_select.return_value = OP_CONVERT_PRECISION
        mock_select.side_effect = [MODEL_ONNX, ACTION_COMMAND]
        mock_ask.side_effect = ["/model.onnx", "./output", False]

        InitWizard().start()
        mock_subprocess.assert_not_called()

    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_full_flow_run_now(self, mock_select, mock_ask, mock_subprocess, mock_onnx_select):
        from olive.cli.init.onnx_flow import OP_CONVERT_PRECISION
        from olive.cli.init.wizard import ACTION_RUN, MODEL_ONNX, InitWizard

        mock_onnx_select.return_value = OP_CONVERT_PRECISION
        mock_select.side_effect = [MODEL_ONNX, ACTION_RUN]
        mock_ask.side_effect = ["/model.onnx", "./output"]

        InitWizard().start()
        mock_subprocess.assert_called_once()

    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_full_flow_generate_config(self, mock_select, mock_ask, mock_subprocess, mock_onnx_select):
        from olive.cli.init.onnx_flow import OP_CONVERT_PRECISION
        from olive.cli.init.wizard import ACTION_CONFIG, MODEL_ONNX, InitWizard

        mock_onnx_select.return_value = OP_CONVERT_PRECISION
        mock_select.side_effect = [MODEL_ONNX, ACTION_CONFIG]
        mock_ask.side_effect = ["/model.onnx", "./output"]

        InitWizard().start()
        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        assert "--save_config_file" in cmd
        assert "--dry_run" in cmd

    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_go_back(self, mock_select, mock_ask, mock_onnx_select):
        from olive.cli.init.onnx_flow import OP_CONVERT_PRECISION
        from olive.cli.init.wizard import ACTION_COMMAND, MODEL_ONNX, GoBackError, InitWizard

        mock_onnx_select.return_value = OP_CONVERT_PRECISION
        select_values = [MODEL_ONNX, GoBackError, MODEL_ONNX, ACTION_COMMAND]

        def select_with_goback(*args, **kwargs):
            val = select_values.pop(0)
            if val is GoBackError:
                raise GoBackError
            return val

        mock_select.side_effect = select_with_goback
        mock_ask.side_effect = ["/model.onnx", "./output", False]

        InitWizard().start()

    @patch("olive.cli.init.wizard.sys.exit")
    @patch("olive.cli.init.wizard._ask_select")
    def test_keyboard_interrupt(self, mock_select, mock_exit):
        from olive.cli.init.wizard import InitWizard

        mock_select.side_effect = KeyboardInterrupt
        InitWizard().start()
        mock_exit.assert_called_once_with(0)

    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_command_then_run_now(self, mock_select, mock_ask, mock_subprocess, mock_onnx_select):
        from olive.cli.init.onnx_flow import OP_CONVERT_PRECISION
        from olive.cli.init.wizard import ACTION_COMMAND, MODEL_ONNX, InitWizard

        mock_onnx_select.return_value = OP_CONVERT_PRECISION
        mock_select.side_effect = [MODEL_ONNX, ACTION_COMMAND]
        mock_ask.side_effect = ["/model.onnx", "./output", True]

        InitWizard().start()
        mock_subprocess.assert_called_once()

    @patch("olive.cli.init.wizard.Path")
    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_config_with_existing_file(self, mock_select, mock_ask, mock_subprocess, mock_onnx_select, mock_path):
        from olive.cli.init.onnx_flow import OP_CONVERT_PRECISION
        from olive.cli.init.wizard import ACTION_CONFIG, MODEL_ONNX, InitWizard

        mock_onnx_select.return_value = OP_CONVERT_PRECISION
        mock_select.side_effect = [MODEL_ONNX, ACTION_CONFIG]
        mock_ask.side_effect = ["/model.onnx", "./output"]
        mock_path.return_value.__truediv__ = lambda self, x: MagicMock(exists=lambda: True)

        InitWizard().start()
        mock_subprocess.assert_called_once()

    def test_no_command_raises_go_back(self):
        """When _run_model_flow returns {} the wizard should go back, not silently finish."""
        from olive.cli.init.wizard import GoBackError, InitWizard

        wizard = InitWizard()
        result = {"not_command": True}  # no "command" key
        with pytest.raises(GoBackError):
            wizard._prompt_output(result)

    @patch("olive.cli.init.pytorch_flow._ask_select")
    @patch("olive.cli.init.pytorch_flow._optimize_flow", return_value={"command": "olive optimize -m m"})
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_pytorch_flow_dispatch(self, mock_select, mock_ask, mock_subprocess, mock_opt, mock_pt_select):
        from olive.cli.init.pytorch_flow import OP_OPTIMIZE
        from olive.cli.init.wizard import ACTION_RUN, MODEL_PYTORCH, SOURCE_HF, InitWizard

        mock_pt_select.return_value = OP_OPTIMIZE
        mock_select.side_effect = [MODEL_PYTORCH, SOURCE_HF, ACTION_RUN]
        mock_ask.side_effect = ["meta-llama/Llama-3.1-8B", "./output"]

        InitWizard().start()
        mock_subprocess.assert_called_once()

    @patch("olive.cli.init.diffusers_flow._ask_select")
    @patch("olive.cli.init.diffusers_flow._ask")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_diffusers_flow_dispatch(self, mock_select, mock_ask, mock_subprocess, mock_diff_ask, mock_diff_select):
        from olive.cli.init.diffusers_flow import OP_EXPORT
        from olive.cli.init.wizard import ACTION_RUN, MODEL_DIFFUSERS, VARIANT_AUTO, InitWizard

        mock_diff_select.return_value = OP_EXPORT
        mock_select.side_effect = [MODEL_DIFFUSERS, VARIANT_AUTO, ACTION_RUN]
        mock_ask.side_effect = ["my-model", "./output"]
        mock_diff_ask.return_value = "float16"

        InitWizard().start()
        mock_subprocess.assert_called_once()


class TestPromptPytorchSource:
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_hf_source(self, mock_select, mock_ask):
        from olive.cli.init.wizard import SOURCE_HF, InitWizard

        mock_select.return_value = SOURCE_HF
        mock_ask.return_value = "meta-llama/Llama-3.1-8B"
        result = InitWizard()._prompt_pytorch_source()
        assert result == {"source_type": SOURCE_HF, "model_path": "meta-llama/Llama-3.1-8B"}

    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_local_source(self, mock_select, mock_ask):
        from olive.cli.init.wizard import SOURCE_LOCAL, InitWizard

        mock_select.return_value = SOURCE_LOCAL
        mock_ask.return_value = "./my-model/"
        result = InitWizard()._prompt_pytorch_source()
        assert result == {"source_type": SOURCE_LOCAL, "model_path": "./my-model/"}

    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_azureml_source(self, mock_select, mock_ask):
        from olive.cli.init.wizard import SOURCE_AZUREML, InitWizard

        mock_select.return_value = SOURCE_AZUREML
        mock_ask.return_value = "azureml://registries/r/models/m/versions/1"
        result = InitWizard()._prompt_pytorch_source()
        assert result == {"source_type": SOURCE_AZUREML, "model_path": "azureml://registries/r/models/m/versions/1"}

    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_script_source_full(self, mock_select, mock_ask):
        from olive.cli.init.wizard import SOURCE_SCRIPT, InitWizard

        mock_select.return_value = SOURCE_SCRIPT
        mock_ask.side_effect = ["train.py", "./src", "my-model"]
        result = InitWizard()._prompt_pytorch_source()
        assert result == {
            "source_type": SOURCE_SCRIPT,
            "model_script": "train.py",
            "script_dir": "./src",
            "model_path": "my-model",
        }

    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_script_source_minimal(self, mock_select, mock_ask):
        from olive.cli.init.wizard import SOURCE_SCRIPT, InitWizard

        mock_select.return_value = SOURCE_SCRIPT
        mock_ask.side_effect = ["train.py", "", ""]  # no script_dir, no model_path
        result = InitWizard()._prompt_pytorch_source()
        assert result == {"source_type": SOURCE_SCRIPT, "model_script": "train.py"}
        assert "script_dir" not in result
        assert "model_path" not in result


class TestPromptDiffusersSource:
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_diffusers_source(self, mock_select, mock_ask):
        from olive.cli.init.wizard import SOURCE_HF, InitWizard

        mock_select.return_value = "sdxl"
        mock_ask.return_value = "stabilityai/sdxl-base-1.0"
        result = InitWizard()._prompt_diffusers_source()
        assert result == {
            "source_type": SOURCE_HF,
            "model_path": "stabilityai/sdxl-base-1.0",
            "variant": "sdxl",
        }


class TestPromptModelSource:
    def test_unknown_model_type_returns_empty(self):
        from olive.cli.init.wizard import InitWizard

        result = InitWizard()._prompt_model_source("unknown")
        assert result == {}


class TestRunModelFlow:
    def test_unknown_model_type_returns_empty(self):
        from olive.cli.init.wizard import InitWizard

        result = InitWizard()._run_model_flow("unknown", {})
        assert result == {}
