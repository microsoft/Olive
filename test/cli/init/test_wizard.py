# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest


class TestBuildCalibrationArgs:
    def test_hf_source_with_subset(self):
        from olive.cli.init.helpers import SourceType, build_calibration_args

        calib = {
            "source": SourceType.HF,
            "data_name": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train",
            "num_samples": "128",
        }
        result = build_calibration_args(calib)
        assert result == " -d Salesforce/wikitext --subset wikitext-2-raw-v1 --split train --max_samples 128"

    def test_hf_source_without_subset(self):
        from olive.cli.init.helpers import SourceType, build_calibration_args

        calib = {
            "source": SourceType.HF,
            "data_name": "Salesforce/wikitext",
            "subset": "",
            "split": "train",
            "num_samples": "64",
        }
        result = build_calibration_args(calib)
        assert "--subset" not in result
        assert result == " -d Salesforce/wikitext --split train --max_samples 64"

    def test_local_source(self):
        from olive.cli.init.helpers import SourceType, build_calibration_args

        calib = {"source": SourceType.LOCAL, "data_files": "/data/calib.json"}
        result = build_calibration_args(calib)
        assert result == " --data_files /data/calib.json"

    def test_unknown_source(self):
        from olive.cli.init.helpers import build_calibration_args

        result = build_calibration_args({"source": "unknown"})
        assert result == ""


class TestPromptCalibrationSource:
    @patch("olive.cli.init.helpers._ask")
    def test_default_returns_none(self, mock_ask):
        from olive.cli.init.helpers import SourceType, prompt_calibration_source

        mock_ask.return_value = SourceType.DEFAULT
        result = prompt_calibration_source()
        assert result is None

    @patch("olive.cli.init.helpers._ask")
    def test_hf_source(self, mock_ask):
        from olive.cli.init.helpers import SourceType, prompt_calibration_source

        mock_ask.side_effect = [SourceType.HF, "my_dataset", "my_subset", "validation", "64"]
        result = prompt_calibration_source()
        assert result == {
            "source": SourceType.HF,
            "data_name": "my_dataset",
            "subset": "my_subset",
            "split": "validation",
            "num_samples": "64",
        }

    @patch("olive.cli.init.helpers._ask")
    def test_local_source(self, mock_ask):
        from olive.cli.init.helpers import SourceType, prompt_calibration_source

        mock_ask.side_effect = [SourceType.LOCAL, "/data/calib.json"]
        result = prompt_calibration_source()
        assert result == {"source": SourceType.LOCAL, "data_files": "/data/calib.json"}


class TestAskHelpers:
    @patch("olive.cli.init.helpers.sys.exit")
    def test_ask_exits_on_none(self, mock_exit):
        from olive.cli.init.helpers import _ask

        question = MagicMock()
        question.ask.return_value = None
        _ask(question)
        mock_exit.assert_called_once_with(0)

    def test_ask_returns_value(self):
        from olive.cli.init.helpers import _ask

        question = MagicMock()
        question.ask.return_value = "hello"
        assert _ask(question) == "hello"

    @patch("olive.cli.init.helpers._ask")
    def test_ask_select_raises_go_back(self, mock_ask):
        from olive.cli.init.helpers import GoBackError, _ask_select

        mock_ask.return_value = "__back__"
        with pytest.raises(GoBackError):
            _ask_select("Pick one:", choices=["a", "b"])

    @patch("olive.cli.init.helpers._ask")
    def test_ask_select_returns_value(self, mock_ask):
        from olive.cli.init.helpers import _ask_select

        mock_ask.return_value = "a"
        result = _ask_select("Pick one:", choices=["a", "b"])
        assert result == "a"

    @patch("olive.cli.init.helpers._ask")
    def test_ask_select_no_back(self, mock_ask):
        from olive.cli.init.helpers import _ask_select

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
        from olive.cli.init.onnx_flow import OnnxOperation
        from olive.cli.init.wizard import InitWizard, ModelType, OutputAction

        mock_onnx_select.return_value = OnnxOperation.CONVERT_PRECISION
        mock_select.side_effect = [ModelType.ONNX, OutputAction.COMMAND]
        mock_ask.side_effect = ["/model.onnx", "./output", False]

        InitWizard().start()
        mock_subprocess.assert_not_called()

    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_full_flow_run_now(self, mock_select, mock_ask, mock_subprocess, mock_onnx_select):
        from olive.cli.init.onnx_flow import OnnxOperation
        from olive.cli.init.wizard import InitWizard, ModelType, OutputAction

        mock_onnx_select.return_value = OnnxOperation.CONVERT_PRECISION
        mock_select.side_effect = [ModelType.ONNX, OutputAction.RUN]
        mock_ask.side_effect = ["/model.onnx", "./output"]

        InitWizard().start()
        mock_subprocess.assert_called_once()

    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_full_flow_generate_config(self, mock_select, mock_ask, mock_subprocess, mock_onnx_select):
        from olive.cli.init.onnx_flow import OnnxOperation
        from olive.cli.init.wizard import InitWizard, ModelType, OutputAction

        mock_onnx_select.return_value = OnnxOperation.CONVERT_PRECISION
        mock_select.side_effect = [ModelType.ONNX, OutputAction.CONFIG]
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
        from olive.cli.init.helpers import GoBackError
        from olive.cli.init.onnx_flow import OnnxOperation
        from olive.cli.init.wizard import InitWizard, ModelType, OutputAction

        mock_onnx_select.return_value = OnnxOperation.CONVERT_PRECISION
        select_values = [ModelType.ONNX, GoBackError, ModelType.ONNX, OutputAction.COMMAND]

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
        from olive.cli.init.onnx_flow import OnnxOperation
        from olive.cli.init.wizard import InitWizard, ModelType, OutputAction

        mock_onnx_select.return_value = OnnxOperation.CONVERT_PRECISION
        mock_select.side_effect = [ModelType.ONNX, OutputAction.COMMAND]
        mock_ask.side_effect = ["/model.onnx", "./output", True]

        InitWizard().start()
        mock_subprocess.assert_called_once()

    @patch("olive.cli.init.wizard.Path")
    @patch("olive.cli.init.onnx_flow._ask_select")
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_config_with_existing_file(self, mock_select, mock_ask, mock_subprocess, mock_onnx_select, mock_path):
        from olive.cli.init.onnx_flow import OnnxOperation
        from olive.cli.init.wizard import InitWizard, ModelType, OutputAction

        mock_onnx_select.return_value = OnnxOperation.CONVERT_PRECISION
        mock_select.side_effect = [ModelType.ONNX, OutputAction.CONFIG]
        mock_ask.side_effect = ["/model.onnx", "./output"]
        mock_path.return_value.__truediv__ = lambda self, x: MagicMock(exists=lambda: True)

        InitWizard().start()
        mock_subprocess.assert_called_once()

    def test_no_command_raises_go_back(self):
        """When _run_model_flow returns {} the wizard should go back, not silently finish."""
        from olive.cli.init.helpers import GoBackError
        from olive.cli.init.wizard import InitWizard

        wizard = InitWizard()
        result = {"not_command": True}  # no "command" key
        with pytest.raises(GoBackError):
            wizard._prompt_output(result)  # pylint: disable=protected-access

    @patch("olive.cli.init.pytorch_flow._ask_select")
    @patch("olive.cli.init.pytorch_flow._optimize_flow", return_value={"command": "olive optimize -m m"})
    @patch("olive.cli.init.wizard.subprocess.run")
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_pytorch_flow_dispatch(self, mock_select, mock_ask, mock_subprocess, mock_opt, mock_pt_select):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.pytorch_flow import OP_OPTIMIZE
        from olive.cli.init.wizard import InitWizard, ModelType, OutputAction

        mock_pt_select.return_value = OP_OPTIMIZE
        mock_select.side_effect = [ModelType.PYTORCH, SourceType.HF, OutputAction.RUN]
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
        from olive.cli.init.helpers import DiffuserVariant
        from olive.cli.init.wizard import InitWizard, ModelType, OutputAction

        mock_diff_select.return_value = OP_EXPORT
        mock_select.side_effect = [ModelType.DIFFUSERS, DiffuserVariant.AUTO, OutputAction.RUN]
        mock_ask.side_effect = ["my-model", "./output"]
        mock_diff_ask.return_value = "float16"

        InitWizard().start()
        mock_subprocess.assert_called_once()


class TestPromptPytorchSource:
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_hf_source(self, mock_select, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.wizard import InitWizard

        mock_select.return_value = SourceType.HF
        mock_ask.return_value = "meta-llama/Llama-3.1-8B"
        result = InitWizard()._prompt_pytorch_source()  # pylint: disable=protected-access
        assert result == {"source_type": SourceType.HF, "model_path": "meta-llama/Llama-3.1-8B"}

    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_local_source(self, mock_select, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.wizard import InitWizard

        mock_select.return_value = SourceType.LOCAL
        mock_ask.return_value = "./my-model/"
        result = InitWizard()._prompt_pytorch_source()  # pylint: disable=protected-access
        assert result == {"source_type": SourceType.LOCAL, "model_path": "./my-model/"}

    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_azureml_source(self, mock_select, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.wizard import InitWizard

        mock_select.return_value = SourceType.AZUREML
        mock_ask.return_value = "azureml://registries/r/models/m/versions/1"
        result = InitWizard()._prompt_pytorch_source()  # pylint: disable=protected-access
        assert result == {"source_type": SourceType.AZUREML, "model_path": "azureml://registries/r/models/m/versions/1"}

    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_script_source_full(self, mock_select, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.wizard import InitWizard

        mock_select.return_value = SourceType.SCRIPT
        mock_ask.side_effect = ["train.py", "./src", "my-model"]
        result = InitWizard()._prompt_pytorch_source()  # pylint: disable=protected-access
        assert result == {
            "source_type": SourceType.SCRIPT,
            "model_script": "train.py",
            "script_dir": "./src",
            "model_path": "my-model",
        }

    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_script_source_minimal(self, mock_select, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.wizard import InitWizard

        mock_select.return_value = SourceType.SCRIPT
        mock_ask.side_effect = ["train.py", "", ""]  # no script_dir, no model_path
        result = InitWizard()._prompt_pytorch_source()  # pylint: disable=protected-access
        assert result == {"source_type": SourceType.SCRIPT, "model_script": "train.py"}
        assert "script_dir" not in result
        assert "model_path" not in result


class TestPromptDiffusersSource:
    @patch("olive.cli.init.wizard._ask")
    @patch("olive.cli.init.wizard._ask_select")
    def test_diffusers_source(self, mock_select, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.wizard import InitWizard

        mock_select.return_value = "sdxl"
        mock_ask.return_value = "stabilityai/sdxl-base-1.0"
        result = InitWizard()._prompt_diffusers_source()  # pylint: disable=protected-access
        assert result == {
            "source_type": SourceType.HF,
            "model_path": "stabilityai/sdxl-base-1.0",
            "variant": "sdxl",
        }


class TestPromptModelSource:
    def test_unknown_model_type_returns_empty(self):
        from olive.cli.init.wizard import InitWizard

        result = InitWizard()._prompt_model_source("unknown")  # pylint: disable=protected-access
        assert not result


class TestRunModelFlow:
    def test_unknown_model_type_returns_empty(self):
        from olive.cli.init.wizard import InitWizard

        result = InitWizard()._run_model_flow("unknown", {})  # pylint: disable=protected-access
        assert not result
