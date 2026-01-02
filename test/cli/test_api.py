# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest

# pylint: disable=W0611, W0212


@pytest.mark.unit
class TestOlivePythonAPI:
    """Test cases for Olive Python API functions."""

    def test_api_module_structure(self):
        """Test that API module has expected functions."""
        from olive import (
            capture_onnx_graph,
            convert_adapters,
            diffusion_lora,
            extract_adapters,
            finetune,
            generate_adapter,
            generate_cost_model,
            quantize,
            tune_session_params,
        )

        # Test that all functions are callable
        api_functions = [
            capture_onnx_graph,
            convert_adapters,
            diffusion_lora,
            extract_adapters,
            finetune,
            generate_adapter,
            generate_cost_model,
            quantize,
            tune_session_params,
        ]

        for func in api_functions:
            assert callable(func), f"{func.__name__} should be callable"

    @patch("olive.cli.api.FineTuneCommand")
    def test_finetune_function_basic(self, mock_command_class):
        """Test basic functionality of finetune function with mocked dependencies."""
        from olive import finetune

        # Mock command and output
        mock_command = MagicMock()
        mock_output = MagicMock()
        mock_command.run.return_value = mock_output
        mock_command_class.return_value = mock_command

        # Test with minimal args
        result = finetune("test_model", data_name="test_data")

        # Verify command was created and run
        mock_command_class.assert_called_once()
        mock_command.run.assert_called_once()
        assert result is mock_output

    @patch("olive.cli.api.QuantizeCommand")
    def test_quantize_function_basic(self, mock_cmd_cls):
        from olive import quantize

        mock_cmd = MagicMock()
        mock_output = MagicMock()
        mock_cmd.run.return_value = mock_output
        mock_cmd_cls.return_value = mock_cmd

        result = quantize("test_model")

        mock_cmd_cls.assert_called_once()
        mock_cmd.run.assert_called_once()
        assert result is mock_output

    @patch("olive.cli.api.CaptureOnnxGraphCommand")
    def test_capture_cmd_basic(self, mock_cmd_cls):
        from olive import capture_onnx_graph

        mock_cmd = MagicMock()
        mock_output = MagicMock()
        mock_cmd.run.return_value = mock_output
        mock_cmd_cls.return_value = mock_cmd

        result = capture_onnx_graph("test_model", task="text-classification", output_path="dummy")

        mock_cmd_cls.assert_called_once()
        mock_cmd.run.assert_called_once()
        assert result is mock_output

    @patch("olive.cli.api.GenerateAdapterCommand")
    def test_generate_adapter_basic(self, mock_cmd_cls):
        from olive import generate_adapter

        mock_cmd = MagicMock()
        mock_output = MagicMock()
        mock_cmd.run.return_value = mock_output
        mock_cmd_cls.return_value = mock_cmd

        result = generate_adapter("dummy.onnx")

        mock_cmd_cls.assert_called_once()
        mock_cmd.run.assert_called_once()
        assert result is mock_output

    @patch("olive.cli.api.SessionParamsTuningCommand")
    def test_tune_session_params_basic(self, mock_cmd_cls):
        from olive import tune_session_params

        mock_cmd = MagicMock()
        mock_output = MagicMock()
        mock_cmd.run.return_value = mock_output
        mock_cmd_cls.return_value = mock_cmd

        tune_session_params("dummy.onnx")

        mock_cmd_cls.assert_called_once()
        mock_cmd.run.assert_called_once()

    @patch("olive.cli.api.GenerateCostModelCommand")
    def test_generate_cost_model_basic(self, mock_cmd_cls):
        from olive import generate_cost_model

        mock_cmd = MagicMock()
        mock_cmd_cls.return_value = mock_cmd

        generate_cost_model("hf-model")

        mock_cmd_cls.assert_called_once()
        mock_cmd.run.assert_called_once()

    @patch("olive.cli.api.ConvertAdaptersCommand")
    def test_convert_adapters_basic(self, mock_cmd_cls):
        from olive import convert_adapters

        mock_cmd = MagicMock()
        mock_cmd_cls.return_value = mock_cmd

        convert_adapters("adapter_folder", output_path="out.bin")

        mock_cmd_cls.assert_called_once()
        mock_cmd.run.assert_called_once()

    @patch("olive.cli.api.ExtractAdaptersCommand")
    def test_extract_adapters_basic(self, mock_cmd_cls):
        from olive import extract_adapters

        mock_cmd = MagicMock()
        mock_cmd_cls.return_value = mock_cmd

        extract_adapters("hf-model")

        mock_cmd_cls.assert_called_once()
        mock_cmd.run.assert_called_once()

    @patch("olive.cli.api.DiffusionLoraCommand")
    def test_diffusion_lora_basic(self, mock_cmd_cls):
        from olive import diffusion_lora

        mock_cmd = MagicMock()
        mock_output = MagicMock()
        mock_cmd.run.return_value = mock_output
        mock_cmd_cls.return_value = mock_cmd

        result = diffusion_lora("runwayml/stable-diffusion-v1-5", data_dir="/path/to/images")

        mock_cmd_cls.assert_called_once()
        mock_cmd.run.assert_called_once()
        assert result is mock_output

    def test_capture_onnx_graph_integration(self, tmp_path):
        """Test capture_onnx_graph integration with a tiny model."""
        from olive import capture_onnx_graph

        # Use a tiny stub model to minimise download / conversion time
        model_id = "hf-internal-testing/tiny-random-bert"

        output_dir = tmp_path / "onnx-model"

        # Run end-to-end. Should return a WorkflowOutput without raising.
        output = capture_onnx_graph(
            model_id,
            task="text-classification",
            output_path=str(output_dir),
        )

        # Basic sanity: output directory exists and contains at least one .onnx file
        assert output_dir.exists(), "Output directory should be created"
        assert list(output_dir.rglob("*.onnx")), "No ONNX model produced"

        # WorkflowOutput type check (import inside try for safety)
        from olive.engine.output import WorkflowOutput

        assert isinstance(output, WorkflowOutput), "capture_onnx_graph should return WorkflowOutput"
