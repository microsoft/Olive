# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for Olive Python API.

These tests validate that the Python API functions are properly structured,
can be imported, and have the expected signatures.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

# pylint: disable=W0611, W0212


@pytest.mark.unit
class TestOlivePythonAPI:
    """Test cases for Olive Python API functions."""

    def test_api_module_structure(self):
        """Test that API module has expected functions."""
        from olive import (
            auto_opt,
            capture_onnx_graph,
            convert_adapters,
            extract_adapters,
            finetune,
            generate_adapter,
            generate_cost_model,
            quantize,
            run,
            session_params_tuning,
        )

        # Test that all functions are callable
        api_functions = [
            auto_opt,
            capture_onnx_graph,
            convert_adapters,
            extract_adapters,
            finetune,
            generate_adapter,
            generate_cost_model,
            quantize,
            run,
            session_params_tuning,
        ]

        for func in api_functions:
            assert callable(func), f"{func.__name__} should be callable"

    def test_workflow_function_signatures(self):
        """Test that workflow functions have expected parameters."""
        try:
            from olive.cli.api import auto_opt, finetune, quantize, run

            # Test run function signature
            run_sig = inspect.signature(run)
            assert "config" in run_sig.parameters
            assert "input_model" in run_sig.parameters
            assert "output_path" in run_sig.parameters

            # Test auto_opt function signature
            auto_opt_sig = inspect.signature(auto_opt)
            assert "model_path" in auto_opt_sig.parameters
            assert "output_path" in auto_opt_sig.parameters
            assert "device" in auto_opt_sig.parameters
            assert "provider" in auto_opt_sig.parameters

            # Test finetune function signature
            finetune_sig = inspect.signature(finetune)
            assert "model_path" in finetune_sig.parameters
            assert "method" in finetune_sig.parameters
            assert "lora_r" in finetune_sig.parameters

            # Test quantize function signature
            quantize_sig = inspect.signature(quantize)
            assert "model_path" in quantize_sig.parameters
            assert "algorithm" in quantize_sig.parameters
            assert "precision" in quantize_sig.parameters

        except ImportError as e:
            pytest.skip(f"Skipping signature test due to missing dependencies: {e}")

    @patch("olive.cli.api.olive_run")
    def test_run_function_basic(self, mock_olive_run):
        """Test basic functionality of run function with mocked dependencies."""
        try:
            from olive import run

            # Mock workflow output
            mock_output = MagicMock()
            mock_olive_run.return_value = mock_output

            # Test with dict config
            config = {"input_model": {"type": "HfModel"}}
            result = run(config)

            # Verify olive_run was called
            mock_olive_run.assert_called_once()
            assert result is mock_output

        except ImportError as e:
            pytest.skip(f"Skipping run function test due to missing dependencies: {e}")

    @patch("olive.cli.api.AutoOptCommand")
    def test_auto_opt_function_basic(self, mock_command_class):
        """Test basic functionality of auto_opt function with mocked dependencies."""
        try:
            from olive import auto_opt

            # Mock command and output
            mock_command = MagicMock()
            mock_output = MagicMock()
            mock_command.run.return_value = mock_output
            mock_command_class.return_value = mock_command

            # Test with minimal args
            result = auto_opt("test_model")

            # Verify command was created and run
            mock_command_class.assert_called_once()
            mock_command.run.assert_called_once()
            assert result is mock_output

        except ImportError as e:
            pytest.skip(f"Skipping auto_opt function test due to missing dependencies: {e}")

    @patch("olive.cli.api.FineTuneCommand")
    def test_finetune_function_basic(self, mock_command_class):
        """Test basic functionality of finetune function with mocked dependencies."""
        try:
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

        except ImportError as e:
            pytest.skip(f"Skipping finetune function test due to missing dependencies: {e}")

    def test_utility_function_signatures(self):
        """Test that utility functions have expected parameters."""
        try:
            from olive import convert_adapters

            # Test convert_adapters signature
            convert_sig = inspect.signature(convert_adapters)
            assert "adapter_path" in convert_sig.parameters
            assert "output_path" in convert_sig.parameters

        except ImportError as e:
            pytest.skip(f"Skipping utility function test due to missing dependencies: {e}")

    def test_function_docstrings(self):
        """Test that API functions have proper docstrings."""
        try:
            from olive import auto_opt, finetune, run

            # Check that functions have docstrings
            assert auto_opt.__doc__ is not None, "auto_opt should have a docstring"
            assert finetune.__doc__ is not None, "finetune should have a docstring"
            assert run.__doc__ is not None, "run should have a docstring"

            # Check docstring content
            assert "Args:" in auto_opt.__doc__, "auto_opt docstring should have Args section"
            assert "Returns:" in auto_opt.__doc__, "auto_opt docstring should have Returns section"
            assert "WorkflowOutput" in auto_opt.__doc__, "auto_opt should mention WorkflowOutput"

        except ImportError as e:
            pytest.skip(f"Skipping docstring test due to missing dependencies: {e}")
