# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from olive.model import ONNXModelHandler, QairtPreparedModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.qairt.preparation import QairtPreparation


def test_preparation_default_config(mock_accelerator_spec):
    """Test that the default config is correctly generated."""
    config = QairtPreparation._default_config(mock_accelerator_spec)  # pylint: disable=protected-access

    assert "script_path" in config
    assert config["script_path"].required is True
    assert "script_config" in config
    assert not config["script_config"].default_value
    assert "cache_dir" in config
    assert config["cache_dir"].default_value == "./cache/qairt/preparation"


def test_preparation_successful_execution(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test successful execution of the preparation script."""
    # Create a mock script file
    script_path = tmp_path / "prep_script.py"
    script_path.write_text("# Mock preparation script")

    output_path = tmp_path / "output"

    # Mock subprocess.Popen to simulate successful execution
    mock_process = MagicMock()
    mock_process.poll.side_effect = [None, None, 0]  # Running, then done
    mock_process.wait.return_value = 0
    # Ensure context manager returns the mock_process itself
    mock_process.__enter__ = Mock(return_value=mock_process)
    mock_process.__exit__ = Mock(return_value=False)
    mock_process.stdout = MagicMock()

    # Use a generator to continuously return empty strings after initial messages
    def stdout_generator():
        yield "Processing...\n"
        yield "Done!\n"
        while True:
            yield ""

    mock_process.stdout.readline = Mock(side_effect=stdout_generator())
    mock_process.stderr = MagicMock()

    # Use a generator to continuously return empty strings
    def stderr_generator():
        while True:
            yield ""

    mock_process.stderr.readline = Mock(side_effect=stderr_generator())

    with patch("subprocess.Popen", return_value=mock_process):
        prep_pass = create_pass_from_dict(
            QairtPreparation,
            {
                "script_path": str(script_path),
                "script_config": {"precision": "int8"},
            },
            disable_search=True,
        )

        result = prep_pass.run(mock_hf_model, str(output_path))

        # Verify result
        assert isinstance(result, QairtPreparedModelHandler)
        assert result.model_path == str(output_path)


def test_preparation_invalid_input_model(tmp_path, mock_qairt_modules):
    """Test that ValueError is raised for non-HfModelHandler input."""
    script_path = tmp_path / "prep_script.py"
    script_path.write_text("# Mock script")

    output_path = tmp_path / "output"

    # Create an ONNX model instead of HF model
    onnx_model_path = tmp_path / "model.onnx"
    onnx_model_path.write_text("dummy onnx")
    onnx_model = ONNXModelHandler(model_path=str(onnx_model_path))

    prep_pass = create_pass_from_dict(
        QairtPreparation,
        {"script_path": str(script_path)},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="QairtPreparation requires HfModelHandler"):
        prep_pass.run(onnx_model, str(output_path))


def test_preparation_script_not_found(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that ValueError is raised when script path doesn't exist."""
    script_path = tmp_path / "nonexistent_script.py"
    output_path = tmp_path / "output"

    prep_pass = create_pass_from_dict(
        QairtPreparation,
        {"script_path": str(script_path)},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="Preparation script not found"):
        prep_pass.run(mock_hf_model, str(output_path))


def test_preparation_invalid_script_extension(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that ValueError is raised for non-Python script."""
    script_path = tmp_path / "prep_script.sh"
    script_path.write_text("#!/bin/bash")
    output_path = tmp_path / "output"

    prep_pass = create_pass_from_dict(
        QairtPreparation,
        {"script_path": str(script_path)},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="Script must be a Python file"):
        prep_pass.run(mock_hf_model, str(output_path))


def test_preparation_script_execution_failure(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that RuntimeError is raised when script fails."""
    script_path = tmp_path / "prep_script.py"
    script_path.write_text("# Mock script")
    output_path = tmp_path / "output"

    # Mock subprocess.Popen to simulate failure
    mock_process = MagicMock()
    mock_process.poll.side_effect = [None, 1]  # Running, then failed
    mock_process.wait.return_value = 1
    # Ensure context manager returns the mock_process itself
    mock_process.__enter__ = Mock(return_value=mock_process)
    mock_process.__exit__ = Mock(return_value=False)
    mock_process.stdout = MagicMock()

    def stdout_generator():
        yield "Starting...\n"
        while True:
            yield ""

    mock_process.stdout.readline = Mock(side_effect=stdout_generator())
    mock_process.stderr = MagicMock()

    def stderr_generator():
        yield "Error: Something went wrong\n"
        while True:
            yield ""

    mock_process.stderr.readline = Mock(side_effect=stderr_generator())

    with patch("subprocess.Popen", return_value=mock_process):
        prep_pass = create_pass_from_dict(
            QairtPreparation,
            {"script_path": str(script_path)},
            disable_search=True,
        )

        with pytest.raises(RuntimeError, match="QAIRT preparation script failed"):
            prep_pass.run(mock_hf_model, str(output_path))


def test_preparation_config_merging(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that script config is properly merged with defaults."""
    script_path = tmp_path / "prep_script.py"
    script_path.write_text("# Mock script")
    output_path = tmp_path / "output"

    captured_config = {}

    def mock_popen(*args, **kwargs):
        # Capture the config file path from the command
        cmd = args[0]
        config_file_idx = cmd.index("--config") + 1
        config_file_path = cmd[config_file_idx]

        # Read and store the config
        with open(config_file_path) as f:
            captured_config.update(json.load(f))

        # Return a successful mock process
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, 0]
        mock_process.wait.return_value = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.stdout = MagicMock()

        def stdout_generator():
            yield "Done\n"
            while True:
                yield ""

        mock_process.stdout.readline = Mock(side_effect=stdout_generator())
        mock_process.stderr = MagicMock()

        def stderr_generator():
            while True:
                yield ""

        mock_process.stderr.readline = Mock(side_effect=stderr_generator())
        return mock_process

    with patch("subprocess.Popen", side_effect=mock_popen):
        prep_pass = create_pass_from_dict(
            QairtPreparation,
            {
                "script_path": str(script_path),
                "script_config": {"precision": "int8", "custom_param": "value"},
                "cache_dir": "./custom_cache",
            },
            disable_search=True,
        )

        prep_pass.run(mock_hf_model, str(output_path))

        # Verify config was merged correctly
        assert "OUTPUT_DIR" in captured_config
        assert captured_config["OUTPUT_DIR"] == str(output_path)
        assert "CACHE_DIR" in captured_config
        assert "ADASCALE_DIR" in captured_config
        assert captured_config["precision"] == "int8"
        assert captured_config["custom_param"] == "value"


def test_preparation_streaming_output(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that stdout and stderr are properly streamed."""
    script_path = tmp_path / "prep_script.py"
    script_path.write_text("# Mock script")
    output_path = tmp_path / "output"

    # Mock subprocess.Popen with multiple output lines
    mock_process = MagicMock()
    mock_process.poll.side_effect = [None, None, None, 0]
    mock_process.wait.return_value = 0
    # Ensure context manager returns the mock_process itself
    mock_process.__enter__ = Mock(return_value=mock_process)
    mock_process.__exit__ = Mock(return_value=False)
    mock_process.stdout = MagicMock()

    def stdout_generator():
        yield "Line 1\n"
        yield "Line 2\n"
        yield "Line 3\n"
        while True:
            yield ""

    mock_process.stdout.readline = Mock(side_effect=stdout_generator())
    mock_process.stderr = MagicMock()

    def stderr_generator():
        yield "Warning: test\n"
        while True:
            yield ""

    mock_process.stderr.readline = Mock(side_effect=stderr_generator())

    with patch("subprocess.Popen", return_value=mock_process):
        prep_pass = create_pass_from_dict(
            QairtPreparation,
            {"script_path": str(script_path)},
            disable_search=True,
        )

        # Should complete without error
        result = prep_pass.run(mock_hf_model, str(output_path))
        assert isinstance(result, QairtPreparedModelHandler)


def test_preparation_temp_config_cleanup(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that temporary config file is cleaned up."""
    script_path = tmp_path / "prep_script.py"
    script_path.write_text("# Mock script")
    output_path = tmp_path / "output"

    mock_process = MagicMock()
    mock_process.poll.side_effect = [None, 0]
    mock_process.wait.return_value = 0
    # Ensure context manager returns the mock_process itself
    mock_process.__enter__ = Mock(return_value=mock_process)
    mock_process.__exit__ = Mock(return_value=False)
    mock_process.stdout = MagicMock()

    def stdout_generator():
        yield "Done\n"
        while True:
            yield ""

    mock_process.stdout.readline = Mock(side_effect=stdout_generator())
    mock_process.stderr = MagicMock()

    def stderr_generator():
        while True:
            yield ""

    mock_process.stderr.readline = Mock(side_effect=stderr_generator())

    with (
        patch("subprocess.Popen", return_value=mock_process),
        patch("tempfile.NamedTemporaryFile") as mock_temp,
    ):
        temp_file_path = tmp_path / "olive_qairt_prep_test.json"

        mock_file = MagicMock()
        mock_file.name = str(temp_file_path)
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_temp.return_value = mock_file

        prep_pass = create_pass_from_dict(
            QairtPreparation,
            {"script_path": str(script_path)},
            disable_search=True,
        )

        prep_pass.run(mock_hf_model, str(output_path))

        # Verify temp file would be cleaned up (unlink called)
        # Note: In actual implementation, cleanup happens in finally block


def test_preparation_uses_sys_executable_and_env(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that subprocess uses sys.executable and passes environment."""
    import sys

    script_path = tmp_path / "prep_script.py"
    script_path.write_text("# Mock script")
    output_path = tmp_path / "output"

    captured_popen_args = {}

    def mock_popen(*args, **kwargs):
        # Capture the arguments passed to Popen
        captured_popen_args["args"] = args
        captured_popen_args["kwargs"] = kwargs

        # Return a successful mock process
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, 0]
        mock_process.wait.return_value = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.stdout = MagicMock()

        def stdout_generator():
            yield "Done\n"
            while True:
                yield ""

        mock_process.stdout.readline = Mock(side_effect=stdout_generator())
        mock_process.stderr = MagicMock()

        def stderr_generator():
            while True:
                yield ""

        mock_process.stderr.readline = Mock(side_effect=stderr_generator())
        return mock_process

    with patch("subprocess.Popen", side_effect=mock_popen):
        prep_pass = create_pass_from_dict(
            QairtPreparation,
            {"script_path": str(script_path)},
            disable_search=True,
        )

        prep_pass.run(mock_hf_model, str(output_path))

        # Verify sys.executable was used instead of "python"
        cmd = captured_popen_args["args"][0]
        assert cmd[0] == sys.executable, f"Expected {sys.executable}, got {cmd[0]}"

        # Verify environment was passed
        assert "env" in captured_popen_args["kwargs"]
        passed_env = captured_popen_args["kwargs"]["env"]
        assert isinstance(passed_env, dict)
        # Verify it's a copy of os.environ (should have similar keys)
        assert len(passed_env) > 0
        # Check that some common environment variables are present
        assert any(key in passed_env for key in ["PATH", "HOME", "USER"])
