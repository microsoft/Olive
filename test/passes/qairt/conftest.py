# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(name="mock_container")
def mock_container_fixture():
    """Provide a pre-configured LLMContainer mock with input/output metadata and an export stub.

    Tests are responsible for wiring LLMContainer.load.return_value so they can customise
    container attributes before the pass runs.
    """
    container = MagicMock()
    container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, *args, **kwargs):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model.dlc").write_text("dummy dlc content")

    container.export.side_effect = mock_export
    return container


@pytest.fixture(name="mock_qairt_modules")
def mock_qairt_modules_fixture():
    """Mock the qairt and qairt.gen_ai_api modules.

    Returns the mock modules for further configuration in tests.
    """
    mock_qairt = MagicMock()
    mock_gen_ai_api = MagicMock()

    # Set up common enums and types
    mock_qairt.BackendType = MagicMock()
    mock_qairt.BackendType.CPU = MagicMock()
    mock_qairt.BackendType.CPU.value = "CPU"
    mock_qairt.BackendType.HTP = MagicMock()
    mock_qairt.BackendType.HTP.value = "HTP"
    mock_qairt.ExportFormat = MagicMock()
    mock_qairt.ExportFormat.LM_EXECUTOR = MagicMock()

    # Mock SDK version (required for version checks in passes)
    mock_qairt.__sdk_version__ = "2.45.0"

    # Make gen_ai_api part of qairt module hierarchy
    mock_qairt.gen_ai_api = mock_gen_ai_api

    # Patch sys.modules
    patcher_qairt = patch.dict("sys.modules", {"qairt": mock_qairt})
    patcher_gen_ai = patch.dict("sys.modules", {"qairt.gen_ai_api": mock_gen_ai_api})

    patcher_qairt.start()
    patcher_gen_ai.start()

    yield {"qairt": mock_qairt, "gen_ai_api": mock_gen_ai_api}

    # Teardown
    patcher_qairt.stop()
    patcher_gen_ai.stop()


@pytest.fixture(name="mock_hf_model")
def mock_hf_model_fixture(tmp_path):
    """Create a mock HfModelHandler."""
    from olive.model import HfModelHandler

    model_path = tmp_path / "hf_model"
    model_path.mkdir(parents=True, exist_ok=True)

    # Create minimal required files
    (model_path / "config.json").write_text('{"model_type": "llama"}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    return HfModelHandler(model_path=str(model_path))


@pytest.fixture(name="mock_qairt_prepared_model")
def mock_qairt_prepared_model_fixture(tmp_path):
    """Create a mock QairtPreparedModelHandler."""
    from olive.model import QairtPreparedModelHandler

    model_path = tmp_path / "qairt_prepared_model"
    model_path.mkdir(parents=True, exist_ok=True)

    # Create minimal required files
    (model_path / "config.json").write_text('{"model_type": "llama"}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Create base/onnx directory structure for GenAIBuilder
    base_onnx_path = model_path / "base" / "onnx"
    base_onnx_path.mkdir(parents=True, exist_ok=True)

    return QairtPreparedModelHandler(model_path=str(model_path))


@pytest.fixture(name="mock_qairt_model")
def mock_qairt_model_fixture(tmp_path):
    """Create a mock QairtModelHandler."""
    from olive.model import QairtModelHandler

    model_path = tmp_path / "qairt_model"
    model_path.mkdir(parents=True, exist_ok=True)

    # Create minimal required files
    (model_path / "config.json").write_text('{"model_type": "llama"}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Create a dummy .dlc file
    (model_path / "model.dlc").write_text("dummy dlc content")

    return QairtModelHandler(model_path=str(model_path))


@pytest.fixture(name="mock_accelerator_spec")
def mock_accelerator_spec_fixture():
    """Provide a mock AcceleratorSpec."""
    return MagicMock()
