# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------
# pylint: disable=protected-access

import builtins
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import onnx
import pytest

from olive.model import ONNXModelHandler, QairtModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.qairt.encapsulation import QairtEncapsulation
from test.passes.qairt.utils import make_minimal_onnx as _make_minimal_onnx


@pytest.mark.parametrize(
    ("key", "expected_default", "expected_required"),
    [
        ("log_level", None, False),
        ("run_checker", False, False),
        ("opset_imports", [["com.microsoft", 1]], False),
        ("genie_overrides", None, False),
        ("backend_extensions_overrides", None, False),
        ("engine_config_overrides", None, False),
    ],
)
def test_encapsulation_default_config(key, expected_default, expected_required, mock_accelerator_spec):
    config = QairtEncapsulation._default_config(mock_accelerator_spec)
    assert key in config
    assert config[key].default_value == expected_default
    assert config[key].required is expected_required


def test_encapsulation_successful_execution(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """Test successful encapsulation of a QAIRT model."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files in the model directory
    model_path = Path(mock_qairt_model.model_path)
    config_data = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "max_position_embeddings": 2048,
        "bos_token_id": 1,
        "vocab_size": 32000,
    }
    (model_path / "config.json").write_text(json.dumps(config_data))

    gen_config_data = {"eos_token_id": 2, "pad_token_id": 0}
    (model_path / "generation_config.json").write_text(json.dumps(gen_config_data))

    # Configure LLMContainer.load to return our mock_container
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)) as mock_save,
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {},
            disable_search=True,
        )

        result = encap_pass.run(mock_qairt_model, str(output_path))

        # Verify result
        assert isinstance(result, ONNXModelHandler)
        # The model_path should point to the output directory
        # ONNXModelHandler may append the model filename
        assert str(output_path) in result.model_path

        # Verify LLMContainer was loaded with the correct path
        mock_qairt_modules["gen_ai_api"].LLMContainer.load.assert_called_with(mock_qairt_model.model_path)

        # Verify export was called
        mock_container.export.assert_called_once()

        # Verify ONNX model was saved
        mock_save.assert_called_once()


def test_encapsulation_multiple_dlc_files(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """Test that a warning is logged when multiple DLC files are found."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Override export to create multiple .dlc files
    def mock_export(output_dir, *args, **kwargs):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model1.dlc").write_text("dummy dlc 1")
        (output_dir_path / "model2.dlc").write_text("dummy dlc 2")

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
        patch("olive.passes.qairt.encapsulation.logger") as mock_logger,
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {},
            disable_search=True,
        )

        result = encap_pass.run(mock_qairt_model, str(output_path))

        # Verify warning was logged
        mock_logger.warning.assert_called()
        assert isinstance(result, ONNXModelHandler)


def test_encapsulation_no_dlc_file(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """Test that FileNotFoundError is raised when no DLC file is found."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama"}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Override export to NOT create any .dlc files
    def mock_export(output_dir, *args, **kwargs):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        # Don't create any .dlc files

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    encap_pass = create_pass_from_dict(
        QairtEncapsulation,
        {},
        disable_search=True,
    )

    with pytest.raises(FileNotFoundError, match=r"No \.dlc file found"):
        encap_pass.run(mock_qairt_model, str(output_path))


def test_encapsulation_with_checker(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """Test that ONNX checker is called when run_checker is True."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker") as mock_checker,
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"run_checker": True},
            disable_search=True,
        )

        result = encap_pass.run(mock_qairt_model, str(output_path))

        # Verify checker was called
        mock_checker.check_model.assert_called_once()
        assert isinstance(result, ONNXModelHandler)


def test_encapsulation_custom_opset_imports(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """Test that custom opset imports are used."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        custom_opsets = [["com.microsoft", 1], ["ai.onnx", 14]]

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"opset_imports": custom_opsets},
            disable_search=True,
        )

        result = encap_pass.run(mock_qairt_model, str(output_path))

        # Verify make_opsetid was called for each opset
        assert mock_helper.make_opsetid.call_count == len(custom_opsets)
        assert isinstance(result, ONNXModelHandler)


@pytest.mark.parametrize(
    ("file_to_remove", "expected_match"),
    [
        ("config.json", r"Cannot create genai_config\.json if source model config doesn't exist"),
        ("generation_config.json", r"Cannot create genai_config\.json if generation config doesn't exist\."),
    ],
)
def test_encapsulation_missing_required_file(
    file_to_remove, expected_match, tmp_path, mock_qairt_model, mock_qairt_modules, mock_container
):
    """Test that ValueError is raised when a required source config file is missing."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Populate both required files, then remove the one under test
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')
    (model_path / file_to_remove).unlink()

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {},
            disable_search=True,
        )

        with pytest.raises(ValueError, match=expected_match):
            encap_pass.run(mock_qairt_model, str(output_path))


def test_encapsulation_import_error_qairt(tmp_path, mock_qairt_model):
    """Test that ImportError is raised if qairt cannot be imported."""

    def import_side_effect(name, *args, **kwargs):
        if name in ["qairt", "qairt.gen_ai_api"]:
            raise ImportError("Mock import error")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__

    with patch("builtins.__import__", side_effect=import_side_effect):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {},
            disable_search=True,
        )

        with pytest.raises(ImportError, match="Failed to import QAIRT GenAIBuilder API"):
            encap_pass.run(mock_qairt_model, str(tmp_path / "output"))


def test_encapsulation_passthrough_files(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """Test that passthrough files are copied to output."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create passthrough files and required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')
    (model_path / "chat_template.jinja").write_text("template content")
    (model_path / "tokenizer.json").write_text('{"vocab": {}}')
    (model_path / "tokenizer_config.json").write_text('{"model_max_length": 2048}')

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {},
            disable_search=True,
        )

        _ = encap_pass.run(mock_qairt_model, str(output_path))

        # Verify passthrough files were copied
        assert (Path(output_path) / "chat_template.jinja").exists()
        assert (Path(output_path) / "tokenizer.json").exists()
        assert (Path(output_path) / "tokenizer_config.json").exists()


def test_encapsulation_sdk_version_check_old_version(tmp_path, mock_qairt_model, mock_qairt_modules):
    """Test that OSError is raised for QAIRT SDK version < 2.45.0."""
    output_path = tmp_path / "output"

    # Mock SDK version to be less than 2.45.0
    mock_qairt_modules["qairt"].__sdk_version__ = "2.44.0"

    encap_pass = create_pass_from_dict(
        QairtEncapsulation,
        {},
        disable_search=True,
    )

    with pytest.raises(OSError, match=r"QairtGenAIBuilder pass is unsupported for QAIRT versions < 2\.45\.0"):
        encap_pass.run(mock_qairt_model, str(output_path))


def test_encapsulation_epcontext_node_outputs(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """The encapsulated model contains a single EPContext node with the expected attributes."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    # Note: helper and save are NOT mocked, so the real ONNX model is written to disk.
    with patch("olive.passes.qairt.encapsulation.checker"):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

    dlc_files = list(output_path.glob("*.dlc"))
    assert len(dlc_files) == 1
    dlc_filename = dlc_files[0].name

    model_def = onnx.load(output_path / "model.onnx")

    assert len(model_def.graph.node) == 1
    node = model_def.graph.node[0]
    assert node.op_type == "EPContext"
    assert node.domain == "com.microsoft"

    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    assert attrs["ep_context_type"] == b"dlc"
    assert attrs["ep_dlc_context"] == dlc_filename.encode()
    assert attrs["source"] == b"QAIRTExport"


def test_encapsulation_log_level_sets_env(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """Setting log_level exports QAIRT_LOG_LEVEL into the environment."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch.dict(os.environ, {"QAIRT_LOG_LEVEL": ""}, clear=False),
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        os.environ.pop("QAIRT_LOG_LEVEL", None)
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"log_level": "ERROR"},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

        assert os.environ.get("QAIRT_LOG_LEVEL") == "ERROR"


def test_encapsulation_sequence_lengths_wiring(tmp_path, mock_qairt_modules, mock_container):
    """sequence_lengths from the model is plumbed into create_genai_config to cap max_length."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    source_path = tmp_path / "source_model"
    source_path.mkdir(parents=True, exist_ok=True)
    (source_path / "config.json").write_text(json.dumps({"model_type": "llama", "max_position_embeddings": 4096}))
    (source_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
    (source_path / "model.dlc").write_text("dummy dlc content")
    model = QairtModelHandler(model_path=str(source_path), model_attributes={"sequence_lengths": [32, 128]})

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {},
            disable_search=True,
        )
        encap_pass.run(model, str(output_path))

    with open(output_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["search"]["max_length"] == 4096 - 32


def test_create_genai_config_head_size_valid(tmp_path):
    """Valid hidden_size and num_attention_heads → head_size == hidden_size // num_attention_heads."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 4096, "num_attention_heads": 32, "model_type": "llama"})
    )
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["model"]["decoder"]["head_size"] == 4096 // 32


def test_create_genai_config_head_size_missing_keys(tmp_path):
    """Missing hidden_size and num_attention_heads → head_size == -1, warning logged."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    with patch("olive.passes.qairt.encapsulation.logger") as mock_logger:
        create_genai_config(model_name, str(tmp_path), None)
        mock_logger.warning.assert_called()

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["model"]["decoder"]["head_size"] == -1


def test_create_genai_config_head_size_zero_attention_heads(tmp_path):
    """num_attention_heads == 0 → head_size == -1, warning logged."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 4096, "num_attention_heads": 0, "model_type": "llama"})
    )
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    with patch("olive.passes.qairt.encapsulation.logger") as mock_logger:
        create_genai_config(model_name, str(tmp_path), None)
        warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("num_attention_heads" in msg for msg in warning_messages)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["model"]["decoder"]["head_size"] == -1


def test_create_genai_config_head_size_non_int(tmp_path):
    """Non-int num_attention_heads → head_size == -1, warning logged."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 4096, "num_attention_heads": "32", "model_type": "llama"})
    )
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    with patch("olive.passes.qairt.encapsulation.logger") as mock_logger:
        create_genai_config(model_name, str(tmp_path), None)
        warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("num_attention_heads" in msg for msg in warning_messages)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["model"]["decoder"]["head_size"] == -1


def test_create_genai_config_pad_token_id_present(tmp_path):
    """pad_token_id in generation_config.json → written as-is to genai_config.json."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2, "pad_token_id": 0}))

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["model"]["pad_token_id"] == 0


def test_create_genai_config_pad_token_id_absent_scalar_eos(tmp_path):
    """No pad_token_id, scalar eos_token_id → pad_token_id falls back to eos_token_id."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["model"]["pad_token_id"] == 2


def test_create_genai_config_pad_token_id_absent_list_eos(tmp_path):
    """No pad_token_id, list eos_token_id → pad_token_id falls back to first element."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 3]}))

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["model"]["pad_token_id"] == 2


def test_create_genai_config_pad_token_id_absent_no_eos(tmp_path):
    """No pad_token_id and no eos_token_id → pad_token_id == -1."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({}))

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["model"]["pad_token_id"] == -1


def test_create_genai_config_search_fields_from_gen_config(tmp_path):
    """Search fields do_sample, temperature, top_k, top_p are read from generation_config.json."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 2, "do_sample": True, "temperature": 0.8, "top_k": 50, "top_p": 0.95})
    )

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["search"]["do_sample"] is True
    assert result["search"]["temperature"] == 0.8
    assert result["search"]["top_k"] == 50
    assert result["search"]["top_p"] == 0.95


def test_create_genai_config_search_fields_defaults_when_absent(tmp_path):
    """Search fields retain template defaults when absent from generation_config.json."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["search"]["do_sample"] is True
    assert result["search"]["temperature"] == 1.0
    assert result["search"]["top_k"] == 50
    assert result["search"]["top_p"] == 1.0


def test_create_genai_config_max_length_with_sequence_lengths(tmp_path):
    """max_length = MAX_GENIE_CONTEXT_LENGTH - min(sequence_lengths) when sequence_lengths provided."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "max_position_embeddings": 4096}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    create_genai_config(model_name, str(tmp_path), None, sequence_lengths=[32, 128])

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["search"]["max_length"] == 4096 - 32


def test_create_genai_config_max_length_without_sequence_lengths(tmp_path):
    """max_length falls back to MAX_GENIE_CONTEXT_LENGTH when sequence_lengths not provided."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "max_position_embeddings": 4096}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    assert result["search"]["max_length"] == 4096


def test_create_genai_config_provider_options_key_lowercase(tmp_path):
    """Provider options key is 'qnn' (lowercase), not 'QNN'."""
    from olive.passes.qairt.encapsulation import create_genai_config

    model_name = "model.onnx"
    _make_minimal_onnx(tmp_path / model_name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    create_genai_config(model_name, str(tmp_path), None)

    with open(tmp_path / "genai_config.json") as f:
        result = json.load(f)

    provider_options = result["model"]["decoder"]["session_options"]["provider_options"]
    assert len(provider_options) == 1
    assert "qnn" in provider_options[0]
    assert "QNN" not in provider_options[0]


# ---------------------------------------------------------------------------
# _deep_merge unit tests
# ---------------------------------------------------------------------------


def test_deep_merge_flat():
    """Flat keys in overrides replace or add keys in base."""
    from olive.passes.qairt.encapsulation import _deep_merge

    result = _deep_merge({"a": 1, "b": 2}, {"b": 99, "c": 3})
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested_dicts_are_merged_not_replaced():
    """Nested dicts are recursively merged, preserving keys not in overrides."""
    from olive.passes.qairt.encapsulation import _deep_merge

    base = {"positional_encoding": {"type": "rope", "rope_dim": 64, "rope_theta": 10000.0}}
    overrides = {"positional_encoding": {"rope_theta": 500000.0}}
    result = _deep_merge(base, overrides)
    assert result["positional_encoding"] == {"type": "rope", "rope_dim": 64, "rope_theta": 500000.0}


def test_deep_merge_nested_override_replaces_non_dict():
    """A dict override replaces a non-dict base value at the same key."""
    from olive.passes.qairt.encapsulation import _deep_merge

    result = _deep_merge({"a": 42}, {"a": {"nested": 1}})
    assert result == {"a": {"nested": 1}}


def test_deep_merge_base_unmodified():
    """_deep_merge does not mutate base, and the result shares no references with base."""
    from olive.passes.qairt.encapsulation import _deep_merge

    base = {"a": {"b": 1}, "c": [{"d": 2}]}
    overrides = {"a": {"b": 2}}
    result = _deep_merge(base, overrides)
    # base is unchanged
    assert base["a"]["b"] == 1
    # mutating the result's untouched branch does not affect base
    result["c"][0]["d"] = 99
    assert base["c"][0]["d"] == 2


def test_deep_merge_list_of_dicts_merged_elementwise():
    """List-of-dicts elements are merged element-wise; base keys absent from the override are preserved.

    This is the core regression test for the backend_extensions_overrides bug where
    context[weight_sharing_enabled] and devices[device_id/soc_model/dsp_arch/cores] were
    being dropped because the override list fully replaced the base list.
    """
    from olive.passes.qairt.encapsulation import _deep_merge

    base = {
        "context": [{"weight_sharing_enabled": True}],
        "devices": [{"device_id": 0, "soc_model": 60, "dsp_arch": "v73", "cores": [{"core_id": 0}]}],
        "memory": {"mem_type": "shared_buffer"},
    }
    overrides = {
        "context": [{"reused_io_limit_mb": 100}],
        "devices": [{"pd_session": "unsigned"}],
    }
    result = _deep_merge(base, overrides)

    # context: override key added, base key preserved
    assert result["context"] == [{"weight_sharing_enabled": True, "reused_io_limit_mb": 100}]
    # devices: override key added, all base keys preserved
    assert result["devices"] == [
        {"device_id": 0, "soc_model": 60, "dsp_arch": "v73", "cores": [{"core_id": 0}], "pd_session": "unsigned"}
    ]
    # memory: untouched
    assert result["memory"] == {"mem_type": "shared_buffer"}


def test_deep_merge_list_override_shorter_than_base_preserves_tail():
    """Extra base list elements beyond the override length are appended to the result."""
    from olive.passes.qairt.encapsulation import _deep_merge

    base = {"graphs": [{"vtcm_mb": 8, "graph_names": ["g1"]}, {"vtcm_mb": 8, "graph_names": ["g2"]}]}
    overrides = {"graphs": [{"vtcm_mb": 16}]}
    result = _deep_merge(base, overrides)

    assert result["graphs"] == [
        {"vtcm_mb": 16, "graph_names": ["g1"]},
        {"vtcm_mb": 8, "graph_names": ["g2"]},
    ]


def test_deep_merge_list_override_longer_than_base_appends_extras():
    """Extra override list elements beyond the base length are appended."""
    from olive.passes.qairt.encapsulation import _deep_merge

    base = {"items": [{"a": 1}]}
    overrides = {"items": [{"a": 2}, {"b": 3}]}
    result = _deep_merge(base, overrides)

    assert result["items"] == [{"a": 2}, {"b": 3}]


def test_deep_merge_none_override_deletes_key():
    """A None override value removes the key from the result entirely."""
    from olive.passes.qairt.encapsulation import _deep_merge

    base = {"context": [{"weight_sharing_enabled": True}], "graphs": [{"vtcm_mb": 8}], "memory": {}}
    overrides = {"graphs": None}
    result = _deep_merge(base, overrides)

    assert "graphs" not in result
    assert "context" in result
    assert "memory" in result


# ---------------------------------------------------------------------------
# genie_overrides integration tests
# ---------------------------------------------------------------------------


def test_encapsulation_genie_overrides_applied(tmp_path, mock_qairt_model, mock_qairt_modules, mock_container):
    """When genie_overrides is set, _gen_ai_config is deep-merged before export."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    # Represent the existing GenAIConfig state after LLMContainer.load()
    initial_gen_ai_state = {
        "context_length": 4096,
        "n_vocab": 32000,
        "bos_token": 1,
        "eos_token": 2,
        "tokenizer_path": str(tmp_path / "tokenizer.json"),
        "kv_dim": None,
        "positional_encoding": {"type": "rope", "rope_dim": 64},
    }
    mock_container._gen_ai_config.model_dump.return_value = initial_gen_ai_state  # pylint: disable=protected-access
    original_gen_ai_config_mock = mock_container._gen_ai_config  # pylint: disable=protected-access

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    overrides = {"kv_dim": 128, "positional_encoding": {"rope_theta": 500000.0}}

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"genie_overrides": overrides},
            disable_search=True,
        )

        encap_pass.run(mock_qairt_model, str(output_path))

    # model_dump was called to capture current state
    original_gen_ai_config_mock.model_dump.assert_called_once_with(mode="json", by_alias=False, exclude_none=True)
    # model_validate was called with the deep-merged result
    expected_merged = {
        **initial_gen_ai_state,
        "kv_dim": 128,
        "positional_encoding": {"type": "rope", "rope_dim": 64, "rope_theta": 500000.0},
    }
    original_gen_ai_config_mock.model_validate.assert_called_once_with(expected_merged)


def test_encapsulation_no_genie_overrides_leaves_gen_ai_config_untouched(
    tmp_path, mock_qairt_model, mock_qairt_modules, mock_container
):
    """When genie_overrides is None, _gen_ai_config.model_dump is never called."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {},
            disable_search=True,
        )

        encap_pass.run(mock_qairt_model, str(output_path))

    mock_container._gen_ai_config.model_dump.assert_not_called()


# ---------------------------------------------------------------------------
# backend_extensions_override integration tests
# ---------------------------------------------------------------------------


def test_encapsulation_backend_extensions_overrides_merges_into_existing(
    tmp_path, mock_qairt_model, mock_qairt_modules, mock_container
):
    """Override is deep-merged into the existing _backend_extensions_config."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_container._backend_extensions_config = {"context": {"n-threads": 6, "kv-cache-size": 512}}
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend_extensions_overrides": {"context": {"n-threads": 4}}},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

    # n-threads overridden; kv-cache-size from original preserved
    assert mock_container._backend_extensions_config == {"context": {"n-threads": 4, "kv-cache-size": 512}}


def test_encapsulation_backend_extensions_overrides_from_empty(
    tmp_path, mock_qairt_model, mock_qairt_modules, mock_container
):
    """When the container has no existing backend extensions config, the override becomes the config."""
    mock_container._backend_extensions_config = None

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend_extensions_overrides": {"context": {"n-threads": 4}}},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

    assert mock_container._backend_extensions_config == {"context": {"n-threads": 4}}


def test_encapsulation_no_backend_extensions_overrides_leaves_config_untouched(
    tmp_path, mock_qairt_model, mock_qairt_modules, mock_container
):
    """When backend_extensions_overrides is None the container config is not touched."""
    original_ext_cfg = {"context": {"n-threads": 6}}
    mock_container._backend_extensions_config = original_ext_cfg

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        encap_pass = create_pass_from_dict(QairtEncapsulation, {}, disable_search=True)
        encap_pass.run(mock_qairt_model, str(output_path))

    assert mock_container._backend_extensions_config is original_ext_cfg


# ---------------------------------------------------------------------------
# engine_config_overrides integration tests
# ---------------------------------------------------------------------------


def test_encapsulation_engine_config_overrides_applied_when_supported(tmp_path, mock_qairt_model, mock_qairt_modules):
    """engine_config_overrides is passed to export() when the installed qairt supports it."""
    import inspect

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, export_format, engine_config=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.dlc").write_text("dummy dlc")

    mock_container.export = MagicMock(side_effect=mock_export)
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    fake_sig = inspect.signature(mock_export)

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
        patch("olive.passes.qairt.encapsulation.inspect.signature", return_value=fake_sig),
    ):
        fake_engine_cfg = MagicMock()
        mock_qairt_modules["gen_ai_api"].EngineConfig.return_value = fake_engine_cfg
        mock_qairt_modules["gen_ai_api"].HTPEngineConfig.return_value = MagicMock()

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"engine_config_overrides": {"n_threads": 4, "htp": {"cpu_mask": "0x3"}}},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

    mock_qairt_modules["gen_ai_api"].HTPEngineConfig.assert_called_once_with(cpu_mask="0x3")
    mock_qairt_modules["gen_ai_api"].EngineConfig.assert_called_once_with(
        n_threads=4, htp=mock_qairt_modules["gen_ai_api"].HTPEngineConfig.return_value
    )
    _, call_kwargs = mock_container.export.call_args
    assert call_kwargs.get("engine_config") is fake_engine_cfg


def test_encapsulation_engine_config_overrides_without_htp(tmp_path, mock_qairt_model, mock_qairt_modules):
    """engine_config_overrides without an 'htp' key constructs EngineConfig with htp=None."""
    import inspect

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, export_format, engine_config=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.dlc").write_text("dummy dlc")

    mock_container.export = MagicMock(side_effect=mock_export)
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    fake_sig = inspect.signature(mock_export)

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
        patch("olive.passes.qairt.encapsulation.inspect.signature", return_value=fake_sig),
    ):
        fake_engine_cfg = MagicMock()
        mock_qairt_modules["gen_ai_api"].EngineConfig.return_value = fake_engine_cfg

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"engine_config_overrides": {"n_threads": 4}},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

    mock_qairt_modules["gen_ai_api"].HTPEngineConfig.assert_not_called()
    mock_qairt_modules["gen_ai_api"].EngineConfig.assert_called_once_with(n_threads=4, htp=None)
    _, call_kwargs = mock_container.export.call_args
    assert call_kwargs.get("engine_config") is fake_engine_cfg


def test_encapsulation_engine_config_overrides_skipped_when_not_supported(
    tmp_path, mock_qairt_model, mock_qairt_modules
):
    """engine_config_overrides is ignored with a warning when export() has no engine_config param."""
    import inspect

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export_old(output_dir, export_format):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.dlc").write_text("dummy dlc")

    mock_container.export = MagicMock(side_effect=mock_export_old)
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    fake_sig = inspect.signature(mock_export_old)

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
        patch("olive.passes.qairt.encapsulation.inspect.signature", return_value=fake_sig),
        patch("olive.passes.qairt.encapsulation.logger") as mock_logger,
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"engine_config_overrides": {"n_threads": 4}},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

    warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
    assert any("engine_config_overrides ignored" in msg for msg in warning_messages)
    _, call_kwargs = mock_container.export.call_args
    assert "engine_config" not in call_kwargs


# ---------------------------------------------------------------------------
# hasattr guard tests
# ---------------------------------------------------------------------------


def test_encapsulation_genie_overrides_skipped_when_private_attr_missing(
    tmp_path, mock_qairt_model, mock_qairt_modules
):
    """genie_overrides is ignored with a warning when _gen_ai_config is absent from the container."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_container = MagicMock(spec=["inputs", "outputs", "export"])
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.dlc").write_text("dummy dlc")

    mock_container.export = MagicMock(side_effect=mock_export)
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
        patch("olive.passes.qairt.encapsulation.logger") as mock_logger,
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"genie_overrides": {"kv_dim": 128}},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

    warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
    assert any("genie_overrides ignored" in msg for msg in warning_messages)


def test_encapsulation_backend_extensions_overrides_skipped_when_private_attr_missing(
    tmp_path, mock_qairt_model, mock_qairt_modules
):
    """backend_extensions_overrides is ignored with a warning when _backend_extensions_config is absent."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
    (model_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))

    mock_container = MagicMock(spec=["inputs", "outputs", "export"])
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.dlc").write_text("dummy dlc")

    mock_container.export = MagicMock(side_effect=mock_export)
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    with (
        patch("olive.passes.qairt.encapsulation.helper"),
        patch("olive.passes.qairt.encapsulation.save", side_effect=lambda _m, p: _make_minimal_onnx(p)),
        patch("olive.passes.qairt.encapsulation.checker"),
        patch("olive.passes.qairt.encapsulation.logger") as mock_logger,
    ):
        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend_extensions_overrides": {"context": {"n-threads": 4}}},
            disable_search=True,
        )
        encap_pass.run(mock_qairt_model, str(output_path))

    warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
    assert any("backend_extensions_overrides ignored" in msg for msg in warning_messages)
