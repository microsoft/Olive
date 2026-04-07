# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import builtins
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.qairt.encapsulation import QairtEncapsulation


def test_encapsulation_default_config(mock_accelerator_spec):
    """Test that the default config is correctly generated."""
    config = QairtEncapsulation._default_config(mock_accelerator_spec)  # pylint: disable=protected-access

    assert "log_level" in config
    assert "run_checker" in config
    assert config["run_checker"].default_value is False
    assert "opset_imports" in config
    assert config["opset_imports"].default_value == [["com.microsoft", 1]]


def test_encapsulation_successful_execution(tmp_path, mock_qairt_model, mock_qairt_modules):
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

    # Mock export to create a .dlc file
    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model.dlc").write_text("dummy dlc content")

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.export = MagicMock(side_effect=mock_export)
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    # Configure LLMContainer.load to return our mock_container
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    # Mock onnx module - but let save actually create a file
    def mock_save_func(model_def, path):
        # Create a minimal valid ONNX file so create_genai_config can load it
        import onnx
        from onnx import TensorProto

        # Create a minimal valid ONNX model
        input_tensor = onnx.helper.make_tensor_value_info(
            "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])

        node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])

        onnx.save(model, path)

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=mock_save_func) as mock_save,
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        mock_helper.make_node.return_value = MagicMock()
        mock_helper.make_attribute.return_value = MagicMock()
        mock_helper.make_tensor_value_info.return_value = MagicMock()
        mock_helper.make_graph.return_value = MagicMock()
        mock_helper.make_opsetid.return_value = MagicMock()
        mock_helper.make_model.return_value = MagicMock()

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend": "CPU"},
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


def test_encapsulation_multiple_dlc_files(tmp_path, mock_qairt_model, mock_qairt_modules):
    """Test that a warning is logged when multiple DLC files are found."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    # Mock export to create multiple .dlc files
    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model1.dlc").write_text("dummy dlc 1")
        (output_dir_path / "model2.dlc").write_text("dummy dlc 2")

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    # Mock save to create actual ONNX file
    def mock_save_func(model_def, path):
        import onnx
        from onnx import TensorProto

        input_tensor = onnx.helper.make_tensor_value_info(
            "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
        node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
        onnx.save(model, path)

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=mock_save_func),
        patch("olive.passes.qairt.encapsulation.checker"),
        patch("olive.passes.qairt.encapsulation.logger") as mock_logger,
    ):
        mock_helper.make_node.return_value = MagicMock()
        mock_helper.make_attribute.return_value = MagicMock()
        mock_helper.make_tensor_value_info.return_value = MagicMock()
        mock_helper.make_graph.return_value = MagicMock()
        mock_helper.make_opsetid.return_value = MagicMock()
        mock_helper.make_model.return_value = MagicMock()

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend": "CPU"},
            disable_search=True,
        )

        result = encap_pass.run(mock_qairt_model, str(output_path))

        # Verify warning was logged
        mock_logger.warning.assert_called()
        assert isinstance(result, ONNXModelHandler)


def test_encapsulation_no_dlc_file(tmp_path, mock_qairt_model, mock_qairt_modules):
    """Test that FileNotFoundError is raised when no DLC file is found."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama"}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    # Mock export to NOT create any .dlc files
    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        # Don't create any .dlc files

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    encap_pass = create_pass_from_dict(
        QairtEncapsulation,
        {"backend": "CPU"},
        disable_search=True,
    )

    with pytest.raises(FileNotFoundError, match="No .dlc file found"):
        encap_pass.run(mock_qairt_model, str(output_path))


def test_encapsulation_with_checker(tmp_path, mock_qairt_model, mock_qairt_modules):
    """Test that ONNX checker is called when run_checker is True."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model.dlc").write_text("dummy dlc")

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    def mock_save_func(model_def, path):
        import onnx
        from onnx import TensorProto

        input_tensor = onnx.helper.make_tensor_value_info(
            "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
        node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
        onnx.save(model, path)

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=mock_save_func),
        patch("olive.passes.qairt.encapsulation.checker") as mock_checker,
    ):
        mock_helper.make_node.return_value = MagicMock()
        mock_helper.make_attribute.return_value = MagicMock()
        mock_helper.make_tensor_value_info.return_value = MagicMock()
        mock_helper.make_graph.return_value = MagicMock()
        mock_helper.make_opsetid.return_value = MagicMock()
        mock_helper.make_model.return_value = MagicMock()

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend": "CPU", "run_checker": True},
            disable_search=True,
        )

        result = encap_pass.run(mock_qairt_model, str(output_path))

        # Verify checker was called
        mock_checker.check_model.assert_called_once()
        assert isinstance(result, ONNXModelHandler)


def test_encapsulation_custom_opset_imports(tmp_path, mock_qairt_model, mock_qairt_modules):
    """Test that custom opset imports are used."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create required config files
    model_path = Path(mock_qairt_model.model_path)
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model.dlc").write_text("dummy dlc")

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    def mock_save_func(model_def, path):
        import onnx
        from onnx import TensorProto

        input_tensor = onnx.helper.make_tensor_value_info(
            "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
        node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
        onnx.save(model, path)

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=mock_save_func),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        mock_helper.make_node.return_value = MagicMock()
        mock_helper.make_attribute.return_value = MagicMock()
        mock_helper.make_tensor_value_info.return_value = MagicMock()
        mock_helper.make_graph.return_value = MagicMock()
        mock_helper.make_opsetid.return_value = MagicMock()
        mock_helper.make_model.return_value = MagicMock()

        custom_opsets = [["com.microsoft", 1], ["ai.onnx", 14]]

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend": "CPU", "opset_imports": custom_opsets},
            disable_search=True,
        )

        result = encap_pass.run(mock_qairt_model, str(output_path))

        # Verify make_opsetid was called for each opset
        assert mock_helper.make_opsetid.call_count == len(custom_opsets)
        assert isinstance(result, ONNXModelHandler)


def test_encapsulation_missing_config_json(tmp_path, mock_qairt_model, mock_qairt_modules):
    """Test that ValueError is raised when config.json is missing."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Don't create config.json in the source model directory
    model_path = Path(mock_qairt_model.model_path)
    # Remove config.json if it exists
    config_file = model_path / "config.json"
    if config_file.exists():
        config_file.unlink()
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model.dlc").write_text("dummy dlc")

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    def mock_save_func(model_def, path):
        import onnx
        from onnx import TensorProto

        input_tensor = onnx.helper.make_tensor_value_info(
            "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
        node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
        onnx.save(model, path)

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=mock_save_func),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        mock_helper.make_node.return_value = MagicMock()
        mock_helper.make_attribute.return_value = MagicMock()
        mock_helper.make_tensor_value_info.return_value = MagicMock()
        mock_helper.make_graph.return_value = MagicMock()
        mock_helper.make_opsetid.return_value = MagicMock()
        mock_helper.make_model.return_value = MagicMock()

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend": "CPU"},
            disable_search=True,
        )

        with pytest.raises(ValueError, match="Cannot create gen_ai_config.json if source model config doesn't exist"):
            encap_pass.run(mock_qairt_model, str(output_path))


def test_encapsulation_missing_generation_config(tmp_path, mock_qairt_model, mock_qairt_modules):
    """Test that ValueError is raised when generation_config.json is missing."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create config.json but not generation_config.json in the source model directory
    model_path = Path(mock_qairt_model.model_path)
    # Remove generation_config.json if it exists
    gen_config_file = model_path / "generation_config.json"
    if gen_config_file.exists():
        gen_config_file.unlink()
    (model_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model.dlc").write_text("dummy dlc")

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    def mock_save_func(model_def, path):
        import onnx
        from onnx import TensorProto

        input_tensor = onnx.helper.make_tensor_value_info(
            "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
        node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
        onnx.save(model, path)

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=mock_save_func),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        mock_helper.make_node.return_value = MagicMock()
        mock_helper.make_attribute.return_value = MagicMock()
        mock_helper.make_tensor_value_info.return_value = MagicMock()
        mock_helper.make_graph.return_value = MagicMock()
        mock_helper.make_opsetid.return_value = MagicMock()
        mock_helper.make_model.return_value = MagicMock()

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend": "CPU"},
            disable_search=True,
        )

        with pytest.raises(ValueError, match="Cannot create gen_ai_config.json if generation config doesn't exist"):
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
            {"backend": "CPU"},
            disable_search=True,
        )

        with pytest.raises(ImportError, match="Failed to import QAIRT GenAIBuilder API"):
            encap_pass.run(mock_qairt_model, str(tmp_path / "output"))


def test_encapsulation_passthrough_files(tmp_path, mock_qairt_model, mock_qairt_modules):
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

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model.dlc").write_text("dummy dlc")

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    def mock_save_func(model_def, path):
        import onnx
        from onnx import TensorProto

        input_tensor = onnx.helper.make_tensor_value_info(
            "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
        node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
        onnx.save(model, path)

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=mock_save_func),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        mock_helper.make_node.return_value = MagicMock()
        mock_helper.make_attribute.return_value = MagicMock()
        mock_helper.make_tensor_value_info.return_value = MagicMock()
        mock_helper.make_graph.return_value = MagicMock()
        mock_helper.make_opsetid.return_value = MagicMock()
        mock_helper.make_model.return_value = MagicMock()

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend": "CPU"},
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
        {"backend": "CPU"},
        disable_search=True,
    )

    with pytest.raises(OSError, match="QairtGenAIBuilder pass is unsupported for QAIRT versions < 2.45.0"):
        encap_pass.run(mock_qairt_model, str(output_path))


def test_encapsulation_sdk_version_check_valid_version(tmp_path, mock_qairt_model, mock_qairt_modules):
    """Test that encapsulation works with QAIRT SDK version >= 2.45.0."""
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Mock SDK version to be >= 2.45.0
    mock_qairt_modules["qairt"].__sdk_version__ = "2.45.0"

    # Create required config files
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
    (model_path / "generation_config.json").write_text('{"eos_token_id": 2}')

    # Mock LLMContainer
    mock_container = MagicMock()
    mock_container.inputs = [("input_ids", 7, ["batch_size", "sequence_length"])]
    mock_container.outputs = [("logits", 1, ["batch_size", 1, "vocab_size"])]

    def mock_export(output_dir, export_format):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "model.dlc").write_text("dummy dlc")

    mock_container.export.side_effect = mock_export
    mock_qairt_modules["gen_ai_api"].LLMContainer.load.return_value = mock_container

    def mock_save_func(model_def, path):
        import onnx
        from onnx import TensorProto

        input_tensor = onnx.helper.make_tensor_value_info(
            "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
        node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
        onnx.save(model, path)

    with (
        patch("olive.passes.qairt.encapsulation.helper") as mock_helper,
        patch("olive.passes.qairt.encapsulation.save", side_effect=mock_save_func),
        patch("olive.passes.qairt.encapsulation.checker"),
    ):
        mock_helper.make_node.return_value = MagicMock()
        mock_helper.make_attribute.return_value = MagicMock()
        mock_helper.make_tensor_value_info.return_value = MagicMock()
        mock_helper.make_graph.return_value = MagicMock()
        mock_helper.make_opsetid.return_value = MagicMock()
        mock_helper.make_model.return_value = MagicMock()

        encap_pass = create_pass_from_dict(
            QairtEncapsulation,
            {"backend": "CPU"},
            disable_search=True,
        )

        # Should not raise an error
        result = encap_pass.run(mock_qairt_model, str(output_path))
        assert result is not None


def _make_minimal_onnx(path):
    """Write a minimal ONNX model to *path* so create_genai_config can load it."""
    import onnx
    from onnx import TensorProto

    input_tensor = onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "seq"])
    output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab"])
    node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
    graph = onnx.helper.make_graph([node], "g", [input_tensor], [output_tensor])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
    onnx.save(model, path)


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
