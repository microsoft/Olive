# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------
# pylint: disable=protected-access

import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.model import QairtModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.qairt.gen_ai_builder import QairtGenAIBuilder


def test_gen_ai_builder_default_config(mock_accelerator_spec):
    """Test that the default config is correctly generated."""
    config = QairtGenAIBuilder._default_config(mock_accelerator_spec)  # pylint: disable=protected-access

    assert "cache_dir" in config
    assert config["cache_dir"].default_value == "./cache/qairt/gen_ai_builder"
    assert "log_level" in config
    assert "backend" in config
    assert config["backend"].default_value == "CPU"
    assert "soc_details" in config
    assert "vtcm_size_in_mb" in config
    assert config["vtcm_size_in_mb"].default_value == 0
    assert "hvx_threads" in config
    assert config["hvx_threads"].default_value == 0
    assert "extended_udma" in config
    assert config["extended_udma"].default_value is False
    assert "sequence_lengths" in config
    assert "num_splits" in config
    assert config["num_splits"].default_value == -1
    assert "multi_graph" in config
    assert config["multi_graph"].default_value is False


def test_gen_ai_builder_cpu_backend_success(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test successful execution with CPU backend and HfModelHandler."""
    output_path = tmp_path / "output"

    # Mock GenAIBuilderFactory and container
    mock_builder = MagicMock()
    mock_container = MagicMock()
    mock_builder.build.return_value = mock_container

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU"},
        disable_search=True,
    )

    result = gen_ai_pass.run(mock_hf_model, str(output_path))

    # Verify result
    assert isinstance(result, QairtModelHandler)
    assert result.model_path == str(output_path)

    # Verify GenAIBuilderFactory was called correctly
    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.assert_called_once()
    call_kwargs = mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.call_args.kwargs
    assert call_kwargs["pretrained_model_path"] == Path(mock_hf_model.model_path)
    assert call_kwargs["backend_type"] == "CPU"

    # Verify build and save were called
    mock_builder.build.assert_called_once()
    mock_container.save.assert_called_once_with(str(output_path), exist_ok=True)


def test_gen_ai_builder_htp_backend_success(tmp_path, mock_qairt_prepared_model, mock_qairt_modules):
    """Test successful execution with HTP backend and QairtPreparedModelHandler."""
    output_path = tmp_path / "output"

    # Mock GenAIBuilderFactory and container
    mock_builder = MagicMock()
    mock_container = MagicMock()
    mock_builder.build.return_value = mock_container
    mock_builder._transformation_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config.arn_cl_options = MagicMock()
    mock_builder._transformation_config.model_transformer_config.split_model = MagicMock()
    mock_builder._compilation_config = MagicMock()
    mock_builder._compilation_config.graph_custom_configs = [MagicMock()]
    mock_builder._compilation_config.device_custom_configs = [MagicMock()]
    mock_builder._compilation_config.context_custom_configs = [MagicMock()]

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {
            "backend": "HTP",
            "soc_details": "chipset:SC8380XP",
            "vtcm_size_in_mb": 8,
            "hvx_threads": 4,
        },
        disable_search=True,
    )

    result = gen_ai_pass.run(mock_qairt_prepared_model, str(output_path))

    # Verify result
    assert isinstance(result, QairtModelHandler)
    assert result.model_path == str(output_path)

    # Verify GenAIBuilderFactory was called with base/onnx path for HTP
    call_kwargs = mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.call_args.kwargs
    assert call_kwargs["pretrained_model_path"] == Path(mock_qairt_prepared_model.model_path) / "base" / "onnx"

    # Verify HTP-specific configurations were set
    mock_builder.set_targets.assert_called_once_with(["chipset:SC8380XP"])
    assert mock_builder._compilation_config.graph_custom_configs[0].vtcm_size_in_mb == 8
    assert mock_builder._compilation_config.graph_custom_configs[0].hvx_threads == 4


def test_gen_ai_builder_cpu_invalid_input_model(tmp_path, mock_qairt_prepared_model, mock_qairt_modules):
    """Test that ValueError is raised when CPU backend receives QairtPreparedModelHandler."""
    output_path = tmp_path / "output"

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU"},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="QAIRT CPU GenAIBuilder can only consume HfModelHandler"):
        gen_ai_pass.run(mock_qairt_prepared_model, str(output_path))


def test_gen_ai_builder_htp_invalid_input_model(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that ValueError is raised when HTP backend receives HfModelHandler."""
    output_path = tmp_path / "output"

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "HTP"},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="QAIRT HTP GenAIBuilder can only consume QairtPreparedModelHandler"):
        gen_ai_pass.run(mock_hf_model, str(output_path))


def test_gen_ai_builder_validate_config_cpu_rejects_htp_options(mock_accelerator_spec, mock_qairt_modules):
    """Test that validate_config rejects HTP-only options for CPU backend."""
    # Test extended_udma
    config = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU", "extended_udma": True},
        disable_search=True,
    ).config
    assert QairtGenAIBuilder.validate_config(config, mock_accelerator_spec) is False

    # Test vtcm_size_in_mb
    config = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU", "vtcm_size_in_mb": 8},
        disable_search=True,
    ).config
    assert QairtGenAIBuilder.validate_config(config, mock_accelerator_spec) is False

    # Test hvx_threads
    config = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU", "hvx_threads": 4},
        disable_search=True,
    ).config
    assert QairtGenAIBuilder.validate_config(config, mock_accelerator_spec) is False

    # Test sequence_lengths
    config = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU", "sequence_lengths": [128, 256]},
        disable_search=True,
    ).config
    assert QairtGenAIBuilder.validate_config(config, mock_accelerator_spec) is False

    # Test num_splits
    config = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU", "num_splits": 2},
        disable_search=True,
    ).config
    assert QairtGenAIBuilder.validate_config(config, mock_accelerator_spec) is False

    # Test multi_graph
    config = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU", "multi_graph": True},
        disable_search=True,
    ).config
    assert QairtGenAIBuilder.validate_config(config, mock_accelerator_spec) is False


def test_gen_ai_builder_extended_udma_dsp_arch_validation(tmp_path, mock_qairt_prepared_model, mock_qairt_modules):
    """Test that extended_udma requires DSP arch >= v81."""
    output_path = tmp_path / "output"

    # Mock builder with DSP arch < v81
    mock_builder = MagicMock()
    mock_builder._compilation_config = MagicMock()
    mock_builder._compilation_config.device_custom_configs = [MagicMock()]
    mock_builder._compilation_config.device_custom_configs[0].dsp_arch = "v73"
    mock_builder._compilation_config.context_custom_configs = [MagicMock()]
    mock_builder._compilation_config.graph_custom_configs = [MagicMock()]
    mock_builder._transformation_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config.arn_cl_options = MagicMock()
    mock_builder._transformation_config.model_transformer_config.split_model = MagicMock()

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "HTP", "extended_udma": True},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="extended_udma is unsupported on DSP architectures less than v81"):
        gen_ai_pass.run(mock_qairt_prepared_model, str(output_path))


def test_gen_ai_builder_extended_udma_success(tmp_path, mock_qairt_prepared_model, mock_qairt_modules):
    """Test that extended_udma works with DSP arch >= v81."""
    output_path = tmp_path / "output"

    # Mock builder with DSP arch >= v81
    mock_builder = MagicMock()
    mock_container = MagicMock()
    mock_builder.build.return_value = mock_container
    mock_builder._compilation_config = MagicMock()
    mock_builder._compilation_config.device_custom_configs = [MagicMock()]
    mock_builder._compilation_config.device_custom_configs[0].dsp_arch = "v81"
    mock_builder._compilation_config.context_custom_configs = [MagicMock()]
    mock_builder._compilation_config.graph_custom_configs = [MagicMock()]
    mock_builder._transformation_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config.arn_cl_options = MagicMock()
    mock_builder._transformation_config.model_transformer_config.split_model = MagicMock()

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "HTP", "extended_udma": True},
        disable_search=True,
    )

    result = gen_ai_pass.run(mock_qairt_prepared_model, str(output_path))

    # Verify extended_udma was set
    assert mock_builder._compilation_config.context_custom_configs[0].extended_udma is True
    assert isinstance(result, QairtModelHandler)


def test_gen_ai_builder_htp_configurations(tmp_path, mock_qairt_prepared_model, mock_qairt_modules):
    """Test that all HTP-specific configurations are properly applied."""
    output_path = tmp_path / "output"

    # Mock builder
    mock_builder = MagicMock()
    mock_container = MagicMock()
    mock_builder.build.return_value = mock_container
    mock_builder._compilation_config = MagicMock()
    mock_builder._compilation_config.graph_custom_configs = [MagicMock()]
    mock_builder._compilation_config.device_custom_configs = [MagicMock()]
    mock_builder._compilation_config.context_custom_configs = [MagicMock()]
    mock_builder._transformation_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config.arn_cl_options = MagicMock()
    mock_builder._transformation_config.model_transformer_config.split_model = MagicMock()

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {
            "backend": "HTP",
            "sequence_lengths": [128, 256, 512],
            "num_splits": 4,
            "multi_graph": True,
        },
        disable_search=True,
    )

    result = gen_ai_pass.run(mock_qairt_prepared_model, str(output_path))

    # Verify configurations were set
    assert mock_builder._transformation_config.model_transformer_config.arn_cl_options.auto_regression_number == [
        128,
        256,
        512,
    ]
    assert mock_builder._transformation_config.model_transformer_config.split_model.num_splits == 4
    assert mock_builder.multi_graph is True
    assert isinstance(result, QairtModelHandler)


def test_gen_ai_builder_import_error(tmp_path, mock_hf_model):
    """Test that ImportError is raised if qairt cannot be imported."""

    def import_side_effect(name, *args, **kwargs):
        if name in ["qairt", "qairt.gen_ai_api"]:
            raise ImportError("Mock import error")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__

    with patch("builtins.__import__", side_effect=import_side_effect):
        gen_ai_pass = create_pass_from_dict(
            QairtGenAIBuilder,
            {"backend": "CPU"},
            disable_search=True,
        )

        with pytest.raises(ImportError, match="Failed to import QAIRT GenAIBuilder API"):
            gen_ai_pass.run(mock_hf_model, str(tmp_path / "output"))


def test_gen_ai_builder_passthrough_files(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that passthrough files are copied to output."""
    output_path = tmp_path / "output"

    # Create passthrough files in the model directory
    model_path = Path(mock_hf_model.model_path)
    (model_path / "chat_template.jinja").write_text("template content")
    (model_path / "tokenizer.json").write_text('{"vocab": {}}')
    (model_path / "tokenizer_config.json").write_text('{"model_max_length": 2048}')

    # Mock GenAIBuilderFactory and container
    mock_builder = MagicMock()
    mock_container = MagicMock()
    mock_builder.build.return_value = mock_container

    # Mock container.save to create the output directory (mimicking real behavior)
    def mock_save(path, exist_ok=False):
        Path(path).mkdir(parents=True, exist_ok=exist_ok)

    mock_container.save.side_effect = mock_save

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU"},
        disable_search=True,
    )

    _ = gen_ai_pass.run(mock_hf_model, str(output_path))

    # Verify passthrough files were copied
    assert (Path(output_path) / "chat_template.jinja").exists()
    assert (Path(output_path) / "tokenizer.json").exists()
    assert (Path(output_path) / "tokenizer_config.json").exists()


def test_gen_ai_builder_sdk_version_check_old_version(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that OSError is raised for QAIRT SDK version < 2.45.0."""
    output_path = tmp_path / "output"

    # Mock SDK version to be less than 2.45.0
    mock_qairt_modules["qairt"].__sdk_version__ = "2.44.0"

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU"},
        disable_search=True,
    )

    with pytest.raises(OSError, match="QairtGenAIBuilder pass is unsupported for QAIRT versions < 2.45.0"):
        gen_ai_pass.run(mock_hf_model, str(output_path))


def test_gen_ai_builder_sdk_version_check_valid_version(tmp_path, mock_hf_model, mock_qairt_modules):
    """Test that gen_ai_builder works with QAIRT SDK version >= 2.45.0."""
    output_path = tmp_path / "output"

    # Mock SDK version to be >= 2.45.0
    mock_qairt_modules["qairt"].__sdk_version__ = "2.45.0"

    # Mock GenAIBuilderFactory and container
    mock_builder = MagicMock()
    mock_container = MagicMock()
    mock_builder.build.return_value = mock_container

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "CPU"},
        disable_search=True,
    )

    # Should not raise an error
    result = gen_ai_pass.run(mock_hf_model, str(output_path))
    assert result is not None


def test_gen_ai_builder_native_kv_configuration(tmp_path, mock_qairt_prepared_model, mock_qairt_modules):
    """Test that native_kv is properly set when enabled with valid sequence_lengths."""
    output_path = tmp_path / "output"

    # Mock builder
    mock_builder = MagicMock()
    mock_container = MagicMock()
    mock_builder.build.return_value = mock_container
    mock_builder._compilation_config = MagicMock()
    mock_builder._compilation_config.graph_custom_configs = [MagicMock()]
    mock_builder._compilation_config.device_custom_configs = [MagicMock()]
    mock_builder._compilation_config.context_custom_configs = [MagicMock()]
    mock_builder._transformation_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config.arn_cl_options = MagicMock()
    mock_builder._transformation_config.model_transformer_config.split_model = MagicMock()

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {
            "backend": "HTP",
            "sequence_lengths": [32, 128],
            "native_kv": True,
        },
        disable_search=True,
    )

    result = gen_ai_pass.run(mock_qairt_prepared_model, str(output_path))

    # Verify native_kv was set
    assert mock_builder.native_kv is True
    assert result is not None


def test_gen_ai_builder_native_kv_validation_invalid_sequence_lengths(mock_accelerator_spec, mock_qairt_modules):
    """Test that validation fails for native_kv with invalid sequence_lengths."""
    config = create_pass_from_dict(
        QairtGenAIBuilder,
        {
            "backend": "HTP",
            "sequence_lengths": [64, 256],  # Invalid for native_kv
            "native_kv": True,
        },
        disable_search=True,
    ).config

    assert QairtGenAIBuilder.validate_config(config, mock_accelerator_spec) is False


def test_gen_ai_builder_htp_no_soc_details_skips_set_targets(tmp_path, mock_qairt_prepared_model, mock_qairt_modules):
    """Test that set_targets is not called when soc_details is None (relies on QAIRT defaults)."""
    output_path = tmp_path / "output"

    mock_builder = MagicMock()
    mock_container = MagicMock()
    mock_builder.build.return_value = mock_container
    mock_builder._compilation_config = MagicMock()
    mock_builder._compilation_config.graph_custom_configs = [MagicMock()]
    mock_builder._compilation_config.device_custom_configs = [MagicMock()]
    mock_builder._compilation_config.context_custom_configs = [MagicMock()]
    mock_builder._transformation_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config = MagicMock()
    mock_builder._transformation_config.model_transformer_config.arn_cl_options = MagicMock()
    mock_builder._transformation_config.model_transformer_config.split_model = MagicMock()

    mock_qairt_modules["gen_ai_api"].GenAIBuilderFactory.create.return_value = mock_builder

    gen_ai_pass = create_pass_from_dict(
        QairtGenAIBuilder,
        {"backend": "HTP"},  # soc_details not set (defaults to None)
        disable_search=True,
    )

    result = gen_ai_pass.run(mock_qairt_prepared_model, str(output_path))

    mock_builder.set_targets.assert_not_called()
    assert isinstance(result, QairtModelHandler)


def test_gen_ai_builder_native_kv_validation_valid_sequence_lengths(mock_accelerator_spec, mock_qairt_modules):
    """Test that validation passes for native_kv with valid sequence_lengths."""
    config = create_pass_from_dict(
        QairtGenAIBuilder,
        {
            "backend": "HTP",
            "sequence_lengths": [32, 128],  # Valid for native_kv
            "native_kv": True,
        },
        disable_search=True,
    ).config

    assert QairtGenAIBuilder.validate_config(config, mock_accelerator_spec) is True
