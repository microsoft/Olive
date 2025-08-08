# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from unittest.mock import patch

import onnx
import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.add_metadata import AddOliveMetadata
from test.utils import get_onnx_model


class TestAddOliveMetadata:
    """Test cases for AddOliveMetadata pass."""

    def test_add_metadata_basic(self, tmp_path):
        """Test basic metadata addition with required graph_name and automatic Olive version."""
        # Setup
        input_model = get_onnx_model()
        config = {"graph_name": "test_graph"}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        assert Path(output_model.model_path).exists()
        # Load the output model and check graph name
        onnx_model = onnx.load_model(output_model.model_path)
        assert onnx_model.graph.name == "test_graph"

        # Check that Olive version and model hash are always added
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert "olive_version" in metadata_dict
        assert "model_hash" in metadata_dict

    def test_add_metadata_missing_graph_name(self, tmp_path):
        """Test that missing graph_name raises ValidationError during pass creation."""
        # Setup
        from olive.common.pydantic_v1 import ValidationError

        config = {}  # Missing required graph_name

        # Execute and Assert - Should fail during pass creation, not execution
        with pytest.raises(ValidationError, match="field required"):
            create_pass_from_dict(AddOliveMetadata, config, disable_search=True)

    def test_add_metadata_with_custom_metadata(self, tmp_path):
        """Test adding custom metadata."""
        # Setup
        input_model = get_onnx_model()
        config = {
            "graph_name": "custom_graph",
            "custom_metadata": {"author": "test_user", "version": "1.0.0", "description": "test model"},
        }
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        assert onnx_model.graph.name == "custom_graph"

        # Check custom metadata
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert metadata_dict["author"] == "test_user"
        assert metadata_dict["version"] == "1.0.0"
        assert metadata_dict["description"] == "test model"

    @patch("olive.__version__", "1.2.3")
    def test_add_metadata_with_olive_version(self, tmp_path):
        """Test that Olive version metadata is always added."""
        # Setup
        input_model = get_onnx_model()
        config = {"graph_name": "version_test_graph"}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert metadata_dict["olive_version"] == "1.2.3"

    def test_add_metadata_with_model_attributes(self, tmp_path):
        """Test adding optimization information from model attributes."""
        # Setup
        model_attributes = {
            "optimization_passes": ["OnnxTransformerOptimization", "OnnxQuantization"],
            "hf_task": "text-classification",
            "quantization_config": "dynamic_int8",
        }
        input_model = get_onnx_model(model_attributes=model_attributes)
        config = {"graph_name": "optimized_graph", "add_optimization_info": True}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

        assert metadata_dict["optimization_passes"] == "OnnxTransformerOptimization, OnnxQuantization"
        assert metadata_dict["hf_task"] == "text-classification"
        assert metadata_dict["quantization_config"] == "dynamic_int8"

    def test_add_metadata_without_optimization_info(self, tmp_path):
        """Test that optimization info is not added when disabled."""
        # Setup
        model_attributes = {
            "optimization_passes": ["OnnxTransformerOptimization"],
        }
        input_model = get_onnx_model(model_attributes=model_attributes)
        config = {"graph_name": "no_opt_info_graph", "add_optimization_info": False}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

        assert "optimization_passes" not in metadata_dict

    def test_add_metadata_optimization_passes_string(self, tmp_path):
        """Test handling of optimization_passes as string instead of list."""
        # Setup
        model_attributes = {"optimization_passes": "SinglePassAsString"}
        input_model = get_onnx_model(model_attributes=model_attributes)
        config = {"graph_name": "string_pass_graph", "add_optimization_info": True}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert metadata_dict["optimization_passes"] == "SinglePassAsString"

    def test_add_metadata_no_model_attributes(self, tmp_path):
        """Test behavior when model has no model_attributes."""
        # Setup
        input_model = get_onnx_model(model_attributes=None)
        config = {"graph_name": "no_attrs_graph", "add_optimization_info": True}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert - should complete successfully without errors
        onnx_model = onnx.load_model(output_model.model_path)
        assert onnx_model.graph.name == "no_attrs_graph"
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

        # Should not have optimization info since no model_attributes
        assert "optimization_passes" not in metadata_dict

    def test_add_metadata_existing_metadata_overwrite(self, tmp_path):
        """Test that new metadata overwrites existing metadata with same keys."""
        # Setup - create a model with existing metadata
        input_model = get_onnx_model()

        # Add some existing metadata to the model
        onnx_model = onnx.load_model(input_model.model_path)
        onnx.helper.set_model_props(onnx_model, {"author": "original_author", "version": "0.1.0"})

        # Save model with existing metadata
        temp_model_path = tmp_path / "input_with_metadata.onnx"
        onnx.save_model(onnx_model, str(temp_model_path))

        # Create new model handler with this modified model
        from olive.model import ONNXModelHandler

        input_model_with_metadata = ONNXModelHandler(model_path=str(temp_model_path))

        config = {
            "graph_name": "overwrite_test_graph",
            "custom_metadata": {
                "author": "new_author",  # This should overwrite
                "license": "MIT",  # This should be added
            },
        }
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model_with_metadata, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

        assert metadata_dict["author"] == "new_author"  # Overwritten
        assert metadata_dict["version"] == "0.1.0"  # Preserved
        assert metadata_dict["license"] == "MIT"  # Added

    def test_add_metadata_invalid_model_type(self, tmp_path):
        """Test that non-ONNX model types raise ValueError."""
        # Setup
        from test.utils import get_pytorch_model

        input_model = get_pytorch_model()
        config = {"graph_name": "test_graph"}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute and Assert
        with pytest.raises(ValueError, match="Model must be an instance of ONNXModelHandler"):
            p.run(input_model, output_folder)

    @patch("olive.passes.onnx.add_metadata.getattr")
    def test_add_metadata_olive_version_exception(self, mock_getattr, tmp_path):
        """Test handling of exception when getting Olive version."""
        # Setup
        mock_getattr.side_effect = AttributeError("No version attribute")

        input_model = get_onnx_model()
        config = {"graph_name": "exception_test_graph"}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert metadata_dict["olive_version"] == "unknown"

    def test_add_metadata_with_model_hashes(self, tmp_path):
        """Test that model hashes are always included in metadata."""
        # Setup
        input_model = get_onnx_model()
        config = {
            "graph_name": "test_graph_with_hashes",
            "custom_metadata": {"test_key": "test_value"},
        }
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        assert Path(output_model.model_path).exists()
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

        # Verify hash fields are always present
        assert "model_hash" in metadata_dict

        # Verify hashes are valid SHA256 (64 hex characters)
        assert len(metadata_dict["model_hash"]) == 64
        assert metadata_dict["model_hash"] != "unknown"

        # Verify other metadata is still present
        assert metadata_dict["test_key"] == "test_value"
        assert "olive_version" in metadata_dict

    def test_add_metadata_no_custom_config(self, tmp_path):
        """Test that hashes are always included even with minimal config."""
        # Setup
        input_model = get_onnx_model()
        config = {
            "graph_name": "test_graph_minimal",
        }
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

        # Verify hash fields are always present even with minimal config
        assert "model_hash" in metadata_dict

        # Verify olive_version is always present
        assert "olive_version" in metadata_dict

    def test_add_metadata_always_includes_hashes(self, tmp_path):
        """Test that hashes are always included by default."""
        # Setup
        input_model = get_onnx_model()
        config = {
            "graph_name": "test_graph_default_hashes"
            # No hash-related config specified - should be included by default
        }
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

        # Verify hash fields are always present (default behavior)
        assert "model_hash" in metadata_dict

    @patch("olive.passes.onnx.add_metadata.logger")
    def test_add_metadata_hash_calculation_error(self, mock_logger, tmp_path):
        """Test handling of hash calculation errors."""
        # Setup
        input_model = get_onnx_model()
        config = {
            "graph_name": "test_graph_hash_error",
        }

        # Create pass and patch the ModelConfig.parse_obj method to raise an exception
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)

        with patch("olive.model.config.model_config.ModelConfig.parse_obj", side_effect=Exception("Hash error")):
            output_folder = str(tmp_path / "onnx")

            # Execute - should not fail despite hash calculation error
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Hash should be "unknown" due to error (but still present)
            assert metadata_dict["model_hash"] == "unknown"

            # Verify warning was logged
            mock_logger.warning.assert_called()

    def test_calculate_model_hash_consistency(self, tmp_path):
        """Test that hash calculation is consistent for the same model."""
        # Setup
        input_model = get_onnx_model()

        # Calculate hash twice for same model using the actual implementation approach
        from olive.model.config.model_config import ModelConfig

        model_config1 = ModelConfig.parse_obj(input_model.to_json())
        hash1 = model_config1.get_model_identifier()

        model_config2 = ModelConfig.parse_obj(input_model.to_json())
        hash2 = model_config2.get_model_identifier()

        # Hashes should be identical
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 length
        assert isinstance(hash1, str)

    def test_add_metadata_with_hf_model_name(self, tmp_path):
        """Test that HF model name is automatically included when model type is HfModel."""
        # Setup - Mock model.to_json() to return HfModel config
        input_model = get_onnx_model()

        # Patch the to_json method to return HfModel configuration with model_attributes
        with patch.object(
            input_model,
            "to_json",
            return_value={
                "config": {
                    "type": "ONNXModel",  # This would be ONNXModel for converted models
                    "model_attributes": {
                        "type": "hfmodel",  # Preserved original model type
                        "_name_or_path": "microsoft/Phi-3.5-mini-instruct",
                        "hf_task": "text-generation",
                    },
                }
            },
        ):
            config = {"graph_name": "hf_model_graph"}
            p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
            output_folder = str(tmp_path / "onnx")

            # Execute
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Verify HF model name is included
            assert "hf_model_name" in metadata_dict
            assert metadata_dict["hf_model_name"] == "microsoft/Phi-3.5-mini-instruct"

    def test_add_metadata_without_hf_model_name(self, tmp_path):
        """Test that non-HF models don't include HF model name."""
        # Setup - Mock model.to_json() to return non-HfModel config
        input_model = get_onnx_model()

        # Patch the to_json method to return ONNXModel configuration
        with patch.object(
            input_model, "to_json", return_value={"config": {"type": "ONNXModel", "model_path": "/path/to/model.onnx"}}
        ):
            config = {"graph_name": "non_hf_model_graph"}
            p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
            output_folder = str(tmp_path / "onnx")

            # Execute
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Verify HF model name is not included for non-HF models
            assert "hf_model_name" not in metadata_dict

    def test_add_metadata_hf_model_with_pytorch_type(self, tmp_path):
        """Test that PyTorchModel with HF model path doesn't include HF model name."""
        # Setup - Mock model.to_json() to return PyTorchModel config with HF model path
        input_model = get_onnx_model()

        # Patch the to_json method to return PyTorchModel configuration
        with patch.object(
            input_model,
            "to_json",
            return_value={
                "config": {
                    "type": "PyTorchModel",
                    "model_path": "microsoft/Phi-3.5-mini-instruct",  # HF path but wrong type
                }
            },
        ):
            config = {"graph_name": "pytorch_with_hf_path_graph"}
            p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
            output_folder = str(tmp_path / "onnx")

            # Execute
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Verify HF model name is not included since type is not HfModel
            assert "hf_model_name" not in metadata_dict

    def test_add_metadata_hf_model_missing_model_path(self, tmp_path):
        """Test that HfModel without model_path doesn't include HF model name."""
        # Setup - Mock model.to_json() to return HfModel config without model_path
        input_model = get_onnx_model()

        # Patch the to_json method to return HfModel configuration without model_path
        with patch.object(
            input_model,
            "to_json",
            return_value={
                "config": {
                    "type": "HfModel",
                    "task": "text-generation",
                    # No model_path field
                }
            },
        ):
            config = {"graph_name": "hf_model_no_path_graph"}
            p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
            output_folder = str(tmp_path / "onnx")

            # Execute
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Verify HF model name is not included since model_path is missing
            assert "hf_model_name" not in metadata_dict

    def test_add_metadata_hf_model_empty_model_path(self, tmp_path):
        """Test that HfModel with empty model_path doesn't include HF model name."""
        # Setup - Mock model.to_json() to return HfModel config with empty model_path
        input_model = get_onnx_model()

        # Patch the to_json method to return HfModel configuration with empty model_path
        with patch.object(
            input_model,
            "to_json",
            return_value={
                "config": {
                    "type": "HfModel",
                    "model_path": "",  # Empty string
                    "task": "text-generation",
                }
            },
        ):
            config = {"graph_name": "hf_model_empty_path_graph"}
            p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
            output_folder = str(tmp_path / "onnx")

            # Execute
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Verify HF model name is not included since model_path is empty
            assert "hf_model_name" not in metadata_dict

    def test_add_metadata_hf_model_non_string_model_path(self, tmp_path):
        """Test that HfModel with non-string model_path doesn't include HF model name."""
        # Setup - Mock model.to_json() to return HfModel config with non-string model_path
        input_model = get_onnx_model()

        # Patch the to_json method to return HfModel configuration with non-string model_path
        with patch.object(
            input_model,
            "to_json",
            return_value={
                "config": {
                    "type": "HfModel",
                    "model_path": 12345,  # Non-string type
                    "task": "text-generation",
                }
            },
        ):
            config = {"graph_name": "hf_model_non_string_path_graph"}
            p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
            output_folder = str(tmp_path / "onnx")

            # Execute
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Verify HF model name is not included since model_path is not a string
            assert "hf_model_name" not in metadata_dict

    def test_add_metadata_hf_model_no_config(self, tmp_path):
        """Test that model without config doesn't include HF model name."""
        # Setup - Mock model.to_json() to return empty response
        input_model = get_onnx_model()

        # Patch the to_json method to return empty configuration
        with patch.object(input_model, "to_json", return_value={}):
            config = {"graph_name": "no_config_graph"}
            p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
            output_folder = str(tmp_path / "onnx")

            # Execute
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Verify HF model name is not included since there's no config
            assert "hf_model_name" not in metadata_dict

    def test_add_metadata_hf_model_to_json_exception(self, tmp_path):
        """Test that exception in to_json() doesn't include HF model name."""
        # Setup - Mock model.to_json() to raise exception
        input_model = get_onnx_model()

        # Patch the to_json method to raise an exception
        with patch.object(input_model, "to_json", side_effect=Exception("JSON conversion failed")):
            config = {"graph_name": "json_exception_graph"}
            p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
            output_folder = str(tmp_path / "onnx")

            # Execute
            output_model = p.run(input_model, output_folder)

            # Assert
            onnx_model = onnx.load_model(output_model.model_path)
            metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}

            # Verify HF model name is not included due to exception
            assert "hf_model_name" not in metadata_dict

    def test_add_metadata_with_external_data_files(self, tmp_path):
        """Test that external data files are preserved when adding metadata to directory-based models."""
        import shutil

        from olive.model import ONNXModelHandler

        # Create a test model directory with external data files
        model_dir = tmp_path / "test_model_with_external_data"
        model_dir.mkdir()

        # Create a simple ONNX model
        input_model = get_onnx_model()
        test_onnx_path = model_dir / "model.onnx"
        shutil.copy(input_model.model_path, test_onnx_path)

        # Create mock external data files (simulating OpenVINO model files)
        external_files = {
            "model.bin": b"mock binary data for weights",
            "model.xml": b"<xml>mock xml configuration</xml>",
            "config.json": b'{"model_type": "openvino"}',
        }

        for filename, content in external_files.items():
            (model_dir / filename).write_bytes(content)

        # Create ONNXModelHandler pointing to the directory
        input_model_handler = ONNXModelHandler(model_path=str(model_dir), onnx_file_name="model.onnx")

        # Setup pass
        config = {"graph_name": "external_data_test_graph"}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        # Use a directory path (without .onnx extension) to ensure directory-based output
        output_folder = str(tmp_path / "output_model_dir")

        # Execute
        output_model = p.run(input_model_handler, output_folder)

        # The framework may return either a directory-based model or a file-based model
        # Both are valid as long as all external files are preserved
        output_path = Path(output_model.model_path)

        if output_path.is_file():
            # If it's a file, it should be in the expected output directory
            expected_dir = Path(output_folder)
            actual_parent = output_path.parent
            assert actual_parent == expected_dir, f"Expected parent {expected_dir} but got {actual_parent}"
            assert output_path.name == "model.onnx"
            output_dir = actual_parent
        else:
            # If it's a directory, use it directly
            assert output_path.is_dir(), f"Expected directory but got: {output_path}"
            assert output_model.onnx_file_name == "model.onnx"
            output_dir = output_path

        # Verify ONNX file exists and has correct metadata
        output_onnx_path = output_dir / "model.onnx"
        assert output_onnx_path.exists()

        onnx_model = onnx.load_model(str(output_onnx_path))
        assert onnx_model.graph.name == "external_data_test_graph"

        # Check metadata was added
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert "olive_version" in metadata_dict
        assert "model_hash" in metadata_dict

        # Verify all external files were preserved
        for filename, original_content in external_files.items():
            external_file_path = output_dir / filename
            assert external_file_path.exists(), f"External file {filename} was not preserved"

            # Verify file content is identical
            preserved_content = external_file_path.read_bytes()
            assert preserved_content == original_content, f"Content of {filename} was modified"

    def test_add_metadata_single_file_to_directory_conversion(self, tmp_path):
        """Test that metadata is correctly added to single ONNX files."""
        # Setup single file model
        input_model = get_onnx_model()
        config = {"graph_name": "single_to_dir_test"}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "output")

        # Execute
        output_model = p.run(input_model, output_folder)

        # For single-file models without external data, the framework may return either
        # a directory-based model or a file-based model
        output_path = Path(output_model.model_path)

        if output_path.is_file():
            # If it's a file, verify it's in the expected location
            assert output_path.parent == Path(output_folder)
            onnx_file_path = output_path
        else:
            # If it's a directory, verify the structure
            assert output_path.is_dir()
            assert output_model.onnx_file_name is not None
            onnx_file_path = output_path / output_model.onnx_file_name

        # Verify the ONNX file exists and has correct metadata
        assert onnx_file_path.exists()
        onnx_model = onnx.load_model(str(onnx_file_path))
        assert onnx_model.graph.name == "single_to_dir_test"

        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert "olive_version" in metadata_dict
        assert "model_hash" in metadata_dict
