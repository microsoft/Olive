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
from test.unit_test.utils import create_onnx_model_file, delete_onnx_model_files, get_onnx_model


@pytest.fixture(scope="module", autouse=True)
def setup_onnx_model():
    """Create ONNX model file for testing."""
    create_onnx_model_file()
    yield
    delete_onnx_model_files()


class TestAddOliveMetadata:
    """Test cases for AddOliveMetadata pass."""

    def test_add_metadata_basic(self, tmp_path):
        """Test basic metadata addition with required graph_name."""
        # Setup
        input_model = get_onnx_model()
        config = {"graph_name": "test_graph"}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)        # Assert
        assert Path(output_model.model_path).exists()
        # Load the output model and check graph name
        onnx_model = onnx.load_model(output_model.model_path)
        assert onnx_model.graph.name == "test_graph"

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
        """Test adding Olive version metadata."""
        # Setup
        input_model = get_onnx_model()
        config = {"graph_name": "version_test_graph", "add_olive_version": True}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert metadata_dict["olive_version"] == "1.2.3"

    def test_add_metadata_without_olive_version(self, tmp_path):
        """Test that Olive version is not added when disabled."""
        # Setup
        input_model = get_onnx_model()
        config = {"graph_name": "no_version_graph", "add_olive_version": False}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert "olive_version" not in metadata_dict

    def test_add_metadata_with_model_attributes(self, tmp_path):
        """Test adding optimization information from model attributes."""
        # Setup
        model_attributes = {
            "original_model_path": "/path/to/original/model.onnx",
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

        assert metadata_dict["original_model_path"] == "/path/to/original/model.onnx"
        assert metadata_dict["optimization_passes"] == "OnnxTransformerOptimization, OnnxQuantization"
        assert metadata_dict["hf_task"] == "text-classification"
        assert metadata_dict["quantization_config"] == "dynamic_int8"

    def test_add_metadata_without_optimization_info(self, tmp_path):
        """Test that optimization info is not added when disabled."""
        # Setup
        model_attributes = {
            "original_model_path": "/path/to/original/model.onnx",
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

        assert "original_model_path" not in metadata_dict
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
        assert "original_model_path" not in metadata_dict
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
        from test.unit_test.utils import get_pytorch_model

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
        config = {"graph_name": "exception_test_graph", "add_olive_version": True}
        p = create_pass_from_dict(AddOliveMetadata, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert
        onnx_model = onnx.load_model(output_model.model_path)
        metadata_dict = {entry.key: entry.value for entry in onnx_model.metadata_props}
        assert metadata_dict["olive_version"] == "unknown"
