# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.model import ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.model.handler.model_package import ModelPackageModelHandler


def _make_onnx_handler(tmp_path, name="model", model_attributes=None):
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{name}.onnx"
    model_file.write_text("dummy")
    return ONNXModelHandler(model_path=str(model_file), model_attributes=model_attributes)


class TestModelPackageModelHandler:
    def test_creation_and_target_iteration(self, tmp_path):
        """Handler stores targets and iterates (name, model) pairs correctly."""
        # setup
        h1 = _make_onnx_handler(tmp_path, "t1")
        h2 = _make_onnx_handler(tmp_path, "t2")

        # execute
        mt = ModelPackageModelHandler([h1, h2], ["t1", "t2"], model_path=tmp_path)

        # assert
        assert mt.target_names == ["t1", "t2"]
        pairs = list(mt.get_target_models())
        assert len(pairs) == 2
        assert pairs[0][0] == "t1"
        assert pairs[1][0] == "t2"

    def test_parent_attributes_merged_into_targets(self, tmp_path):
        """Parent-level model_attributes are merged into each target while preserving target-specific ones."""
        # setup
        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})
        mt = ModelPackageModelHandler(
            [h1, h2],
            ["t1", "t2"],
            model_path=tmp_path,
            model_attributes={"ep": "QNNExecutionProvider", "device": "NPU"},
        )

        # execute
        pairs = list(mt.get_target_models())

        # assert: parent attrs merged
        assert pairs[0][1].model_attributes["ep"] == "QNNExecutionProvider"
        assert pairs[1][1].model_attributes["device"] == "NPU"
        # assert: target-specific attrs preserved
        assert pairs[0][1].model_attributes["architecture"] == "60"
        assert pairs[1][1].model_attributes["architecture"] == "73"

    def test_to_json_round_trip(self, tmp_path):
        """to_json produces correct structure with parent/target attribute separation."""
        # setup
        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})
        mt = ModelPackageModelHandler(
            [h1, h2],
            ["t1", "t2"],
            model_path=tmp_path,
            model_attributes={"ep": "QNNExecutionProvider"},
        )

        # execute
        json_dict = mt.to_json()

        # assert
        assert json_dict["type"].lower() == "modelpackagemodel"
        assert json_dict["config"]["target_names"] == ["t1", "t2"]
        assert len(json_dict["config"]["target_models"]) == 2
        assert json_dict["config"]["model_attributes"]["ep"] == "QNNExecutionProvider"

    def test_mismatched_names_raises(self, tmp_path):
        """Mismatch between target count and name count raises AssertionError."""
        # setup
        h1 = _make_onnx_handler(tmp_path, "t1")

        # execute + assert
        with pytest.raises(AssertionError, match="Number of target models and names must match"):
            ModelPackageModelHandler([h1], ["t1", "t2"], model_path=tmp_path)

    def test_is_composite_false_for_onnx_targets(self, tmp_path):
        """is_composite returns False when all targets are ONNXModelHandler."""
        # setup
        mt = ModelPackageModelHandler(
            [_make_onnx_handler(tmp_path, "t1"), _make_onnx_handler(tmp_path, "t2")],
            ["t1", "t2"],
            model_path=tmp_path,
        )

        # execute + assert
        assert mt.is_composite is False

    def test_is_composite_true_for_composite_targets(self, tmp_path):
        """is_composite returns True when all targets are CompositeModelHandler."""
        # setup
        c1 = CompositeModelHandler([_make_onnx_handler(tmp_path, "e1")], ["enc"], model_path=str(tmp_path / "c1"))
        c2 = CompositeModelHandler([_make_onnx_handler(tmp_path, "e2")], ["enc"], model_path=str(tmp_path / "c2"))
        mt = ModelPackageModelHandler([c1, c2], ["soc_a", "soc_b"], model_path=tmp_path)

        # execute + assert
        assert mt.is_composite is True

    def test_is_composite_mixed_types_raises(self, tmp_path):
        """Mixed ONNX and Composite targets raise AssertionError."""
        # setup
        plain = _make_onnx_handler(tmp_path, "plain")
        composite = CompositeModelHandler(
            [_make_onnx_handler(tmp_path, "enc")], ["encoder"], model_path=str(tmp_path / "comp")
        )
        mt = ModelPackageModelHandler([plain, composite], ["t1", "t2"], model_path=tmp_path)

        # execute + assert
        with pytest.raises(AssertionError, match="All target models must be the same type"):
            _ = mt.is_composite

    def test_is_composite_empty_targets(self, tmp_path):
        """is_composite returns False for empty target list."""
        # setup
        mt = ModelPackageModelHandler([], [], model_path=tmp_path)

        # execute + assert
        assert mt.is_composite is False
