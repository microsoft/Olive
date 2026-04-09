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
    def test_create_model_package_handler(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1")
        h2 = _make_onnx_handler(tmp_path, "t2")

        mt = ModelPackageModelHandler([h1, h2], ["t1", "t2"], model_path=tmp_path)

        assert mt.target_names == ["t1", "t2"]
        pairs = list(mt.get_target_models())
        assert len(pairs) == 2
        assert pairs[0][0] == "t1"
        assert pairs[1][0] == "t2"

    def test_model_package_handler_inherits_attributes(self, tmp_path):
        """Parent-level model_attributes are merged into each target model."""
        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})

        mt = ModelPackageModelHandler(
            [h1, h2],
            ["t1", "t2"],
            model_path=tmp_path,
            model_attributes={"ep": "QNNExecutionProvider", "device": "NPU"},
        )

        for _, target in mt.get_target_models():
            # Parent attributes are merged in
            assert target.model_attributes["ep"] == "QNNExecutionProvider"
            assert target.model_attributes["device"] == "NPU"

        # Target-specific attributes are preserved
        pairs = list(mt.get_target_models())
        assert pairs[0][1].model_attributes["architecture"] == "60"
        assert pairs[1][1].model_attributes["architecture"] == "73"

    def test_model_package_handler_to_json(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})

        mt = ModelPackageModelHandler(
            [h1, h2],
            ["t1", "t2"],
            model_path=tmp_path,
            model_attributes={"ep": "QNNExecutionProvider"},
        )

        json_dict = mt.to_json()

        assert json_dict["type"].lower() == "modelpackagemodel"
        assert json_dict["config"]["target_names"] == ["t1", "t2"]
        assert len(json_dict["config"]["target_models"]) == 2
        # Parent-level "ep" is in the parent config, not duplicated in targets
        assert json_dict["config"]["model_attributes"]["ep"] == "QNNExecutionProvider"

    def test_model_package_handler_mismatched_names_raises(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1")
        with pytest.raises(AssertionError, match="Number of target models and names must match"):
            ModelPackageModelHandler([h1], ["t1", "t2"], model_path=tmp_path)

    def test_is_composite_false_for_plain_targets(self, tmp_path):
        # setup
        h1 = _make_onnx_handler(tmp_path, "t1")
        h2 = _make_onnx_handler(tmp_path, "t2")
        mt = ModelPackageModelHandler([h1, h2], ["t1", "t2"], model_path=tmp_path)

        # execute / assert
        assert mt.is_composite is False

    def test_is_composite_true_for_composite_targets(self, tmp_path):
        # setup
        c1 = CompositeModelHandler([_make_onnx_handler(tmp_path, "enc1")], ["encoder"], model_path=str(tmp_path / "c1"))
        c2 = CompositeModelHandler([_make_onnx_handler(tmp_path, "enc2")], ["encoder"], model_path=str(tmp_path / "c2"))
        mt = ModelPackageModelHandler([c1, c2], ["soc_a", "soc_b"], model_path=tmp_path)

        # execute / assert
        assert mt.is_composite is True

    def test_is_composite_empty_targets(self, tmp_path):
        # setup
        mt = ModelPackageModelHandler([], [], model_path=tmp_path)

        # execute / assert
        assert mt.is_composite is False

    def test_is_composite_mixed_types_raises(self, tmp_path):
        # setup
        plain = _make_onnx_handler(tmp_path, "plain")
        composite = CompositeModelHandler(
            [_make_onnx_handler(tmp_path, "enc")], ["encoder"], model_path=str(tmp_path / "comp")
        )
        mt = ModelPackageModelHandler([plain, composite], ["t1", "t2"], model_path=tmp_path)

        # execute / assert
        with pytest.raises(AssertionError, match="All target models must be the same type"):
            _ = mt.is_composite
