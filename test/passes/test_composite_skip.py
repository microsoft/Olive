# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the generic ``components_to_skip`` support in the base ``Pass`` class.

These tests use trivial dummy passes (not a real optimization pass) so that they exercise the generic
per-component composite loop in :class:`olive.passes.olive_pass.Pass`, not any pass-specific logic.
"""

import os
from pathlib import Path

import numpy as np
import onnx
import pytest

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes.olive_pass import Pass, create_pass_from_dict
from olive.passes.onnx.common import get_external_data_file_names
from olive.passes.pass_config import PassConfigParam, get_components_to_skip_config

ORIGINAL_PRODUCER = "original"
PROCESSED_PRODUCER = "processed"


class DummySkipAwarePass(Pass):
    """Dummy pass that opts into ``components_to_skip`` and marks every model it processes."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {**get_components_to_skip_config()}

    def _run_for_config(self, model, config, output_model_path: str) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
        model_proto = onnx.load(model.model_path)
        model_proto.producer_name = PROCESSED_PRODUCER
        onnx.save(model_proto, output_model_path)
        return ONNXModelHandler(model_path=output_model_path)


class DummyNoSkipPass(DummySkipAwarePass):
    """Dummy pass that does NOT opt into ``components_to_skip`` (the default for every Olive pass)."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {}


class DummyCompositeAcceptingPass(DummySkipAwarePass):
    """Invalid pass: processes composite models as a whole yet declares ``components_to_skip``."""

    _accepts_composite_model = True


def make_pass(pass_cls: type[Pass], **config):
    accelerator_spec = AcceleratorSpec(accelerator_type="CPU", execution_provider="CPUExecutionProvider")
    return create_pass_from_dict(pass_cls, config, disable_search=True, accelerator_spec=accelerator_spec)


def make_onnx_model(model_dir: Path, external_data_location: str = None) -> ONNXModelHandler:
    """Create a tiny MatMul ONNX model with ``producer_name == ORIGINAL_PRODUCER``."""
    model_dir.mkdir(parents=True, exist_ok=True)
    weight = np.random.randn(4, 8).astype(np.float32)
    inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])
    out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])
    weight_init = onnx.numpy_helper.from_array(weight, name="weight")
    node = onnx.helper.make_node("MatMul", ["input", "weight"], ["output"], name="MatMul_Node")
    graph = onnx.helper.make_graph([node], "g", [inp], [out], initializer=[weight_init])
    model_proto = onnx.helper.make_model(graph, producer_name=ORIGINAL_PRODUCER)
    model_proto.opset_import[0].version = 13

    if external_data_location:
        (model_dir / external_data_location).parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(
            model_proto,
            str(model_dir / "model.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_location,
            size_threshold=0,
        )
    else:
        onnx.save(model_proto, str(model_dir / "model.onnx"))
    return ONNXModelHandler(model_path=str(model_dir), onnx_file_name="model.onnx")


def get_component(model: CompositeModelHandler, name: str):
    return next(component for component_name, component in model.get_model_components() if component_name == name)


def get_producer(model: ONNXModelHandler) -> str:
    return onnx.load(model.model_path, load_external_data=False).producer_name


class TestComponentsToSkip:
    def test_skipped_component_is_passed_through_unchanged(self, tmp_path):
        composite = CompositeModelHandler(
            model_components=[make_onnx_model(tmp_path / "src" / "decoder"), make_onnx_model(tmp_path / "src" / "emb")],
            model_component_names=["decoder", "embedding"],
            model_path=str(tmp_path / "src"),
        )

        result = make_pass(DummySkipAwarePass, components_to_skip=["embedding"]).run(composite, str(tmp_path / "out"))

        assert isinstance(result, CompositeModelHandler)
        assert result.model_component_names == ["decoder", "embedding"]
        assert get_producer(get_component(result, "decoder")) == PROCESSED_PRODUCER
        assert get_producer(get_component(result, "embedding")) == ORIGINAL_PRODUCER

    def test_components_to_skip_none_processes_all(self, tmp_path):
        composite = CompositeModelHandler(
            model_components=[make_onnx_model(tmp_path / "src" / "decoder"), make_onnx_model(tmp_path / "src" / "emb")],
            model_component_names=["decoder", "embedding"],
            model_path=str(tmp_path / "src"),
        )

        result = make_pass(DummySkipAwarePass, components_to_skip=None).run(composite, str(tmp_path / "out"))

        for _, component in result.get_model_components():
            assert get_producer(component) == PROCESSED_PRODUCER

    def test_single_model_is_unaffected(self, tmp_path):
        model = make_onnx_model(tmp_path / "src" / "single")

        result = make_pass(DummySkipAwarePass, components_to_skip=["single"]).run(model, str(tmp_path / "out"))

        assert get_producer(result) == PROCESSED_PRODUCER

    def test_unknown_skip_name_raises(self, tmp_path):
        composite = CompositeModelHandler(
            model_components=[make_onnx_model(tmp_path / "src" / "decoder"), make_onnx_model(tmp_path / "src" / "vis")],
            model_component_names=["decoder", "vision"],
        )

        p = make_pass(DummySkipAwarePass, components_to_skip=["typo_component"])
        with pytest.raises(ValueError, match="typo_component") as exc_info:
            p.run(composite, str(tmp_path / "out"))

        # The error must list the available names so the user can fix the typo.
        message = str(exc_info.value)
        assert "decoder" in message
        assert "vision" in message
        # Nothing may have been written before the failure.
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize(
        "malicious_name",
        ["../evil", "..", "sub/dir", "a/../../evil", os.sep + "tmp" + os.sep + "evil"],
    )
    def test_malicious_component_name_raises_before_filesystem_mutation(self, tmp_path, malicious_name):
        composite = CompositeModelHandler(
            model_components=[
                make_onnx_model(tmp_path / "src" / "decoder"),
                make_onnx_model(tmp_path / "src" / "evil"),
            ],
            model_component_names=["decoder", malicious_name],
        )

        # A sibling directory of the output dir that must not be deleted/overwritten.
        victim_dir = tmp_path / "evil"
        victim_dir.mkdir(parents=True, exist_ok=True)
        victim_file = victim_dir / "important.txt"
        victim_file.write_text("do not delete me")

        p = make_pass(DummySkipAwarePass, components_to_skip=[malicious_name])
        with pytest.raises(ValueError, match="component_name must be a simple identifier"):
            p.run(composite, str(tmp_path / "out" / "model"))

        assert victim_file.read_text() == "do not delete me"
        assert not (tmp_path / "out").exists()

    def test_malicious_component_name_raises_when_not_skipped(self, tmp_path):
        """Name validation applies to every component, not only the skipped ones."""
        composite = CompositeModelHandler(
            model_components=[
                make_onnx_model(tmp_path / "src" / "decoder"),
                make_onnx_model(tmp_path / "src" / "evil"),
            ],
            model_component_names=["decoder", "../evil"],
        )

        p = make_pass(DummySkipAwarePass, components_to_skip=["decoder"])
        with pytest.raises(ValueError, match="component_name must be a simple identifier"):
            p.run(composite, str(tmp_path / "out"))

        assert not (tmp_path / "out").exists()

    def test_skipping_non_onnx_component_raises(self, tmp_path):
        nested = CompositeModelHandler(
            model_components=[make_onnx_model(tmp_path / "src" / "inner")],
            model_component_names=["inner"],
        )
        composite = CompositeModelHandler(
            model_components=[make_onnx_model(tmp_path / "src" / "decoder"), nested],
            model_component_names=["decoder", "nested"],
        )

        p = make_pass(DummySkipAwarePass, components_to_skip=["nested"])
        with pytest.raises(ValueError, match="only supports ONNXModelHandler"):
            p.run(composite, str(tmp_path / "out"))

    def test_external_data_of_skipped_component_is_copied(self, tmp_path):
        """A skipped component with a non-default external-data filename stays loadable."""
        embedding = make_onnx_model(tmp_path / "src" / "embedding", external_data_location="weights.bin")
        composite = CompositeModelHandler(
            model_components=[make_onnx_model(tmp_path / "src" / "decoder"), embedding],
            model_component_names=["decoder", "embedding"],
            model_path=str(tmp_path / "src"),
        )

        result = make_pass(DummySkipAwarePass, components_to_skip=["embedding"]).run(composite, str(tmp_path / "out"))

        emb_out = get_component(result, "embedding")
        # resave_model normalizes the external-data file name to "<model>.onnx.data" and rewrites the
        # location in the model accordingly, so the weights are preserved regardless of the source name.
        copied_data = Path(emb_out.model_path).parent / "model.onnx.data"
        assert copied_data.exists()
        assert copied_data.read_bytes() == (tmp_path / "src" / "embedding" / "weights.bin").read_bytes()
        loaded = onnx.load(emb_out.model_path, load_external_data=False)
        assert loaded.producer_name == ORIGINAL_PRODUCER
        assert get_external_data_file_names(emb_out.model_path) == ["model.onnx.data"]

    def test_external_data_in_subdirectory_of_skipped_component_is_copied(self, tmp_path):
        """The external-data ``location`` may contain a sub-directory."""
        embedding = make_onnx_model(tmp_path / "src" / "embedding", external_data_location="weights/data.bin")
        composite = CompositeModelHandler(
            model_components=[embedding],
            model_component_names=["embedding"],
            model_path=str(tmp_path / "src"),
        )

        result = make_pass(DummySkipAwarePass, components_to_skip=["embedding"]).run(composite, str(tmp_path / "out"))

        emb_out = get_component(result, "embedding")
        copied_data = Path(emb_out.model_path).parent / "model.onnx.data"
        assert copied_data.exists()
        assert copied_data.read_bytes() == (tmp_path / "src" / "embedding" / "weights" / "data.bin").read_bytes()
        assert get_external_data_file_names(emb_out.model_path) == ["model.onnx.data"]


class TestComponentsToSkipNestedComposite:
    @staticmethod
    def make_nested_composite(tmp_path) -> CompositeModelHandler:
        """decoder(composite: prefill, embedding) + lm_head."""
        inner = CompositeModelHandler(
            model_components=[
                make_onnx_model(tmp_path / "src" / "prefill"),
                make_onnx_model(tmp_path / "src" / "embedding"),
            ],
            model_component_names=["prefill", "embedding"],
            model_path=str(tmp_path / "src"),
        )
        return CompositeModelHandler(
            model_components=[inner, make_onnx_model(tmp_path / "src" / "lm_head")],
            model_component_names=["decoder", "lm_head"],
            model_path=str(tmp_path / "src"),
        )

    def test_skip_name_inside_nested_composite_matches(self, tmp_path):
        composite = self.make_nested_composite(tmp_path)

        result = make_pass(DummySkipAwarePass, components_to_skip=["embedding"]).run(composite, str(tmp_path / "out"))

        inner_out = get_component(result, "decoder")
        assert isinstance(inner_out, CompositeModelHandler)
        assert get_producer(get_component(inner_out, "embedding")) == ORIGINAL_PRODUCER
        assert get_producer(get_component(inner_out, "prefill")) == PROCESSED_PRODUCER
        assert get_producer(get_component(result, "lm_head")) == PROCESSED_PRODUCER

    def test_nested_only_name_does_not_trigger_unknown_name_error(self, tmp_path):
        """A name that exists only at a nested level must not be reported as unknown."""
        composite = self.make_nested_composite(tmp_path)

        # Would raise if the unknown-name check only looked at the top-level component names.
        result = make_pass(DummySkipAwarePass, components_to_skip=["prefill"]).run(composite, str(tmp_path / "out"))

        inner_out = get_component(result, "decoder")
        assert get_producer(get_component(inner_out, "prefill")) == ORIGINAL_PRODUCER

    def test_unknown_name_in_nested_composite_raises(self, tmp_path):
        composite = self.make_nested_composite(tmp_path)

        p = make_pass(DummySkipAwarePass, components_to_skip=["nope"])
        with pytest.raises(ValueError, match="nope"):
            p.run(composite, str(tmp_path / "out"))


class TestPassWithoutComponentsToSkip:
    def test_pass_without_opt_in_processes_all_components(self, tmp_path):
        """Backward-compat guarantee: passes that don't opt in are unaffected."""
        composite = CompositeModelHandler(
            model_components=[make_onnx_model(tmp_path / "src" / "decoder"), make_onnx_model(tmp_path / "src" / "emb")],
            model_component_names=["decoder", "embedding"],
            model_path=str(tmp_path / "src"),
        )

        p = make_pass(DummyNoSkipPass)
        assert "components_to_skip" not in DummyNoSkipPass._default_config(  # pylint: disable=protected-access
            p.accelerator_spec
        )
        assert getattr(p.config, "components_to_skip", None) is None

        result = p.run(composite, str(tmp_path / "out"))

        assert result.model_component_names == ["decoder", "embedding"]
        for _, component in result.get_model_components():
            assert get_producer(component) == PROCESSED_PRODUCER

    def test_nested_composite_without_opt_in(self, tmp_path):
        composite = TestComponentsToSkipNestedComposite.make_nested_composite(tmp_path)

        result = make_pass(DummyNoSkipPass).run(composite, str(tmp_path / "out"))

        inner_out = get_component(result, "decoder")
        assert isinstance(inner_out, CompositeModelHandler)
        for _, component in inner_out.get_model_components():
            assert get_producer(component) == PROCESSED_PRODUCER
        assert get_producer(get_component(result, "lm_head")) == PROCESSED_PRODUCER


class TestAcceptsCompositeModelGuardrail:
    def test_pass_accepting_composite_model_cannot_declare_components_to_skip(self):
        with pytest.raises(AssertionError, match="components_to_skip"):
            make_pass(DummyCompositeAcceptingPass, components_to_skip=["embedding"])

    def test_guardrail_fires_even_without_a_value(self):
        with pytest.raises(AssertionError, match="_accepts_composite_model"):
            make_pass(DummyCompositeAcceptingPass)
