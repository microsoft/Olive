# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

# The lmeval_ort module imports lm_eval at module load time.
pytest.importorskip("lm_eval")

from olive.evaluator.lmeval_ort import LMEvalOnnxBase  # pylint: disable=wrong-import-position


def test_device_property_has_setter():
    """LMEvalOnnxBase must override lm-eval's read-only ``device`` property.

    lm-eval's ``LM`` base class exposes ``device`` as a read-only property,
    but the ONNX evaluators assign ``self.device`` in ``__init__``. Without a
    setter this raises ``AttributeError: property 'device' ... has no setter``.
    This is a regression test for that incompatibility.
    """
    device_prop = LMEvalOnnxBase.__dict__.get("device")
    assert isinstance(device_prop, property)
    assert device_prop.fset is not None, "device property must define a setter"


def test_device_property_round_trips():
    """The setter stores and the getter returns the assigned value."""
    device_prop = LMEvalOnnxBase.__dict__["device"]

    # Use a bare holder to exercise the descriptor without instantiating the
    # abstract base class (which has unrelated abstract methods).
    class _Holder:
        pass

    holder = _Holder()
    assert device_prop.fget(holder) is None  # unset -> None

    device_prop.fset(holder, "cuda")
    assert device_prop.fget(holder) == "cuda"

    device_prop.fset(holder, "cpu")
    assert device_prop.fget(holder) == "cpu"


class TestResolvePastPresentShareBuffer:
    """The exported genai_config value is the default; an explicit override wins."""

    @staticmethod
    def _resolve(override, genai_config):
        from olive.evaluator.lmeval_ort import LMEvalORTGenAIEvaluator

        # pylint: disable=protected-access
        return LMEvalORTGenAIEvaluator._resolve_past_present_share_buffer(override, genai_config)

    @pytest.mark.parametrize("config_value", [True, False])
    def test_uses_config_value_when_no_override(self, config_value):
        genai_config = {"search": {"past_present_share_buffer": config_value}}
        assert self._resolve(None, genai_config) is config_value

    def test_defaults_to_false_when_absent(self):
        assert self._resolve(None, {"search": {}}) is False
        assert self._resolve(None, {}) is False

    @pytest.mark.parametrize(
        ("override", "config_value"),
        [(False, True), (True, False)],
    )
    def test_override_takes_precedence_over_config(self, override, config_value):
        # Gemma 4 exports with shared buffers enabled but requires it disabled for evaluation.
        genai_config = {"search": {"past_present_share_buffer": config_value}}
        assert self._resolve(override, genai_config) is override
