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
