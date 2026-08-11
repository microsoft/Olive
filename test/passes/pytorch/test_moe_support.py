# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for shared fused-MoE layout safety checks."""

import pytest
import torch

from olive.passes.pytorch.moe_support import MoeSupportError, check_moe_layout_support


class _FakeExperts(torch.nn.Module):
    def __init__(self, **flags):
        super().__init__()
        for key, value in flags.items():
            setattr(self, key, value)


def make_fake_experts(class_name: str = "FakeExperts", **flags) -> torch.nn.Module:
    """Build a stand-in experts module whose class name is ``class_name``."""
    return type(class_name, (_FakeExperts,), {})(**flags)


@pytest.mark.parametrize("model_type", ["mixtral", "qwen3_moe", "some_future_moe_architecture"])
def test_check_support_accepts_any_model_type_when_not_transposed(model_type: str):
    """``is_transposed=False`` is accepted for any model_type, not just a fixed allow-list."""
    check_moe_layout_support(
        [make_fake_experts(is_transposed=False)],
        model_type=model_type,
        operation="Test MoE quantization",
    )


def test_check_support_rejects_missing_capability():
    with pytest.raises(MoeSupportError, match="is_transposed") as exc_info:
        check_moe_layout_support([make_fake_experts()], model_type="qwen3_moe", operation="Test MoE quantization")
    assert "moe=False" in str(exc_info.value)


@pytest.mark.parametrize("bad_value", [None, 0, 1, "False", "", [], torch.tensor(False)])
def test_check_support_rejects_non_bool_is_transposed(bad_value):
    """Non-boolean values (e.g. a stray ``None``/tensor/string) must not be treated as falsy-safe."""
    with pytest.raises(MoeSupportError, match="is_transposed") as exc_info:
        check_moe_layout_support(
            [make_fake_experts(is_transposed=bad_value)], model_type="qwen3_moe", operation="Test MoE quantization"
        )
    assert "moe=False" in str(exc_info.value)


def test_check_support_rejects_transposed_layout():
    with pytest.raises(MoeSupportError, match=r"\(E, K, OUT\)") as exc_info:
        check_moe_layout_support(
            [make_fake_experts(is_transposed=True)],
            model_type="gpt_oss",
            operation="Test MoE quantization",
        )
    message = str(exc_info.value)
    assert "moe=False" in message
    assert "gpt_oss" in message


def test_check_support_exempts_module_list_experts_with_no_direct_3d_param():
    """Classic per-expert ``nn.ModuleList`` experts are exempt from the layout check.

    E.g. PhiMoE/Mixtral on transformers releases without the fused-experts refactor are
    unconditionally K-last (each child is a plain 2D ``nn.Linear``), so they must not be
    rejected for a missing ``is_transposed`` attribute.
    """
    module_list = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
    check_moe_layout_support([module_list], model_type="any_model_type", operation="Test MoE quantization")


def test_check_support_rejects_module_list_with_direct_3d_param():
    """A ``ModuleList`` that also owns a direct 3D parameter is not exempt.

    Olive's quantizer selection would group that parameter's last dimension directly, so it
    needs the same ``is_transposed`` guarantee as any other fused-experts module -- and a
    bare ``ModuleList`` has no such attribute to trust.
    """
    module_list = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    module_list.fused_weight = torch.nn.Parameter(torch.zeros(2, 4, 4))
    with pytest.raises(MoeSupportError, match="is_transposed"):
        check_moe_layout_support([module_list], model_type="any_model_type", operation="Test MoE quantization")


def test_check_support_checks_every_module_in_a_mixed_list():
    """A single transposed module among otherwise-safe ones still triggers rejection."""
    safe = make_fake_experts("SafeExperts", is_transposed=False)
    unsafe = make_fake_experts("UnsafeExperts", is_transposed=True)
    with pytest.raises(MoeSupportError, match="UnsafeExperts"):
        check_moe_layout_support([safe, unsafe], model_type="mixtral", operation="Test MoE quantization")
