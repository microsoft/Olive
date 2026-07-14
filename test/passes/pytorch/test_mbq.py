# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access,unexpected-keyword-arg
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.mbq import Mbq
from olive.passes.pytorch.rtn import Rtn


def _make_tiny_qwen(tmp_path):
    model = Qwen2ForCausalLM(
        Qwen2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=32,
        )
    )
    model.save_pretrained(tmp_path)
    return HfModelHandler(tmp_path)


def _calibration_batch():
    input_ids = torch.tensor([[1, 4, 5, 7, 8, 2]])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.tensor([[-100, -100, -100, -100, 8, 2]]),
        "vision_mask": torch.tensor([[False, True, True, False, False, False]]),
    }


def test_apply_scale_preserves_norm_linear_output():
    torch.manual_seed(0)
    norm = torch.nn.RMSNorm(8)
    linears = (torch.nn.Linear(8, 4), torch.nn.Linear(8, 6))
    inputs = torch.randn(2, 3, 8)
    reference = tuple(linear(norm(inputs)) for linear in linears)
    scale = torch.rand(8) + 0.5

    Mbq._apply_scale(norm, linears, scale)

    for expected, linear in zip(reference, linears):
        assert torch.allclose(linear(norm(inputs)), expected, atol=1e-5, rtol=1e-5)


def test_apply_scale_preserves_linear_linear_output():
    torch.manual_seed(0)
    previous = torch.nn.Linear(8, 12)
    following = torch.nn.Linear(12, 8)
    inputs = torch.randn(2, 3, 8)
    reference = following(previous(inputs))
    scale = torch.rand(12) + 0.5

    Mbq._apply_scale(previous, (following,), scale)

    assert torch.allclose(following(previous(inputs)), reference, atol=1e-5, rtol=1e-5)


def test_mbq_checkpoint_reload_and_matching_rtn(tmp_path):
    input_model = _make_tiny_qwen(tmp_path / "input")
    original = input_model.load_model()
    batch = _calibration_batch()
    with torch.no_grad():
        original_logits = original(input_ids=batch["input_ids"]).logits

    mbq = create_pass_from_dict(
        Mbq,
        {
            "bits": 4,
            "group_size": 16,
            "sym": False,
            "n_grid": 4,
            "data_config": {"name": "unused"},
            "device": "cpu",
            "save_processor": False,
        },
        disable_search=True,
    )
    with patch("olive.passes.pytorch.mbq.get_calibration_dataset", return_value=[batch]):
        mbq_output = mbq.run(input_model, str(tmp_path / "mbq"))

    reloaded = mbq_output.load_model()
    with torch.no_grad():
        reloaded_logits = reloaded(input_ids=batch["input_ids"]).logits
    assert torch.allclose(reloaded_logits, original_logits, atol=2e-5, rtol=2e-5)
    assert reloaded.config.mbq_config["bits"] == 4

    rtn = create_pass_from_dict(
        Rtn,
        {"bits": 4, "group_size": 16, "sym": False},
        disable_search=True,
    )
    rtn_output = rtn.run(mbq_output, str(tmp_path / "rtn"))
    assert isinstance(rtn_output, HfModelHandler)


def test_mbq_rejects_mismatched_downstream_quantization(tmp_path):
    input_model = _make_tiny_qwen(tmp_path / "input")
    input_model.model_attributes["mbq_config"] = {
        "bits": 4,
        "group_size": 16,
        "symmetric": False,
    }
    model = input_model.load_model()
    model.config.mbq_config = input_model.model_attributes["mbq_config"]
    model.save_pretrained(tmp_path / "mbq")
    mbq_model = HfModelHandler(tmp_path / "mbq")

    rtn = create_pass_from_dict(
        Rtn,
        {"bits": 8, "group_size": 16, "sym": False},
        disable_search=True,
    )
    with pytest.raises(ValueError, match="must match"):
        rtn.run(mbq_model, str(tmp_path / "rtn"))
