# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=unexpected-keyword-arg

from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.multimodal_activation_range import MultimodalActivationRangeCalibration


def test_multimodal_activation_range_records_component_metadata(tmp_path):
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
    model.save_pretrained(tmp_path / "input")
    input_model = HfModelHandler(tmp_path / "input")
    input_ids = torch.tensor([[1, 2, 3, 4]])
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    calibration = create_pass_from_dict(
        MultimodalActivationRangeCalibration,
        {
            "data_config": {"name": "unused"},
            "components": ["vision", "text"],
            "component_map": {"vision": ["model.layers.0"]},
            "device": "cpu",
        },
        disable_search=True,
    )
    with patch(
        "olive.passes.pytorch.multimodal_activation_range.get_calibration_dataset",
        return_value=[batch],
    ):
        output = calibration.run(input_model, str(tmp_path / "unused"))

    metadata = output.model_attributes["multimodal_activation_ranges"]
    assert metadata["schema_version"] == 1
    assert metadata["provenance"]["runtime_consumer"] is False
    assert metadata["component_rules"]["linear_module_counts"]["vision"] > 0
    assert metadata["component_rules"]["linear_module_counts"]["text"] > 0
    assert metadata["ranges"]["vision"]["numel"] > 0
    assert metadata["ranges"]["text"]["numel"] > 0


def test_multimodal_activation_range_skips_absent_default_components(tmp_path):
    model = Qwen2ForCausalLM(
        Qwen2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=32,
        )
    )
    model.save_pretrained(tmp_path / "input")
    input_model = HfModelHandler(tmp_path / "input")
    input_ids = torch.tensor([[1, 2, 3, 4]])
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    calibration = create_pass_from_dict(
        MultimodalActivationRangeCalibration,
        {"data_config": {"name": "unused"}, "device": "cpu"},
        disable_search=True,
    )
    with patch(
        "olive.passes.pytorch.multimodal_activation_range.get_calibration_dataset",
        return_value=[batch],
    ):
        output = calibration.run(input_model, str(tmp_path / "unused"))

    metadata = output.model_attributes["multimodal_activation_ranges"]
    assert metadata["component_rules"]["observed"] == ["text"]
    assert metadata["component_rules"]["missing"] == ["audio", "projector", "vision"]
    assert set(metadata["ranges"]) == {"text"}
