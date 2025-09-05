# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from olive.constants import PrecisionBits
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.selective_mixed_precision import SelectiveMixedPrecision


@pytest.fixture(name="input_model", scope="module")
def input_model_fixture(tmp_path_factory):
    save_path = tmp_path_factory.mktemp("selective-mixed-precision-test")
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=16,
            intermediate_size=64,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=32000,
        )
    )
    model.save_pretrained(save_path)
    return HfModelHandler(save_path)


@pytest.mark.parametrize(
    ("algorithm", "expected_layer_indices", "include_qkv"),
    [
        ("k_quant_down", [0, 3, 6, 7], False),  # first 1/8, every 3rd, and last 1/8
        ("k_quant_mixed", [0, 3, 6, 7], True),
        ("k_quant_last", [], False),
    ],
)
def test_selective_mixed_precision(algorithm, expected_layer_indices, include_qkv, input_model, tmp_path):
    """Test SelectiveMixedPrecision pass with different algorithms."""
    config = {"algorithm": algorithm}
    p = create_pass_from_dict(SelectiveMixedPrecision, config, disable_search=True)

    output_model = p.run(input_model, str(tmp_path))

    # Check that mixed_precision_info was added
    assert "mixed_precision_info" in output_model.model_attributes
    expected_mp_info = {
        "default": {"bits": PrecisionBits.BITS4},
        "overrides": {
            "lm_head": {"bits": 8},
        },
    }
    for idx in expected_layer_indices:
        expected_mp_info["overrides"].update(
            {
                **(
                    {
                        f"model.layers.{idx}.self_attn.q_proj": {"bits": PrecisionBits.BITS8},
                        f"model.layers.{idx}.self_attn.k_proj": {"bits": PrecisionBits.BITS8},
                        f"model.layers.{idx}.self_attn.v_proj": {"bits": PrecisionBits.BITS8},
                    }
                    if include_qkv
                    else {}
                ),
                f"model.layers.{idx}.mlp.down_proj": {"bits": PrecisionBits.BITS8},
            }
        )
    assert output_model.model_attributes["mixed_precision_info"] == expected_mp_info
