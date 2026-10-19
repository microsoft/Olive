# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from olive.data.component.dataset import TransformersDummyDataset


@pytest.mark.parametrize("num_key_value_heads", [2, 4])
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("head_dim", [None, 12])
@pytest.mark.parametrize("use_fp16", [False, True])
@pytest.mark.parametrize("past_seq_len", [0, 3])
def test_get_past_kv_inputs_preserves_head_dimension(num_key_value_heads, world_size, head_dim, use_fp16, past_seq_len):
    model_attributes = {
        "model_type": "llama",
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": num_key_value_heads,
        "num_hidden_layers": 2,
    }
    if head_dim is not None:
        model_attributes["head_dim"] = head_dim
    dataset = TransformersDummyDataset("unused", seq_len=1, past_seq_len=3, max_seq_len=8)

    past_key_values = dataset.get_past_kv_inputs(model_attributes, past_seq_len, use_fp16, world_size)

    expected_shape = (num_key_value_heads // world_size, past_seq_len, head_dim or 8)
    expected_dtype = torch.float16 if use_fp16 else torch.float32
    assert len(past_key_values) == 2
    for key, value in past_key_values:
        assert key.shape == value.shape == expected_shape
        assert key.dtype == value.dtype == expected_dtype


@pytest.mark.parametrize(("num_key_value_heads", "world_size"), [(None, 1), (None, 2), (1, 1)])
@pytest.mark.parametrize("use_fp16", [False, True])
@pytest.mark.parametrize("past_seq_len", [0, 3])
def test_get_past_kv_inputs_supports_default_and_single_kv_head(
    num_key_value_heads, world_size, use_fp16, past_seq_len
):
    model_attributes = {
        "model_type": "llama",
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
    }
    if num_key_value_heads is not None:
        model_attributes["num_key_value_heads"] = num_key_value_heads
    dataset = TransformersDummyDataset("unused", seq_len=1, past_seq_len=past_seq_len, max_seq_len=8)

    past_key_values = dataset.get_past_kv_inputs(model_attributes, past_seq_len, use_fp16, world_size)

    expected_shape = ((num_key_value_heads or 4) // world_size, past_seq_len, 8)
    expected_dtype = torch.float16 if use_fp16 else torch.float32
    assert len(past_key_values) == 2
    for key, value in past_key_values:
        assert key.shape == value.shape == expected_shape
        assert key.dtype == value.dtype == expected_dtype


def test_get_past_kv_inputs_can_extend_local_llama_cache_with_explicit_head_dimension():
    config = LlamaConfig.from_dict(
        {
            "vocab_size": 16,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 12,
        }
    )
    dataset = TransformersDummyDataset("unused", seq_len=1, past_seq_len=3, max_seq_len=8)
    past_key_values = dataset.get_past_kv_inputs(config.to_dict(), 3, use_fp16=False)
    cache = DynamicCache(config=config)
    for layer_index, (key, value) in enumerate(past_key_values):
        cache.update(key.unsqueeze(0), value.unsqueeze(0), layer_index)
    model = LlamaForCausalLM(config).eval()

    with torch.no_grad():
        output = model(input_ids=torch.tensor([[1]]), past_key_values=cache, use_cache=True)

    assert output.logits.shape == (1, 1, 16)
    assert output.past_key_values.get_seq_length() == 4
