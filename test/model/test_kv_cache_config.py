# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.model.config.kv_cache_config import KVCacheConfig


@pytest.mark.parametrize("num_hidden_layers", [16, 32])
@pytest.mark.parametrize("shared_kv", [True, False])
@pytest.mark.parametrize("sequence_length_idx", [1, 2])
def test_kv_cache_dynamic_axes(num_hidden_layers, shared_kv, sequence_length_idx):
    config = KVCacheConfig(
        world_size=1,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=32,
        hidden_size=2560,
        past_sequence_length=128,
        batch_size=2,
        dtype="float32",
        shared_kv=shared_kv,
        sequence_length_idx=sequence_length_idx,
    )
    dynamic_axes = config.get_dynamic_axes()
    # 2 * 2 * num_hidden_layers because we have past and present key/values
    assert len(dynamic_axes) == 2 * 2 * num_hidden_layers

    past_sequence_length_name = "past_sequence_length" if not shared_kv else "max_sequence_length"
    present_sequence_length_name = "past_sequence_length + sequence_length" if not shared_kv else "max_sequence_length"
    assert dynamic_axes["past_key_values.0.key"] == {
        "0": "batch_size",
        str(sequence_length_idx): past_sequence_length_name,
    }
    assert dynamic_axes["past_key_values.0.value"] == {
        "0": "batch_size",
        str(sequence_length_idx): past_sequence_length_name,
    }
    assert dynamic_axes["present.0.key"] == {"0": "batch_size", str(sequence_length_idx): present_sequence_length_name}
    assert dynamic_axes["present.0.value"] == {
        "0": "batch_size",
        str(sequence_length_idx): present_sequence_length_name,
    }


@pytest.mark.parametrize("num_hidden_layers", [16, 32])
def test_kv_cache_output_names(num_hidden_layers):
    config = KVCacheConfig(
        world_size=1,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=32,
        hidden_size=2560,
        past_sequence_length=128,
        batch_size=2,
        dtype="float32",
        shared_kv=True,
        sequence_length_idx=2,
    )
    output_names = config.get_output_names()
    for i in range(num_hidden_layers):
        # key and value come in pairs
        assert output_names[i * 2] == f"present.{i}.key"
        assert output_names[i * 2 + 1] == f"present.{i}.value"


@pytest.mark.parametrize("num_hidden_layers", [16, 32])
@pytest.mark.parametrize("num_attention_heads", [16, 32])
@pytest.mark.parametrize("hidden_size", [2560, 5120])
def test_kv_cache_input_names_shape_types(num_hidden_layers, num_attention_heads, hidden_size):
    config = KVCacheConfig(
        world_size=1,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        past_sequence_length=128,
        batch_size=2,
        dtype="float32",
        shared_kv=False,
        sequence_length_idx=2,
    )
    input_names, input_shape, input_types = config.get_input_names_shapes_types()
    for i in range(num_hidden_layers):
        assert input_names[i * 2] == f"past_key_values.{i}.key"
        assert input_names[i * 2 + 1] == f"past_key_values.{i}.value"
        assert input_shape[i * 2] == [2, num_attention_heads // 1, 128, hidden_size // num_attention_heads]
        assert input_shape[i * 2 + 1] == [
            2,
            num_attention_heads // 1,
            128,
            hidden_size // num_attention_heads,
        ]
        assert input_types[i * 2] == "float32"
        assert input_types[i * 2 + 1] == "float32"
