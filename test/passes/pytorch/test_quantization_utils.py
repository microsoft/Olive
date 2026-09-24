# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import torch

from olive.common.quant.tensor import QuantTensor
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.selective_mixed_precision import SelectiveMixedPrecision

DENSE_INT2_GROUP_SIZE = 16


@Registry.register_dataset("dense_int2_calibration_dataset")
def _dense_int2_calibration_dataset(seq_len: int, max_samples: int, vocab_size: int = 32):
    """Create deterministic, distinct token sequences for local GPTQ calibration."""
    generator = torch.Generator().manual_seed(0)
    return [
        (
            {
                "input_ids": torch.randint(1, vocab_size, (1, seq_len), generator=generator),
                "attention_mask": torch.ones((1, seq_len), dtype=torch.int64),
            },
            torch.tensor([0]),
        )
        for _ in range(max_samples)
    ]


def make_local_tiny_dense_llama(save_path: Path, *, tie_word_embeddings: bool = False) -> HfModelHandler:
    """Save a tiny dense Llama checkpoint and tokenizer without accessing the hub."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(0)
    save_path.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        **({"tie_word_embeddings": True} if tie_word_embeddings else {}),
    )
    LlamaForCausalLM(config).save_pretrained(save_path)

    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(config.vocab_size)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", pad_token="t0").save_pretrained(save_path)
    return HfModelHandler(model_path=str(save_path))


def tied_word_embedding_group(embedding_parameter: str = "model.embed_tokens.weight") -> dict:
    """Describe a tied token table owned by a separate embedding component."""
    return {
        "name": "word_embeddings",
        "kind": "tied_word_embeddings",
        "canonical": {
            "component": "embedding",
            "parameter": embedding_parameter,
        },
        "aliases": [{"component": "decoder", "parameter": "lm_head.weight"}],
    }


def make_local_calibration_data_config(seq_len: int = 16, max_samples: int = 4) -> DataConfig:
    """Return deterministic token calibration data generated entirely in-process."""
    return DataConfig(
        name="local_dense_calibration_data",
        type="DummyDataContainer",
        load_dataset_config=DataComponentConfig(
            type="dense_int2_calibration_dataset",
            params={
                "seq_len": seq_len,
                "max_samples": max_samples,
            },
        ),
        pre_process_data_config=DataComponentConfig(type="skip_pre_process"),
        post_process_data_config=DataComponentConfig(type="skip_post_process"),
    )


def load_quant_tensor_from_disk(
    model_dir: Path,
    pname: str,
    *,
    bits: int,
    symmetric: bool,
    group_size: int,
    shape: tuple[int, ...],
) -> QuantTensor:
    """Rebuild a QuantTensor directly from its saved safetensors buffers."""
    from safetensors.torch import load_file

    from olive.common.quant.state_dict import buffer_names

    qname, sname, zname = buffer_names(pname)
    weights = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        weights.update(load_file(shard))
    qweight, scales = weights[qname], weights[sname]
    return QuantTensor.from_packed(
        qweight=qweight,
        scales=scales,
        qzeros=weights.get(zname),
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        shape=shape,
        dtype=scales.dtype,
    )


def assert_packed_quant_module(
    module: torch.nn.Linear,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> QuantTensor:
    """Assert effective precision and lossless uint8 bit packing on a reloaded module."""
    from olive.common.quant.utils import pack_to_uint8, unpack_from_uint8

    quant_tensor = module._parameters["weight"]  # pylint: disable=protected-access
    assert isinstance(quant_tensor, QuantTensor)
    assert quant_tensor.bits == bits
    assert quant_tensor.group_size == group_size
    assert quant_tensor.symmetric is symmetric
    assert quant_tensor.is_placeholder is False
    assert quant_tensor.qweight.dtype == torch.uint8

    packing_factor = 8 // bits
    assert quant_tensor.shape[-1] % packing_factor == 0
    assert quant_tensor.qweight.shape == (*quant_tensor.shape[:-1], quant_tensor.shape[-1] // packing_factor)
    assert quant_tensor.qweight.numel() * packing_factor == quant_tensor.numel()

    unpacked = unpack_from_uint8(quant_tensor.qweight, bits, tuple(quant_tensor.shape))
    assert unpacked.dtype == torch.int32
    assert unpacked.min().item() >= 0
    assert unpacked.max().item() <= (1 << bits) - 1
    assert unpacked.unique().numel() > 1
    assert torch.equal(pack_to_uint8(unpacked, bits), quant_tensor.qweight)
    return quant_tensor


def assert_saved_quant_tensor_matches(
    output_path: Path,
    module_name: str,
    quant_tensor: QuantTensor,
) -> QuantTensor:
    """Assert a reloaded QuantTensor matches its serialized safetensors buffers."""
    disk_tensor = load_quant_tensor_from_disk(
        output_path,
        f"{module_name}.weight",
        bits=quant_tensor.bits,
        symmetric=quant_tensor.symmetric,
        group_size=quant_tensor.group_size,
        shape=tuple(quant_tensor.shape),
    )
    assert torch.equal(quant_tensor.qweight, disk_tensor.qweight)
    assert torch.equal(quant_tensor.scales, disk_tensor.scales)
    if quant_tensor.qzeros is None:
        assert disk_tensor.qzeros is None
    else:
        assert torch.equal(quant_tensor.qzeros, disk_tensor.qzeros)
    torch.testing.assert_close(quant_tensor.to_dense(), disk_tensor.to_dense(), rtol=0, atol=0)
    return disk_tensor


def assert_uniform_int2_checkpoint(loaded, output_path: Path) -> None:
    """Assert every decoder linear is a serialized symmetric INT2 tensor."""
    quantized_linears = {
        f"model.layers.0.{name}": module
        for name, module in loaded.model.layers[0].named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    assert quantized_linears
    for module_name, module in quantized_linears.items():
        quant_tensor = assert_packed_quant_module(
            module,
            bits=2,
            group_size=DENSE_INT2_GROUP_SIZE,
            symmetric=True,
        )
        assert loaded.config.quantization_config.get_qlinear_init_args(module_name) == {
            "bits": 2,
            "symmetric": True,
            "group_size": DENSE_INT2_GROUP_SIZE,
        }
        assert_saved_quant_tensor_matches(output_path, module_name, quant_tensor)


def plan_dense_int2_mixed_precision(input_model: HfModelHandler, output_path: Path) -> HfModelHandler:
    """Plan dense INT2 defaults with selected INT4 and explicit consumer INT8 coverage."""
    planner = create_pass_from_dict(
        SelectiveMixedPrecision,
        {"algorithm": "high_precision_mlp_down", "bits": 2, "high_bits": 4},
        disable_search=True,
    )
    planned = planner.run(input_model, str(output_path))
    assert planned.model_attributes["mixed_precision_info"]["default"] == {"bits": 2}
    assert planned.model_attributes["mixed_precision_info"]["overrides"]["model.layers.0.mlp.down_proj"] == {"bits": 4}
    return planned


def assert_dense_int2_mixed_precision_checkpoint(loaded, output_path: Path) -> None:
    """Assert INT2 defaults, selected INT4, and explicit INT8 survive save/reload."""
    expected_bits = {
        "model.layers.0.mlp.up_proj": 2,
        "model.layers.0.mlp.down_proj": 4,
        "model.layers.0.mlp.gate_proj": 8,
    }
    for module_name, bits in expected_bits.items():
        quant_tensor = assert_packed_quant_module(
            loaded.get_submodule(module_name),
            bits=bits,
            group_size=DENSE_INT2_GROUP_SIZE,
            symmetric=True,
        )
        assert loaded.config.quantization_config.get_qlinear_init_args(module_name) == {
            "bits": bits,
            "symmetric": True,
            "group_size": DENSE_INT2_GROUP_SIZE,
        }
        assert_saved_quant_tensor_matches(output_path, module_name, quant_tensor)

    # Keep the established QKV grouping behavior unchanged: absent a QKV-specific
    # override, all three projections consume the SMP INT2 default.
    for projection in ("q_proj", "k_proj", "v_proj"):
        module_name = f"model.layers.0.self_attn.{projection}"
        quant_tensor = assert_packed_quant_module(
            loaded.get_submodule(module_name),
            bits=2,
            group_size=DENSE_INT2_GROUP_SIZE,
            symmetric=True,
        )
        assert loaded.config.quantization_config.get_qlinear_init_args(module_name)["bits"] == 2
        assert_saved_quant_tensor_matches(output_path, module_name, quant_tensor)
