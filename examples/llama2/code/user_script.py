# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from itertools import chain

import parallel_layers  # noqa: F401
import patching_llama  # noqa: F401
import torch
import torch.nn.init
from transformers import LlamaConfig, LlamaForCausalLM

torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x


def _load_pytorch_model(model_name):
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    config = LlamaConfig.from_pretrained(model_name)

    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=config.torch_dtype, config=config)
    model.parallel_model()
    model.to(torch.device(rank))
    model.eval()
    model.requires_grad_(False)
    # model.generate = torch.compile(model.generate, backend="inductor")
    return model


def get_decoder_io_config(model_name):
    config = LlamaConfig.from_pretrained(model_name)

    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(config.num_hidden_layers))),
    ]

    dynamic_axes = {}
    for name in input_names + output_names:
        if name in input_names:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + 1, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length + 1"}
        else:
            raise Exception("Unknown input or output name found")

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
    }


def get_decoder_component(model_name):
    return _load_pytorch_model(model_name)


def get_decoder_dummy_inputs(model):
    model = model.load_model()
    config = model.config
    device = model.device

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64, device=device
    )
    attn_mask = torch.randint(low=0, high=2, size=(batch_size, seq_len), dtype=torch.int64, device=device)
    # pos_ids is of shape (batch_size, seq_len)
    pos_ids = attn_mask.long().cumsum(-1) - 1
    pos_ids.masked_fill_(attn_mask == 0, 1)

    return (input_ids, attn_mask, pos_ids)


def get_decoder_with_past_io_config(model_name):
    config = LlamaConfig.from_pretrained(model_name)

    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *list(
            chain.from_iterable(
                (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(config.num_hidden_layers)
            )
        ),
    ]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(config.num_hidden_layers))),
    ]

    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"input_ids", "position_ids"}:
            # shape is (batch_size, 1)
            dynamic_axes[name] = {0: "batch_size"}
        elif name == "attention_mask":
            # shape is (batch_size, past_sequence_length + 1)
            dynamic_axes[name] = {0: "batch_size", 1: "past_sequence_length + 1"}
        elif "past" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, 1, vocab_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + 1, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length + 1"}
        else:
            raise Exception("Unknown input or output name found")

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
    }


def get_decoder_with_past_component(model_name):
    return _load_pytorch_model(model_name)


def get_decoder_with_past_dummy_inputs(model):
    from mpi4py import MPI

    model = model.load_model()
    config = model.config
    device = model.device
    world_size = MPI.COMM_WORLD.Get_size()

    batch_size, past_seq_len = 2, 8
    num_heads = config.num_key_value_heads // world_size
    head_size = config.hidden_size // config.num_attention_heads
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, 1), dtype=torch.int64, device=device)
    attn_mask = torch.randint(low=0, high=2, size=(batch_size, past_seq_len + 1), dtype=torch.int64, device=device)
    # pos_ids is of shape (batch_size, 1)
    pos_ids = attn_mask.long().cumsum(-1) - 1
    pos_ids.masked_fill_(attn_mask == 0, 1)
    pos_ids = pos_ids[:, -1].unsqueeze(-1)
    past_kv = [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=config.torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=config.torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]

    return (input_ids, attn_mask, pos_ids, past_kv)
