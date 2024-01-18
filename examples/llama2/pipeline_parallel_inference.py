# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
import platform
import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel, LlamaTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.getLogger(__name__)
prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?'


def llamamodel___init__(self, config: LlamaConfig):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

    super(LlamaModel, self).__init__(config)

    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.rank = config.rank if hasattr(config, "rank") else 0
    self.world_size = config.world_size if hasattr(config, "world_size") else 1

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList(
        [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers // self.world_size)]
    )
    self._use_sdpa = config._attn_implementation == "sdpa"
    self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
    self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()


def llamamodel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    if self.world_size > 1 and self.rank != 0:
        hidden_states = hidden_states.contiguous()
        dist.recv(tensor=hidden_states, src=self.rank - 1)

        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        pkv_size = (batch_size, num_heads, seq_length, head_dim)

        past_key_values = DynamicCache()
        for i in range(self.config.num_hidden_layers):
            k = torch.zeros(pkv_size, dtype=torch.float, device=hidden_states.device).contiguous()
            dist.recv(tensor=k, src=self.rank - 1)

            v = torch.zeros(pkv_size, dtype=torch.float, device=hidden_states.device).contiguous()
            dist.recv(tensor=v, src=self.rank - 1)

            past_key_values.update(k, v, i)

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    if self.world_size > 1 and self.rank != (self.world_size - 1):
        dist.send(tensor=hidden_states.contiguous(), dst=self.rank + 1)

        for k, v in past_key_values:
            dist.send(tensor=k.contiguous(), dst=self.rank + 1)
            dist.send(tensor=v.contiguous(), dst=self.rank + 1)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _run_one_rank(model_id, model_path, rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29506"

    if platform.system() == "Windows":
        backend = "gloo"
        device = torch.device("cpu")
    else:
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    LlamaModel.__init__ = llamamodel___init__
    LlamaModel.forward = llamamodel_forward

    config = LlamaConfig.from_pretrained(model_path, local_files_only=True)

    model = LlamaForCausalLM.from_pretrained(model_path, config=config, local_files_only=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    tokens = tokenizer(prompt, return_tensors="pt")

    position_ids = tokens.attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(tokens.attention_mask == 0, 1)

    inputs = {
        "input_ids": tokens.input_ids.to(device),
        "attention_mask": tokens.attention_mask.to(device),
        "position_ids": position_ids.to(device),
    }

    model(**inputs)
    return 0


def main():
    model_id = "meta-llama/Llama-2-7b-hf"

    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        model_path = f"models/pipeline_parallel/pipeline_parallel/gpu-cuda_model/model/model_{rank:02d}"
        p = mp.Process(target=_run_one_rank, args=(model_id, model_path, rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return 0


if __name__ == "__main__":
    sys.exit(main())
