# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math

import config
import numpy as np
import torch


class DecoderModel(torch.nn.Module):
    def __init__(self, use_embeddings: bool = False) -> None:
        super().__init__()
        self.use_embeddings = use_embeddings
        self.model = Model(use_embeddings)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=config.has_lm_head_bias)

    def forward_common(self, use_cache, tokens_or_embeddings, position_ids_increment, attention_mask, past_key_values):
        hidden_states, k_caches, v_caches = self.model(
            use_cache, tokens_or_embeddings, position_ids_increment, attention_mask, past_key_values
        )
        logits = self.lm_head(hidden_states)

        return_values = [logits]

        for k_cache, v_cache in zip(k_caches, v_caches):
            return_values.append(k_cache)
            return_values.append(v_cache)

        return return_values

    def forward(self, input_ids, position_ids, attention_mask, past_key_values):
        use_cache = False
        return self.forward_common(use_cache, input_ids, position_ids, attention_mask, past_key_values)

    def forward_no_cache_embeddings(self, embeddings, position_ids, attention_mask, past_key_values):
        use_cache = False
        return self.forward_common(use_cache, embeddings, position_ids, attention_mask, past_key_values)

    def get_embeddings(self):
        return self.model.embed_tokens


class Model(torch.nn.Module):
    def __init__(
        self,
        use_embeddings: bool,
    ) -> None:
        super().__init__()
        self.use_embeddings = use_embeddings
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)

        self.norm = {
            "layer_norm": torch.nn.LayerNorm(config.hidden_size, config.epsilon, bias=config.has_norm_bias),
            "rms": RMSNorm(config.hidden_size, config.epsilon),
        }[config.normalization_type]

        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_layers):
            layer = TransformerLayer()
            self.layers.append(layer)

    def forward(self, use_cache, tokens_or_embeddings, position_ids, attention_mask, past_key_values):
        k_caches = []
        v_caches = []

        x = tokens_or_embeddings if self.use_embeddings else self.embed_tokens(tokens_or_embeddings)

        if config.model_type == "gemma":
            x *= np.round(np.sqrt(config.hidden_size), decimals=2)

        for layer_idx, layer in enumerate(self.layers):
            k_cache = past_key_values[layer_idx]["key"].clone().detach()
            v_cache = past_key_values[layer_idx]["value"].clone().detach()

            x, k_cache, v_cache = layer(use_cache, x, position_ids, attention_mask, k_cache, v_cache)

            k_caches.append(k_cache)
            v_caches.append(v_cache)

        return self.norm(x), k_caches, v_caches


def rotary_mat(
    max_seq_len: int,
    theta: float = 10000.0,
    head_scale=1.0,
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_head_dim = head_scale * config.head_dim

    pos = torch.arange(0, scaled_head_dim, step=2, dtype=dtype)
    freqs = 1.0 / (theta ** (pos / scaled_head_dim))

    idx = torch.arange(max_seq_len)
    freqs = torch.outer(idx.to(dtype), freqs)
    freqs = torch.cat((freqs, freqs), dim=-1)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    dtype = torch.get_default_dtype()

    return cos.to(dtype), sin.to(dtype)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if config.model_type == "gemma":
            return (1.0 + self.weight) * hidden_states.to(input_dtype)

        return self.weight * hidden_states.to(input_dtype)


class TransformerLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input_layernorm = {
            "layer_norm": torch.nn.LayerNorm(config.hidden_size, config.epsilon, bias=config.has_input_layernorm_bias),
            "rms": RMSNorm(config.hidden_size, config.epsilon),
        }[config.normalization_type]

        if config.apply_residual_connection_post_layernorm:
            self.post_attention_layernorm = {
                "layer_norm": torch.nn.LayerNorm(
                    config.hidden_size, config.epsilon, bias=config.has_input_layernorm_bias
                ),
                "rms": RMSNorm(config.hidden_size, config.epsilon),
            }[config.normalization_type]

        self.cos, self.sin = rotary_mat(config.max_position_embeddings, head_scale=config.partial_rotary_factor)

        self.self_attn = SelfAttention()
        self.mlp = MLP()

    def forward(
        self,
        use_cache: bool,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Dimension of x is [batch_size, sequence_length, hidden_size] Dimension of
        # k_cache and v_cache is [batch_size, num_layers, pos, num_heads, head_dim]
        attn_norm_output = self.input_layernorm(x)
        h, k_out, v_out = self.self_attn(
            use_cache, attn_norm_output, position_ids, attention_mask, self.cos, self.sin, k_cache, v_cache
        )

        h = x + h

        if config.apply_residual_connection_post_layernorm:
            attn_norm_output = self.post_attention_layernorm(h)

        return h + self.mlp(attn_norm_output), k_out, v_out


class ApplyMask(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, use_cache, score, attention_mask, sequence_length, dtype=torch.float32):
        # The mask contains 1's for values that should stay intact, and 0's for values that should get added to -10000
        expanded_mask = attention_mask[:, None, None, :].expand(-1, 1, sequence_length, -1).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        mask_score = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(torch.float16).min)

        if not use_cache:
            batch_size, max_seq_len = attention_mask.size()
            causal_mask = torch.tril(
                torch.ones((batch_size, 1, sequence_length, max_seq_len)), diagonal=max_seq_len - sequence_length
            )
            inverted_causal_mask = 1.0 - causal_mask
            mask_score += inverted_causal_mask.masked_fill(
                inverted_causal_mask.to(torch.bool), torch.finfo(torch.float16).min
            )

        mask_score = mask_score.expand(-1, config.num_heads, -1, -1)

        return score + mask_score


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin, position_ids):
    head_dim = x.shape[-1]
    cos = cos[:, :head_dim]
    sin = sin[:, :head_dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, sequence_length, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, sequence_length, dim]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return apply_rope(x, cos, sin, position_ids)


def broadcast_key_value(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SelfAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=config.use_bias)
        self.k_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.use_bias
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.use_bias
        )
        self.o_proj = torch.nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=config.use_bias)
        self.apply_mask = ApplyMask()
        self.rotary_embedding = RotaryEmbedding()

    def forward(
        self,
        use_cache: bool,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        # Split the attention heads
        query = torch.reshape(query, [batch_size, sequence_length, config.num_heads, config.head_dim]).transpose(1, 2)
        key = torch.reshape(key, [batch_size, sequence_length, config.num_key_value_heads, config.head_dim]).transpose(
            1, 2
        )
        value = torch.reshape(
            value, [batch_size, sequence_length, config.num_key_value_heads, config.head_dim]
        ).transpose(1, 2)

        if config.partial_rotary_factor != 1.0:
            # Partial rotary embedding
            partial_dim = int(config.partial_rotary_factor * config.head_dim)

            query_rot, query_pass = (
                query[..., :partial_dim],
                query[..., partial_dim:],
            )
            key_rot, key_pass = (
                key[..., :partial_dim],
                key[..., partial_dim:],
            )

            query_rot = self.rotary_embedding(query_rot, cos, sin, position_ids)
            key_rot = self.rotary_embedding(key_rot, cos, sin, position_ids)

            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            # Apply rotary positional embedding
            query = self.rotary_embedding(query, cos, sin, position_ids)
            key = self.rotary_embedding(key, cos, sin, position_ids)

        # Append new entries to the end of k, v cache
        k_cache = torch.cat((k_cache, key), dim=2)
        v_cache = torch.cat((v_cache, value), dim=2)

        key = k_cache
        value = v_cache

        # Broadcast key and value from num_key_value_heads to match the query's num_heads
        if config.num_heads != config.num_key_value_heads:
            n_reps = config.num_heads // config.num_key_value_heads
            key = broadcast_key_value(key, n_reps)
            value = broadcast_key_value(value, n_reps)

        key = key.permute([0, 1, 3, 2])

        # Calculate attention scores
        score = torch.matmul(query, key) / np.sqrt(config.head_dim)

        # Apply the mask
        score = self.apply_mask(use_cache, score, attention_mask, sequence_length)

        # Calculate attention values
        prob = torch.nn.functional.softmax(score, dim=-1)
        attn = torch.matmul(prob, value)

        # Merge attention heads
        attn = attn.permute([0, 2, 1, 3]).reshape([batch_size, sequence_length, config.num_heads * config.head_dim])

        return self.o_proj(attn), k_cache, v_cache


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        if config.use_split_sigmoid:
            self.gate_proj = torch.nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=config.use_bias)
        else:
            self.gate_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)

        self.down_proj = torch.nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

        self.act = {
            "silu": SILUActivation(),
            "gelu_new": NewGELUActivation(),
            "gelu": torch.nn.GELU(),
        }[config.hidden_act]

        if config.has_up_proj:
            self.up_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)

    def forward(self, x):
        w1x = self.gate_proj(x)

        if config.has_up_proj:
            return self.down_proj(self.act(w1x) * self.up_proj(x))
        else:
            return self.down_proj(self.act(w1x))


class NewGELUActivation(torch.nn.Module):
    """Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).

    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SILUActivation(torch.nn.Module):
    def forward(self, x):
        if config.use_split_sigmoid:
            gate, y = torch.split(x, [config.intermediate_size, config.intermediate_size], dim=-1)
            return y * gate * torch.sigmoid(gate)
        else:
            return x * torch.sigmoid(x)
