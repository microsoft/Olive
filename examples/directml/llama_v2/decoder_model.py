# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import torch

# pylint: disable=abstract-method, not-callable, unsubscriptable-object


class DecoderModel(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        hidden_size: int,
        n_heads: int,
        scale_type: str,
        normalization_type: str,
    ) -> None:
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(vocab_size, hidden_size)

        self.norm = {
            "layer_norm": LayerNorm(hidden_size, eps=1e-5),
            "rms": RMSNorm(hidden_size, eps=1e-5),
        }[normalization_type]

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerLayer(
                hidden_size,
                n_heads,
                scale_type,
                normalization_type,
            )
            self.layers.append(layer)

        self.output = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        self.hidden_size = hidden_size
        self.n_heads = n_heads

    def forward_common(self, use_cache, tokens, position_ids, attn_mask, past_key_values):
        k_caches = []
        v_caches = []

        x = self.tok_embeddings(tokens)

        for layer_idx, layer in enumerate(self.layers):
            k_cache = past_key_values[layer_idx]["key"].clone().detach()
            v_cache = past_key_values[layer_idx]["value"].clone().detach()

            x, k_cache, v_cache = layer(use_cache, x, position_ids, attn_mask, k_cache, v_cache)

            k_caches.append(k_cache)
            v_caches.append(v_cache)

        x = self.norm(x)
        logits = self.output(x[:, -1, :])

        # For the increment iterations, we only ever use the last row of the mask. This row should look something like
        # this: [0, 0, 0, 0, 1, 1, 1, 1]. By rolling it to the left, it becomes [0, 0, 0, 1, 1, 1, 1, 0] and then all
        # we have to do is set the last element to 1 so it becomes [0, 0, 0, 1, 1, 1, 1, 1].
        attn_mask = torch.nn.functional.pad(attn_mask[:, 1:], (0, 1, 0, 0), value=1)
        return_values = [logits, attn_mask]

        for k_cache, v_cache in zip(k_caches, v_caches):
            return_values.append(k_cache)
            return_values.append(v_cache)

        return tuple(return_values)

    def forward_no_cache(self, tokens, position_ids, attn_mask, past_key_values):
        use_cache = False
        return self.forward_common(use_cache, tokens, position_ids, attn_mask, past_key_values)

    def forward_use_cache(self, tokens_increment, position_ids_increment, attn_mask, past_key_values):
        use_cache = True
        return self.forward_common(use_cache, tokens_increment, position_ids_increment, attn_mask, past_key_values)

    def set_use_cache(self, use_cache):
        self.forward = self.forward_use_cache if use_cache else self.forward_no_cache


def rotary_mat(
    hidden_size: int,
    n_heads: int,
    max_seq_len: int,
    theta: float = 10000.0,
    head_scale=1.0,
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = head_scale * hidden_size / n_heads

    pos = torch.arange(0, head_dim, step=2, dtype=dtype)
    freqs = 1.0 / (theta ** (pos / head_dim))

    idx = torch.arange(max_seq_len)
    freqs = torch.outer(idx.to(dtype), freqs)
    freqs = torch.cat((freqs, freqs), dim=-1)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    dtype = torch.get_default_dtype()

    return cos.to(dtype), sin.to(dtype)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.zeros(dim)

    def forward(self, hidden_states):
        diff = hidden_states - hidden_states.mean(-1, keepdim=True)
        variance = diff.pow(2).mean(-1, keepdim=True)
        hidden_states = diff / torch.sqrt(variance + self.eps)
        return self.weight * hidden_states + self.bias


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        scale_type: str,
        normalization_type: str,
    ) -> None:
        super().__init__()
        self.attention_norm = {
            "layer_norm": LayerNorm(hidden_size, eps=1e-6),
            "rms": RMSNorm(hidden_size, eps=1e-6),
        }[normalization_type]

        self.ffn_norm = {
            "layer_norm": LayerNorm(hidden_size, eps=1e-6),
            "rms": RMSNorm(hidden_size, eps=1e-6),
        }[normalization_type]

        self.cos, self.sin = rotary_mat(hidden_size, n_heads, 4096, head_scale=1.0)

        self.attention = SelfAttention(
            hidden_size,
            n_heads,
            scale_type,
        )
        proj_dim = hidden_size * 4
        proj_dim = int(2 * proj_dim / 3)
        proj_dim = 256 * ((proj_dim + 256 - 1) // 256)
        self.feed_forward = ProjLayerSiluMatMul(hidden_size, proj_dim)

    def forward(
        self,
        use_cache: bool,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Dimension of x is [batch_size, seq_len, hidden_size] Dimension of
        # k_cache and v_cache is [batch_size, n_layers, pos, n_heads, head_dim]
        h, k_out, v_out = self.attention(
            use_cache, self.attention_norm(x), position_ids, attn_mask, self.cos, self.sin, k_cache, v_cache
        )

        h = x + h
        return h + self.feed_forward(self.ffn_norm(h)), k_out, v_out


class ApplyMask(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, use_cache, score, attn_mask, seq_len, num_heads, dtype=torch.float32):
        # The mask contains 1's for values that should stay intact, and 0's for values that should get added to -10000
        expanded_mask = attn_mask[:, None, None, :].expand(-1, 1, seq_len, -1).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        mask_score = inverted_mask.masked_fill(inverted_mask.to(torch.bool), -10000.0)

        if not use_cache:
            batch_size, max_seq_len = attn_mask.size()
            causal_mask = torch.tril(torch.ones((batch_size, 1, max_seq_len, max_seq_len)))
            causal_mask = causal_mask[:, :, -seq_len:, :]
            inverted_causal_mask = 1.0 - causal_mask
            mask_score += inverted_causal_mask.masked_fill(inverted_causal_mask.to(torch.bool), -10000.0)

        mask_score = mask_score.expand(-1, num_heads, -1, -1)

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
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
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


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        scale_type: str,
    ) -> None:
        super().__init__()
        self.wq = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.apply_mask = ApplyMask()
        self.rotary_embedding = RotaryEmbedding()

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = int(hidden_size / n_heads)

        if scale_type == "HeadDim":
            self.scale = self.head_dim
        elif scale_type == "SquareRootHeadDim":
            self.scale = np.sqrt(self.head_dim)
        else:
            raise ValueError(f"Unknown scale type {scale_type}")

    def forward(
        self,
        use_cache: bool,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Split the attention heads
        query = torch.reshape(query, [batch_size, seq_len, self.n_heads, self.head_dim]).transpose(1, 2)
        key = torch.reshape(key, [batch_size, seq_len, self.n_heads, self.head_dim]).transpose(1, 2)
        value = torch.reshape(value, [batch_size, seq_len, self.n_heads, self.head_dim]).transpose(1, 2)

        # Apply rotary positional embedding
        query = self.rotary_embedding(query, cos, sin, position_ids)
        key = self.rotary_embedding(key, cos, sin, position_ids)

        # Append new entries to the end of k, v cache
        k_cache = k_cache[:, :, seq_len:, :]
        v_cache = v_cache[:, :, seq_len:, :]
        k_cache = torch.cat((k_cache, key), dim=2)
        v_cache = torch.cat((v_cache, value), dim=2)

        key = k_cache
        value = v_cache

        key = key.permute([0, 1, 3, 2])

        # Calculate attention scores
        score = torch.matmul(query, key) / self.scale

        # Apply the mask
        score = self.apply_mask(use_cache, score, attn_mask, seq_len, self.n_heads)

        # Calculate attention values
        prob = torch.nn.functional.softmax(score, dim=-1)
        attn = torch.matmul(prob, value)

        # Merge attention heads
        attn = attn.permute([0, 2, 1, 3]).reshape([batch_size, seq_len, self.hidden_size])

        return self.wo(attn), k_cache, v_cache


class ProjLayerSiluMatMul(torch.nn.Module):
    def __init__(
        self,
        in_feature_size: int,
        hidden_feature_size: int,
    ) -> None:
        super().__init__()
        self.hidden_feature_size = hidden_feature_size
        self.in_feature_size = in_feature_size

        self.w1 = torch.nn.Linear(in_feature_size, hidden_feature_size, bias=False)
        self.w2 = torch.nn.Linear(hidden_feature_size, in_feature_size, bias=False)
        self.w3 = torch.nn.Linear(in_feature_size, hidden_feature_size, bias=False)

    def forward(self, x):
        w1x = self.w1(x)

        return self.w2(w1x * torch.sigmoid(w1x) * self.w3(x))
