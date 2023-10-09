# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Union

import numpy as np
import torch


class DecoderModel(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        hidden_size: int,
        n_heads: int,
        scale_type: str,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        use_biases = False
        interleaved = True
        self.tok_embeddings = torch.nn.Linear(hidden_size, vocab_size, bias=False, device=device)

        self.norm = RMSNorm(hidden_size, eps=1e-5)

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerLayer(
                hidden_size,
                n_heads,
                scale_type,
                device=device,
                use_biases=use_biases,
                interleaved=interleaved,
            )
            self.layers.append(layer)

        self.output = torch.nn.Linear(hidden_size, vocab_size, bias=False, device=device)

        self.hidden_size = hidden_size
        self.n_heads = n_heads

    def get_input_embeddings(self) -> torch.Tensor:
        assert self.tok_embeddings is not None
        return self.tok_embeddings.weight

    def forward_use_cache(self, x_increment, attn_mask, cos, sin, cache):
        use_cache = True
        k_caches = []
        v_caches = []

        # For the cache model, we always work on the last element
        pos_end = attn_mask.size(2)
        pos = pos_end - 1

        sliced_cos = cos[:, pos:pos_end, :, :]
        sliced_sin = sin[:, pos:pos_end, :, :]

        for layer_idx, layer in enumerate(self.layers):
            k_cache = cache[layer_idx]["key"].clone().detach()
            v_cache = cache[layer_idx]["value"].clone().detach()

            x_increment, k_cache, v_cache = layer(
                use_cache, x_increment, attn_mask, sliced_cos, sliced_sin, k_cache, v_cache, pos
            )

            k_cache = torch.roll(k_cache, shifts=-1, dims=2)
            v_cache = torch.roll(v_cache, shifts=-1, dims=2)
            k_caches.append(k_cache)
            v_caches.append(v_cache)

        x_increment = self.norm(x_increment)

        logits = self.output(x_increment[:, -1, :])

        cos = torch.roll(cos, shifts=-1, dims=1)
        sin = torch.roll(sin, shifts=-1, dims=1)

        # For the increment iterations, we only ever use the last row of the mask. This row should look something
        # like this: [-10000, -10000, -10000, -10000, 0, 0, 0, 0]. By rollowing it to the left, it becomes
        # [-10000, -10000, -10000, 0, 0, 0, 0, -10000] and then all we have to do is set the last element to 0
        # so it becomes [-10000, -10000, -10000, 0, 0, 0, 0, 0].
        attn_mask = torch.roll(attn_mask, shifts=-1, dims=2)
        attn_mask[:, -1, -1] = 0.0

        return_values = [logits, attn_mask, cos, sin]

        for k_cache, v_cache in zip(k_caches, v_caches):
            return_values.append(k_cache)
            return_values.append(v_cache)

        return tuple(return_values)

    def forward_no_cache(self, x, attn_mask, cache):
        use_cache = False
        k_caches = []
        v_caches = []

        max_seq_len = attn_mask.size(1)
        cos, sin = rotary_mat(self.hidden_size, self.n_heads, max_seq_len, head_scale=1.0)

        seq_len = x.size(1)
        next_seq_len = seq_len + 1
        remaining_seq_len = max_seq_len - next_seq_len

        sliced_cos = cos[:, :seq_len, :, :]
        sliced_sin = sin[:, :seq_len, :, :]

        for layer_idx, layer in enumerate(self.layers):
            k_cache = cache[layer_idx]["key"].clone().detach()
            v_cache = cache[layer_idx]["value"].clone().detach()

            x, k_cache, v_cache = layer(use_cache, x, attn_mask, sliced_cos, sliced_sin, k_cache, v_cache, 0)

            # torch.roll doesn't support a tensor as the shifts argument, so we manually slice and concat instead
            k_cache_parts = torch.split(k_cache, [next_seq_len, remaining_seq_len], dim=2)
            k_cache = torch.cat([k_cache_parts[1], k_cache_parts[0]], dim=2)

            v_cache_parts = torch.split(v_cache, [next_seq_len, remaining_seq_len], dim=2)
            v_cache = torch.cat([v_cache_parts[1], v_cache_parts[0]], dim=2)

            k_caches.append(k_cache)
            v_caches.append(v_cache)

        x = self.norm(x)
        logits = self.output(x[:, -1, :])

        cos_parts = torch.split(cos, [next_seq_len, remaining_seq_len], dim=1)
        cos = torch.cat([cos_parts[1], cos_parts[0]], dim=1)

        sin_parts = torch.split(sin, [next_seq_len, remaining_seq_len], dim=1)
        sin = torch.cat([sin_parts[1], sin_parts[0]], dim=1)

        attn_mask_top = attn_mask[:, :-1, :]
        unpadded_attn_mask = -10000.0 * torch.ones([attn_mask.shape[0], 1, attn_mask.shape[2] - next_seq_len])
        attn_mask = torch.nn.functional.pad(unpadded_attn_mask, (0, next_seq_len))
        attn_mask = torch.cat([attn_mask_top, attn_mask], dim=1)

        return_values = [logits, attn_mask, cos, sin]

        for k_cache, v_cache in zip(k_caches, v_caches):
            return_values.append(k_cache)
            return_values.append(v_cache)

        return tuple(return_values)

    def set_use_cache(self, use_cache):
        self.forward = self.forward_use_cache if use_cache else self.forward_no_cache


def rotary_mat(
    hidden_size: int,
    n_heads: int,
    max_seq_len: int,
    theta: float = 10000.0,
    head_scale=1.0,
    device=torch.device("cpu"),
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = head_scale * hidden_size / n_heads

    pos = torch.arange(0, 2 * (head_dim // 2), step=2, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (pos / head_dim))

    idx = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(idx.to(dtype), freqs)

    cos = torch.reshape(torch.cos(freqs), [1, max_seq_len, 1, -1])
    sin = torch.reshape(torch.sin(freqs), [1, max_seq_len, 1, -1])
    dtype = torch.get_default_dtype()

    return cos.to(dtype), sin.to(dtype)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        scale_type: str,
        device: Union[torch.device, None] = None,
        use_biases: bool = True,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        # these should have variable eps.
        self.attention_norm = RMSNorm(hidden_size, eps=1e-5)
        self.ffn_norm = RMSNorm(hidden_size, eps=1e-5)

        self.attention = SelfAttention(
            hidden_size,
            n_heads,
            scale_type,
            device=device,
            use_biases=use_biases,
            interleaved=interleaved,
        )
        proj_dim = hidden_size * 4
        proj_dim = int(2 * proj_dim / 3)
        proj_dim = 256 * ((proj_dim + 256 - 1) // 256)
        self.feed_forward = ProjLayerSiluMatMul(hidden_size, proj_dim, device=device)

    def forward(
        self,
        use_cache: bool,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        sliced_cos: torch.Tensor,
        sliced_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Dimension of x is [batch_size, seq_len, hidden_size] Dimension of
        # k_cache and v_cache is [batch_size, n_layers, pos, n_heads, head_dim]
        h, k_out, v_out = self.attention(
            use_cache, self.attention_norm(x), attn_mask, sliced_cos, sliced_sin, k_cache, v_cache, pos
        )

        h = x + h
        return h + self.feed_forward(self.ffn_norm(h)), k_out, v_out


class UpdateCache(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, use_cache, k_cache, v_cache, key, value, pos, pos_end):
        if use_cache:
            k_cache[:, 0, -1, :, :] = key
            v_cache[:, 0, -1, :, :] = value
        else:
            k_cache[:, 0, pos:pos_end, :, :] = key
            v_cache[:, 0, pos:pos_end, :, :] = value

        return k_cache, v_cache


class ApplyMask(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, score, attn_mask, pos, pos_end):
        score = score + attn_mask[:, pos:pos_end, :pos_end]
        return score


class RotateTensor(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        sliced_cos: torch.Tensor,
        sliced_sin: torch.Tensor,
        interleaved: bool = False,
    ) -> torch.Tensor:
        rot_dim = 2 * sliced_cos.shape[3]

        x_rot = x[:, :, :, :rot_dim]

        if interleaved:
            x1 = x_rot[:, :, :, 0::2]
            x2 = x_rot[:, :, :, 1::2]
        else:
            half = x_rot.shape[-1] // 2
            whole = 2 * half
            x1 = x[:, :, :, 0:half]
            x2 = x[:, :, :, half:whole]

        real = sliced_cos * x1 - sliced_sin * x2
        imag = sliced_sin * x1 + sliced_cos * x2

        if interleaved:
            x_rot[:, :, :, 0::2] = real
            x_rot[:, :, :, 1::2] = imag
        else:
            x_rot = torch.cat((real, imag), dim=-1)

        return x_rot


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        scale_type: str,
        device: Union[torch.device, None] = None,
        use_biases: bool = True,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.wq = torch.nn.Linear(hidden_size, hidden_size, bias=use_biases, device=device)
        self.wk = torch.nn.Linear(hidden_size, hidden_size, bias=use_biases, device=device)
        self.wv = torch.nn.Linear(hidden_size, hidden_size, bias=use_biases, device=device)
        self.wo = torch.nn.Linear(hidden_size, hidden_size, bias=use_biases, device=device)
        self.update_cache = UpdateCache()
        self.apply_mask = ApplyMask()
        self.rotate_tensor = RotateTensor()

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = int(hidden_size / n_heads)

        if scale_type == "HeadDim":
            self.scale = self.head_dim
        elif scale_type == "SquareRootHeadDim":
            self.scale = np.sqrt(self.head_dim)
        else:
            raise ValueError(f"Unknown scale type {scale_type}")

        self.interleaved = interleaved

    def forward(
        self,
        use_cache: bool,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        sliced_cos: torch.Tensor,
        sliced_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Dimension of x is [batch_size, seq_len, hidden_size]
        # Dimension of attn_mask is [batch_size, max_seq_len, max_seq_len]
        # Dimension of k_cache and v_cache is
        #   [batch_size, n_layers, pos, n_heads, head_dim]
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Split the attention heads
        query = torch.reshape(query, [batch_size, seq_len, self.n_heads, self.head_dim])
        key = torch.reshape(key, [batch_size, seq_len, self.n_heads, self.head_dim])
        value = torch.reshape(value, [batch_size, seq_len, self.n_heads, self.head_dim])

        # Apply rotary positional embedding
        query = self.rotate_tensor(query, sliced_cos, sliced_sin, self.interleaved)
        key = self.rotate_tensor(key, sliced_cos, sliced_sin, self.interleaved)

        # Append new entries to the end of k, v cache
        pos_end = pos + seq_len
        k_cache, v_cache = self.update_cache(use_cache, k_cache, v_cache, key, value, pos, pos_end)

        if use_cache:
            key = torch.reshape(k_cache, [batch_size, pos_end, self.n_heads, self.head_dim])
            value = torch.reshape(v_cache, [batch_size, pos_end, self.n_heads, self.head_dim])
        else:
            key = k_cache[:, 0, :pos_end, :, :]
            value = v_cache[:, 0, :pos_end, :, :]

        query = query.permute([0, 2, 1, 3]).reshape([batch_size * self.n_heads, seq_len, self.head_dim])
        key = key.permute([0, 2, 3, 1]).reshape([batch_size * self.n_heads, self.head_dim, pos_end])
        value = value.permute([0, 2, 1, 3]).reshape([batch_size * self.n_heads, pos_end, self.head_dim])

        # Calculate attention scores
        score = torch.matmul(query, key) / self.scale

        # Dimension of score is [n_heads, seq_len, pos + seq_len]
        # score = score + attn_mask[:, pos:pos_end, :pos_end]
        score = self.apply_mask(score, attn_mask, pos, pos_end)

        # Calculate attention values
        prob = torch.nn.functional.softmax(score, dim=-1)
        attn = torch.matmul(prob, value)

        # Merge attention heads
        attn = attn.reshape(batch_size, self.n_heads, seq_len, self.head_dim)
        attn = attn.permute([0, 2, 1, 3]).reshape([batch_size, seq_len, self.hidden_size])

        return self.wo(attn), k_cache, v_cache


class ProjLayerSiluMatMul(torch.nn.Module):
    def __init__(
        self,
        in_feature_size: int,
        hidden_feature_size: int,
        device: Union[torch.device, None] = None,
    ) -> None:
        super().__init__()
        self.hidden_feature_size = hidden_feature_size
        self.in_feature_size = in_feature_size

        self.w1 = torch.nn.Linear(in_feature_size, hidden_feature_size, bias=False, device=device)
        self.w2 = torch.nn.Linear(hidden_feature_size, in_feature_size, bias=False, device=device)
        self.w3 = torch.nn.Linear(in_feature_size, hidden_feature_size, bias=False, device=device)

    def forward(self, x):
        w1x = self.w1(x)

        return self.w2(w1x * torch.sigmoid(w1x) * self.w3(x))
