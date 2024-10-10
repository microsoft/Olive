# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Automatically distribute HF Llama model using Tensor Parallelism
# --------------------------------------------------------------------------

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from packaging import version

from olive.passes.pytorch.tensor_parallel import TensorParallel
from olive.passes.pytorch.tensor_parallel_layers import TensorParallelColumnLinear, TensorParallelRowLinear

if TYPE_CHECKING:
    import torch

# pylint: disable=not-callable

logger = logging.getLogger(__name__)


# Tensor Parallel layers. This layers will replace corresponding layer in the model.
def tp_llama_mlp_init(self, config):
    from transformers.activations import ACT2FN
    from transformers.models.llama.modeling_llama import LlamaMLP

    super(LlamaMLP, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.world_size = config.world_size if hasattr(config, "world_size") else 1

    # Original
    # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.gate_proj = TensorParallelColumnLinear(
        self.hidden_size, self.intermediate_size, bias=False, world_size=self.world_size
    )
    self.up_proj = TensorParallelColumnLinear(
        self.hidden_size, self.intermediate_size, bias=False, world_size=self.world_size
    )
    self.down_proj = TensorParallelRowLinear(
        self.intermediate_size, self.hidden_size, bias=False, world_size=self.world_size
    )

    self.act_fn = ACT2FN[config.hidden_act]


def tp_llama_attention_init(self, config, layer_idx: Optional[int] = None):
    from transformers.models.llama.modeling_llama import LlamaAttention

    super(LlamaAttention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
        logger.warning_once(
            f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
            "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
            "when creating this class."
        )

    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True
    self.world_size = config.world_size if hasattr(config, "world_size") else 1

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
            f" and `num_heads`: {self.num_heads})."
        )

    # Original
    # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
    # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
    # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
    # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    self.q_proj = TensorParallelColumnLinear(
        self.hidden_size, self.num_heads * self.head_dim, bias=False, world_size=self.world_size
    )
    self.k_proj = TensorParallelColumnLinear(
        self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, world_size=self.world_size
    )
    self.v_proj = TensorParallelColumnLinear(
        self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, world_size=self.world_size
    )
    self.o_proj = TensorParallelRowLinear(
        self.num_heads * self.head_dim, self.hidden_size, bias=False, world_size=self.world_size
    )

    self._init_rope()  # pylint: disable=protected-access


# Overwrite original functions
def tp_llama_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    from transformers.models.llama.modeling_llama import rotate_half

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def tp_llama_attention_forward(
    self,
    hidden_states: "torch.Tensor",
    attention_mask: Optional["torch.Tensor"] = None,
    position_ids: Optional["torch.LongTensor"] = None,
    past_key_value: Optional[Tuple["torch.Tensor"]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[Tuple["torch.Tensor"]]]:
    import torch
    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import repeat_kv

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])  # pylint: disable=not-callable
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])  # pylint: disable=not-callable
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])  # pylint: disable=not-callable
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads // self.world_size, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads // self.world_size, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads // self.world_size, self.head_dim).transpose(
        1, 2
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len = kv_seq_len + past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = tp_llama_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads // self.world_size, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads // self.world_size, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(  # pylint: disable=not-callable
        query_states.dtype
    )
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads // self.world_size, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads // self.world_size, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    # Original
    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(
            F.linear(attn_output[i], o_proj_slices[i])  # pylint: disable=not-callable
            for i in range(self.config.pretraining_tp)
        )
    else:
        attn_output = self.o_proj(attn_output)

    attn_output = attn_output.contiguous()
    attn_weights = attn_weights.contiguous()

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Adapted from LlamaSdpaAttention.forward
def tp_llama_sdpa_attention_forward(
    self,
    hidden_states: "torch.Tensor",
    attention_mask: Optional["torch.Tensor"] = None,
    position_ids: Optional["torch.LongTensor"] = None,
    past_key_value: Optional["Cache"] = None,  # noqa: F821
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[Tuple["torch.Tensor"]]]:
    import torch
    from transformers.models.llama.modeling_llama import repeat_kv

    if output_attentions:
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
            "support `output_attentions=True`. Falling back to the manual attention implementation, "
            "but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
            'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads // self.world_size, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads // self.world_size, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads // self.world_size, self.head_dim).transpose(
        1, 2
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len = kv_seq_len + past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = tp_llama_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None and attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        raise ValueError(
            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        )

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal
        # mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2)
    # Original
    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output).contiguous()

    return attn_output, None, past_key_value


def tp_llama_attention_parallel_split(self, world_size):
    self.num_heads = self.num_heads // world_size
    self.num_key_value_heads = self.num_key_value_heads // world_size


def replace_llama2_tensor_parallel_layers():
    import torch
    from transformers import __version__ as tf_version

    if version.parse(tf_version) >= version.parse("4.38"):
        raise ImportError("Llama Tensor Parallelism is not supported for Transformers version >= 4.38.0.")

    from transformers.models import llama

    originals = {
        "mlp_init": llama.modeling_llama.LlamaMLP.__init__,
        "attention_init": llama.modeling_llama.LlamaAttention.__init__,
        "attention_forward": llama.modeling_llama.LlamaAttention.forward,
        "parallel_split": None,
        "sdpa_attention_forward": llama.modeling_llama.LlamaSdpaAttention.forward,
        "apply_rotary_pos_emb": llama.modeling_llama.apply_rotary_pos_emb,
        "uniform_": torch.nn.init.uniform_,
        "kaiming_normal_": torch.nn.init.kaiming_normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
    }

    llama.modeling_llama.LlamaMLP.__init__ = tp_llama_mlp_init
    llama.modeling_llama.LlamaAttention.__init__ = tp_llama_attention_init
    llama.modeling_llama.LlamaAttention.forward = tp_llama_attention_forward
    llama.modeling_llama.LlamaAttention.parallel_split = tp_llama_attention_parallel_split
    llama.modeling_llama.LlamaSdpaAttention.forward = tp_llama_sdpa_attention_forward
    llama.modeling_llama.apply_rotary_pos_emb = tp_llama_apply_rotary_pos_emb

    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x

    return originals


def restore_llama2_tensor_parallel_layers(originals: Dict[str, Any]):
    import torch
    from transformers.models import llama

    llama.modeling_llama.LlamaMLP.__init__ = originals["mlp_init"]
    llama.modeling_llama.LlamaAttention.__init__ = originals["attention_init"]
    llama.modeling_llama.LlamaAttention.forward = originals["attention_forward"]
    llama.modeling_llama.LlamaAttention.parallel_split = originals["parallel_split"]
    llama.modeling_llama.LlamaSdpaAttention.forward = originals["sdpa_attention_forward"]
    llama.modeling_llama.apply_rotary_pos_emb = originals["apply_rotary_pos_emb"]

    torch.nn.init.uniform_ = originals["uniform_"]
    torch.nn.init.kaiming_normal_ = originals["kaiming_normal_"]
    torch.nn.init.kaiming_uniform_ = originals["kaiming_uniform_"]


class LlamaPyTorchTensorParallel(TensorParallel):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.originals = {}

    def replace_layers(self):
        self.originals = replace_llama2_tensor_parallel_layers()

    def restore_layers(self):
        restore_llama2_tensor_parallel_layers(self.originals)

    def split_weights(self, model: "torch.nn.Module"):
        from transformers.models.llama.modeling_llama import LlamaAttention

        def _split_weights(m):
            if isinstance(m, LlamaAttention):
                m.parallel_split(self.world_size)

            for mm in m._modules.values():  # pylint: disable=protected-access
                _split_weights(mm)

        _split_weights(model)

    def load_rank_weights(self, model: "torch.nn.Module"):
        def _load_rank_weights(m):
            if isinstance(m, (TensorParallelColumnLinear, TensorParallelRowLinear)):
                m.load_rank_weights(self.rank, self.world_size)

            for mm in m._modules.values():  # pylint: disable=protected-access
                _load_rank_weights(mm)

        _load_rank_weights(model)
