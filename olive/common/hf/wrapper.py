# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from olive.common.hf.io_config.io_resolver import resolve_alias
from olive.common.utils import find_first_matched_value, get_attr, replace_submodules, set_attr

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# ruff: noqa: RUF012


# TODO(jambayk): consider always returning the name of the submodule
def get_submodules(
    module: nn.Module,
    mapping: dict,
    key: str,
    return_name: bool = False,
    return_name_prefix: str = "",
    fail_on_not_found: bool = True,
):
    names = mapping.get(key, mapping["default"])

    if isinstance(names, str):
        submodules = get_attr(module, names, fail_on_not_found=fail_on_not_found)
        names = f"{return_name_prefix}{names}"
    else:
        submodules = [get_attr(module, name, fail_on_not_found=fail_on_not_found) for name in names]
        names = [f"{return_name_prefix}{name}" for name in names]

    return submodules if not return_name else (submodules, names)


class UnpackedQKV(nn.Module):
    """Unpacks the QKV projection matrix into separate projections."""

    def __init__(self, qkv: nn.Module, num_attn_heads: int, num_key_value_heads: int, head_dim: int):
        super().__init__()
        q_size = num_attn_heads * head_dim
        kv_size = num_key_value_heads * head_dim

        def create_proj(start, end):
            proj = nn.Linear(q_size, end - start)
            proj.weight = nn.Parameter(qkv.weight[start:end], requires_grad=qkv.weight.requires_grad)
            proj.bias = (
                None if qkv.bias is None else nn.Parameter(qkv.bias[start:end], requires_grad=qkv.bias.requires_grad)
            )
            return proj

        self.q_proj = create_proj(0, q_size)
        self.k_proj = create_proj(q_size, q_size + kv_size)
        self.v_proj = create_proj(q_size + kv_size, q_size + 2 * kv_size)

    def forward(self, hidden_states):
        return torch.cat([self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)], dim=-1)

    @torch.no_grad()
    def create_packed(self) -> nn.Linear:
        """Repacks the QKV projections into a single projection matrix."""
        qkv = nn.Linear(
            self.q_proj.in_features + self.k_proj.in_features + self.v_proj.in_features,
            self.q_proj.out_features,
        )
        qkv.weight = nn.Parameter(
            torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0),
            requires_grad=self.q_proj.weight.requires_grad,
        )
        qkv.bias = (
            None
            if self.q_proj.bias is None
            else nn.Parameter(
                torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                requires_grad=self.q_proj.bias.requires_grad,
            )
        )
        return qkv


class LayerWrapper:
    """Wrapper for transformer layer block."""

    FIRST_LAYER_NORM = {
        "default": "input_layernorm",
        "gpt2": "ln_1",
        "lfm2": "operator_norm",
        "opt": "self_attn_layer_norm",
        "qwen": "ln_1",
    }
    SECOND_LAYER_NORM = {
        "default": "post_attention_layernorm",
        "gemma2": "pre_feedforward_layernorm",
        "gpt2": "ln_2",
        "lfm2": "ffn_norm",
        "opt": "final_layer_norm",
        "qwen": "ln_2",
    }
    ATTENTION = {"default": "self_attn", "bloom": "self_attention", "gpt2": "attn", "qwen": "attn"}
    ATTENTION_INPUTS = {
        "default": ["q_proj", "k_proj", "v_proj"],
        "bloom": ["query_key_value"],
        "gpt2": ["c_attn"],
        "phi3": ["qkv_proj"],
        "qwen": ["c_attn"],
    }
    ATTENTION_OUTPUTS = {
        "default": ["o_proj"],
        "bloom": ["dense"],
        "gpt2": ["c_proj"],
        "lfm2": ["out_proj"],
        "opt": ["out_proj"],
        "qwen": ["c_proj"],
    }
    # ``granitemoe``/``jamba`` keep their (MoE or dense) feed-forward block under a
    # non-``mlp`` attribute:
    #   * ``GraniteMoeDecoderLayer.block_sparse_moe = GraniteMoeMoE(config)``
    #   * ``Jamba{Attention,Mamba}DecoderLayer.feed_forward = JambaSparseMoeBlock(...)``
    #     or ``JambaMLP(...)`` -- Jamba interleaves MoE and dense layers, so the dense
    #     ones simply resolve to a block without ``.experts``/``.router`` and
    #     ``get_experts()``/``get_router()`` return ``None``.
    MLP = {
        "default": "mlp",
        "granitemoe": "block_sparse_moe",
        "jamba": "feed_forward",
        "lfm2": "feed_forward",
        "opt": "",
    }
    MLP_INPUTS = {
        "default": ["gate_proj", "up_proj"],
        "bloom": ["dense_h_to_4h"],
        "gpt2": ["c_fc"],
        "lfm2": ["w1", "w3"],
        "opt": ["fc1"],
        "phi3": ["gate_up_proj"],
        "qwen": ["w1", "w2"],
    }
    MLP_OUTPUTS = {
        "default": ["down_proj"],
        "bloom": ["dense_4h_to_h"],
        "gpt2": ["c_proj"],
        "lfm2": ["w2"],
        "opt": ["fc2"],
        "qwen": ["c_proj"],
    }
    # MoE-block conventions. These are resolved relative to ``self.mlp``
    # (i.e., the layer's MLP attribute) because every modern HF MoE
    # transformer block lives at ``layer.mlp``.
    #
    # ``EXPERTS`` is the experts sub-module:
    #   - For fused-3D MoEs (Mixtral, Qwen3-MoE, GPT-OSS) it owns 3D
    #     ``nn.Parameter`` tensors such as ``gate_up_proj`` of shape
    #     ``(num_experts, ...)``.
    #   - For ``ModuleList(Expert)`` MoEs (PhiMoE, DeepSeek-V3, classic
    #     Mixtral) it is the ``ModuleList`` whose children are the
    #     per-expert ``nn.Module`` blocks (each with their own
    #     ``nn.Linear``s).
    #
    # ``ROUTER`` is the routing module ("gate" in most, "router" in
    # GPT-OSS). It is usually an ``nn.Linear`` (or a small custom module
    # containing one) and should typically be kept in full precision.
    EXPERTS = {
        "default": "experts",
    }
    #
    # Router attribute names verified against transformers 5.14.1:
    #   * ``gate``   -- qwen2_moe, qwen3_moe, mixtral, deepseek_v3, olmoe (the default)
    #   * ``router`` -- gpt_oss, phimoe, granitemoe, jamba. Note that Jamba's router is a
    #     bare ``nn.Linear`` (not a wrapped router module), so it would otherwise be swept
    #     into the ordinary 2D quantization walk; ``iter_quant_targets`` excludes every
    #     resolved router of an MoE layer by identity.
    ROUTER = {
        "default": "gate",
        "gpt_oss": "router",
        "granitemoe": "router",
        "jamba": "router",
        "phimoe": "router",
    }
    # ``MAMBA`` is a decoder layer's state-space-model (SSM) sub-module, when present --
    # e.g. Jamba interleaves ``JambaMambaDecoderLayer`` (has ``.mamba``) with
    # ``JambaAttentionDecoderLayer`` (has ``.self_attn`` instead). A Mamba block's
    # ``nn.Linear`` projections (``in_proj``/``x_proj``/``dt_proj``/``out_proj``) feed a
    # state-space recursion rather than a plain matmul, so they were never intentionally
    # supported by the generic 2D quantization walk -- resolved (when present) purely so
    # ``iter_quant_targets`` can exclude them, the same way it excludes routers.
    MAMBA = {
        "default": "mamba",
    }

    def __init__(self, layer: nn.Module, model_type: str):
        # TODO(jambayk): use _layer and property to get the layer?
        self.layer = layer
        self.model_type = model_type

        # Use fail_on_not_found=False to support hybrid architectures (e.g., Qwen3.5, LFM2)
        # where some layers lack standard self-attention (linear attention or conv layers)
        self.attn, self.attn_name = get_submodules(
            layer, self.ATTENTION, self.model_type, return_name=True, fail_on_not_found=False
        )
        self.mlp, self.mlp_name = get_submodules(layer, self.MLP, self.model_type, return_name=True)

    def get_first_layer_norm(self, return_name: bool = True):
        return get_submodules(self.layer, self.FIRST_LAYER_NORM, self.model_type, return_name=return_name)

    def get_second_layer_norm(self, return_name: bool = True):
        return get_submodules(self.layer, self.SECOND_LAYER_NORM, self.model_type, return_name=return_name)

    def get_attention_inputs(self, return_name: bool = True, partial_ok: bool = False):
        """Return the attention input projections of this layer.

        Args:
            return_name: Whether to also return the resolved module names.
            partial_ok: When ``False`` (the default) every projection named in
                ``ATTENTION_INPUTS`` must exist, otherwise an ``AttributeError`` is raised.
                This keeps the returned list positional, which callers such as
                ``olive.passes.pytorch.rotate`` rely on (they identify ``v_proj`` by index).
                When ``True`` missing projections are dropped from the returned list, which
                is only correct for callers that treat the list as an unordered set --
                e.g. QKV-group normalization, which needs to tolerate architectures with a
                non-QKV attention (DeepSeek-V3's MLA exposes ``q_proj`` /
                ``kv_a_proj_with_mqa`` / ``kv_b_proj``, so ``k_proj``/``v_proj`` are absent).
                Those extra projections are still quantized by the generic ``nn.Linear``
                walk, and QKV normalization is a no-op for a group of fewer than two members.

        """
        if self.attn is None:
            return ([], []) if return_name else []
        attention_inputs, names = get_submodules(
            self.attn,
            self.ATTENTION_INPUTS,
            self.model_type,
            return_name=True,
            return_name_prefix=f"{self.attn_name}.",
            fail_on_not_found=not partial_ok,
        )
        if partial_ok:
            keep = [i for i, module in enumerate(attention_inputs) if module is not None]
            attention_inputs = [attention_inputs[i] for i in keep]
            names = [names[i] for i in keep]
        if attention_inputs and isinstance(attention_inputs[0], UnpackedQKV):
            names = [f"{names[0]}.{part}" for part in ["q_proj", "k_proj", "v_proj"]]
            attention_inputs = [attention_inputs[0].q_proj, attention_inputs[0].k_proj, attention_inputs[0].v_proj]
        return attention_inputs if not return_name else (attention_inputs, names)

    def get_attention_outputs(self, return_name: bool = True):
        if self.attn is None:
            return ([], []) if return_name else []
        return get_submodules(
            self.attn,
            self.ATTENTION_OUTPUTS,
            self.model_type,
            return_name=return_name,
            return_name_prefix=f"{self.attn_name}.",
        )

    def get_mlp_inputs(self, return_name: bool = True, partial_ok: bool = False):
        return self._get_mlp_projections(self.MLP_INPUTS, return_name, partial_ok)

    def get_mlp_outputs(self, return_name: bool = True, partial_ok: bool = False):
        return self._get_mlp_projections(self.MLP_OUTPUTS, return_name, partial_ok)

    def _get_mlp_projections(self, mapping: dict, return_name: bool, partial_ok: bool):
        """Resolve the MLP projections named in ``mapping``.

        By default every mapped projection must exist. ``partial_ok=True`` drops missing
        projections for MoE-aware callers that intentionally handle layers without a single
        dense MLP path.
        """
        modules, names = get_submodules(
            self.mlp,
            mapping,
            self.model_type,
            return_name=True,
            return_name_prefix=f"{self.mlp_name}.",
            fail_on_not_found=not partial_ok,
        )
        if partial_ok:
            keep = [i for i, module in enumerate(modules) if module is not None]
            modules = [modules[i] for i in keep]
            names = [names[i] for i in keep]
        return modules if not return_name else (modules, names)

    def get_experts(self, return_name: bool = True):
        """Return the experts sub-module of this layer (or ``None`` if not MoE).

        The experts sub-module is the parent of every per-expert weight
        (fused 3D ``nn.Parameter``s, or a ``ModuleList`` of per-expert
        ``nn.Module``s). The caller can use the returned module to (a)
        collect ids of every ``nn.Module`` under the experts subtree
        (for the ``moe=False`` skip set) and (b) iterate the
        ``nn.Parameter``s to quantize when ``moe=True``.

        Returns ``None`` (and an empty name when ``return_name=True``)
        for layers without an experts sub-module.
        """
        if self.mlp is None:
            return (None, "") if return_name else None
        module = get_submodules(self.mlp, self.EXPERTS, self.model_type, return_name=False, fail_on_not_found=False)
        if module is None:
            return (None, "") if return_name else None
        name = f"{self.mlp_name}.{self.EXPERTS.get(self.model_type, self.EXPERTS['default'])}"
        return (module, name) if return_name else module

    def get_router(self, return_name: bool = True):
        """Return the router sub-module of this layer (or ``None`` if not MoE).

        Routers are typically small modules (e.g., a single
        ``nn.Linear``) and should usually be kept in full precision.
        Olive does not quantize routers automatically — this accessor is
        used by callers that want to skip the router via
        ``modules_to_not_convert``.
        """
        if self.mlp is None:
            return (None, "") if return_name else None
        module = get_submodules(self.mlp, self.ROUTER, self.model_type, return_name=False, fail_on_not_found=False)
        if module is None:
            return (None, "") if return_name else None
        name = f"{self.mlp_name}.{self.ROUTER.get(self.model_type, self.ROUTER['default'])}"
        return (module, name) if return_name else module

    def get_mamba(self, return_name: bool = True):
        """Return this layer's Mamba/SSM sub-module (or ``None`` for a non-Mamba layer).

        Resolved relative to ``self.layer`` (not ``self.mlp``, unlike ``EXPERTS``/``ROUTER``):
        a Mamba block is a sibling of the MLP/attention block, not nested inside either.
        Used by ``iter_quant_targets`` to exclude the block's projections from the generic
        2D quantization walk -- see ``MAMBA``'s docstring for why.
        """
        module = get_submodules(self.layer, self.MAMBA, self.model_type, return_name=False, fail_on_not_found=False)
        if module is None:
            return (None, "") if return_name else None
        name = self.MAMBA.get(self.model_type, self.MAMBA["default"])
        return (module, name) if return_name else module


class ModelWrapper:
    """Wrapper for transformer model."""

    # Model-type specific mappings (cannot use aliases)
    MAX_LENGTH = {
        "default": "max_position_embeddings",
        "gpt2": "n_positions",
        "gptj": "n_positions",
        "qwen": "seq_length",
    }
    EMBEDDINGS = {
        "default": ["model.embed_tokens"],
        "bloom": ["transformer.word_embeddings", "transformer.word_embeddings_layernorm"],
        "falcon": ["transformer.word_embeddings"],
        "gpt2": ["transformer.wte", "transformer.wpe"],
        "gpt_neox": ["gpt_neox.embed_in"],
        "gptj": ["transformer.wte"],
        "opt": ["model.decoder.embed_tokens", "model.decoder.embed_positions"],
        "qwen": ["transformer.wte"],
        "qwen3_vl_text": ["embed_tokens"],
    }
    # in newer transformers versions, there is one rotary embedding per model
    ROTARY_EMBEDDING = {
        "default": "model.rotary_emb",
        "falcon": "transformer.rotary_emb",
        "gpt_neox": "gpt_neox.rotary_emb",
        "qwen": "transformer.rotary_emb",
        "qwen3_vl_text": "rotary_emb",
    }
    LM_HEAD = {"default": "lm_head"}
    PRE_HEAD_LAYERNORM = {
        "default": "model.norm",
        "gpt2": "transformer.ln_f",
        "lfm2": "model.embedding_norm",
        "qwen": "transformer.ln_f",
        "qwen3_vl_text": "norm",
    }
    LAYERS = {
        "default": "model.layers",
        "bloom": "transformer.h",
        "falcon": "transformer.h",
        "gpt2": "transformer.h",
        "gpt_neox": "gpt_neox.layers",
        "gptj": "transformer.h",
        "opt": "model.decoder.layers",
        "qwen": "transformer.h",
        "qwen3_vl_text": "layers",
    }

    def __init__(self, config: Union[PretrainedConfig, dict]):
        self.config = config if isinstance(config, PretrainedConfig) else PretrainedConfig.from_dict(config)
        self.model_type = getattr(self.config, "model_type", None)

        # model attributes (using unified aliases from defaults.yaml)
        self.hidden_size = resolve_alias(self.config, "hidden_size")
        self.num_attention_heads = resolve_alias(self.config, "num_attention_heads")
        self.num_key_value_heads = resolve_alias(self.config, "num_kv_heads") or self.num_attention_heads
        self.head_dim = resolve_alias(self.config, "head_dim") or self.hidden_size // self.num_attention_heads
        self.num_hidden_layers = resolve_alias(self.config, "num_layers")
        # MAX_LENGTH uses model_type-based mapping, not aliases
        self.max_length = find_first_matched_value(self.config, self.MAX_LENGTH)

        self._model = None
        self._layer_wrappers = None
        self.olive_root_model: Optional[nn.Module] = None
        self.olive_component_path: Optional[str] = None
        self.olive_component_role: Optional[str] = None

    @property
    def model(self) -> "PreTrainedModel":
        if self._model is None:
            raise ValueError("Model is not set. Please set the model using set_model method.")

        return self._model

    def set_model(self, model: "PreTrainedModel", initialize_layer_wrappers: bool = True):
        self._model = model
        self._layer_wrappers = (
            [LayerWrapper(layer, self.model_type) for layer in self.get_layers(False)]
            if initialize_layer_wrappers
            else []
        )

    def get_embeds(self, return_name: bool = True):
        return get_submodules(self.model, self.EMBEDDINGS, self.model_type, return_name=return_name)

    def get_rotary_embed(self, return_name: bool = True):
        return get_submodules(
            self.model, self.ROTARY_EMBEDDING, self.model_type, return_name=return_name, fail_on_not_found=False
        )

    def get_lm_head(self, return_name: bool = True):
        return get_submodules(self.model, self.LM_HEAD, self.model_type, return_name=return_name)

    def get_pre_head_layernorm(self, return_name: bool = True):
        return get_submodules(self.model, self.PRE_HEAD_LAYERNORM, self.model_type, return_name=return_name)

    def get_layers(self, return_name: bool = True):
        return get_submodules(self.model, self.LAYERS, self.model_type, return_name=return_name)

    def get_layer_wrappers(self):
        if self._layer_wrappers is None:
            raise ValueError("Layer wrappers are not set. Please set the model using set_model method.")

        return self._layer_wrappers

    def maybe_untie_word_embeddings(self):
        """Untie the word embeddings if they are tied."""
        if getattr(self.config, "tie_word_embeddings", False):
            self.config.tie_word_embeddings = False
            self.model.config.tie_word_embeddings = False

            self.get_lm_head(False).weight = nn.Parameter(self.get_embeds(False)[0].weight.clone().detach())
            logger.debug("Untied word embeddings.")

    def maybe_unpack_qkv(self):
        """Unpack the QKV projection matrix into separate projections for models like phi3."""
        for layer_wrapper in self.get_layer_wrappers():
            if layer_wrapper.attn is None:
                continue
            attn_inputs, attn_input_names = layer_wrapper.get_attention_inputs()

            if len(attn_inputs) != 1 or not isinstance(attn_inputs[0], nn.Linear):
                return

            set_attr(
                layer_wrapper.layer,
                attn_input_names[0],
                UnpackedQKV(
                    attn_inputs[0],
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                ),
            )

    def save_model(self, output_model_path: str, replacements: list[tuple[nn.Module, Callable]] = None):
        """Save the model to the output_model_path with the specified replacements.

        :param output_model_path: Path to save the model.
        :param replacements: List of replacements to apply before saving the model. Each replacement is a tuple of
            (submodule_type, replacement_fn). The replacement_fn should take the submodule as input and return the
            replacement module.
        """
        replacements = replacements or []
        # unpack qkv before saving
        replacements.append([UnpackedQKV, lambda module: module.create_packed()])

        for submodule_type, replacement_fn in replacements:
            logger.debug("Replacing %s with %s", submodule_type, replacement_fn)
            replace_submodules(self.model, submodule_type, replacement_fn)

        self.model.save_pretrained(output_model_path)

    @classmethod
    def from_model(cls, model: "PreTrainedModel") -> "ModelWrapper":
        model_wrapper = cls(model.config)
        model_wrapper.set_model(model)
        return model_wrapper
