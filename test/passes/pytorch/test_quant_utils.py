# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BertConfig,
    BertForSequenceClassification,
    LlamaConfig,
    LlamaForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
)

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.nn import QuantEmbedding, QuantLinear
from olive.constants import PrecisionBits
from olive.model import HfModelHandler
from olive.passes.pytorch import quant_utils as quant_utils_module
from olive.passes.pytorch.quant_utils import (
    _quant_config_rank,
    finalize,
    normalize_qkv_quant_config,
    prepare_model,
    run_layerwise_quantization,
)
from test.utils import get_tiny_phi3

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(name="input_model", scope="module")
def input_model_fixture(tmp_path_factory):
    save_path = tmp_path_factory.mktemp("quant-utils-test")
    model = LlamaForCausalLM(
        LlamaConfig(  # pylint: disable=unexpected-keyword-arg
            hidden_size=16,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=32000,
        )
    )
    model.save_pretrained(save_path)
    return HfModelHandler(save_path)


def _baseline_pass_config(overrides=None, *, embeds=False):
    return SimpleNamespace(
        bits=PrecisionBits.BITS4,
        sym=False,
        group_size=16,
        lm_head=False,
        embeds=embeds,
        overrides=overrides,
    )


def _with_existing_quantization_config(monkeypatch, existing):
    """Patch ``HfModelHandler.get_hf_model_config`` to attach an existing quantization_config."""
    real = HfModelHandler.get_hf_model_config

    def fake(self, exclude_load_keys=None):
        cfg = real(self, exclude_load_keys=exclude_load_keys)
        cfg.quantization_config = existing
        return cfg

    monkeypatch.setattr(HfModelHandler, "get_hf_model_config", fake)


class _NestedDecoderRoot(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.vision = torch.nn.Linear(2, 2)
        self.config = decoder.config
        self.saved_state_keys = set()

    def save_pretrained(self, output_dir):
        self.saved_state_keys = set(self.state_dict())
        self.config.save_pretrained(output_dir)


def _make_nested_decoder_root(input_model):
    return _NestedDecoderRoot(LlamaForCausalLM.from_pretrained(input_model.model_path))


class _NestedBackboneRoot(torch.nn.Module):
    """VLM-like root where the text backbone and LM head are disjoint."""

    def __init__(self, causal_lm):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = causal_lm.model
        self.lm_head = causal_lm.lm_head
        self.vision = torch.nn.Linear(2, 2)
        self.config = causal_lm.config
        self.saved_state_keys = set()

    def save_pretrained(self, output_dir):
        self.saved_state_keys = set(self.state_dict())
        self.config.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# _quant_config_rank
# ---------------------------------------------------------------------------


def test_quant_config_rank_prefers_bits_then_smaller_positive_group_size():
    """Higher bits win; among equal bits, smaller positive group sizes win; per-tensor is worst."""
    symmetric_qargs = {"bits": PrecisionBits.BITS4, "group_size": 16, "symmetric": True}
    asymmetric_qargs = {"bits": PrecisionBits.BITS4, "group_size": 16, "symmetric": False}
    group_size_qargs = [
        {"bits": PrecisionBits.BITS4, "group_size": 128, "symmetric": True},
        {"bits": PrecisionBits.BITS4, "group_size": 32, "symmetric": True},
        {"bits": PrecisionBits.BITS4, "group_size": -1, "symmetric": True},
        {"bits": PrecisionBits.BITS4, "group_size": 0, "symmetric": True},
    ]
    higher_bit_qargs = {"bits": PrecisionBits.BITS8, "group_size": 128, "symmetric": True}

    assert _quant_config_rank(symmetric_qargs) == _quant_config_rank(asymmetric_qargs)
    assert max(group_size_qargs, key=_quant_config_rank) == group_size_qargs[1]
    assert max(group_size_qargs[2:], key=_quant_config_rank) == group_size_qargs[2]
    assert max([*group_size_qargs, higher_bit_qargs], key=_quant_config_rank) == higher_bit_qargs


# ---------------------------------------------------------------------------
# normalize_qkv_quant_config
# ---------------------------------------------------------------------------


def test_normalize_qkv_quant_config_does_not_rewrite_locked_overrides(input_model):
    """Locked QKV overrides preserved; others match the locked member.

    A locked (already-quantized) QKV member's override is preserved; new members are
    promoted/demoted to match it instead of the rank-based winner.
    """
    model = input_model.load_model()
    wrapper = ModelWrapper.from_model(model)
    qcfg = OliveHfQuantizationConfig(
        bits=PrecisionBits.BITS8,
        symmetric=True,
        group_size=16,
        overrides={
            # Locked: already physically quantized at 4-bit asymmetric.
            "model.layers.0.self_attn.q_proj": {
                "bits": PrecisionBits.BITS4,
                "symmetric": False,
                "group_size": 16,
            },
        },
    )
    locked = {"model.layers.0.self_attn.q_proj"}

    normalize_qkv_quant_config(wrapper, qcfg, locked_modules=locked)

    expected = {"bits": PrecisionBits.BITS4, "symmetric": False, "group_size": 16}
    for proj in ("q_proj", "k_proj", "v_proj"):
        assert qcfg.get_qlinear_init_args(f"model.layers.0.self_attn.{proj}") == expected


def test_normalize_qkv_quant_config_skips_group_with_conflicting_locked_members(input_model):
    """Conflicting locked members in a QKV group → skip with debug log."""
    model = input_model.load_model()
    wrapper = ModelWrapper.from_model(model)
    qcfg = OliveHfQuantizationConfig(
        bits=PrecisionBits.BITS8,
        symmetric=True,
        group_size=16,
        overrides={
            "model.layers.0.self_attn.q_proj": {
                "bits": PrecisionBits.BITS4,
                "symmetric": False,
                "group_size": 16,
            },
            "model.layers.0.self_attn.k_proj": {
                "bits": PrecisionBits.BITS8,
                "symmetric": True,
                "group_size": 32,
            },
        },
    )
    locked = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    }

    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler(level=logging.DEBUG)
    quant_utils_module.logger.addHandler(handler)
    quant_utils_module.logger.setLevel(logging.DEBUG)
    try:
        normalize_qkv_quant_config(wrapper, qcfg, locked_modules=locked)
    finally:
        quant_utils_module.logger.removeHandler(handler)

    # Locked overrides untouched.
    assert qcfg.get_qlinear_init_args("model.layers.0.self_attn.q_proj") == {
        "bits": PrecisionBits.BITS4,
        "symmetric": False,
        "group_size": 16,
    }
    assert qcfg.get_qlinear_init_args("model.layers.0.self_attn.k_proj") == {
        "bits": PrecisionBits.BITS8,
        "symmetric": True,
        "group_size": 32,
    }
    # V was not in the locked set; since the group is skipped, V keeps the base config.
    assert qcfg.get_qlinear_init_args("model.layers.0.self_attn.v_proj") == {
        "bits": PrecisionBits.BITS8,
        "symmetric": True,
        "group_size": 16,
    }
    assert any("conflicting configs" in rec.getMessage() for rec in records)


# ---------------------------------------------------------------------------
# prepare_model: basic / fresh-model cases
# ---------------------------------------------------------------------------


def test_prepare_model_no_existing_quant_config_no_overrides_quantizes_all_linears(input_model):
    """Uniform default config with no overrides: every Linear gets quant_info; overrides empty."""
    wrapper, qcfg, eligible = prepare_model(input_model, _baseline_pass_config())

    for name, module in wrapper.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == "lm_head":
                assert not hasattr(module, "quant_info")
            else:
                assert module.quant_info.quantizer.bits == PrecisionBits.BITS4
    assert qcfg.overrides == {}
    assert qcfg.bits == PrecisionBits.BITS4
    assert eligible is False


def test_prepare_model_component_source_path_quantizes_only_selected_component(input_model, monkeypatch):
    """A selected HfModel component should not attach quant_info outside that submodule."""
    root_model = _make_nested_decoder_root(input_model)
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={"component_source_paths": ["decoder"]},
    )

    wrapper, qcfg, _ = prepare_model(model, _baseline_pass_config())

    assert wrapper.model is root_model.decoder
    assert hasattr(root_model.decoder.model.layers[0].self_attn.q_proj, "quant_info")
    assert not hasattr(root_model.vision, "quant_info")
    assert "vision" in qcfg.modules_to_not_convert
    assert "decoder.model.embed_tokens" in qcfg.modules_to_not_convert
    assert "decoder.lm_head" in qcfg.modules_to_not_convert
    assert not any(name.startswith("decoder.model.layers") for name in qcfg.modules_to_not_convert)


def test_prepare_model_rejects_component_source_path_missing_at_runtime(input_model, monkeypatch):
    root_model = _make_nested_decoder_root(input_model)
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={"component_source_paths": ["model.language_model"]},
    )

    with pytest.raises(ValueError, match=r"model\.language_model.*named_modules"):
        prepare_model(model, _baseline_pass_config())


def test_prepare_model_rejects_selected_component_without_source_paths(input_model):
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={"component_name": "decoder", "component_role": "decoder"},
    )

    with pytest.raises(ValueError, match="no runtime source paths"):
        prepare_model(model, _baseline_pass_config())


def test_prepare_model_whole_encoder_component_uses_generic_wrapper(input_model, monkeypatch):
    root_model = BertForSequenceClassification(
        BertConfig(  # pylint: disable=unexpected-keyword-arg
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=128,
        )
    )
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={"component_name": "model", "component_role": "encoder"},
    )

    wrapper, qcfg, _ = prepare_model(model, _baseline_pass_config())

    assert wrapper.hidden_size == root_model.config.hidden_size
    assert wrapper.num_attention_heads == root_model.config.num_attention_heads
    assert wrapper.num_hidden_layers == root_model.config.num_hidden_layers
    assert wrapper.get_layer_wrappers() == []
    assert hasattr(root_model.bert.encoder.layer[0].attention.self.query, "quant_info")
    assert hasattr(root_model.classifier, "quant_info")
    assert "bert.embeddings.word_embeddings" in qcfg.modules_to_not_convert
    assert "bert.embeddings.position_embeddings" in qcfg.modules_to_not_convert
    assert "bert.embeddings.token_type_embeddings" in qcfg.modules_to_not_convert


def test_finalize_whole_encoder_reloads_all_embeddings_as_float(
    input_model,
    monkeypatch,
    tmp_path,
):
    root_model = BertForSequenceClassification(
        BertConfig(  # pylint: disable=unexpected-keyword-arg
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=128,
        )
    )
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        task="text-classification",
        model_attributes={"component_name": "model", "component_role": "encoder"},
    )
    model.save_metadata = lambda *_, **__: []
    wrapper, qcfg, _ = prepare_model(model, _baseline_pass_config())

    output_model = finalize(model, str(tmp_path), wrapper, qcfg, device="cpu")
    reloaded = output_model.load_model()

    assert isinstance(reloaded.bert.encoder.layer[0].attention.self.query, QuantLinear)
    assert isinstance(reloaded.classifier, QuantLinear)
    assert isinstance(reloaded.bert.embeddings.word_embeddings, torch.nn.Embedding)
    assert isinstance(reloaded.bert.embeddings.position_embeddings, torch.nn.Embedding)
    assert isinstance(reloaded.bert.embeddings.token_type_embeddings, torch.nn.Embedding)


def test_layerwise_quantization_rejects_embedding_role_with_decoder_wrapper(input_model, monkeypatch):
    root_model = _NestedBackboneRoot(LlamaForCausalLM.from_pretrained(input_model.model_path))
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={
            "component_role": "embedding",
            "component_source_paths": ["model.language_model.embed_tokens"],
        },
    )
    wrapper, _, _ = prepare_model(model, _baseline_pass_config(embeds=True))

    with pytest.raises(ValueError, match="Layerwise calibration requires a decoder"):
        run_layerwise_quantization(
            model,
            wrapper,
            data_config=None,
            input_hook=lambda *_: None,
            process_module=lambda *_: None,
            update_before_process=False,
            include_lm_head=False,
        )


def test_prepare_model_multi_path_component_slices_common_ancestor(input_model, monkeypatch):
    """A multi-path component slices to the common ancestor, quantizing only declared sub-trees."""
    root_model = _make_nested_decoder_root(input_model)
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    # decoder.model.layers (transformer blocks) + decoder.lm_head, common ancestor = "decoder".
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={"component_source_paths": ["decoder.model.layers", "decoder.lm_head"]},
    )

    wrapper, qcfg, _ = prepare_model(model, _baseline_pass_config())

    # Sliced to the common ancestor submodule, not the whole root.
    assert wrapper.model is root_model.decoder
    # Linear inside a declared sub-tree is quantized.
    assert hasattr(root_model.decoder.model.layers[0].self_attn.q_proj, "quant_info")
    # A sibling module inside the slice but outside the declared sub-trees is excluded.
    assert "decoder.model.embed_tokens" in qcfg.modules_to_not_convert
    # A module outside the slice entirely is excluded.
    assert "vision" in qcfg.modules_to_not_convert


def test_finalize_multi_path_vlm_decoder_quantizes_and_saves_full_model(
    input_model,
    monkeypatch,
    tmp_path,
):
    root_model = _NestedBackboneRoot(LlamaForCausalLM.from_pretrained(input_model.model_path))
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={
            "component_source_paths": [
                "model.language_model.layers",
                "model.language_model.norm",
                "lm_head",
            ]
        },
    )
    model.save_metadata = lambda *_, **__: []
    wrapper, qcfg, _ = prepare_model(model, _baseline_pass_config())

    finalize(model, str(tmp_path), wrapper, qcfg, device="cpu")

    assert isinstance(root_model.model.language_model.layers[0].self_attn.q_proj, QuantLinear)
    assert isinstance(root_model.model.language_model.embed_tokens, torch.nn.Embedding)
    assert isinstance(root_model.lm_head, torch.nn.Linear)
    assert isinstance(root_model.vision, torch.nn.Linear)
    assert any(
        key.startswith("model.language_model.layers.0.self_attn.q_proj.qweight") for key in root_model.saved_state_keys
    )
    assert "vision.weight" in root_model.saved_state_keys


def test_finalize_bart_decoder_reloads_unquantized_modules_as_float(
    input_model,
    monkeypatch,
    tmp_path,
):
    root_model = BartForConditionalGeneration(
        BartConfig(  # pylint: disable=unexpected-keyword-arg
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            vocab_size=128,
        )
    )
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        task="text2text-generation",
        model_attributes={
            "component_role": "decoder",
            "component_source_paths": ["model.decoder", "lm_head"],
        },
    )
    model.save_metadata = lambda *_, **__: []
    wrapper, qcfg, _ = prepare_model(model, _baseline_pass_config())

    output_model = finalize(model, str(tmp_path), wrapper, qcfg, device="cpu")
    reloaded = output_model.load_model()

    assert isinstance(reloaded.model.decoder.layers[0].self_attn.q_proj, QuantLinear)
    assert isinstance(reloaded.model.decoder.embed_tokens, torch.nn.Embedding)
    assert isinstance(reloaded.model.decoder.embed_positions, torch.nn.Embedding)
    assert isinstance(reloaded.model.encoder.layers[0].self_attn.q_proj, torch.nn.Linear)
    assert isinstance(reloaded.lm_head, torch.nn.Linear)


def test_finalize_bart_decoder_embeddings_preserves_untied_float_weights(
    input_model,
    monkeypatch,
    tmp_path,
):
    root_model = BartForConditionalGeneration(
        BartConfig(  # pylint: disable=unexpected-keyword-arg
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            vocab_size=128,
        )
    )
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        task="text2text-generation",
        model_attributes={
            "component_role": "decoder",
            "component_source_paths": ["model.decoder", "lm_head"],
        },
    )
    model.save_metadata = lambda *_, **__: []
    wrapper, qcfg, retie = prepare_model(model, _baseline_pass_config(embeds=True))
    expected_shared = root_model.model.shared.weight.detach().clone()
    expected_encoder = root_model.model.encoder.embed_tokens.weight.detach().clone()
    expected_head = root_model.lm_head.weight.detach().clone()

    output_model = finalize(
        model,
        str(tmp_path),
        wrapper,
        qcfg,
        device="cpu",
        retie_word_embeddings=retie,
    )
    reloaded = output_model.load_model()

    assert not retie
    assert isinstance(reloaded.model.decoder.embed_tokens, QuantEmbedding)
    assert isinstance(reloaded.model.decoder.embed_positions, QuantEmbedding)
    assert isinstance(reloaded.model.shared, torch.nn.Embedding)
    assert isinstance(reloaded.model.encoder.embed_tokens, torch.nn.Embedding)
    assert isinstance(reloaded.lm_head, torch.nn.Linear)
    torch.testing.assert_close(reloaded.model.shared.weight, expected_shared)
    torch.testing.assert_close(reloaded.model.encoder.embed_tokens.weight, expected_encoder)
    torch.testing.assert_close(reloaded.lm_head.weight, expected_head)


def test_finalize_t5_shared_embedding_preserves_float_aliases(
    input_model,
    monkeypatch,
    tmp_path,
):
    root_model = T5ForConditionalGeneration(
        T5Config(  # pylint: disable=unexpected-keyword-arg
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            vocab_size=128,
        )
    )
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        task="text2text-generation",
        model_attributes={
            "component_role": "embedding",
            "component_source_paths": ["shared"],
        },
    )
    model.save_metadata = lambda *_, **__: []
    wrapper, qcfg, retie = prepare_model(model, _baseline_pass_config(embeds=True))
    expected_encoder = root_model.encoder.embed_tokens.weight.detach().clone()
    expected_decoder = root_model.decoder.embed_tokens.weight.detach().clone()
    expected_head = root_model.lm_head.weight.detach().clone()

    output_model = finalize(
        model,
        str(tmp_path),
        wrapper,
        qcfg,
        device="cpu",
        retie_word_embeddings=retie,
    )
    reloaded = output_model.load_model()

    assert not retie
    assert isinstance(reloaded.shared, QuantEmbedding)
    assert isinstance(reloaded.encoder.embed_tokens, torch.nn.Embedding)
    assert isinstance(reloaded.decoder.embed_tokens, torch.nn.Embedding)
    assert isinstance(reloaded.lm_head, torch.nn.Linear)
    torch.testing.assert_close(reloaded.encoder.embed_tokens.weight, expected_encoder)
    torch.testing.assert_close(reloaded.decoder.embed_tokens.weight, expected_decoder)
    torch.testing.assert_close(reloaded.lm_head.weight, expected_head)


def test_prepare_model_removes_current_component_from_existing_exclusions(
    input_model,
    monkeypatch,
):
    existing = OliveHfQuantizationConfig(
        bits=PrecisionBits.BITS4,
        symmetric=False,
        group_size=16,
        modules_to_not_convert=["model.language_model.embed_tokens", "vision"],
    )
    _with_existing_quantization_config(monkeypatch, existing)
    root_model = _NestedBackboneRoot(LlamaForCausalLM.from_pretrained(input_model.model_path))
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={"component_source_paths": ["model.language_model.embed_tokens"]},
    )

    _, qcfg, _ = prepare_model(
        model,
        _baseline_pass_config(embeds=True),
        allow_quantized=True,
    )

    assert "model.language_model.embed_tokens" not in qcfg.modules_to_not_convert
    assert "vision" in qcfg.modules_to_not_convert


def test_finalize_vlm_encoder_component_only_quantizes_encoder(
    input_model,
    monkeypatch,
    tmp_path,
):
    root_model = _NestedBackboneRoot(LlamaForCausalLM.from_pretrained(input_model.model_path))
    root_model.vision = torch.nn.Linear(16, 16)
    monkeypatch.setattr(quant_utils_module, "load_hf_base_model", lambda _: root_model)
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={
            "component_role": "encoder",
            "component_source_paths": ["vision"],
        },
    )
    model.save_metadata = lambda *_, **__: []
    wrapper, qcfg, _ = prepare_model(model, _baseline_pass_config())

    finalize(model, str(tmp_path), wrapper, qcfg, device="cpu")

    assert isinstance(root_model.vision, QuantLinear)
    assert isinstance(root_model.model.language_model.layers[0].self_attn.q_proj, torch.nn.Linear)
    assert isinstance(root_model.model.language_model.embed_tokens, torch.nn.Embedding)
    assert isinstance(root_model.lm_head, torch.nn.Linear)


def test_prepare_model_promotes_user_override_conflicts_for_qkv(input_model):
    """User-supplied overrides on K/V promote Q to the most-precise shared config."""
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={
            "mixed_precision_info": {
                "default": {"bits": PrecisionBits.BITS4, "group_size": 16, "symmetric": False},
                "overrides": {
                    "model.layers.0.self_attn.k_proj": {
                        "bits": PrecisionBits.BITS8,
                        "group_size": 16,
                        "symmetric": True,
                    },
                    "model.layers.0.self_attn.v_proj": {
                        "bits": PrecisionBits.BITS8,
                        "group_size": 16,
                        "symmetric": True,
                    },
                },
            }
        },
    )
    config = _baseline_pass_config(
        overrides={"model.layers.0.self_attn.q_proj": {"bits": PrecisionBits.BITS4}},
    )

    wrapper, qcfg, _ = prepare_model(model, config)

    expected = {"bits": PrecisionBits.BITS8, "symmetric": True, "group_size": 16}
    for proj in ("q_proj", "k_proj", "v_proj"):
        assert qcfg.get_qlinear_init_args(f"model.layers.0.self_attn.{proj}") == expected
        attached = getattr(wrapper.model.model.layers[0].self_attn, proj)
        assert attached.quant_info.quantizer.bits == PrecisionBits.BITS8


def test_prepare_model_attaches_quant_info_matching_final_post_normalize_config(input_model):
    """Each module's attached ``quant_info`` must match the final, post-normalize qcfg config."""
    config = _baseline_pass_config(
        overrides={
            "model.layers.0.self_attn.q_proj": {"bits": PrecisionBits.BITS8, "symmetric": True, "group_size": 16},
        },
    )

    wrapper, qcfg, _ = prepare_model(input_model, config)

    attn = wrapper.model.model.layers[0].self_attn
    for proj in ("q_proj", "k_proj", "v_proj"):
        attached_bits = getattr(attn, proj).quant_info.quantizer.bits
        cfg_bits = qcfg.get_qlinear_init_args(f"model.layers.0.self_attn.{proj}")["bits"]
        assert attached_bits == cfg_bits == PrecisionBits.BITS8


# ---------------------------------------------------------------------------
# prepare_model: leakage-prevention cases (modules that won't be quantized
# must not appear in the final qcfg overrides).
# ---------------------------------------------------------------------------


def test_prepare_model_drops_override_for_nonexistent_module(input_model):
    """Overrides referencing modules that don't exist must not survive in the final qcfg."""
    bogus_name = "model.layers.99.self_attn.totally_made_up"
    config = _baseline_pass_config(overrides={bogus_name: {"bits": PrecisionBits.BITS8}})

    _, qcfg, _ = prepare_model(input_model, config)

    assert bogus_name not in (qcfg.overrides or {})


def test_prepare_model_drops_override_for_lm_head_when_lm_head_disabled(input_model):
    """lm_head isn't quantized when ``lm_head=False``; its override should not leak into qcfg."""
    config = _baseline_pass_config(overrides={"lm_head": {"bits": PrecisionBits.BITS8}})

    wrapper, qcfg, _ = prepare_model(input_model, config)

    assert "lm_head" not in (qcfg.overrides or {})
    assert not hasattr(wrapper.model.lm_head, "quant_info")


def test_prepare_model_drops_embedding_override_when_embeds_disabled(input_model):
    """Embedding overrides shouldn't leak in when embeddings aren't being quantized."""
    config = _baseline_pass_config(overrides={"model.embed_tokens": {"bits": PrecisionBits.BITS8}})

    _, qcfg, _ = prepare_model(input_model, config)

    assert "model.embed_tokens" not in (qcfg.overrides or {})


def test_prepare_model_drops_qkv_overrides_for_modules_excluded_via_exclude_attn_inputs(input_model):
    """``exclude_attn_inputs=True`` should not leak q/k overrides into the final qcfg.

    Q/K aren't quantized this pass; the follow-up pass re-derives their config from the
    quantized (locked) V member via ``normalize_qkv_quant_config``.
    """
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={
            "mixed_precision_info": {
                "default": {"bits": PrecisionBits.BITS4, "group_size": 16, "symmetric": False},
                "overrides": {
                    "model.layers.0.self_attn.q_proj": {
                        "bits": PrecisionBits.BITS8,
                        "group_size": 16,
                        "symmetric": True,
                    },
                    "model.layers.0.self_attn.k_proj": {
                        "bits": PrecisionBits.BITS8,
                        "group_size": 16,
                        "symmetric": True,
                    },
                },
            }
        },
    )

    wrapper, qcfg, _ = prepare_model(model, _baseline_pass_config(), exclude_attn_inputs=True)
    attention = wrapper.model.model.layers[0].self_attn

    assert not hasattr(attention.q_proj, "quant_info")
    assert not hasattr(attention.k_proj, "quant_info")
    # V is quantized and promoted to the group-wide 8-bit config.
    assert qcfg.get_qlinear_init_args("model.layers.0.self_attn.v_proj") == {
        "bits": PrecisionBits.BITS8,
        "symmetric": True,
        "group_size": 16,
    }
    assert attention.v_proj.quant_info.quantizer.bits == PrecisionBits.BITS8
    # Q/K overrides dropped; follow-up pass will rebuild them from V (locked).
    assert "model.layers.0.self_attn.q_proj" not in (qcfg.overrides or {})
    assert "model.layers.0.self_attn.k_proj" not in (qcfg.overrides or {})


def test_prepare_model_exclude_attn_inputs_fused_qkv_does_not_create_overrides_for_fused_module():
    """When attn inputs are fused into a single qkv_proj, it's excluded and no override is created."""
    model_handler = get_tiny_phi3()
    wrapper, qcfg, _ = prepare_model(model_handler, _baseline_pass_config(), exclude_attn_inputs=True)

    qkv_proj = wrapper.model.model.layers[0].self_attn.qkv_proj
    assert not hasattr(qkv_proj, "quant_info")
    assert "model.layers.0.self_attn.qkv_proj" not in (qcfg.overrides or {})


# ---------------------------------------------------------------------------
# prepare_model: existing quantization_config (allow_quantized=True) cases
# ---------------------------------------------------------------------------


def test_prepare_model_raises_when_existing_quant_config_present_without_allow_quantized(input_model, monkeypatch):
    """Pre-existing quantization config without ``allow_quantized`` must raise."""
    existing = {"quant_method": "olive", "bits": PrecisionBits.BITS4, "symmetric": False, "group_size": 16}
    _with_existing_quantization_config(monkeypatch, existing)

    with pytest.raises(ValueError, match="already quantized"):
        prepare_model(input_model, _baseline_pass_config(), allow_quantized=False)


def test_prepare_model_raises_when_existing_quant_method_is_unsupported(input_model, monkeypatch):
    """A non-olive ``quant_method`` must raise even with ``allow_quantized=True``."""
    existing = {"quant_method": "awq", "bits": PrecisionBits.BITS4, "symmetric": False, "group_size": 16}
    _with_existing_quantization_config(monkeypatch, existing)

    with pytest.raises(ValueError, match="not compatible"):
        prepare_model(input_model, _baseline_pass_config(), allow_quantized=True)


def test_prepare_model_does_not_mutate_caller_existing_quantization_config_dict(input_model, monkeypatch):
    """If the HF config holds a dict ``quantization_config``, prepare_model must not mutate it."""
    existing = {
        "quant_method": "olive",
        "bits": PrecisionBits.BITS4,
        "symmetric": False,
        "group_size": 16,
        "lm_head": False,
        "embeds": False,
        "overrides": {
            "model.layers.0.self_attn.q_proj": {
                "bits": PrecisionBits.BITS8,
                "symmetric": True,
                "group_size": 16,
            },
        },
    }
    snapshot = deepcopy(existing)
    _with_existing_quantization_config(monkeypatch, existing)

    prepare_model(input_model, _baseline_pass_config(), allow_quantized=True)

    assert existing == snapshot, "prepare_model mutated the caller's existing quantization_config dict"


def test_prepare_model_does_not_mutate_caller_existing_quantization_config_object(input_model, monkeypatch):
    """If the HF config holds an ``OliveHfQuantizationConfig`` object, prepare_model must not mutate it."""
    existing = OliveHfQuantizationConfig(
        bits=PrecisionBits.BITS4,
        symmetric=False,
        group_size=16,
        overrides={
            "model.layers.0.self_attn.q_proj": {
                "bits": PrecisionBits.BITS8,
                "symmetric": True,
                "group_size": 16,
            },
        },
    )
    snapshot = existing.to_dict()
    _with_existing_quantization_config(monkeypatch, existing)

    prepare_model(input_model, _baseline_pass_config(), allow_quantized=True)

    assert existing.to_dict() == snapshot, "prepare_model mutated the caller's existing OliveHfQuantizationConfig"


def test_prepare_model_preserves_pre_existing_overrides_verbatim(input_model, monkeypatch):
    """Pre-existing overrides describe already-quantized weights and must survive untouched."""
    locked_override = {"bits": PrecisionBits.BITS8, "symmetric": True, "group_size": 16}
    existing = {
        "quant_method": "olive",
        "bits": PrecisionBits.BITS4,
        "symmetric": False,
        "group_size": 16,
        "lm_head": False,
        "embeds": False,
        "overrides": {
            "model.layers.0.mlp.down_proj": dict(locked_override),
        },
    }
    _with_existing_quantization_config(monkeypatch, existing)

    _, qcfg, _ = prepare_model(input_model, _baseline_pass_config(), allow_quantized=True)

    assert qcfg.get_qlinear_init_args("model.layers.0.mlp.down_proj") == locked_override


def test_prepare_model_renormalizes_qkv_after_merging_existing_quant_config(input_model, monkeypatch):
    """After merging a pre-existing qcfg, QKV is renormalized to the locked member's config."""
    existing_quantization_config = {
        "quant_method": "olive",
        "bits": PrecisionBits.BITS4,
        "symmetric": False,
        "group_size": 16,
        "lm_head": False,
        "embeds": False,
        "overrides": {
            "model.layers.0.self_attn.q_proj": {
                "bits": PrecisionBits.BITS8,
                "symmetric": True,
                "group_size": 16,
            },
        },
    }
    _with_existing_quantization_config(monkeypatch, dict(existing_quantization_config))

    _, qcfg, _ = prepare_model(input_model, _baseline_pass_config(), allow_quantized=True)

    expected = {"bits": PrecisionBits.BITS8, "symmetric": True, "group_size": 16}
    for proj in ("q_proj", "k_proj", "v_proj"):
        assert qcfg.get_qlinear_init_args(f"model.layers.0.self_attn.{proj}") == expected


def test_prepare_model_existing_quant_config_drops_fresh_overrides_for_non_quantized_modules(input_model, monkeypatch):
    """Merging with existing qcfg drops overrides for modules that won't be quantized.

    Overrides for modules that are neither quantized this pass nor already-on-disk must not
    leak into the final qcfg.
    """
    existing = {
        "quant_method": "olive",
        "bits": PrecisionBits.BITS4,
        "symmetric": False,
        "group_size": 16,
        "lm_head": False,
        "embeds": False,
        "overrides": {},
    }
    _with_existing_quantization_config(monkeypatch, existing)
    config = _baseline_pass_config(overrides={"model.does.not.exist": {"bits": PrecisionBits.BITS8}})

    _, qcfg, _ = prepare_model(input_model, config, allow_quantized=True)

    assert "model.does.not.exist" not in (qcfg.overrides or {})


def test_prepare_model_locks_default_quantized_qkv_member_without_override(input_model, monkeypatch):
    """A default-quantized (no override entry) QKV member is still locked.

    If V was quantized at the existing config's defaults (so it has no entry in
    ``existing_qcfg['overrides']`` but IS a ``QuantLinear`` after load) and a fresh pass
    promotes Q/K to higher precision, the QKV normalization must NOT write a new override
    for V -- that would disagree with V's on-disk weights. Instead, Q/K should be demoted
    to V's existing default config.
    """
    qu = quant_utils_module

    existing = {
        "quant_method": "olive",
        "bits": PrecisionBits.BITS4,
        "symmetric": False,
        "group_size": 16,
        "lm_head": False,
        "embeds": False,
        "overrides": {},
    }
    _with_existing_quantization_config(monkeypatch, existing)

    real_loader = qu.load_hf_base_model

    def fake_load(model_handler, **kwargs):
        loaded = real_loader(model_handler, **kwargs)
        # Replace v_proj of layer 0 with a QuantLinear so it looks already-quantized on disk.
        attn = loaded.model.layers[0].self_attn
        v = attn.v_proj
        attn.v_proj = QuantLinear(
            in_features=v.in_features,
            out_features=v.out_features,
            bias=v.bias is not None,
            bits=PrecisionBits.BITS4,
            symmetric=False,
            group_size=16,
            device=v.weight.device,
            dtype=v.weight.dtype,
        )
        return loaded

    monkeypatch.setattr(qu, "load_hf_base_model", fake_load)

    config = _baseline_pass_config(
        overrides={
            "model.layers.0.self_attn.q_proj": {
                "bits": PrecisionBits.BITS8,
                "symmetric": True,
                "group_size": 16,
            },
        },
    )

    _, qcfg, _ = prepare_model(input_model, config, allow_quantized=True)

    # V's on-disk config (existing defaults) is the locked promotion target; Q/K must match.
    default = {"bits": PrecisionBits.BITS4, "symmetric": False, "group_size": 16}
    for proj in ("q_proj", "k_proj", "v_proj"):
        assert qcfg.get_qlinear_init_args(f"model.layers.0.self_attn.{proj}") == default
    # No new override added for V (it stays at defaults on disk).
    assert "model.layers.0.self_attn.v_proj" not in (qcfg.overrides or {})
