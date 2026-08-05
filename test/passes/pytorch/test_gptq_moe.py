# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
"""Tests for calibrated (GPTQ) quantization of fused-MoE expert weights.

Every model here is a randomly-initialised tiny model built from the architecture's own HF
config (no hub checkpoints), following the precedent in ``test_rtn.py``.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.selection import iter_quant_targets
from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import WeightQuantizer
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.gptq import Gptq
from olive.passes.pytorch.moe_calib import (
    SUPPORTED_MOE_MODEL_TYPES,
    MoeCalibrationError,
    MoeCalibrationSession,
    check_moe_gptq_support,
)
from olive.passes.pytorch.quant_utils import QuantInfo

# every architecture on the K-last allow-list
ALLOW_LISTED_MODEL_TYPES = [
    "deepseek_v3",
    "granitemoe",
    "jamba",
    "mixtral",
    "olmoe",
    "phimoe",
    "qwen2_moe",
    "qwen3_moe",
]

VOCAB_SIZE = 64
NUM_EXPERTS = 4


def build_tiny_moe_model(model_type: str) -> torch.nn.Module:
    """Build a tiny, randomly-initialised model for one allow-listed MoE architecture."""
    import transformers as tf

    torch.manual_seed(0)
    common = {
        "vocab_size": VOCAB_SIZE,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
    }
    builders = {
        "qwen2_moe": lambda: tf.Qwen2MoeForCausalLM(
            tf.Qwen2MoeConfig(
                **common,
                moe_intermediate_size=16,
                shared_expert_intermediate_size=16,
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=2,
                decoder_sparse_step=1,
            )
        ),
        "qwen3_moe": lambda: tf.Qwen3MoeForCausalLM(
            tf.Qwen3MoeConfig(
                **common,
                moe_intermediate_size=16,
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=2,
                decoder_sparse_step=1,
                head_dim=16,
            )
        ),
        "phimoe": lambda: tf.PhimoeForCausalLM(
            tf.PhimoeConfig(**common, num_local_experts=NUM_EXPERTS, num_experts_per_tok=2)
        ),
        "mixtral": lambda: tf.MixtralForCausalLM(
            tf.MixtralConfig(**common, num_local_experts=NUM_EXPERTS, num_experts_per_tok=2)
        ),
        "deepseek_v3": lambda: tf.DeepseekV3ForCausalLM(
            tf.DeepseekV3Config(
                **common,
                moe_intermediate_size=16,
                n_routed_experts=NUM_EXPERTS,
                num_experts_per_tok=2,
                first_k_dense_replace=1,
                n_group=1,
                topk_group=1,
                n_shared_experts=1,
                qk_rope_head_dim=8,
                qk_nope_head_dim=8,
                v_head_dim=16,
                kv_lora_rank=16,
                q_lora_rank=None,
            )
        ),
        "granitemoe": lambda: tf.GraniteMoeForCausalLM(
            tf.GraniteMoeConfig(**common, num_local_experts=NUM_EXPERTS, num_experts_per_tok=2)
        ),
        "olmoe": lambda: tf.OlmoeForCausalLM(
            tf.OlmoeConfig(**common, num_experts=NUM_EXPERTS, num_experts_per_tok=2)
        ),
        "jamba": lambda: tf.JambaForCausalLM(
            tf.JambaConfig(
                **common,
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=2,
                expert_layer_period=2,
                expert_layer_offset=1,
                attn_layer_period=2,
                attn_layer_offset=1,
                mamba_d_state=8,
                mamba_d_conv=2,
                mamba_dt_rank=16,
            )
        ),
    }
    return builders[model_type]().eval()


def save_tiny_moe_model(save_path, model_type: str) -> HfModelHandler:
    """Save a tiny MoE model (plus a trivial local tokenizer) and return its handler.

    The saved config pins ``experts_implementation="eager"``: the default ``grouped_mm``
    backend transposes the fused expert weight, which an Olive ``QuantTensor`` cannot do
    (quantization is storage-only). Same precedent as ``test_rtn.py``.
    """
    import json

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    build_tiny_moe_model(model_type).save_pretrained(save_path)

    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(VOCAB_SIZE)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", pad_token="t0").save_pretrained(save_path)

    config_path = save_path / "config.json"
    config = json.loads(config_path.read_text())
    config["experts_implementation"] = "eager"
    config_path.write_text(json.dumps(config, indent=2))

    return HfModelHandler(model_path=str(save_path))


@pytest.fixture
def _patched_calibration_dataset(monkeypatch):
    """Replace the wikitext calibration dataset with deterministic local random token batches."""
    import olive.passes.pytorch.quant_utils as quant_utils

    def fake_calibration_dataset(model, data_config=None, **kwargs):
        generator = torch.Generator().manual_seed(0)
        return [
            {
                "input_ids": torch.randint(0, VOCAB_SIZE, (1, 16), generator=generator),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }
            for _ in range(3)
        ]

    monkeypatch.setattr(quant_utils, "get_calibration_dataset", fake_calibration_dataset)


@contextmanager
def capture_logs(logger_name: str):
    """Collect records emitted by ``logger_name`` (Olive's loggers don't propagate to caplog)."""
    records: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    handler = _Handler(level=logging.INFO)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _attach_quant_info(experts: torch.nn.Module, group_size: int = 16) -> None:
    for pname in ("gate_up_proj", "down_proj"):
        experts._parameters[pname].quant_info = QuantInfo(
            quantizer=WeightQuantizer(bits=4, symmetric=True, group_size=group_size)
        )


# ---------------------------------------------------------------------------
# LayerWrapper mappings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ALLOW_LISTED_MODEL_TYPES)
def test_layer_wrapper_resolves_experts_and_router(model_type: str):
    """Every allow-listed architecture must resolve its experts + router submodules.

    Regression for ``granitemoe`` (``layer.block_sparse_moe``) and ``jamba``
    (``layer.feed_forward``), which previously raised ``AttributeError`` in
    ``LayerWrapper.__init__`` before any MoE guard could run.
    """
    wrapper = ModelWrapper.from_model(build_tiny_moe_model(model_type))
    assert wrapper.model_type == model_type

    moe_layers = 0
    for layer_wrapper in wrapper.get_layer_wrappers():
        experts = layer_wrapper.get_experts(return_name=False)
        router = layer_wrapper.get_router(return_name=False)
        if experts is None:
            # architectures that interleave dense layers (deepseek_v3, jamba) resolve neither
            assert router is None
            continue
        moe_layers += 1
        assert router is not None
        assert experts.gate_up_proj.dim() == 3
        assert experts.gate_up_proj.shape[0] == NUM_EXPERTS

    assert moe_layers > 0


@pytest.mark.parametrize("model_type", ALLOW_LISTED_MODEL_TYPES)
def test_router_is_never_a_quant_target(model_type: str):
    """Routers stay full precision, including Jamba's bare ``nn.Linear`` router."""
    model = build_tiny_moe_model(model_type)
    wrapper = ModelWrapper.from_model(model)
    router_ids = {
        id(sub)
        for lw in wrapper.get_layer_wrappers()
        if (router := lw.get_router(return_name=False)) is not None
        for sub in router.modules()
    }
    assert router_ids, "expected at least one resolvable router"

    for quantize_moe in (False, True):
        targets = list(
            iter_quant_targets(
                model, quantize_lm_head=True, quantize_embeds=True, quantize_moe=quantize_moe
            )
        )
        assert not any(id(module) in router_ids for module, _, _ in targets)
        # sanity: the walk is not empty (so the assertion above is meaningful)
        assert targets


def test_jamba_router_linear_would_otherwise_be_selected():
    """Guard the specific Jamba risk: its router *is* an ``nn.Linear`` in the 2D walk."""
    model = build_tiny_moe_model("jamba")
    wrapper = ModelWrapper.from_model(model)
    routers = [
        router for lw in wrapper.get_layer_wrappers() if (router := lw.get_router(return_name=False)) is not None
    ]
    assert routers and all(isinstance(router, torch.nn.Linear) for router in routers)


# ---------------------------------------------------------------------------
# support gating (fail closed)
# ---------------------------------------------------------------------------


class _FakeExperts(torch.nn.Module):
    def __init__(self, **flags):
        super().__init__()
        for key, value in flags.items():
            setattr(self, key, value)


def test_check_support_accepts_allow_listed_model():
    check_moe_gptq_support("qwen3_moe", [_FakeExperts(is_transposed=False, has_bias=False, has_gate=True)])


@pytest.mark.parametrize("model_type", ["gpt_oss", "llama4", "aria", "not_a_model"])
def test_check_support_rejects_non_allow_listed_model(model_type: str):
    with pytest.raises(MoeCalibrationError, match="not supported for model_type"):
        check_moe_gptq_support(model_type, [_FakeExperts(is_transposed=False)])


def test_check_support_rejects_missing_capability():
    """No ``is_transposed`` attribute => transformers too old / class not decorated."""
    with pytest.raises(MoeCalibrationError, match="is_transposed"):
        check_moe_gptq_support("qwen3_moe", [_FakeExperts()])


def test_check_support_rejects_transposed_layout():
    with pytest.raises(MoeCalibrationError, match="transposed fused weights"):
        check_moe_gptq_support("qwen3_moe", [_FakeExperts(is_transposed=True)])


def test_check_support_rejects_old_transformers(monkeypatch):
    import olive.passes.pytorch.moe_calib as moe_calib

    monkeypatch.setattr(moe_calib, "_transformers_supports_experts_registry", lambda: False)
    with pytest.raises(MoeCalibrationError, match="requires transformers >="):
        check_moe_gptq_support("qwen3_moe", [_FakeExperts(is_transposed=False)])


def test_allow_list_matches_documented_architectures():
    assert set(ALLOW_LISTED_MODEL_TYPES) == set(SUPPORTED_MOE_MODEL_TYPES)
    assert "gpt_oss" not in SUPPORTED_MOE_MODEL_TYPES


def test_session_create_returns_none_for_dense_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2)
    )
    assert MoeCalibrationSession.create(ModelWrapper.from_model(model)) is None


# ---------------------------------------------------------------------------
# per-expert recording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ALLOW_LISTED_MODEL_TYPES)
def test_per_expert_hessians_are_isolated(model_type: str):
    """Each expert gets its own (K, K) Hessian built only from the tokens routed to it."""
    model = build_tiny_moe_model(model_type)
    wrapper = ModelWrapper.from_model(model)
    session = MoeCalibrationSession.create(wrapper)
    assert session is not None

    experts = next(
        lw.get_experts(return_name=False)
        for lw in wrapper.get_layer_wrappers()
        if lw.get_experts(return_name=False) is not None
    )
    _attach_quant_info(experts)

    session.start()
    try:
        with session.record([experts]):
            with torch.no_grad():
                model(torch.randint(0, VOCAB_SIZE, (2, 16)))
    finally:
        session.finish()

    data = experts.gate_up_proj.quant_info.data
    assert data["moe"] is True
    assert data["tokens_seen"] == 32
    assert len(data["token_counts"]) == NUM_EXPERTS
    # top_k=2 => every token is counted for exactly two experts
    assert sum(data["token_counts"]) == 32 * 2

    hidden_size = experts.gate_up_proj.shape[-1]
    for expert_idx, entry in data["experts"].items():
        assert entry["H"].shape == (hidden_size, hidden_size)
        # the Hessian saw exactly the tokens routed to that expert -- no cross-expert pooling
        assert entry["N"] == data["token_counts"][expert_idx]

    down_data = experts.down_proj.quant_info.data
    intermediate_size = experts.down_proj.shape[-1]
    for entry in down_data["experts"].values():
        assert entry["H"].shape == (intermediate_size, intermediate_size)


def test_recording_switch_suppresses_second_pass():
    """The true-sequential re-run must not double-count into the Hessians."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    session = MoeCalibrationSession.create(wrapper)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    _attach_quant_info(experts)

    inputs = torch.randint(0, VOCAB_SIZE, (2, 16))
    session.start()
    try:
        with session.record([experts]):
            with torch.no_grad():
                model(inputs)
        recorded = {e: entry["N"] for e, entry in experts.gate_up_proj.quant_info.data["experts"].items()}
        hessians = {e: entry["H"].clone() for e, entry in experts.gate_up_proj.quant_info.data["experts"].items()}

        # second (post-quantization) pass -- recording is off
        with torch.no_grad():
            model(inputs)
    finally:
        session.finish()

    after = {e: entry["N"] for e, entry in experts.gate_up_proj.quant_info.data["experts"].items()}
    assert after == recorded
    for expert_idx, hessian in hessians.items():
        assert torch.equal(experts.gate_up_proj.quant_info.data["experts"][expert_idx]["H"], hessian)


def test_calibration_forward_matches_eager_output():
    """The recording experts forward must return the model's normal output."""
    model = build_tiny_moe_model("qwen3_moe")
    model.set_experts_implementation("eager")
    inputs = torch.randint(0, VOCAB_SIZE, (2, 16))
    with torch.no_grad():
        expected = model(inputs).logits

    wrapper = ModelWrapper.from_model(model)
    session = MoeCalibrationSession.create(wrapper)
    session.start()
    try:
        with torch.no_grad():
            actual = model(inputs).logits
    finally:
        session.finish()

    torch.testing.assert_close(actual, expected)
    assert model.config._experts_implementation == "eager"


def test_coverage_report_flags_unseen_experts():
    """Coverage is reported per layer + summarized, and warns (never raises) when thin."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    session = MoeCalibrationSession.create(wrapper)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    _attach_quant_info(experts)

    # bias routing hard toward expert 0 so at least one expert is never routed to
    router = wrapper.get_layer_wrappers()[0].get_router(return_name=False)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[0] = 10.0

    session.start()
    with capture_logs("olive.passes.pytorch.moe_calib") as records:
        try:
            with session.record([experts]):
                with torch.no_grad():
                    model(torch.randint(0, VOCAB_SIZE, (2, 16)))
            session.add_coverage("model.layers.0", experts)
        finally:
            session.finish()

    counts = experts.gate_up_proj.quant_info.data["token_counts"]
    assert counts[0] > 0
    assert sum(1 for c in counts if c == 0) >= 1
    messages = "\n".join(records)
    assert "MoE coverage [model.layers.0]" in messages
    assert "MoE coverage summary" in messages
    assert "unseen" in messages


# ---------------------------------------------------------------------------
# RTN fallback
# ---------------------------------------------------------------------------


def _run_recording(model, wrapper, experts, fallback_threshold: float):
    session = MoeCalibrationSession.create(wrapper, fallback_threshold=fallback_threshold)
    session.start()
    try:
        with session.record([experts]):
            with torch.no_grad():
                model(torch.randint(0, VOCAB_SIZE, (2, 16)))
    finally:
        session.finish()
    return session


def test_rtn_fallback_when_below_threshold():
    """Below the threshold an expert is RTN-quantized: float weight kept, RTN qparams."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    _attach_quant_info(experts)
    # threshold of 1.0 => an expert must see *every* calibration token to avoid the fallback,
    # which top-k routing can never satisfy for more than top_k experts.
    _run_recording(model, wrapper, experts, fallback_threshold=1.0)

    original = experts.gate_up_proj.data.clone()
    quantizer = experts.gate_up_proj.quant_info.quantizer
    with capture_logs("olive.passes.pytorch.gptq") as records:
        Gptq.process_module(experts, moe_fallback_threshold=1.0)

    assert any("quantized with RTN" in message for message in records)
    # RTN keeps the float weight untouched and derives qparams straight from it
    torch.testing.assert_close(experts.gate_up_proj.data, original)
    expected_scales, expected_zp = quantizer.find_qparams(original.float())
    torch.testing.assert_close(experts.gate_up_proj.quant_info.scales, expected_scales.cpu())
    torch.testing.assert_close(experts.gate_up_proj.quant_info.zero_points, expected_zp.cpu())


def test_gptq_path_used_when_above_threshold():
    """With the default threshold every routed expert is GPTQ-quantized (weights change)."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    _attach_quant_info(experts)
    _run_recording(model, wrapper, experts, fallback_threshold=0.005)

    original = experts.gate_up_proj.data.clone()
    Gptq.process_module(experts, moe_fallback_threshold=0.005)

    assert not torch.equal(experts.gate_up_proj.data, original)
    num_groups = original.shape[-1] // 16
    assert experts.gate_up_proj.quant_info.scales.shape == (NUM_EXPERTS, original.shape[1], num_groups)
    assert experts.gate_up_proj.quant_info.data is None


def test_zero_sample_expert_falls_back_without_hessian():
    """An expert that was never routed to has no Hessian at all and must not crash GPTQ."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    _attach_quant_info(experts)

    # bias routing hard toward expert 0 so at least one expert is never routed to
    router = wrapper.get_layer_wrappers()[0].get_router(return_name=False)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[0] = 10.0
    _run_recording(model, wrapper, experts, fallback_threshold=0.005)

    data = experts.gate_up_proj.quant_info.data
    unseen = [idx for idx, count in enumerate(data["token_counts"]) if count == 0]
    assert unseen
    original = experts.gate_up_proj.data.clone()

    Gptq.process_module(experts, moe_fallback_threshold=0.005)

    for idx in unseen:
        torch.testing.assert_close(experts.gate_up_proj.data[idx], original[idx])


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ALLOW_LISTED_MODEL_TYPES)
@pytest.mark.usefixtures("_patched_calibration_dataset")
def test_gptq_moe_end_to_end(tmp_path: Path, model_type: str):
    """The real ``Gptq`` pass quantizes fused expert weights for every allow-listed model."""
    input_model = save_tiny_moe_model(tmp_path / "input_model", model_type)
    p = create_pass_from_dict(
        Gptq,
        {"bits": 4, "group_size": 16, "sym": True, "moe": True},
        disable_search=True,
    )
    out = p.run(input_model, str(tmp_path / "gptq"))
    assert isinstance(out, HfModelHandler)

    loaded = out.load_model()
    assert loaded.config.quantization_config.moe is True

    wrapper = ModelWrapper.from_model(loaded)
    moe_layers = 0
    for layer_wrapper in wrapper.get_layer_wrappers():
        experts = layer_wrapper.get_experts(return_name=False)
        if experts is None:
            continue
        moe_layers += 1
        for pname in ("gate_up_proj", "down_proj"):
            param = experts._parameters[pname]
            assert isinstance(param.data, QuantTensor), f"{pname} was not quantized"
            assert param.data.scales.dim() == 3
        # routers stay full precision
        router = layer_wrapper.get_router(return_name=False)
        assert not any(isinstance(p.data, QuantTensor) for p in router.parameters())
    assert moe_layers > 0

    # ``finalize()`` re-serializes config.json without the custom ``experts_implementation``
    # field, so re-pin ``eager`` for the forward (the default ``grouped_mm`` transposes the
    # fused weight, which a storage-only QuantTensor cannot do).
    loaded.set_experts_implementation("eager")
    loaded.eval()
    with torch.no_grad():
        logits = loaded(torch.randint(0, VOCAB_SIZE, (1, 8))).logits
    assert torch.isfinite(logits).all()


@pytest.mark.usefixtures("_patched_calibration_dataset")
def test_gptq_moe_disabled_leaves_experts_alone(tmp_path: Path):
    """``moe=False`` (the default) keeps fused expert weights in full precision."""
    input_model = save_tiny_moe_model(tmp_path / "input_model", "qwen3_moe")
    p = create_pass_from_dict(Gptq, {"bits": 4, "group_size": 16, "sym": True}, disable_search=True)
    out = p.run(input_model, str(tmp_path / "gptq"))

    loaded = out.load_model()
    experts = loaded.model.layers[0].mlp.experts
    assert not any(isinstance(param.data, QuantTensor) for param in experts.parameters())
    assert isinstance(loaded.model.layers[0].self_attn.q_proj.weight.data, QuantTensor)


@pytest.mark.usefixtures("_patched_calibration_dataset")
def test_gptq_moe_refuses_unsupported_architecture(tmp_path: Path, monkeypatch):
    """A non-allow-listed ``model_type`` fails closed with an actionable error."""
    input_model = save_tiny_moe_model(tmp_path / "input_model", "qwen3_moe")

    # pretend the model is a transposed-layout architecture; nothing else changes
    original = ModelWrapper.from_model

    def patched_from_model(model):
        wrapper = original(model)
        wrapper.model_type = "gpt_oss"
        return wrapper

    monkeypatch.setattr(ModelWrapper, "from_model", staticmethod(patched_from_model))

    p = create_pass_from_dict(
        Gptq, {"bits": 4, "group_size": 16, "sym": True, "moe": True}, disable_search=True
    )
    with pytest.raises(MoeCalibrationError, match="gpt_oss"):
        p.run(input_model, str(tmp_path / "gptq"))
