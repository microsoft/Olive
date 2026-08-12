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
import threading
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
    OLIVE_MOE_CALIB_IMPLEMENTATION,
    MoeCalibrationError,
    MoeCalibrationSession,
    _register_calib_implementation,
    check_moe_gptq_support,
)
from olive.passes.pytorch.quant_utils import QuantInfo

# Representative K-last architectures covered by the end-to-end tests.
TESTED_MOE_MODEL_TYPES = [
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
    """Build a tiny, randomly-initialised model for one tested MoE architecture."""
    # pylint: disable=unexpected-keyword-arg
    # HF config ``__init__``s are not statically resolvable by astroid (every kwarg below is
    # a real, documented config field); ``test_rtn.py`` suppresses the same false positive.
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
        "olmoe": lambda: tf.OlmoeForCausalLM(tf.OlmoeConfig(**common, num_experts=NUM_EXPERTS, num_experts_per_tok=2)),
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

    def fake_calibration_dataset(model, data_config=None, **kwargs):
        generator = torch.Generator().manual_seed(0)
        return [
            {
                "input_ids": torch.randint(0, VOCAB_SIZE, (1, 16), generator=generator),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }
            for _ in range(3)
        ]

    # ``monkeypatch.setattr`` with a dotted string target resolves the module internally,
    # so this file never needs its own ``import olive.passes.pytorch.quant_utils`` alongside
    # the top-level ``from olive.passes.pytorch.quant_utils import QuantInfo``.
    monkeypatch.setattr("olive.passes.pytorch.quant_utils.get_calibration_dataset", fake_calibration_dataset)


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


@pytest.mark.parametrize("model_type", TESTED_MOE_MODEL_TYPES)
def test_layer_wrapper_resolves_experts_and_router(model_type: str):
    """Every tested architecture must resolve its experts + router submodules.

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


@pytest.mark.parametrize("model_type", TESTED_MOE_MODEL_TYPES)
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
            iter_quant_targets(model, quantize_lm_head=True, quantize_embeds=True, quantize_moe=quantize_moe)
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
    assert routers
    assert all(isinstance(router, torch.nn.Linear) for router in routers)


# ---------------------------------------------------------------------------
# support gating (fail closed)
# ---------------------------------------------------------------------------


class _FakeExperts(torch.nn.Module):
    def __init__(self, **flags):
        super().__init__()
        self.config = object()
        for key, value in flags.items():
            setattr(self, key, value)


class _ShapeOnly:
    """Minimal stand-in for a fused expert parameter: only ``shape`` is needed."""

    def __init__(self, shape):
        self.shape = shape


def make_fake_experts(class_name: str = "Qwen3MoeExperts", **flags) -> torch.nn.Module:
    """Build a stand-in experts module whose *class name* is ``class_name``."""
    return type(class_name, (_FakeExperts,), {})(**flags)


def test_check_support_accepts_model_type_outside_old_allow_list():
    check_moe_gptq_support(
        "some_future_moe_architecture",
        [make_fake_experts("FutureExperts", is_transposed=False, has_bias=False, has_gate=True)],
    )


def test_check_support_rejects_missing_capability():
    """No ``is_transposed`` attribute => transformers too old / class not decorated."""
    with pytest.raises(MoeCalibrationError, match="is_transposed"):
        check_moe_gptq_support("qwen3_moe", [make_fake_experts()])


def test_check_support_rejects_transposed_layout():
    with pytest.raises(MoeCalibrationError, match="transposed fused-weight layout"):
        check_moe_gptq_support("qwen3_moe", [make_fake_experts(is_transposed=True)])


@pytest.mark.parametrize("bad_value", [None, 0, 1, "False", "", torch.tensor(False)])
def test_check_support_rejects_non_bool_is_transposed(bad_value):
    with pytest.raises(MoeCalibrationError, match="cannot verify"):
        check_moe_gptq_support("qwen3_moe", [make_fake_experts(is_transposed=bad_value)])


def test_check_support_rejects_old_transformers(monkeypatch):
    monkeypatch.setattr("olive.passes.pytorch.moe_calib._transformers_supports_experts_registry", lambda: False)
    with pytest.raises(MoeCalibrationError, match="requires transformers >="):
        check_moe_gptq_support("qwen3_moe", [make_fake_experts(is_transposed=False)])


def test_check_support_rejects_module_list_experts():
    experts = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
    with pytest.raises(MoeCalibrationError, match="cannot intercept") as exc_info:
        check_moe_gptq_support("classic_moe", [experts])
    assert "moe=False" in str(exc_info.value)
    assert "Rtn pass" in str(exc_info.value)


def test_check_support_rejects_expert_bias():
    experts = make_fake_experts(is_transposed=False, has_bias=True, has_gate=True)
    with pytest.raises(MoeCalibrationError, match="expert biases"):
        check_moe_gptq_support("qwen3_moe", [experts])


def test_check_support_rejects_non_gated_experts():
    experts = make_fake_experts(is_transposed=False, has_bias=False, has_gate=False)
    with pytest.raises(MoeCalibrationError, match="non-gated experts"):
        check_moe_gptq_support("qwen3_moe", [experts])


def test_hessian_memory_preflight_warns_for_large_configs():
    """A DeepSeek-V3-scale config must warn about Hessian memory before calibration starts."""
    experts = make_fake_experts("DeepseekV3Experts", is_transposed=False, has_bias=False, has_gate=True)
    experts.num_experts = 256
    # shapes only: (num_experts, out, in) with hidden=7168 and moe_intermediate=2048
    experts.gate_up_proj = _ShapeOnly((256, 4096, 7168))
    experts.down_proj = _ShapeOnly((256, 7168, 2048))

    with capture_logs("olive.passes.pytorch.moe_calib") as records:
        check_moe_gptq_support("deepseek_v3", [experts])

    messages = "\n".join(records)
    assert "Hessians" in messages
    # 256 * (7168^2 + 2048^2) * 4 bytes ~= 53.0 GiB
    assert "53." in messages


def test_hessian_memory_preflight_silent_for_small_configs():
    """Tiny models must not emit the memory warning."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    with capture_logs("olive.passes.pytorch.moe_calib") as records:
        assert MoeCalibrationSession.create(wrapper) is not None
    assert not any("Hessians" in message for message in records)


def test_session_create_returns_none_for_dense_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2
    )
    assert MoeCalibrationSession.create(ModelWrapper.from_model(LlamaForCausalLM(config))) is None


# ---------------------------------------------------------------------------
# per-expert recording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", TESTED_MOE_MODEL_TYPES)
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
        with session.record([experts]), torch.no_grad():
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
        with session.record([experts]), torch.no_grad():
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
            with session.record([experts]), torch.no_grad():
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


def test_token_counts_do_not_depend_on_gate_up_proj_being_quantized():
    """Coverage must stay correct when only ``down_proj`` carries ``quant_info``.

    ``token_counts`` is derived from the recorded Hessian sample counts, so excluding
    ``gate_up_proj`` (e.g. via ``modules_to_not_convert``) must not report every expert as
    unseen.
    """
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    session = MoeCalibrationSession.create(wrapper)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    experts._parameters["down_proj"].quant_info = QuantInfo(
        quantizer=WeightQuantizer(bits=4, symmetric=True, group_size=16)
    )

    session.start()
    try:
        with session.record([experts]), torch.no_grad():
            model(torch.randint(0, VOCAB_SIZE, (2, 16)))
    finally:
        session.finish()

    data = experts.down_proj.quant_info.data
    assert not hasattr(experts.gate_up_proj, "quant_info")
    assert sum(data["token_counts"]) == 32 * 2
    for expert_idx, entry in data["experts"].items():
        assert entry["N"] == data["token_counts"][expert_idx]


# ---------------------------------------------------------------------------
# session lifecycle
# ---------------------------------------------------------------------------


def test_start_is_not_re_entrant():
    """A second ``start()`` would clobber the saved implementation, so it must be refused."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    original_implementation = model.get_experts_implementation()
    session = MoeCalibrationSession.create(wrapper)

    session.start()
    try:
        with pytest.raises(MoeCalibrationError, match="already active"):
            session.start()
    finally:
        session.finish()

    assert model.get_experts_implementation() == original_implementation


def test_second_session_on_the_same_model_is_refused():
    """Two overlapping sessions would restore each other's calibration implementation."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    original_implementation = model.get_experts_implementation()
    first = MoeCalibrationSession.create(wrapper)
    second = MoeCalibrationSession.create(wrapper)

    first.start()
    try:
        with pytest.raises(MoeCalibrationError, match="another MoE calibration session"):
            second.start()
    finally:
        first.finish()

    assert model.get_experts_implementation() == original_implementation


def test_concurrent_sessions_cannot_both_claim_the_same_model(monkeypatch):
    """A session reserves the model before swapping, closing the start() check/set race."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    first = MoeCalibrationSession.create(wrapper)
    second = MoeCalibrationSession.create(wrapper)
    original_setter = model.set_experts_implementation
    setter_entered = threading.Event()
    release_setter = threading.Event()
    block_lock = threading.Lock()
    should_block = True

    def blocking_setter(implementation):
        nonlocal should_block
        with block_lock:
            block_this_call = implementation == OLIVE_MOE_CALIB_IMPLEMENTATION and should_block
            if block_this_call:
                should_block = False
        if block_this_call:
            setter_entered.set()
            assert release_setter.wait(timeout=10)
        return original_setter(implementation)

    monkeypatch.setattr(model, "set_experts_implementation", blocking_setter)
    first_errors = []

    def start_first():
        try:
            first.start()
        except BaseException as exc:  # pragma: no cover - asserted below
            first_errors.append(exc)

    thread = threading.Thread(target=start_first)
    thread.start()
    assert setter_entered.wait(timeout=10)
    try:
        with pytest.raises(MoeCalibrationError, match="another MoE calibration session"):
            second.start()
    finally:
        release_setter.set()
        thread.join(timeout=10)
        if first._active:
            first.finish()

    assert not thread.is_alive()
    assert first_errors == []


def test_record_rejects_nested_context_for_same_experts():
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    session = MoeCalibrationSession.create(wrapper)
    experts = session.experts_modules[0]

    session.start()
    try:
        with session.record([experts]):
            with pytest.raises(MoeCalibrationError, match=r"nested or concurrent record\(\) contexts"):
                with session.record([experts]):
                    pass
    finally:
        session.finish()


def test_record_requires_active_session():
    model = build_tiny_moe_model("qwen3_moe")
    session = MoeCalibrationSession.create(ModelWrapper.from_model(model))

    with pytest.raises(MoeCalibrationError, match=r"record\(\) requires an active session"):
        with session.record([session.experts_modules[0]]):
            pass


def test_start_restores_the_model_when_the_swap_is_stale():
    """A failed swap validation must not leave the model in calibration mode."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    original_implementation = model.get_experts_implementation()
    session = MoeCalibrationSession.create(wrapper)

    # an experts module whose config never picks up the swap => stale
    unswitchable = make_fake_experts(is_transposed=False)
    unswitchable.config = type("_Config", (), {"_experts_implementation": "eager"})()
    session.experts_modules = [*session.experts_modules, unswitchable]

    with pytest.raises(MoeCalibrationError, match="could not switch the experts implementation"):
        session.start()

    assert model.get_experts_implementation() == original_implementation
    assert session._active is False


def test_start_reports_stale_swap_when_restore_fails(monkeypatch):
    """A restore failure must not mask the actionable stale-swap error."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    original_implementation = model.get_experts_implementation()
    original_setter = model.set_experts_implementation
    session = MoeCalibrationSession.create(wrapper)

    unswitchable = make_fake_experts(is_transposed=False)
    unswitchable.config = type("_Config", (), {"_experts_implementation": "eager"})()
    session.experts_modules = [*session.experts_modules, unswitchable]

    def setter_with_failed_restore(implementation):
        if implementation == original_implementation:
            raise RuntimeError("restore failed")
        return original_setter(implementation)

    monkeypatch.setattr(model, "set_experts_implementation", setter_with_failed_restore)
    try:
        with (
            capture_logs("olive.passes.pytorch.moe_calib") as records,
            pytest.raises(MoeCalibrationError, match="restoring the original experts implementation also failed"),
        ):
            session.start()
    finally:
        original_setter(original_implementation)

    assert any("Failed to restore the original experts implementation" in message for message in records)
    assert session._active is False


def test_registry_key_collision_is_detected(monkeypatch):
    """A foreign function under Olive's registry key must fail closed, not run silently."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    key = OLIVE_MOE_CALIB_IMPLEMENTATION
    previous = ALL_EXPERTS_FUNCTIONS.get(key)

    def foreign_experts_forward(*args, **kwargs):
        raise AssertionError("this must never be called")

    ALL_EXPERTS_FUNCTIONS.register(key, foreign_experts_forward)
    try:
        with pytest.raises(MoeCalibrationError, match="already maps"):
            _register_calib_implementation()
    finally:
        if previous is None:
            del ALL_EXPERTS_FUNCTIONS[key]
        else:
            ALL_EXPERTS_FUNCTIONS.register(key, previous)


@pytest.mark.usefixtures("_patched_calibration_dataset")
def test_calibration_state_is_restored_when_processing_raises(tmp_path: Path, monkeypatch):
    """An exception mid-calibration must still restore ``experts_implementation`` + ``use_cache``.

    Both are process-global mutations owned by ``run_layerwise_quantization``; leaking them
    would silently corrupt any later use of the same in-memory model.
    """
    import importlib

    gptq_module = importlib.import_module("olive.passes.pytorch.gptq")

    input_model = save_tiny_moe_model(tmp_path / "input_model", "qwen3_moe")

    captured = {}
    original_prepare_model = gptq_module.prepare_model

    def spy_prepare_model(model, config):
        result = original_prepare_model(model, config)
        captured["wrapper"] = result[0]
        captured["experts_implementation"] = result[0].model.config._experts_implementation
        captured["use_cache"] = result[0].model.config.use_cache
        return result

    monkeypatch.setattr(gptq_module, "prepare_model", spy_prepare_model)

    calls = []
    original_process_module = Gptq.process_module

    def flaky_process_module(module, **kwargs):
        calls.append(module)
        if len(calls) == 2:
            raise RuntimeError("injected calibration failure")
        return original_process_module(module, **kwargs)

    monkeypatch.setattr(Gptq, "process_module", staticmethod(flaky_process_module))

    p = create_pass_from_dict(Gptq, {"bits": 4, "group_size": 16, "sym": True, "moe": True}, disable_search=True)
    with pytest.raises(RuntimeError, match="injected calibration failure"):
        p.run(input_model, str(tmp_path / "gptq"))

    config = captured["wrapper"].model.config
    assert config._experts_implementation == captured["experts_implementation"] == "eager"
    assert config.use_cache == captured["use_cache"]


# ---------------------------------------------------------------------------
# LayerWrapper projection accessors
# ---------------------------------------------------------------------------


def test_get_attention_inputs_is_strict_by_default():
    """Positional consumers (rotate.py) must get an error, not a silently shorter list.

    DeepSeek-V3's MLA has no ``k_proj``/``v_proj``; with the strict default the missing
    projections raise instead of shifting ``v_proj`` to another index.
    """
    wrapper = ModelWrapper.from_model(build_tiny_moe_model("deepseek_v3"))
    layer_wrapper = wrapper.get_layer_wrappers()[0]

    with pytest.raises(AttributeError):
        layer_wrapper.get_attention_inputs()

    partial = layer_wrapper.get_attention_inputs(return_name=False, partial_ok=True)
    assert len(partial) == 1  # q_proj only


def test_get_attention_inputs_keeps_qkv_order_for_standard_attention():
    """The positional contract (index 2 == ``v_proj``) still holds for normal attention."""
    wrapper = ModelWrapper.from_model(build_tiny_moe_model("qwen3_moe"))
    _, names = wrapper.get_layer_wrappers()[0].get_attention_inputs()
    assert [name.rsplit(".", 1)[-1] for name in names] == ["q_proj", "k_proj", "v_proj"]


@pytest.mark.parametrize("model_type", TESTED_MOE_MODEL_TYPES)
def test_get_mlp_projections_require_explicit_partial_mode_for_moe_layers(model_type: str):
    """Strict consumers must fail rather than silently skip an MoE layer's projections."""
    wrapper = ModelWrapper.from_model(build_tiny_moe_model(model_type))
    moe_layers = 0
    for layer_wrapper in wrapper.get_layer_wrappers():
        if layer_wrapper.get_experts(return_name=False) is None:
            continue
        moe_layers += 1
        with pytest.raises(AttributeError):
            layer_wrapper.get_mlp_inputs(return_name=False)
        with pytest.raises(AttributeError):
            layer_wrapper.get_mlp_outputs(return_name=False)
        assert layer_wrapper.get_mlp_inputs(return_name=False, partial_ok=True) == []
        assert layer_wrapper.get_mlp_outputs(return_name=False, partial_ok=True) == []
    assert moe_layers > 0


# ---------------------------------------------------------------------------
# RTN fallback
# ---------------------------------------------------------------------------


def _run_recording(model, wrapper, experts, fallback_threshold: float):
    session = MoeCalibrationSession.create(wrapper, fallback_threshold=fallback_threshold)
    session.start()
    try:
        with session.record([experts]), torch.no_grad():
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
    # RTN derives qparams straight from the float weight, and the weight is left
    # fake-quantized (on-grid) so the true-sequential re-run sees the same invariant as the
    # GPTQ experts.
    expected_scales, expected_zp = quantizer.find_qparams(original.float())
    expected_weight = quantizer.fake_quantize(original.float(), expected_scales, expected_zp)
    torch.testing.assert_close(experts.gate_up_proj.data, expected_weight.to(original.dtype))
    torch.testing.assert_close(experts.gate_up_proj.quant_info.scales, expected_scales.cpu())
    torch.testing.assert_close(experts.gate_up_proj.quant_info.zero_points, expected_zp.cpu())


def test_gptq_path_used_when_above_threshold():
    """When both thresholds are satisfied, every routed expert is GPTQ-quantized (not RTN)."""
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    _attach_quant_info(experts)
    _run_recording(model, wrapper, experts, fallback_threshold=0.005)

    original = experts.gate_up_proj.data.clone()
    # min_k_multiple=0.0 isolates this test to the skew condition alone (this test's
    # purpose): every routed expert must clear the *default* skew threshold on this tiny
    # fixture's calibration size, but not necessarily the *default* sufficiency threshold
    # (2.0x K) -- that condition is covered separately below.
    with capture_logs("olive.passes.pytorch.gptq") as records:
        Gptq.process_module(experts, moe_fallback_threshold=0.005, moe_fallback_min_k_multiple=0.0)

    assert not any("quantized with RTN" in message for message in records)
    assert not torch.equal(experts.gate_up_proj.data, original)
    num_groups = original.shape[-1] // 16
    assert experts.gate_up_proj.quant_info.scales.shape == (NUM_EXPERTS, original.shape[1], num_groups)
    assert experts.gate_up_proj.quant_info.data is None


def test_rtn_fallback_when_below_sufficiency_threshold_even_if_skew_passes():
    """Sufficiency (min_k_multiple) must independently gate fallback, even when skew passes.

    Regression for the dual-condition OR-gate: with a wide-open skew threshold (0.0, so it
    never fires) but a large ``moe_fallback_min_k_multiple``, every expert must still fall
    back to RTN once its observed token count is below ``min_k_multiple * K`` -- this is the
    condition added in the dual fallback-threshold commit, and it previously had zero direct
    test coverage.
    """
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    _attach_quant_info(experts)
    # skew_threshold = 0.0 * tokens_seen = 0, so it can never fail; sufficiency_threshold =
    # 1000.0 * K is unreachable by this tiny fixture's calibration set, so every expert must
    # fail on sufficiency alone.
    _run_recording(model, wrapper, experts, fallback_threshold=0.0)

    original = experts.gate_up_proj.data.clone()
    quantizer = experts.gate_up_proj.quant_info.quantizer
    with capture_logs("olive.passes.pytorch.gptq") as records:
        Gptq.process_module(experts, moe_fallback_threshold=0.0, moe_fallback_min_k_multiple=1000.0)

    assert any("quantized with RTN" in message for message in records)
    assert any(f"{NUM_EXPERTS}/{NUM_EXPERTS} experts quantized with RTN" in message for message in records)
    expected_scales, expected_zp = quantizer.find_qparams(original.float())
    expected_weight = quantizer.fake_quantize(original.float(), expected_scales, expected_zp)
    torch.testing.assert_close(experts.gate_up_proj.data, expected_weight.to(original.dtype))


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
    quantizer = experts.gate_up_proj.quant_info.quantizer

    Gptq.process_module(experts, moe_fallback_threshold=0.005)

    for idx in unseen:
        # no Hessian => RTN fallback: the weight is the plain fake-quantized original
        expected = quantizer.fake_quantize(original[idx].float())
        torch.testing.assert_close(experts.gate_up_proj.data[idx], expected.to(original.dtype))


def test_rtn_fallback_weights_are_on_grid_before_true_sequential_rerun():
    """Fallback experts must be fake-quantized (not left float) after ``process_module``.

    The true-sequential loop re-runs the layer after ``process_module``; if fallback experts
    still held raw float weights there, the next layer would be calibrated against a
    higher-precision layer than the one that is actually saved.
    """
    model = build_tiny_moe_model("qwen3_moe")
    wrapper = ModelWrapper.from_model(model)
    experts = wrapper.get_layer_wrappers()[0].get_experts(return_name=False)
    _attach_quant_info(experts)
    _run_recording(model, wrapper, experts, fallback_threshold=1.0)  # forces every expert to fall back

    quantizer = experts.gate_up_proj.quant_info.quantizer
    Gptq.process_module(experts, moe_fallback_threshold=1.0)

    info = experts.gate_up_proj.quant_info
    weight = experts.gate_up_proj.data.float()
    # already on the quantization grid => re-applying the recorded qparams is a no-op,
    # which is exactly what ``finalize`` does when it serializes the weight
    on_grid = quantizer.fake_quantize(weight, info.scales.to(weight.device), info.zero_points.to(weight.device))
    torch.testing.assert_close(on_grid, weight)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", TESTED_MOE_MODEL_TYPES)
@pytest.mark.usefixtures("_patched_calibration_dataset")
def test_gptq_moe_end_to_end(tmp_path: Path, model_type: str):
    """The real ``Gptq`` pass quantizes fused expert weights for every tested model."""
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
def test_gptq_moe_accepts_model_type_outside_old_allow_list(tmp_path: Path, monkeypatch):
    """A K-last architecture is accepted without a ``model_type`` allow-list entry."""
    input_model = save_tiny_moe_model(tmp_path / "input_model", "qwen3_moe")

    # Pretend the K-last model is a future architecture; only its diagnostic name changes.
    original = ModelWrapper.from_model

    def patched_from_model(model):
        wrapper = original(model)
        wrapper.model_type = "some_future_moe_architecture"
        return wrapper

    monkeypatch.setattr(ModelWrapper, "from_model", staticmethod(patched_from_model))

    p = create_pass_from_dict(Gptq, {"bits": 4, "group_size": 16, "sym": True, "moe": True}, disable_search=True)
    out = p.run(input_model, str(tmp_path / "gptq"))
    assert isinstance(out, HfModelHandler)
