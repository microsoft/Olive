# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
from pathlib import Path

import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.tensor import QuantTensor
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.gptq import Gptq
from olive.passes.pytorch.moe_support import MoeSupportError
from olive.passes.pytorch.quant_utils import prepare_model
from olive.passes.pytorch.rtn import Rtn
from test.utils import get_tiny_phi3


def _save_trivial_tokenizer(save_path: Path, vocab_size: int) -> None:
    """Save a local tokenizer so pass metadata serialization never needs the hub."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab_size)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", pad_token="t0").save_pretrained(save_path)


def _is_quant(module: torch.nn.Module) -> bool:
    if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        return False
    weight = module._parameters.get("weight")
    return weight is not None and isinstance(weight.data, QuantTensor)


def _bits(module: torch.nn.Module) -> int:
    return module.weight.data.bits


def _make_local_tiny_mixtral(save_path):
    """Save a local copy of ``yujiepan/mixtral-tiny-random`` forced to the ``eager`` experts implementation.

    The hub model defaults to ``grouped_mm`` for its MoE forward, which hits a CUDA kernel
    stride-alignment limitation on this architecture's tiny (non-16-byte-aligned) hidden
    dims and is unrelated to Olive's quantization code; forcing ``eager`` avoids it while
    keeping this a real HF model / real Gptq+Rtn pass run.
    """
    import json

    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained("yujiepan/mixtral-tiny-random")
    model.save_pretrained(save_path)
    AutoTokenizer.from_pretrained("yujiepan/mixtral-tiny-random").save_pretrained(save_path)

    config_path = save_path / "config.json"
    config = json.loads(config_path.read_text())
    config["experts_implementation"] = "eager"
    config_path.write_text(json.dumps(config, indent=2))

    return HfModelHandler(model_path=str(save_path))


def _make_local_tiny_qwen3_moe(save_path) -> HfModelHandler:
    """Save a tiny K-last fused-experts model without downloading a checkpoint."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    config = Qwen3MoeConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        head_dim=8,
        experts_implementation="eager",
    )
    Qwen3MoeForCausalLM(config).save_pretrained(save_path)
    _save_trivial_tokenizer(save_path, config.vocab_size)
    return HfModelHandler(model_path=str(save_path))


def test_rtn_moe_refuses_transposed_layout_before_finalize(tmp_path: Path, monkeypatch):
    """A transposed-layout (``is_transposed=True``) experts module fails before finalize.

    The rejection must happen before quantization or output serialization.
    """
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")

    def patched_prepare_model(*args, **kwargs):
        wrapper, qcfg, retie = prepare_model(*args, **kwargs)
        for layer in wrapper.get_layer_wrappers():
            experts = layer.get_experts(return_name=False)
            if experts is not None:
                experts.is_transposed = True
        return wrapper, qcfg, retie

    def unexpected_finalize(*args, **kwargs):
        pytest.fail("finalize must not run after MoE layout support is rejected")

    monkeypatch.setattr("olive.passes.pytorch.rtn.prepare_model", patched_prepare_model)
    monkeypatch.setattr("olive.passes.pytorch.rtn.finalize", unexpected_finalize)

    quantizer = create_pass_from_dict(Rtn, {"moe": True}, disable_search=True)
    output_path = tmp_path / "rtn"
    with pytest.raises(MoeSupportError, match=r"\(E, K, OUT\).*moe=False"):
        quantizer.run(input_model, str(output_path))
    assert not output_path.exists()


def test_rtn_moe_false_does_not_run_layout_gate(tmp_path: Path, monkeypatch):
    """The default dense path must not invoke fused-experts layout validation."""
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")

    def unexpected_gate(*args, **kwargs):
        pytest.fail("MoE layout support check must not run when moe=False")

    monkeypatch.setattr("olive.passes.pytorch.rtn.check_moe_layout_support", unexpected_gate)
    quantizer = create_pass_from_dict(Rtn, {"moe": False, "group_size": -1}, disable_search=True)
    out = quantizer.run(input_model, str(tmp_path / "rtn"))

    loaded = out.load_model()
    experts = loaded.model.layers[0].mlp.experts
    assert not any(isinstance(param.data, QuantTensor) for param in experts.parameters())
    assert isinstance(loaded.model.layers[0].self_attn.q_proj.weight.data, QuantTensor)


def test_rtn_moe_gate_ignores_prior_checkpoint_moe_flag(tmp_path: Path, monkeypatch):
    """Regression test: a second RTN pass with moe=False must not re-run the layout gate.

    ``prepare_model`` ORs a pre-existing checkpoint's ``moe`` flag into the merged
    ``qcfg.moe`` (see ``quant_utils.prepare_model``), so gating on ``qcfg.moe`` would make
    this second, moe=False invocation incorrectly re-run fused-experts layout validation
    -- something this run never asked for. The gate must key off this invocation's own
    ``config.moe`` request instead.
    """
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")
    first_pass = create_pass_from_dict(Rtn, {"moe": True, "group_size": -1}, disable_search=True)
    quantized = first_pass.run(input_model, str(tmp_path / "rtn_first"))

    def unexpected_gate(*args, **kwargs):
        pytest.fail("MoE layout support check must not re-run when this invocation requests moe=False")

    monkeypatch.setattr("olive.passes.pytorch.rtn.check_moe_layout_support", unexpected_gate)
    second_pass = create_pass_from_dict(Rtn, {"moe": False, "lm_head": True, "group_size": -1}, disable_search=True)
    out = second_pass.run(quantized, str(tmp_path / "rtn_second"))

    loaded = out.load_model()
    assert isinstance(loaded.lm_head.weight.data, QuantTensor)


def test_rtn_moe_k_last_layout_quantizes_experts(tmp_path: Path):
    """A K-last (``is_transposed=False``) fused-experts model quantizes successfully end to end."""
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")
    quantizer = create_pass_from_dict(Rtn, {"moe": True, "group_size": -1}, disable_search=True)
    out = quantizer.run(input_model, str(tmp_path / "rtn"))

    loaded = out.load_model()
    experts = loaded.model.layers[0].mlp.experts
    assert any(isinstance(param.data, QuantTensor) for param in experts.parameters())


@pytest.mark.parametrize("group_size", [-1, 16])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("lm_head", [True, False])
def test_gptq(tmp_path: Path, group_size: int, sym: bool, lm_head: bool):
    # setup
    input_model = get_tiny_phi3()
    p = create_pass_from_dict(
        Rtn,
        {
            "bits": 4,
            "group_size": group_size,
            "lm_head": lm_head,
            "sym": sym,
            "overrides": {"model.layers.0.self_attn.o_proj": {"bits": 8}},
        },
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    out = p.run(input_model, gptq_out_folder)

    # assert
    assert isinstance(out, HfModelHandler)
    loaded_model = out.load_model()
    assert loaded_model.__class__.__name__ == "Phi3ForCausalLM"
    assert hasattr(loaded_model, "quantization_method")
    assert loaded_model.quantization_method == "olive"
    assert hasattr(loaded_model.config, "quantization_config")
    assert isinstance(loaded_model.config.quantization_config, OliveHfQuantizationConfig)
    assert loaded_model.config.quantization_config.group_size == group_size
    assert not any(isinstance(m, torch.nn.Linear) and not _is_quant(m) for m in loaded_model.model.layers.modules())
    assert _is_quant(loaded_model.model.layers[0].self_attn.o_proj)
    assert _bits(loaded_model.model.layers[0].self_attn.o_proj) == 8
    assert _bits(loaded_model.model.layers[0].mlp.down_proj) == 4
    assert loaded_model.config.quantization_config.lm_head == lm_head
    assert _is_quant(loaded_model.lm_head) == lm_head
    assert isinstance(loaded_model.model.embed_tokens, torch.nn.Embedding)
    assert not _is_quant(loaded_model.model.embed_tokens)

    # compose another rtn pass on top of the partially quantized model
    p2 = create_pass_from_dict(
        Rtn,
        {
            "bits": 8,
            "group_size": group_size,
            "lm_head": True,
            "embeds": True,
            "sym": sym,
        },
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder_2 = str(tmp_path / "gptq2")
    out2 = p2.run(out, gptq_out_folder_2)

    # assert
    assert isinstance(out2, HfModelHandler)
    loaded_model_2 = out2.load_model()
    # check that the embed tokens layer is quantized to 8 bits
    assert _is_quant(loaded_model_2.model.embed_tokens)
    assert _bits(loaded_model_2.model.embed_tokens) == 8
    # check that the lm head is quantized to 8 bits if it was not quantized before
    assert _is_quant(loaded_model_2.lm_head)
    assert _bits(loaded_model_2.lm_head) == (4 if lm_head else 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Gptq requires a CUDA-capable GPU")
def test_gptq_then_rtn_moe_e2e(tmp_path: Path):
    """M5: real end-to-end ``Gptq`` -> ``Rtn(moe=True, embeds=True)`` composition.

    Runs the actual ``Gptq`` pass (calibration-based) on a small, real MoE model
    (``yujiepan/mixtral-tiny-random``), then runs the actual ``Rtn`` pass with
    ``moe=True, embeds=True`` on top. Verifies that:

    * the ``nn.Linear`` attention/MLP-router weights that GPTQ already quantized are
      *not* re-quantized by RTN (still the original GPTQ ``QuantTensor``), and
    * the fused-3D MoE expert weights and the input embeddings -- which GPTQ does not
      touch -- get RTN-quantized.
    * a save -> reload round-trip preserves both sets of quantized weights.
    """
    input_model = _make_local_tiny_mixtral(tmp_path / "tiny_mixtral")

    gptq_pass = create_pass_from_dict(
        Gptq,
        {"group_size": -1, "lm_head": False},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")
    gptq_out = gptq_pass.run(input_model, gptq_out_folder)
    assert isinstance(gptq_out, HfModelHandler)

    gptq_loaded = gptq_out.load_model()
    # GPTQ quantizes attention/MLP nn.Linear weights, not the MoE fused-3D expert params.
    assert _is_quant(gptq_loaded.model.layers[0].self_attn.q_proj)
    router_gate = gptq_loaded.model.layers[0].mlp.gate
    assert not _is_quant(router_gate)  # router (MixtralTopKRouter, not nn.Linear) stays full precision
    experts = gptq_loaded.model.layers[0].mlp.experts
    assert not any(isinstance(p.data, QuantTensor) for p in experts.parameters())
    assert not _is_quant(gptq_loaded.model.embed_tokens)
    # snapshot the GPTQ-quantized q_proj weight for later comparison
    gptq_qproj_before = gptq_loaded.model.layers[0].self_attn.q_proj.weight.data.to_dense().clone()
    gptq_qproj_bits = _bits(gptq_loaded.model.layers[0].self_attn.q_proj)

    rtn_pass = create_pass_from_dict(
        Rtn,
        {"bits": 4, "group_size": -1, "moe": True, "embeds": True, "lm_head": True},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    rtn_out_folder = str(tmp_path / "rtn")
    rtn_out = rtn_pass.run(gptq_out, rtn_out_folder)
    assert isinstance(rtn_out, HfModelHandler)

    rtn_loaded = rtn_out.load_model()

    # The GPTQ-quantized Linear is untouched by RTN (same bits, same dequantized values).
    q_proj = rtn_loaded.model.layers[0].self_attn.q_proj
    assert _is_quant(q_proj)
    assert _bits(q_proj) == gptq_qproj_bits
    torch.testing.assert_close(q_proj.weight.data.to_dense(), gptq_qproj_before, rtol=0, atol=0)

    # MoE experts and embeddings, which GPTQ left untouched, are now RTN-quantized.
    rtn_experts = rtn_loaded.model.layers[0].mlp.experts
    assert any(isinstance(p.data, QuantTensor) for p in rtn_experts.parameters())
    assert _is_quant(rtn_loaded.model.embed_tokens)
    assert _bits(rtn_loaded.model.embed_tokens) == 4

    # Save -> reload round-trip preserves both GPTQ and RTN quantization.
    reload_folder = str(tmp_path / "reload")
    rtn_loaded.save_pretrained(reload_folder)
    reloaded_handler = HfModelHandler(model_path=reload_folder)
    reloaded = reloaded_handler.load_model()

    reloaded_q_proj = reloaded.model.layers[0].self_attn.q_proj
    assert _is_quant(reloaded_q_proj)
    assert _bits(reloaded_q_proj) == gptq_qproj_bits
    torch.testing.assert_close(reloaded_q_proj.weight.data.to_dense(), gptq_qproj_before, rtol=0, atol=0)

    reloaded_experts = reloaded.model.layers[0].mlp.experts
    assert any(isinstance(p.data, QuantTensor) for p in reloaded_experts.parameters())
    assert _is_quant(reloaded.model.embed_tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Rtn(moe=True) real forward needs a CUDA-capable GPU")
def test_rtn_moe_real_forward_after_reload(tmp_path: Path):
    """Regression test for a save/reload round-trip bug in fused-3D MoE quantized weights.

    ``transformers``'s ``save_pretrained`` defaults to ``save_original_format=True``,
    which for Mixtral-family MoE architectures round-trips the on-disk state dict
    through a *legacy* per-expert ``nn.Linear``-shaped layout (splitting the fused-3D
    ``experts.gate_up_proj``/``down_proj`` into ``experts.{i}.w1/w2/w3.weight`` and back
    on load). That reshape machinery assumes plain float weight tensors and silently
    drops the trailing group-size dimension of our quantized ``_scales``/``_qzeros``
    buffers, which crashes real ``forward()`` calls on the reloaded model (previously
    undetected because no existing test called ``forward()`` on a save/reload round trip).

    This test quantizes a real MoE model with ``Rtn(moe=True)``, saves via the actual
    pass output (exercising ``finalize()``'s ``save_pretrained`` call), reloads from
    disk, and calls the model's real ``forward()`` -- asserting no crash, no ``NaN``,
    and that the fused-3D scales buffer keeps its group-size dimension.
    """
    input_model = _make_local_tiny_mixtral(tmp_path / "tiny_mixtral")

    rtn_pass = create_pass_from_dict(
        Rtn,
        {"bits": 4, "group_size": -1, "moe": True, "embeds": True, "lm_head": True},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    rtn_out_folder = str(tmp_path / "rtn")
    rtn_out = rtn_pass.run(input_model, rtn_out_folder)
    assert isinstance(rtn_out, HfModelHandler)

    # ``finalize()`` re-serializes config.json and does not preserve custom,
    # non-quantization-related fields; re-patch ``experts_implementation`` so the
    # reloaded model uses the ``eager`` MoE forward (see ``_make_local_tiny_mixtral``).
    import json

    config_path = Path(rtn_out_folder) / "config.json"
    config = json.loads(config_path.read_text())
    config["experts_implementation"] = "eager"
    config_path.write_text(json.dumps(config, indent=2))

    reloaded_handler = HfModelHandler(model_path=rtn_out_folder)
    reloaded = reloaded_handler.load_model()

    # Guard the exact regression: the fused-3D expert weight's scales/qzeros must keep
    # their trailing group-size dimension (num_experts, out_features, num_groups) after
    # the disk round trip, instead of collapsing to a 2D (num_experts, out_features).
    gate_up_proj = reloaded.model.layers[0].mlp.experts.gate_up_proj
    assert _is_quant(reloaded.model.embed_tokens)  # sanity: quantization actually applied
    assert isinstance(gate_up_proj.data, QuantTensor)
    scales_shape = gate_up_proj.data.scales.shape
    assert len(scales_shape) == 3, f"expected fused-3D scales, got shape {scales_shape}"
    assert scales_shape[:2] == gate_up_proj.data.qweight.shape[:2]

    reloaded = reloaded.cuda().eval()
    input_ids = torch.randint(0, 100, (1, 8), device="cuda")
    with torch.no_grad():
        out = reloaded(input_ids)
    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()


def _make_local_tiny_tied_llama(save_path) -> HfModelHandler:
    """Save a tiny ``LlamaForCausalLM`` with ``tie_word_embeddings=True`` (built locally, no hub access)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    LlamaForCausalLM(config).save_pretrained(save_path)
    _save_trivial_tokenizer(save_path, config.vocab_size)
    return HfModelHandler(model_path=str(save_path))


def _load_quant_tensor_from_disk(model_dir, pname: str, sym: bool, group_size: int) -> QuantTensor:
    """Rebuild the ``QuantTensor`` for ``pname`` straight from the saved safetensors shard.

    Gives a device-independent, bit-exact reference for what ``from_pretrained`` must load
    (recomputing the quantization in-process would not match bit-for-bit, since the pass
    quantizes on GPU when one is available).
    """
    from safetensors.torch import load_file

    from olive.common.quant.state_dict import buffer_names

    qname, sname, zname = buffer_names(pname)
    weights: dict = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        weights.update(load_file(shard))
    qweight, scales = weights[qname], weights[sname]
    return QuantTensor.from_packed(
        qweight=qweight,
        scales=scales,
        qzeros=weights.get(zname),
        bits=4,
        group_size=group_size,
        symmetric=sym,
        shape=(scales.shape[0], qweight.shape[-1] * 2),
        dtype=scales.dtype,
    )


@pytest.mark.parametrize("sym", [True, False])
def test_rtn_tied_word_embeddings_roundtrip(tmp_path: Path, sym: bool):
    """Regression: tied ``lm_head`` / ``embed_tokens`` must survive a save -> reload round trip.

    ``tie_quant_word_embeddings`` installs the *same* ``QuantTensor`` object on both modules
    and aliases ``lm_head``'s buffer dict entries to ``embed_tokens``'s buffer *objects* — a
    one-time snapshot. HF's loader then replaces ``embed_tokens``'s buffer objects, leaving
    ``lm_head``'s entries stale. ``refresh_quant_tensor_refs`` used to rebind the shared
    QuantTensor once per hosting module (last-write-wins over ``named_modules()`` order), so
    the stale ``lm_head`` buffers could win and silently zero/garbage the reloaded weight
    while ``is_placeholder`` still read ``False``.
    """
    group_size = 16
    input_model = _make_local_tiny_tied_llama(tmp_path / "tiny_llama_tied")
    original_embed = input_model.load_model().model.embed_tokens.weight.detach().clone()

    p = create_pass_from_dict(
        Rtn,
        {"bits": 4, "group_size": group_size, "sym": sym, "lm_head": True, "embeds": True},
        disable_search=True,
    )
    out = p.run(input_model, str(tmp_path / "quantized"))
    assert isinstance(out, HfModelHandler)

    loaded = out.load_model()
    assert loaded.config.quantization_config.tie_word_embeddings is True

    embed, lm_head = loaded.model.embed_tokens, loaded.lm_head
    assert _is_quant(embed)
    assert _is_quant(lm_head)
    # Tying must be preserved end to end: same QuantTensor object and same buffer objects.
    # NOTE: read the parameter out of ``_parameters`` -- ``param.data`` on a tensor subclass
    # goes through ``detach()`` and returns a *fresh* QuantTensor whose inner tensors are
    # views, so it cannot be used for identity assertions.
    shared = embed._parameters["weight"]
    assert shared is lm_head._parameters["weight"]
    assert shared.qweight is embed._buffers["weight_qweight"]
    assert shared.scales is embed._buffers["weight_scales"]
    assert embed._buffers["weight_qweight"] is lm_head._buffers["weight_qweight"]
    assert embed._buffers["weight_scales"] is lm_head._buffers["weight_scales"]
    assert shared.is_placeholder is False

    # The reloaded weight must be bit-identical to what was written to disk -- not zeros,
    # not stale placeholder data.
    disk = _load_quant_tensor_from_disk(tmp_path / "quantized", "model.embed_tokens.weight", sym, group_size)
    assert torch.equal(shared.qweight, disk.qweight)
    assert torch.equal(shared.scales, disk.scales)
    expected = disk.to_dense()

    embed_dense = shared.to_dense()
    lm_head_dense = lm_head._parameters["weight"].to_dense()
    assert torch.isfinite(embed_dense).all()
    assert embed_dense.abs().sum() > 0
    torch.testing.assert_close(embed_dense, expected, rtol=0, atol=0)
    torch.testing.assert_close(lm_head_dense, expected, rtol=0, atol=0)
    # ... and it still approximates the original float weight within quantization error.
    torch.testing.assert_close(embed_dense, original_embed, rtol=0, atol=float(disk.scales.max()))

    # A real forward pass on the reloaded model produces finite logits.
    loaded.eval()
    with torch.no_grad():
        logits = loaded(torch.randint(0, 32, (1, 8))).logits
    assert torch.isfinite(logits).all()

    # And a second save -> reload round trip is stable (the alias buffers written to disk
    # agree with the live ones).
    resave_path = tmp_path / "resaved"
    loaded.save_pretrained(resave_path, save_original_format=False)
    input_model.save_metadata(str(resave_path))
    resaved = HfModelHandler(model_path=str(resave_path)).load_model()
    torch.testing.assert_close(resaved.model.embed_tokens._parameters["weight"].to_dense(), expected, rtol=0, atol=0)
    torch.testing.assert_close(resaved.lm_head._parameters["weight"].to_dense(), expected, rtol=0, atol=0)
