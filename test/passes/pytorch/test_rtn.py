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
from olive.passes.pytorch.rtn import Rtn
from test.utils import get_tiny_phi3


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
