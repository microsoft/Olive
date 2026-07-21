# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Numerical-parity tests for QuantTensor-backed modules.

Confirms that ``nn.Linear`` / ``nn.Embedding`` / fused-3D MoE host modules
whose weight has been swapped to a ``QuantTensor`` produce **bit-exact**
outputs vs the same forward run against ``QuantTensor.to_dense()`` (the
canonical eager dequant), and that the ONNX-export wrappers built by
``make_export_compatible_quant`` round-trip through onnxruntime with
matching outputs.
"""

from __future__ import annotations

import copy

import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from olive.common.hf.quant import make_export_compatible_quant
from olive.common.quant.state_dict import install_quant_tensor_param
from olive.common.quant.tensor import QuantTensor


def _quantize_inplace(module: nn.Module, pname: str, *, bits: int, group_size: int, symmetric: bool) -> None:
    param = module._parameters[pname]
    qt = QuantTensor.from_float(
        param.data.detach().clone(),
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    install_quant_tensor_param(module, pname, qt)


def _dense_reference(model: nn.Module) -> nn.Module:
    """Return a deep copy of ``model`` with every QuantTensor weight materialized."""
    ref = copy.deepcopy(model)
    for module in ref.modules():
        weight = module._parameters.get("weight")
        if weight is None:
            continue
        if isinstance(weight.data, QuantTensor):
            module._parameters["weight"] = nn.Parameter(weight.data.to_dense(), requires_grad=False)
    return ref


@pytest.mark.parametrize("bits,group_size,symmetric", [(4, 32, False), (4, -1, True), (8, 32, False)])
def test_full_model_forward_parity(bits, group_size, symmetric):
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(32, 64)
            self.fc1 = nn.Linear(64, 128, bias=False)
            self.fc2 = nn.Linear(128, 64, bias=True)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.silu(self.fc1(self.embed(ids))))

    model = Toy().eval()
    ids = torch.randint(0, 32, (2, 16))
    ref_out = model(ids)

    for sub_module in (model.embed, model.fc1, model.fc2):
        _quantize_inplace(sub_module, "weight", bits=bits, group_size=group_size, symmetric=symmetric)

    quant_out = model(ids)
    dense_out = _dense_reference(model)(ids)

    # vs canonical dequant: must be bit-exact (same kernels, same data)
    torch.testing.assert_close(quant_out, dense_out, rtol=0, atol=0)
    # vs original fp: just a sanity bound on quant error
    assert (quant_out - ref_out).abs().mean().item() < 1.0


@pytest.mark.parametrize("bits,group_size,symmetric", [(4, 16, False), (8, -1, True)])
def test_fused_moe_forward_parity(bits, group_size, symmetric):
    """Fused 3D-expert forward: index into a QuantTensor 3D weight per expert."""
    torch.manual_seed(0)
    num_experts, in_dim, hidden = 4, 32, 64

    class FusedMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up = nn.Parameter(torch.randn(num_experts, hidden, in_dim))
            self.down = nn.Parameter(torch.randn(num_experts, in_dim, hidden))

        def forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
            # x: (tokens, in_dim); expert_ids: (tokens,) routing
            outs = []
            for t in range(x.shape[0]):
                e = int(expert_ids[t].item())
                h = F.linear(x[t : t + 1], self.gate_up[e])  # (1, hidden)
                outs.append(F.linear(F.silu(h), self.down[e]))  # (1, in_dim)
            return torch.cat(outs, dim=0)

    model = FusedMoE().eval()
    x = torch.randn(8, in_dim)
    expert_ids = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3])
    ref_out = model(x, expert_ids)

    # Quantize the 3D expert tensors directly via QuantTensor.from_float.
    for pname in ("gate_up", "down"):
        _quantize_inplace(model, pname, bits=bits, group_size=group_size, symmetric=symmetric)

    quant_out = model(x, expert_ids)

    # Reference: same forward against dense-materialized 3D weights.
    dense_model = FusedMoE()
    with torch.no_grad():
        dense_model.gate_up = nn.Parameter(model.gate_up.data.to_dense(), requires_grad=False)
        dense_model.down = nn.Parameter(model.down.data.to_dense(), requires_grad=False)
    dense_out = dense_model(x, expert_ids)

    torch.testing.assert_close(quant_out, dense_out, rtol=0, atol=0)
    assert (quant_out - ref_out).abs().mean().item() < 5.0


@pytest.mark.parametrize("bits,group_size,symmetric", [(4, 32, False), (4, 32, True), (8, 32, False)])
def test_onnx_export_parity(tmp_path, bits, group_size, symmetric):
    """Olive-quantized nn.Linear -> ONNX MatMulNBits -> onnxruntime matches eager."""
    torch.manual_seed(0)
    in_dim, out_dim = 64, 32

    class LinearOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = LinearOnly().eval()
    _quantize_inplace(model.fc, "weight", bits=bits, group_size=group_size, symmetric=symmetric)

    x = torch.randn(1, 4, in_dim)
    eager_out = model(x).detach()

    # Build an export-compatible variant (QuantLinearNbit wrapper) and export to ONNX.
    export_model = make_export_compatible_quant(copy.deepcopy(model), dynamo=False)

    onnx_path = tmp_path / "model.onnx"
    torch.onnx.export(
        export_model,
        (x,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
        opset_version=21,
        dynamo=False,
    )

    onnx.checker.check_model(str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["output"], {"input": x.numpy()})[0]

    torch.testing.assert_close(torch.from_numpy(ort_out), eager_out, rtol=1e-2, atol=1e-2)


def _quantize_model_targets(model, *, moe, embeds, lm_head, bits=4, group_size=32, symmetric=False):
    """Quantize every ``iter_quant_targets`` selection in place with a QuantTensor."""
    from olive.common.quant.selection import iter_quant_targets

    for module, pname, _ in list(
        iter_quant_targets(
            model,
            quantize_lm_head=lm_head,
            quantize_embeds=embeds,
            quantize_moe=moe,
        )
    ):
        _quantize_inplace(module, pname, bits=bits, group_size=group_size, symmetric=symmetric)


def test_real_hf_mixtral_moe_round_trip():
    """Round-trip a real (config-only, no weight download) HF MoE architecture.

    Builds a tiny ``MixtralForCausalLM`` from config, quantizes it with ``moe=True`` (the
    fused 3D expert weights plus embeddings / lm_head), then saves and reloads the state
    dict and verifies every quantized weight dequantizes bit-identically after reload.

    Note: the model's own ``grouped_mm`` expert forward is not exercised here — Olive's
    storage-only MoE quantization dispatches ``F.linear`` / ``F.embedding`` and does not
    implement fused ``grouped_mm``; ONNX export / execution of the experts is delegated to
    ORT GenAI ModelBuilder / Mobius. This test therefore validates the buffer-backed
    save/reload path (the actual round-trip contract) on a real architecture.
    """
    transformers = pytest.importorskip("transformers")

    from olive.common.quant.state_dict import refresh_quant_tensor_refs

    def build():
        torch.manual_seed(0)
        cfg = transformers.MixtralConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        return transformers.MixtralForCausalLM(cfg).eval()

    model = build()
    _quantize_model_targets(model, moe=True, embeds=True, lm_head=True)

    # every fused expert weight must now be a QuantTensor; biases (if any) stay dense.
    experts = model.model.layers[0].mlp.experts
    assert isinstance(experts._parameters["gate_up_proj"].data, QuantTensor)
    assert isinstance(experts._parameters["down_proj"].data, QuantTensor)

    # Snapshot the dequantized value of every quantized parameter.
    def dequant_snapshot(m):
        snap = {}
        for name, sub in m.named_modules():
            for pname, param in sub._parameters.items():
                if param is not None and isinstance(param.data, QuantTensor):
                    snap[f"{name}.{pname}"] = param.data.to_dense().clone()
        return snap

    before = dequant_snapshot(model)
    assert any("experts" in k for k in before), "expected at least one quantized expert weight"

    # Save state dict (QuantTensor params dropped; only plain buffers persisted) then reload
    # onto a freshly quantized model and re-verify each dequantized weight is bit-identical.
    state = model.state_dict()
    assert not any(isinstance(v, QuantTensor) for v in state.values())

    reloaded = build()
    _quantize_model_targets(reloaded, moe=True, embeds=True, lm_head=True)
    _missing, unexpected = reloaded.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys on reload: {unexpected}"
    refresh_quant_tensor_refs(reloaded)

    after = dequant_snapshot(reloaded)
    assert after.keys() == before.keys()
    for key, ref in before.items():
        torch.testing.assert_close(after[key], ref, rtol=0, atol=0)
