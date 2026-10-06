# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Preserve per-projection Q/K/V quantization on tiny offline models."""

from pathlib import Path

import pytest

from olive.common.quant.tensor import QuantTensor
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.gptq import Gptq
from olive.passes.pytorch.kquant import KQuant
from olive.passes.pytorch.rtn import Rtn
from test.passes.pytorch.test_quantization_utils import (
    assert_packed_quant_module,
    assert_saved_quant_tensor_matches,
    make_local_calibration_data_config,
    make_local_tiny_dense_llama,
)
from test.passes.pytorch.test_rtn import _make_local_tiny_qwen3_moe

QKV = "model.layers.0.self_attn"
V = f"{QKV}.v_proj"
Q = f"{QKV}.q_proj"
K = f"{QKV}.k_proj"


def _run(pass_type, input_model, output_path: Path, **options):
    if pass_type is Gptq:
        options["data_config"] = make_local_calibration_data_config(seq_len=8, max_samples=2)
        options["desc_act"] = False
    return create_pass_from_dict(pass_type, {"group_size": 16, **options}, disable_search=True).run(
        input_model, str(output_path)
    )


def _assert_qkv(output, path: Path, expected: tuple[int, int, int], *, group_size=16, symmetric=False):
    loaded = output.load_model()
    qcfg = loaded.config.quantization_config
    assert "independent_qkv" not in qcfg.to_dict()
    for proj, bits in zip(("q_proj", "k_proj", "v_proj"), expected):
        name = f"{QKV}.{proj}"
        assert qcfg.get_qlinear_init_args(name) == {
            "bits": bits,
            "symmetric": symmetric,
            "group_size": group_size,
        }
        tensor = assert_packed_quant_module(
            loaded.get_submodule(name), bits=bits, group_size=group_size, symmetric=symmetric
        )
        assert_saved_quant_tensor_matches(path, name, tensor)
    return loaded


def test_quantizers_do_not_require_a_qkv_precision_flag():
    for pass_type in (Rtn, KQuant, Gptq):
        quantizer = create_pass_from_dict(pass_type, {}, disable_search=True)
        assert "independent_qkv" not in pass_type.default_config(None)
        assert not hasattr(quantizer.config, "independent_qkv")


@pytest.mark.parametrize("pass_type", [Rtn, KQuant, Gptq])
def test_default_preserves_fresh_qkv(tmp_path: Path, pass_type):
    model = make_local_tiny_dense_llama(tmp_path / "input")
    path = tmp_path / "default"
    output = _run(pass_type, model, path, bits=4, overrides={V: {"bits": 8}})
    _assert_qkv(output, path, (4, 4, 8))


@pytest.mark.parametrize(
    ("pass_type", "v_override"),
    [
        (Rtn, V),
        (Rtn, r"re:.*\.self_attn\.v_proj"),
        (KQuant, r"re:.*\.self_attn\.v_proj"),
        (Gptq, r"re:.*\.self_attn\.v_proj"),
    ],
)
def test_explicit_and_regex_overrides_materialize_independent_qkv(tmp_path: Path, pass_type, v_override):
    model = make_local_tiny_dense_llama(tmp_path / "input")
    path = tmp_path / "independent"
    output = _run(
        pass_type,
        model,
        path,
        bits=4,
        overrides={v_override: {"bits": 8}},
    )
    _assert_qkv(output, path, (4, 4, 8))


def test_qwen3_moe_independent_qkv_preserves_fused_experts(tmp_path: Path):
    model = _make_local_tiny_qwen3_moe(tmp_path / "input")
    path = tmp_path / "qwen3_moe"
    output = _run(
        Rtn,
        model,
        path,
        bits=4,
        group_size=-1,
        moe=True,
        overrides={V: {"bits": 8}},
    )
    loaded = _assert_qkv(output, path, (4, 4, 8), group_size=-1)
    experts = loaded.model.layers[0].mlp.experts
    assert isinstance(experts.gate_up_proj.data, QuantTensor)
    assert experts.gate_up_proj.data.bits == 4
    assert isinstance(experts.down_proj.data, QuantTensor)
    assert experts.down_proj.data.bits == 4


@pytest.mark.parametrize("pass_type", [Rtn, KQuant])
def test_locked_v_from_checkpoint_preserves_qk_defaults(tmp_path: Path, pass_type):
    model = make_local_tiny_dense_llama(tmp_path / "input")
    # V is physically INT8 but has no explicit checkpoint override.
    first_path = tmp_path / "first"
    first = _run(
        Rtn,
        model,
        first_path,
        bits=8,
        modules_to_not_convert=[Q, K],
    )
    original_v = _assert_v(first, first_path)
    path = tmp_path / "second"
    second = _run(
        pass_type,
        first,
        path,
        bits=4,
    )
    loaded = _assert_qkv(second, path, (4, 4, 8))
    assert loaded.get_submodule(V).weight.qweight.equal(original_v.qweight)


def _assert_v(output, path):
    loaded = output.load_model()
    assert V not in (loaded.config.quantization_config.overrides or {})
    tensor = assert_packed_quant_module(loaded.get_submodule(V), bits=8, group_size=16, symmetric=False)
    assert_saved_quant_tensor_matches(path, V, tensor)
    return tensor


@pytest.mark.parametrize("pass_type", [Rtn, KQuant])
def test_conflicting_locked_qv_and_excluded_k(tmp_path: Path, pass_type):
    model = make_local_tiny_dense_llama(tmp_path / "input")
    first_path = tmp_path / "first"
    first = _run(
        Rtn,
        model,
        first_path,
        bits=4,
        overrides={V: {"bits": 8}},
        modules_to_not_convert=[K],
    )
    first_loaded = first.load_model()
    q_weight = first_loaded.get_submodule(Q).weight.qweight.clone()
    v_weight = first_loaded.get_submodule(V).weight.qweight.clone()
    path = tmp_path / "second"
    second = _run(pass_type, first, path, bits=4)
    loaded = _assert_qkv(second, path, (4, 4, 8))
    assert loaded.get_submodule(Q).weight.qweight.equal(q_weight)
    assert loaded.get_submodule(V).weight.qweight.equal(v_weight)


def test_independent_qkv_keeps_exclusions_and_other_quant_settings(tmp_path: Path):
    model = make_local_tiny_dense_llama(tmp_path / "input")
    path = tmp_path / "excluded"
    output = _run(
        Rtn,
        model,
        path,
        bits=4,
        sym=True,
        overrides={V: {"bits": 8, "group_size": 32, "symmetric": False}},
        modules_to_not_convert=[K],
    )
    loaded = output.load_model()
    qcfg = loaded.config.quantization_config
    assert_packed_quant_module(loaded.get_submodule(Q), bits=4, group_size=16, symmetric=True)
    v_tensor = assert_packed_quant_module(loaded.get_submodule(V), bits=8, group_size=32, symmetric=False)
    assert_saved_quant_tensor_matches(path, V, v_tensor)
    assert not isinstance(loaded.get_submodule(K).weight, QuantTensor)
    assert qcfg.get_qlinear_init_args(V) == {"bits": 8, "group_size": 32, "symmetric": False}
