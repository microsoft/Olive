# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Invocation-local independent Q/K/V quantization on tiny offline models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from olive.common.quant.tensor import QuantTensor
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.autoclip import AutoClip
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


def test_only_native_quantizers_expose_independent_qkv():
    for pass_type in (Rtn, KQuant, Gptq):
        quantizer = create_pass_from_dict(pass_type, {}, disable_search=True)
        assert quantizer.config.independent_qkv is False
    assert "independent_qkv" not in AutoClip._default_config(None)
    with pytest.raises(ValidationError, match="independent_qkv"):
        create_pass_from_dict(Rtn, {"independent_qkv": "not-a-bool"}, disable_search=True)


@pytest.mark.parametrize("pass_type", [Rtn, KQuant, Gptq])
@pytest.mark.parametrize("flag", [None, False])
def test_default_still_promotes_fresh_qkv(tmp_path: Path, pass_type, flag):
    model = make_local_tiny_dense_llama(tmp_path / "input")
    options = {"bits": 4, "overrides": {V: {"bits": 8}}}
    if flag is not None:
        options["independent_qkv"] = flag
    path = tmp_path / "default"
    output = _run(pass_type, model, path, **options)
    _assert_qkv(output, path, (8, 8, 8))


@pytest.mark.parametrize(
    ("pass_type", "v_override"),
    [
        (Rtn, V),
        (Rtn, r"re:.*\.self_attn\.v_proj"),
        (KQuant, r"re:.*\.self_attn\.v_proj"),
        (Gptq, r"re:.*\.self_attn\.v_proj"),
    ],
)
def test_opt_in_materializes_independent_qkv(tmp_path: Path, pass_type, v_override):
    model = make_local_tiny_dense_llama(tmp_path / "input")
    path = tmp_path / "independent"
    output = _run(
        pass_type,
        model,
        path,
        bits=4,
        independent_qkv=True,
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
        independent_qkv=True,
        overrides={V: {"bits": 8}},
    )
    loaded = _assert_qkv(output, path, (4, 4, 8), group_size=-1)
    experts = loaded.model.layers[0].mlp.experts
    assert isinstance(experts.gate_up_proj.data, QuantTensor)
    assert experts.gate_up_proj.data.bits == 4
    assert isinstance(experts.down_proj.data, QuantTensor)
    assert experts.down_proj.data.bits == 4


@pytest.mark.parametrize("pass_type", [Rtn, KQuant])
@pytest.mark.parametrize("opt_in", [False, True])
def test_locked_v_from_checkpoint_defaults(tmp_path: Path, pass_type, opt_in):
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
        independent_qkv=opt_in,
    )
    loaded = _assert_qkv(second, path, (4, 4, 8) if opt_in else (8, 8, 8))
    assert loaded.get_submodule(V)._parameters["weight"].qweight.equal(original_v.qweight)


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
        independent_qkv=True,
        overrides={V: {"bits": 8}},
        modules_to_not_convert=[K],
    )
    first_loaded = first.load_model()
    q_weight = first_loaded.get_submodule(Q)._parameters["weight"].qweight.clone()
    v_weight = first_loaded.get_submodule(V)._parameters["weight"].qweight.clone()
    path = tmp_path / "second"
    second = _run(pass_type, first, path, bits=4, independent_qkv=True)
    loaded = _assert_qkv(second, path, (4, 4, 8))
    assert loaded.get_submodule(Q)._parameters["weight"].qweight.equal(q_weight)
    assert loaded.get_submodule(V)._parameters["weight"].qweight.equal(v_weight)


def test_independent_qkv_keeps_exclusions_and_other_quant_settings(tmp_path: Path):
    model = make_local_tiny_dense_llama(tmp_path / "input")
    path = tmp_path / "excluded"
    output = _run(
        Rtn,
        model,
        path,
        bits=4,
        sym=True,
        independent_qkv=True,
        overrides={V: {"bits": 8, "group_size": 32, "symmetric": False}},
        modules_to_not_convert=[K],
    )
    loaded = output.load_model()
    qcfg = loaded.config.quantization_config
    assert_packed_quant_module(loaded.get_submodule(Q), bits=4, group_size=16, symmetric=True)
    assert_packed_quant_module(loaded.get_submodule(V), bits=8, group_size=32, symmetric=False)
    assert_saved_quant_tensor_matches(path, V, loaded.get_submodule(V)._parameters["weight"])
    assert not hasattr(loaded.get_submodule(K)._parameters["weight"].data, "qweight")
    assert qcfg.get_qlinear_init_args(V) == {"bits": 8, "group_size": 32, "symmetric": False}
