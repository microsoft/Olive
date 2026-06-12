# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.vitis_ai.vitis_generate_model_sd import VitisGenerateModelSD
from test.utils import ONNX_MODEL_PATH, get_onnx_model

pytest.importorskip("model_generate", reason="model_generate is not installed; skipping all SD model generation tests")

_PATCH_GEN = "model_generate.generate_model"


def _make_pass(**kwargs):
    cfg = {"model_type": "unet", "resolutions": [], **kwargs}
    return create_pass_from_dict(VitisGenerateModelSD, cfg, disable_search=True)


def _generate_writes_placeholder(**kwargs):
    """Mock generate_model: leave output Olive's _ensure_model_onnx can satisfy."""
    out = Path(kwargs["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "optimized.onnx").write_bytes(b"placeholder")


def test_run_raises_on_missing_model_generate(tmp_path):
    p = _make_pass()
    with (
        patch.dict("sys.modules", {"model_generate": None}),
        pytest.raises(
            ImportError,
            match="model_generate is required for VitisGenerateModelSD",
        ),
    ):
        p.run(get_onnx_model(), str(tmp_path / "out"))


def test_run_includes_resolutions_in_extra_options(tmp_path):
    gen = MagicMock(side_effect=_generate_writes_placeholder)
    with patch(_PATCH_GEN, gen):
        p = create_pass_from_dict(
            VitisGenerateModelSD,
            {
                "model_type": "unet",
                "resolutions": ["512x512", "768x768"],
            },
            disable_search=True,
        )
        p.run(get_onnx_model(), str(tmp_path / "sd_out"))

    gen.assert_called_once()
    kwargs = gen.call_args.kwargs
    assert kwargs["mode"] == "sd"
    assert kwargs["extra_options"]["model_type"] == "unet"
    assert kwargs["extra_options"]["resolutions"] == "512x512,768x768"


def test_run_default_resolutions_passed_when_using_defaults(tmp_path):
    gen = MagicMock(side_effect=_generate_writes_placeholder)
    with patch(_PATCH_GEN, gen):
        p = create_pass_from_dict(
            VitisGenerateModelSD,
            {"model_type": "unet"},
            disable_search=True,
        )
        p.run(get_onnx_model(), str(tmp_path / "out"))

    assert gen.call_args.kwargs["extra_options"].get("resolutions") == "512x512"


def test_run_omits_resolutions_when_empty_list(tmp_path):
    gen = MagicMock(side_effect=_generate_writes_placeholder)
    with patch(_PATCH_GEN, gen):
        p = create_pass_from_dict(
            VitisGenerateModelSD,
            {"model_type": "unet", "resolutions": []},
            disable_search=True,
        )
        p.run(get_onnx_model(), str(tmp_path / "out"))

    assert "resolutions" not in gen.call_args.kwargs["extra_options"]


def test_ensure_model_onnx_copies_optimized(tmp_path):
    def write_optimized(**kwargs):
        out = Path(kwargs["output_dir"])
        (out / "optimized.onnx").write_text("from_optimized", encoding="utf-8")

    gen = MagicMock(side_effect=write_optimized)
    with patch(_PATCH_GEN, gen):
        p = create_pass_from_dict(
            VitisGenerateModelSD,
            {"model_type": "unet", "resolutions": []},
            disable_search=True,
        )
        p.run(get_onnx_model(), str(tmp_path / "out"))

    assert (tmp_path / "out" / "model.onnx").read_text(encoding="utf-8") == "from_optimized"


def test_ensure_model_onnx_prefers_dd_replaced_over_optimized(tmp_path):
    def write_both(**kwargs):
        out = Path(kwargs["output_dir"])
        (out / "optimized.onnx").write_text("from_optimized", encoding="utf-8")
        dd = out / "dd"
        dd.mkdir(parents=True)
        (dd / "replaced.onnx").write_text("from_dd", encoding="utf-8")

    gen = MagicMock(side_effect=write_both)
    with patch(_PATCH_GEN, gen):
        p = create_pass_from_dict(
            VitisGenerateModelSD,
            {"model_type": "unet", "resolutions": []},
            disable_search=True,
        )
        p.run(get_onnx_model(), str(tmp_path / "out"))

    assert (tmp_path / "out" / "model.onnx").read_text(encoding="utf-8") == "from_dd"


def test_ensure_model_onnx_skips_copy_when_model_onnx_exists(tmp_path):
    def write_only_original(**kwargs):
        out = Path(kwargs["output_dir"])
        (out / "model.onnx").write_text("original", encoding="utf-8")
        (out / "optimized.onnx").write_text("optimized", encoding="utf-8")

    gen = MagicMock(side_effect=write_only_original)
    with patch(_PATCH_GEN, gen):
        p = create_pass_from_dict(
            VitisGenerateModelSD,
            {"model_type": "unet", "resolutions": []},
            disable_search=True,
        )
        p.run(get_onnx_model(), str(tmp_path / "out"))

    assert (tmp_path / "out" / "model.onnx").read_text(encoding="utf-8") == "original"


def test_ensure_model_onnx_raises_when_no_candidate_files(tmp_path):
    gen = MagicMock()
    with patch(_PATCH_GEN, gen):
        p = create_pass_from_dict(
            VitisGenerateModelSD,
            {"model_type": "unet", "resolutions": []},
            disable_search=True,
        )
        with pytest.raises(FileNotFoundError, match=r"No optimized\.onnx or dd/replaced\.onnx"):
            p.run(get_onnx_model(), str(tmp_path / "out"))


def test_resolve_onnx_input_path_single_file():
    p = _make_pass()
    h = ONNXModelHandler(model_path=str(ONNX_MODEL_PATH))
    assert p.resolve_onnx_input_path(h) == Path(ONNX_MODEL_PATH)


def test_resolve_onnx_input_path_dir_with_model_onnx(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    p = _make_pass()
    h = ONNXModelHandler(model_path=str(tmp_path))
    assert p.resolve_onnx_input_path(h) == tmp_path / "model.onnx"


def test_resolve_onnx_input_path_dir_with_onnx_file_name(tmp_path):
    (tmp_path / "custom.onnx").write_bytes(b"x")
    p = _make_pass()
    h = ONNXModelHandler(model_path=str(tmp_path), onnx_file_name="custom.onnx")
    assert p.resolve_onnx_input_path(h) == tmp_path / "custom.onnx"


def test_resolve_onnx_input_path_dir_onnx_file_name_missing_raises(tmp_path):
    p = _make_pass()
    h = SimpleNamespace(model_path=str(tmp_path), onnx_file_name="missing.onnx")
    with pytest.raises(FileNotFoundError, match="Specified onnx_file_name"):
        p.resolve_onnx_input_path(h)


def test_resolve_onnx_input_path_dir_single_unnamed_onnx(tmp_path):
    (tmp_path / "only.onnx").write_bytes(b"x")
    p = _make_pass()
    h = ONNXModelHandler(model_path=str(tmp_path))
    assert p.resolve_onnx_input_path(h) == tmp_path / "only.onnx"


def test_resolve_onnx_input_path_dir_multiple_onnx_raises(tmp_path):
    (tmp_path / "a.onnx").write_bytes(b"x")
    (tmp_path / "b.onnx").write_bytes(b"y")
    p = _make_pass()
    h = SimpleNamespace(model_path=str(tmp_path))
    with pytest.raises(ValueError, match=r"Multiple \.onnx model files found"):
        p.resolve_onnx_input_path(h)


def test_resolve_onnx_input_path_dir_no_onnx_raises(tmp_path):
    p = _make_pass()
    h = SimpleNamespace(model_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"No \.onnx file found"):
        p.resolve_onnx_input_path(h)


def test_resolve_onnx_input_path_missing_path_raises(tmp_path):
    p = _make_pass()
    missing = tmp_path / "nope"
    h = SimpleNamespace(model_path=str(missing))
    with pytest.raises(FileNotFoundError, match="Model path does not exist"):
        p.resolve_onnx_input_path(h)


def test_run_requires_onnx_model_handler(tmp_path):
    gen = MagicMock()
    with patch(_PATCH_GEN, gen):
        p = create_pass_from_dict(
            VitisGenerateModelSD,
            {"model_type": "unet", "resolutions": []},
            disable_search=True,
        )
        bad = MagicMock()
        with pytest.raises(TypeError, match="ONNXModelHandler"):
            p.run(bad, str(tmp_path / "out"))
