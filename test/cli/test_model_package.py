# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
"""Tests for ``olive generate-model-package``.

Covers both the CLI argument-parsing / source-validation surface and the
underlying writer (``write_model_package`` and helpers); they live in the
same module (``olive.cli.model_package``).
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from olive.cli.model_package import (
    ModelPackageCommand,
    VariantSpec,
    disambiguate_variant_names,
    parse_compatibility_strings,
    write_model_package,
)

# ---------------------------------------------------------------------------
# ONNX fixture helpers
# ---------------------------------------------------------------------------


def _make_onnx_inline(onnx_path: Path, metadata_props: dict[str, str] | None = None) -> Path:
    """Write a minimal ONNX file with no external data."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    init = helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])
    node = helper.make_node("Identity", inputs=["weight"], outputs=["y"])
    graph = helper.make_graph([node], "test", inputs=[], outputs=[output], initializer=[init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    if metadata_props:
        for k, v in metadata_props.items():
            entry = model.metadata_props.add()
            entry.key = k
            entry.value = v
    onnx.save(model, str(onnx_path))
    return onnx_path


def _make_onnx_with_external(
    onnx_path: Path,
    blob_relpath: str,
    blob_bytes: bytes,
    metadata_props: dict[str, str] | None = None,
) -> Path:
    """Write a minimal ONNX file whose only initializer points at an external-data blob."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path = onnx_path.parent / blob_relpath
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(blob_bytes)

    init = TensorProto()
    init.name = "weight"
    init.data_type = TensorProto.FLOAT
    init.dims.extend([max(1, len(blob_bytes) // 4)])
    init.data_location = TensorProto.EXTERNAL
    for k, v in (("location", blob_relpath), ("offset", "0"), ("length", str(len(blob_bytes)))):
        entry = init.external_data.add()
        entry.key = k
        entry.value = v

    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])
    node = helper.make_node("Identity", inputs=["weight"], outputs=["y"])
    graph = helper.make_graph([node], "test", inputs=[], outputs=[output], initializer=[init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    if metadata_props:
        for k, v in metadata_props.items():
            entry = model.metadata_props.add()
            entry.key = k
            entry.value = v
    onnx.save(model, str(onnx_path))
    return onnx_path


def _create_source_dir(
    tmp_path: Path,
    name: str,
    *,
    ep: str = "CPUExecutionProvider",
    onnx_metadata: dict[str, str] | None = None,
    filename: str = "model.onnx",
    provider_options: dict | None = None,
    session_options_extras: dict | None = None,
    role: str = "decoder",
) -> Path:
    """Create a fake GenAI-shaped source directory.

    Writes a minimal ``genai_config.json`` describing one role (default
    ``decoder``) with ``filename``, plus a real ONNX file at the role's
    filename. Optionally seeds the role's ``session_options.provider_options``
    with the canonical alias for the supplied ``ep`` so the packager's
    EP-derivation logic resolves the variant to that EP. No
    ``model_config.json`` is written — the packager is genai_config-driven.
    """
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)
    onnx_path = source_dir / filename
    _make_onnx_inline(onnx_path, metadata_props=onnx_metadata)

    ep_to_alias = {
        "CPUExecutionProvider": "CPU",
        "CUDAExecutionProvider": "cuda",
        "QNNExecutionProvider": "qnn",
        "OpenVINOExecutionProvider": "OpenVINO",
        "VitisAIExecutionProvider": "VitisAI",
        "WebGpuExecutionProvider": "WebGPU",
        "DmlExecutionProvider": "DML",
        "TensorrtExecutionProvider": "tensorrt",
        "ROCMExecutionProvider": "rocm",
        "CoreMLExecutionProvider": "CoreML",
        "XnnpackExecutionProvider": "XNNPACK",
    }
    alias = ep_to_alias.get(ep, "CPU")
    session_options: dict = dict(session_options_extras or {})
    if alias == "CPU":
        session_options.setdefault("provider_options", [])
    else:
        session_options.setdefault("provider_options", [{alias: provider_options or {}}])

    genai = {
        "model": {
            role: {"filename": filename, "session_options": session_options},
        }
    }
    (source_dir / "genai_config.json").write_text(json.dumps(genai))
    return source_dir


def _make_command(args_list):
    """Create a ModelPackageCommand instance from CLI args."""
    parser = ArgumentParser()
    commands_parser = parser.add_subparsers()
    ModelPackageCommand.register_subcommand(commands_parser)
    parsed_args, unknown = parser.parse_known_args(args_list)
    return parsed_args.func(parser, parsed_args, unknown)


# ---------------------------------------------------------------------------
# CLI: source validation
# ---------------------------------------------------------------------------


class TestSourceValidation:
    def test_accepts_single_source(self, tmp_path):
        src = _create_source_dir(tmp_path, "soc_60", ep="QNNExecutionProvider")
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(tmp_path / "out")])

        sources = cmd._parse_sources()

        assert sources == [("soc_60", src)]

    def test_rejects_missing_genai_config(self, tmp_path):
        """A source without ``genai_config.json`` is rejected.

        The packager is genai_config-driven: it lifts the model layout
        (role filenames, session_options, pipeline) directly from the
        source's genai_config. A directory lacking that file has no way to
        describe its contents to the packager.
        """
        no_config = tmp_path / "no_config"
        no_config.mkdir()
        valid = _create_source_dir(tmp_path, "valid", ep="QNNExecutionProvider")
        cmd = _make_command(
            ["generate-model-package", "-s", str(no_config), "-s", str(valid), "-o", str(tmp_path / "out")]
        )

        with pytest.raises(ValueError, match=r"genai_config\.json"):
            cmd._parse_sources()

    def test_rejects_nonexistent_path(self, tmp_path):
        valid = _create_source_dir(tmp_path, "valid", ep="QNNExecutionProvider")
        cmd = _make_command(
            ["generate-model-package", "-s", "/nonexistent/path", "-s", str(valid), "-o", str(tmp_path / "out")]
        )

        with pytest.raises(ValueError, match="does not exist"):
            cmd._parse_sources()

    def test_rejects_duplicate_source_basenames(self, tmp_path):
        # Two source dirs share basename "soc_60" — variant names would collide.
        src_a = _create_source_dir(tmp_path / "a", "soc_60", ep="QNNExecutionProvider")
        src_b = _create_source_dir(tmp_path / "b", "soc_60", ep="QNNExecutionProvider")
        cmd = _make_command(["generate-model-package", "-s", str(src_a), "-s", str(src_b), "-o", str(tmp_path / "out")])

        with pytest.raises(ValueError, match="share the directory name"):
            cmd._parse_sources()

    def test_parses_two_valid_sources(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "soc_60", ep="QNNExecutionProvider")
        src2 = _create_source_dir(tmp_path, "soc_73", ep="QNNExecutionProvider")
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])

        sources = cmd._parse_sources()

        assert len(sources) == 2
        assert sources[0] == ("soc_60", src1)
        assert sources[1] == ("soc_73", src2)


# ---------------------------------------------------------------------------
# CLI: end-to-end (single component, multi-variant)
# ---------------------------------------------------------------------------


class TestGeneratePackageMultiVariant:
    def test_writes_proposal_layout(self, tmp_path):
        # setup
        src1 = _create_source_dir(tmp_path, "soc_60", ep="QNNExecutionProvider")
        src2 = _create_source_dir(tmp_path, "soc_73", ep="QNNExecutionProvider")
        out = tmp_path / "out.ortpackage"
        cmd = _make_command(
            [
                "generate-model-package",
                "-s",
                str(src1),
                "-s",
                str(src2),
                "-o",
                str(out),
                "--model_name",
                "test_model",
                "--model_version",
                "2.0",
            ]
        )

        # execute
        cmd.run()

        # assert: top-level manifest + components under models/
        assert (out / "manifest.json").is_file()
        assert (out / "models").is_dir()

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["schema_version"] == 1
        # ``decoder`` (not ``model``) — the genai_config role is ``decoder``,
        # so _extract_task -> ``text_generation`` -> component dir ``decoder``.
        assert manifest["components"] == ["decoder"]
        assert manifest["producer"]["model_name"] == "test_model"
        assert manifest["producer"]["model_version"] == "2.0"

        # metadata uses inline EP
        metadata = json.loads((out / "models" / "decoder" / "metadata.json").read_text())
        assert metadata["schema_version"] == 1
        assert metadata["component_name"] == "decoder"
        assert set(metadata["variants"]) == {"soc_60", "soc_73"}
        for variant_payload in metadata["variants"].values():
            assert variant_payload == {"ep": "QNNExecutionProvider"}

        # No variant.json is emitted; the ONNX file lands in the variant
        # directory.
        for v in ("soc_60", "soc_73"):
            assert not (out / "models" / "decoder" / v / "variant.json").exists()
            assert (out / "models" / "decoder" / v / "model.onnx").is_file()


class TestGeneratePackageSingleSource:
    def test_single_source_is_valid_package(self, tmp_path):
        src = _create_source_dir(tmp_path, "cpu_x64", ep="CPUExecutionProvider")
        out = tmp_path / "out.ortpackage"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["components"] == ["decoder"]
        metadata = json.loads((out / "models" / "decoder" / "metadata.json").read_text())
        assert "cpu_x64" in metadata["variants"]
        assert metadata["variants"]["cpu_x64"] == {"ep": "CPUExecutionProvider"}
        # No shared_weights because nothing to dedup.
        assert not (out / "models" / "decoder" / "shared_weights").exists()


# ---------------------------------------------------------------------------
# Writer: layout + manifest + metadata
# ---------------------------------------------------------------------------


class TestWriteModelPackageLayout:
    def test_writes_proposal_shape_for_single_variant(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                    device="cpu",
                )
            ],
            producer_info={"tool": "olive-ai", "model_name": "demo"},
        )

        assert (out / "manifest.json").is_file()
        assert (out / "models" / "decoder" / "metadata.json").is_file()
        # No variant.json is emitted.
        assert not (out / "models" / "decoder" / "cpu" / "variant.json").exists()
        assert (out / "models" / "decoder" / "cpu" / "model.onnx").is_file()

    def test_manifest_uses_proposal_schema(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                )
            ],
            producer_info={"tool": "olive-ai", "tool_version": "1.2.3", "model_name": "demo"},
        )

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["schema_version"] == 1
        assert manifest["components"] == ["decoder"]
        assert manifest["package_name"] == "package"
        assert manifest["package_version"] == "1.0"
        assert manifest["configs_dir"] == "configs"
        assert manifest["producer"] == {
            "tool": "olive-ai",
            "tool_version": "1.2.3",
            "model_name": "demo",
        }
        # No legacy fields
        assert "name" not in manifest
        assert "component_models" not in manifest
        assert "model_version" not in manifest

    def test_metadata_uses_inline_ep(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="qnn-npu",
                    onnx_files=[onnx_path],
                    ep="QNNExecutionProvider",
                    device="NPU",
                    compatibility_string="soc_60,soc_69",
                )
            ],
        )

        metadata = json.loads((out / "models" / "decoder" / "metadata.json").read_text())
        assert metadata["schema_version"] == 1
        assert metadata["component_name"] == "decoder"
        assert metadata["variants"]["qnn-npu"] == {
            "ep": "QNNExecutionProvider",
            "device": "NPU",
            "compatibility_string": "soc_60,soc_69",
        }
        assert "model_variants" not in metadata

    def test_metadata_omits_optional_fields_when_unset(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                )
            ],
        )

        metadata = json.loads((out / "models" / "decoder" / "metadata.json").read_text())
        assert metadata["variants"]["cpu"] == {"ep": "CPUExecutionProvider"}

    def test_overlay_carries_session_and_provider_options(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        inference = {
            "session_options": {"graph_optimization_level": 3},
            "execution_provider": ["CUDAExecutionProvider"],
            "provider_options": [{"device_id": "0"}],
        }

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cuda",
                    onnx_files=[onnx_path],
                    ep="CUDAExecutionProvider",
                    inference_settings=inference,
                )
            ],
        )

        # Runtime fields go to genai_config_overlay.json, not variant.json.
        assert not (out / "models" / "decoder" / "cuda" / "variant.json").exists()
        overlay = json.loads((out / "models" / "decoder" / "cuda" / "genai_config_overlay.json").read_text())
        assert overlay == {
            "model": {
                "decoder": {
                    "filename": "model.onnx",
                    "session_options": {
                        "graph_optimization_level": 3,
                        "provider_options": [{"cuda": {"device_id": "0"}}],
                    },
                }
            }
        }

    def test_overlay_provider_options_match_ep_by_name(self, tmp_path):
        """When inference_settings has multiple EPs, pick the one whose name matches VariantSpec.ep."""
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        inference = {
            "session_options": {},
            "execution_provider": ["CPUExecutionProvider", "QNNExecutionProvider"],
            "provider_options": [{"cpu_only": "1"}, {"backend_path": "QnnHtp.so"}],
        }

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="qnn",
                    onnx_files=[onnx_path],
                    ep="QNNExecutionProvider",
                    inference_settings=inference,
                )
            ],
        )

        overlay = json.loads((out / "models" / "decoder" / "qnn" / "genai_config_overlay.json").read_text())
        assert overlay["model"]["decoder"]["session_options"]["provider_options"] == [
            {"qnn": {"backend_path": "QnnHtp.so"}}
        ]

    def test_overlay_emits_empty_provider_options_for_cpu(self, tmp_path):
        """CPU variants emit ``provider_options: []`` rather than a sentinel entry.

        ``[{"CPU": {}}]`` is not needed: ORT-GenAI's dispatch table has no CPU
        handler (src/models/session_options.cpp), and ORT InferenceSession
        implicitly registers the CPU EP when no other provider is selected
        (onnxruntime/core/session/inference_session.cc), so the explicit entry
        would only trigger a V1 no-op registration. An empty list matches the
        convention used by reference ORT model packages.
        """
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                )
            ],
        )

        overlay = json.loads((out / "models" / "decoder" / "cpu" / "genai_config_overlay.json").read_text())
        assert overlay == {
            "model": {
                "decoder": {
                    "filename": "model.onnx",
                    "session_options": {"provider_options": []},
                }
            }
        }

    def test_overlay_lifts_per_variant_model_level_fields(self, tmp_path):
        """Per-variant ``context_length`` (and similar) flows from source to overlay.

        Each variant's source ``genai_config.json`` is the source of truth for
        model-level scalars that legitimately vary across variants of the same
        model (e.g. an NPU build caps ``context_length`` while the GPU build
        does not). The writer strips these from the base config and re-supplies
        them per variant; without this lift the merged config would silently
        use whichever variant's base happened to win.
        """
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        npu_source_genai = {
            "model": {
                "type": "phi3",
                "context_length": 4224,
                "pad_token_id": 200020,
                "eos_token_id": [200020, 199999],
                "bos_token_id": 199999,
                "vocab_size": 200064,
                "decoder": {"head_size": 128, "filename": "model.onnx", "session_options": {}},
            }
        }

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="npu",
                    onnx_files=[onnx_path],
                    ep="OpenVINOExecutionProvider",
                    source_genai=npu_source_genai,
                )
            ],
        )

        overlay = json.loads((out / "models" / "decoder" / "npu" / "genai_config_overlay.json").read_text())
        model_patch = overlay["model"]
        assert model_patch["context_length"] == 4224
        assert model_patch["pad_token_id"] == 200020
        assert model_patch["eos_token_id"] == [200020, 199999]
        assert model_patch["bos_token_id"] == 199999
        assert model_patch["type"] == "phi3"
        # ``vocab_size`` is structural (shared across all variants of a model)
        # and is not in the per-variant lift list, so it must NOT appear in
        # the overlay — otherwise it would duplicate the base copy.
        assert "vocab_size" not in model_patch

    def test_base_genai_strips_per_variant_model_fields(self, tmp_path):
        """The base ``configs/genai_config.json`` must not carry per-variant fields.

        If ``context_length`` (or similar) lived in the base, GenAI's overlay
        merge would still honor the per-variant value (overlay scalar wins),
        but ``_VARIANT_LEVEL_MODEL_KEYS`` includes arrays (``eos_token_id``)
        whose presence in the base would trigger GenAI's array-append merge
        semantics — the merged result would duplicate the array. So the base
        must be free of every variant-level model key.
        """
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"
        cfg = tmp_path / "configs_src" / "genai_config.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            json.dumps(
                {
                    "model": {
                        "type": "phi3",
                        "context_length": 131072,
                        "pad_token_id": 199999,
                        "eos_token_id": [200020, 199999],
                        "bos_token_id": 199999,
                        "vocab_size": 200064,
                        "decoder": {
                            "head_size": 128,
                            "filename": "model.onnx",
                            "session_options": {"log_id": "x"},
                        },
                    }
                }
            )
        )

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                )
            ],
            config_files={"genai_config.json": cfg},
        )

        base = json.loads((out / "configs" / "genai_config.json").read_text())
        model = base["model"]
        for stripped in ("context_length", "pad_token_id", "eos_token_id", "bos_token_id", "type"):
            assert stripped not in model, f"base genai_config must not contain {stripped!r}"
        # Variant-specific decoder fields also stripped.
        assert "filename" not in model["decoder"]
        assert "session_options" not in model["decoder"]
        # Structural shared fields remain.
        assert model["vocab_size"] == 200064
        assert model["decoder"]["head_size"] == 128


# ---------------------------------------------------------------------------
# Writer: external-data blobs are always kept inline per variant (no dedup)
# ---------------------------------------------------------------------------


class TestExternalDataInline:
    def test_keeps_identical_external_data_inline_in_each_variant(self, tmp_path):
        blob = b"\x00\x01\x02\x03" * 64
        a = _make_onnx_with_external(tmp_path / "a" / "model.onnx", "model.onnx.data", blob)
        b = _make_onnx_with_external(tmp_path / "b" / "model.onnx", "model.onnx.data", blob)
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="v1",
                    onnx_files=[a],
                    ep="CPUExecutionProvider",
                ),
                VariantSpec(
                    component_name="decoder",
                    variant_name="v2",
                    onnx_files=[b],
                    ep="CPUExecutionProvider",
                ),
            ],
        )

        # Each variant keeps its own external-data blob inline; no shared_weights
        # directory or variant.json is emitted.
        assert not (out / "models" / "decoder" / "shared_weights").exists()
        for v in ("v1", "v2"):
            assert (out / "models" / "decoder" / v / "model.onnx.data").is_file()
            assert not (out / "models" / "decoder" / v / "variant.json").exists()

    def test_keeps_distinct_external_data_inline_per_variant(self, tmp_path):
        a = _make_onnx_with_external(tmp_path / "a" / "model.onnx", "model.onnx.data", b"a-bytes" * 32)
        b = _make_onnx_with_external(tmp_path / "b" / "model.onnx", "model.onnx.data", b"b-bytes" * 32)
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="v1",
                    onnx_files=[a],
                    ep="CPUExecutionProvider",
                ),
                VariantSpec(
                    component_name="decoder",
                    variant_name="v2",
                    onnx_files=[b],
                    ep="CPUExecutionProvider",
                ),
            ],
        )

        assert not (out / "models" / "decoder" / "shared_weights").exists()
        assert (out / "models" / "decoder" / "v1" / "model.onnx.data").is_file()
        assert (out / "models" / "decoder" / "v2" / "model.onnx.data").is_file()

        # No variant.json is emitted.
        for v in ("v1", "v2"):
            assert not (out / "models" / "decoder" / v / "variant.json").exists()

    def test_single_variant_keeps_blob_inline(self, tmp_path):
        onnx_path = _make_onnx_with_external(tmp_path / "src" / "model.onnx", "model.onnx.data", b"x" * 128)
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                )
            ],
        )

        assert (out / "models" / "decoder" / "cpu" / "model.onnx.data").is_file()
        assert not (out / "models" / "decoder" / "shared_weights").exists()
        # No variant.json is emitted.
        assert not (out / "models" / "decoder" / "cpu" / "variant.json").exists()

    def test_copies_model_suffix_sidecars_into_variant_dir(self, tmp_path):
        """Sidecars next to an EPContext stub get copied into the variant dir.

        OpenVINO/QNN-style sidecars (e.g. ``.xml``/``.bin`` next to an EPContext stub
        ``.onnx``) aren't referenced through ONNX initializer external_data, so the
        writer sweeps the source directory and copies every model-suffix file next to
        the variant ONNX. Non-model files like ``.bak`` and ``.json`` are left alone.
        """
        src_dir = tmp_path / "src"
        onnx_path = _make_onnx_inline(src_dir / "openvino_model_dy.onnx")
        (src_dir / "openvino_model_dy.xml").write_bytes(b"<openvino-ir/>")
        (src_dir / "openvino_model_dy.bin").write_bytes(b"\x01\x02\x03\x04" * 64)
        # Files that must NOT be picked up by the sidecar sweep:
        (src_dir / "openvino_model_dy.onnx.bak").write_bytes(b"stale")
        (src_dir / "tokenizer.json").write_text("{}")

        out = tmp_path / "package"
        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="openvino_gpu",
                    onnx_files=[onnx_path],
                    ep="OpenVINOExecutionProvider",
                )
            ],
        )

        variant_dir = out / "models" / "decoder" / "openvino_gpu"
        assert (variant_dir / "openvino_model_dy.onnx").is_file()
        assert (variant_dir / "openvino_model_dy.xml").is_file()
        assert (variant_dir / "openvino_model_dy.bin").is_file()
        assert (variant_dir / "openvino_model_dy.bin").read_bytes() == b"\x01\x02\x03\x04" * 64
        # .bak and .json must stay out of the variant dir; .bak has the wrong suffix
        # and .json belongs under configs/, not next to the ONNX.
        assert not (variant_dir / "openvino_model_dy.onnx.bak").exists()
        assert not (variant_dir / "tokenizer.json").exists()

    def test_sidecar_sweep_does_not_overwrite_external_data(self, tmp_path):
        """External-data blobs are not overwritten by the sidecar sweep.

        Blobs already copied through the ONNX initializer path must not be overwritten
        by the broader source-directory sweep — the existing copy is authoritative
        (it came from the ONNX it belongs to).
        """
        blob = b"\xaa" * 256
        onnx_path = _make_onnx_with_external(tmp_path / "src" / "model.onnx", "model.onnx.data", blob)
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                )
            ],
        )

        copied = out / "models" / "decoder" / "cpu" / "model.onnx.data"
        assert copied.is_file()
        assert copied.read_bytes() == blob


# ---------------------------------------------------------------------------
# Writer: configs/ + safety
# ---------------------------------------------------------------------------


class TestConfigsAndSafety:
    def test_copies_config_files_into_configs_dir(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        cfg_a = tmp_path / "configs_src" / "tokenizer.json"
        cfg_a.parent.mkdir(parents=True)
        cfg_a.write_text("{}")
        cfg_b = tmp_path / "configs_src" / "genai_config.json"
        cfg_b.write_text("{}")
        out = tmp_path / "package"

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                )
            ],
            config_files={"tokenizer.json": cfg_a, "genai_config.json": cfg_b},
        )

        assert (out / "configs" / "tokenizer.json").is_file()
        assert (out / "configs" / "genai_config.json").is_file()

    def test_rejects_non_empty_output_dir(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"
        out.mkdir()
        (out / "stale.txt").write_text("stale")

        with pytest.raises(ValueError, match="not empty"):
            write_model_package(
                output_dir=out,
                variants=[
                    VariantSpec(
                        component_name="decoder",
                        variant_name="cpu",
                        onnx_files=[onnx_path],
                        ep="CPUExecutionProvider",
                    )
                ],
            )

    def test_rejects_invalid_component_name(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        with pytest.raises(ValueError, match="component name"):
            write_model_package(
                output_dir=out,
                variants=[
                    VariantSpec(
                        component_name="../escape",
                        variant_name="cpu",
                        onnx_files=[onnx_path],
                        ep="CPUExecutionProvider",
                    )
                ],
            )

    def test_rejects_invalid_variant_name(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        with pytest.raises(ValueError, match="variant name"):
            write_model_package(
                output_dir=out,
                variants=[
                    VariantSpec(
                        component_name="decoder",
                        variant_name="bad/name",
                        onnx_files=[onnx_path],
                        ep="CPUExecutionProvider",
                    )
                ],
            )

    def test_rejects_duplicate_variant_names_per_component(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        with pytest.raises(ValueError, match="Duplicate variant name"):
            write_model_package(
                output_dir=out,
                variants=[
                    VariantSpec(
                        component_name="decoder",
                        variant_name="cpu",
                        onnx_files=[onnx_path],
                        ep="CPUExecutionProvider",
                    ),
                    VariantSpec(
                        component_name="decoder",
                        variant_name="cpu",
                        onnx_files=[onnx_path],
                        ep="CPUExecutionProvider",
                    ),
                ],
            )

    def test_rejects_empty_variants(self, tmp_path):
        with pytest.raises(ValueError, match="at least one variant"):
            write_model_package(output_dir=tmp_path / "package", variants=[])

    def test_skips_config_file_with_unsafe_key(self, tmp_path):
        # setup: a real source plus a config_files map with a path-escaping key.
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        bad = tmp_path / "configs_src" / "evil.txt"
        bad.parent.mkdir(parents=True)
        bad.write_text("oops")
        out = tmp_path / "package"

        # execute
        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                )
            ],
            config_files={"../escape.txt": bad, "subdir/nested.txt": bad, "ok.txt": bad},
        )

        # assert: unsafe keys are dropped, safe key copied
        assert not (out.parent / "escape.txt").exists()
        assert not (out / "configs" / "subdir").exists()
        assert not (out / "configs" / "..").is_dir() or not (out / ".." / "escape.txt").exists()
        assert (out / "configs" / "ok.txt").exists()
        # configs/ should contain only the one safe entry
        assert sorted(p.name for p in (out / "configs").iterdir()) == ["ok.txt"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestParseCompatibilityStrings:
    def test_splits_comma_delimited_string(self):
        assert parse_compatibility_strings("sm_80,sm_86,sm_90") == ["sm_80", "sm_86", "sm_90"]

    def test_strips_whitespace_and_drops_empty(self):
        assert parse_compatibility_strings(" sm_80 , , sm_86 ") == ["sm_80", "sm_86"]

    def test_returns_empty_for_none_or_empty(self):
        assert parse_compatibility_strings(None) == []
        assert parse_compatibility_strings("") == []


class TestDisambiguateVariantNames:
    def test_passes_unique_names_through(self):
        assert disambiguate_variant_names([("c", "a"), ("c", "b")]) == ["a", "b"]

    def test_appends_rank_suffix_on_collision(self):
        out = disambiguate_variant_names([("c", "a"), ("c", "a"), ("c", "a")])
        assert out == ["a_rank1", "a_rank2", "a_rank3"]

    def test_isolates_collisions_per_component(self):
        out = disambiguate_variant_names([("c1", "a"), ("c2", "a")])
        assert out == ["a", "a"]


# ---------------------------------------------------------------------------
# CLI: comma-delimited compatibility from ONNX metadata
# ---------------------------------------------------------------------------


class TestCompatibilityFromOnnxMetadata:
    def test_passes_through_comma_delimited_metadata(self, tmp_path):
        # setup: source with QNNExecutionProvider compat info in ONNX metadata_props
        src = _create_source_dir(
            tmp_path,
            "soc_60",
            ep="QNNExecutionProvider",
            onnx_metadata={"ep_compatibility_info.QNNExecutionProvider": "soc_60,soc_69,soc_73"},
        )
        out = tmp_path / "out.ortpackage"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        # execute
        cmd.run()

        # assert: compatibility_string passes the raw opaque string through verbatim
        metadata = json.loads((out / "models" / "decoder" / "metadata.json").read_text())
        variant = metadata["variants"]["soc_60"]
        assert variant["ep"] == "QNNExecutionProvider"
        assert variant["compatibility_string"] == "soc_60,soc_69,soc_73"


# ---------------------------------------------------------------------------
# Pipeline sources (multi-stage exports, e.g. QNN) and VLM multi-role overlay
# ---------------------------------------------------------------------------


def _create_pipeline_source(
    tmp_path: Path,
    name: str,
    *,
    stage_filenames: list[str],
    stage_with_options: str,
    provider_alias: str,
    provider_options: dict,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Build a fake GenAI-shaped multi-stage source dir (e.g. QNN pipeline).

    The source has ONE genai_config.json + N real ONNX stage files (no
    ``model_config.json``). ``stage_with_options`` is the only stage carrying
    provider_options (per QNN convention where embedding / transformer-head
    run on CPU and only the prompt / iter stages carry the HTP options).
    """
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)
    for fname in stage_filenames:
        _make_onnx_inline(source_dir / fname)

    pipeline_stages = []
    stage_names = ["embedding", "prompt-processor", "token-generator", "transformer-head"][: len(stage_filenames)]
    for stage_name, fname in zip(stage_names, stage_filenames):
        body: dict = {"filename": fname, "inputs": [], "outputs": []}
        if stage_name == stage_with_options:
            body["session_options"] = {
                "provider_options": [{provider_alias: provider_options}],
            }
        pipeline_stages.append({stage_name: body})

    genai = {
        "model": {
            "type": "phi3-pipeline",
            "context_length": 4096,
            "pad_token_id": 199999,
            "eos_token_id": [200020, 199999],
            "bos_token_id": 199999,
            "vocab_size": 200064,
            "decoder": {
                "head_size": 128,
                "session_options": {"log_id": "onnxruntime-genai"},
                "pipeline": pipeline_stages,
            },
        }
    }
    (source_dir / "genai_config.json").write_text(json.dumps(genai))

    if extra_files:
        for fname, content in extra_files.items():
            (source_dir / fname).write_text(content)
    return source_dir


def _create_vlm_source(tmp_path: Path, name: str) -> Path:
    """Build a fake flat VLM source (vision + embedding + decoder ONNXs in one dir).

    Mirrors the shape of real-world VLM packages where a single source dir
    holds multiple roles' ONNX files alongside one ``genai_config.json`` that
    references each role's ``filename``. The packager must restore EVERY
    role's filename in the per-variant overlay — not just the primary one —
    or the GenAI loader cannot locate the vision/embedding ONNXs at load
    time.
    """
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)
    for fname in ("vision.onnx", "embedding.onnx", "text.onnx"):
        _make_onnx_inline(source_dir / fname)
    genai = {
        "model": {
            "type": "qwen3vl",
            "vocab_size": 151936,
            "vision": {
                "filename": "vision.onnx",
                "session_options": {"provider_options": []},
            },
            "embedding": {
                "filename": "embedding.onnx",
                "session_options": {"provider_options": []},
            },
            "decoder": {
                "head_size": 128,
                "filename": "text.onnx",
                "session_options": {"provider_options": []},
            },
        }
    }
    (source_dir / "genai_config.json").write_text(json.dumps(genai))
    return source_dir


class TestPipelineSources:
    """Pipeline multi-stage sources (e.g. QNN)."""

    def test_rejects_source_without_genai_config(self, tmp_path):
        """A source without ``genai_config.json`` is rejected with a clear error."""
        empty = tmp_path / "empty"
        empty.mkdir()
        _make_onnx_inline(empty / "model.onnx")
        cmd = _make_command(["generate-model-package", "-s", str(empty), "-o", str(tmp_path / "out")])

        with pytest.raises(ValueError, match=r"no genai_config\.json"):
            cmd._parse_sources()

    def test_packs_pipeline_with_all_stage_onnx_files(self, tmp_path):
        """All pipeline-stage ONNX files land in the variant directory.

        The single-ONNX resolver would fail because the source has >1 ONNX;
        the pipeline resolver enumerates stage filenames from the source
        genai_config so every stage is copied next to the variant's
        overlay.
        """
        stage_files = ["phi_embed.onnx", "phi_ctx.onnx", "phi_iter.onnx", "phi_head.onnx"]
        src = _create_pipeline_source(
            tmp_path,
            "qnn_npu",
            stage_filenames=stage_files,
            stage_with_options="prompt-processor",
            provider_alias="qnn",
            provider_options={"soc_model": "60"},
        )
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out), "--model_name", "phi-pipe"])

        cmd.run()

        variant_dir = out.with_suffix(".ortpackage") / "models" / "decoder" / "qnn_npu"
        assert variant_dir.is_dir()
        for fname in stage_files:
            assert (variant_dir / fname).is_file(), f"missing stage file {fname}"

    def test_pipeline_overlay_lifts_full_stage_structure_from_source(self, tmp_path):
        """The variant overlay carries the pipeline list with per-stage options.

        The producing toolchain decided per-stage EP knobs (soc_model,
        htp_performance_mode, etc.); copying them verbatim avoids the
        overlay writer having to re-derive each one and guarantees the
        loader sees the exact same configuration the source intended.
        """
        src = _create_pipeline_source(
            tmp_path,
            "qnn_npu",
            stage_filenames=["e.onnx", "c.onnx", "i.onnx", "h.onnx"],
            stage_with_options="prompt-processor",
            provider_alias="qnn",
            provider_options={"htp_performance_mode": "burst", "soc_model": "60"},
        )
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        overlay_path = out.with_suffix(".ortpackage") / "models" / "decoder" / "qnn_npu" / "genai_config_overlay.json"
        overlay = json.loads(overlay_path.read_text())
        decoder = overlay["model"]["decoder"]
        assert "pipeline" in decoder
        stage_names = [next(iter(stage)) for stage in decoder["pipeline"]]
        assert stage_names == ["embedding", "prompt-processor", "token-generator", "transformer-head"]
        prompt_stage = decoder["pipeline"][1]["prompt-processor"]
        assert prompt_stage["filename"] == "c.onnx"
        assert prompt_stage["session_options"]["provider_options"] == [
            {"qnn": {"htp_performance_mode": "burst", "soc_model": "60"}}
        ]
        # decoder-level session_options also lifted from source so log_id etc. survive.
        assert decoder["session_options"]["log_id"] == "onnxruntime-genai"

    def test_base_genai_strips_pipeline_field(self, tmp_path):
        """``pipeline`` lives only in the overlay; base must not duplicate it.

        GenAI's overlay parser appends arrays rather than replacing them
        (``src/config.cpp:PipelineModelObject_Element``), so a ``pipeline``
        in both base and overlay would double every stage. The strip is the
        guard.
        """
        src = _create_pipeline_source(
            tmp_path,
            "qnn_npu",
            stage_filenames=["e.onnx", "c.onnx", "i.onnx", "h.onnx"],
            stage_with_options="prompt-processor",
            provider_alias="qnn",
            provider_options={"soc_model": "60"},
        )
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        base = json.loads((out.with_suffix(".ortpackage") / "configs" / "genai_config.json").read_text())
        decoder = base["model"]["decoder"]
        assert "pipeline" not in decoder, "base genai_config must not retain the pipeline array"

    def test_flat_source_ep_derived_from_source_genai_when_attrs_missing(self, tmp_path):
        """For flat sources, source genai's ``provider_options`` overrules name guess.

        A directory named ``vitia_npu`` would otherwise be heuristically
        classified as QNN (the ``npu`` substring wins by accident); the
        source genai_config saying ``provider_options: [{"VitisAI": {}}]``
        is the authoritative signal.
        """
        source_dir = tmp_path / "vitia_npu"
        source_dir.mkdir()
        _make_onnx_inline(source_dir / "model.onnx")
        (source_dir / "genai_config.json").write_text(
            json.dumps(
                {
                    "model": {
                        "type": "phi3",
                        "vocab_size": 200064,
                        "decoder": {
                            "head_size": 128,
                            "filename": "model.onnx",
                            "session_options": {"provider_options": [{"VitisAI": {}}]},
                        },
                    }
                }
            )
        )
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(source_dir), "-o", str(out)])

        cmd.run()

        metadata = json.loads((out.with_suffix(".ortpackage") / "models" / "decoder" / "metadata.json").read_text())
        assert metadata["variants"]["vitia_npu"]["ep"] == "VitisAIExecutionProvider"
        overlay = json.loads(
            (
                out.with_suffix(".ortpackage") / "models" / "decoder" / "vitia_npu" / "genai_config_overlay.json"
            ).read_text()
        )
        assert overlay["model"]["decoder"]["session_options"]["provider_options"] == [{"VitisAI": {}}]


class TestLiftRoleOverlayBodyPipelineWins:
    """Pipeline takes precedence over a role-level ``filename`` in overlay lift.

    A role body that carries BOTH ``filename`` and a non-empty ``pipeline``
    is malformed input; the artifact collector already prefers the
    pipeline shape in that case, so the overlay writer must do the same.
    Lifting both would emit ``{"filename": ..., "pipeline": [...]}`` to
    the overlay (invalid for the loader, which expects exactly one
    shape per role) AND silently alias ``onnx_rel_paths[0]`` between the
    role-level filename and stage 0's filename.
    """

    def test_pipeline_present_drops_role_level_filename(self):
        from olive.cli.model_package import _lift_role_overlay_body

        # Role declares both — pipeline wins. onnx_rel_paths reflects what
        # the per-role artifact collector would emit (one entry per
        # pipeline stage, basename-only).
        role_body = {
            "filename": "old_flat.onnx",  # stale role-level fallback
            "pipeline": [
                {"prompt": {"filename": "qnn/prompt.onnx"}},
                {"token": {"filename": "qnn/token.onnx"}},
            ],
        }
        rel_paths = ["prompt.onnx", "token.onnx"]

        patch = _lift_role_overlay_body(role_body, rel_paths)

        # Bug guard: no ``filename`` key at all when pipeline is present.
        assert "filename" not in patch, f"role-level filename leaked into overlay even though pipeline wins: {patch!r}"
        # Pipeline shape preserved with each stage's filename mapped to
        # its writer-known basename in order.
        assert patch["pipeline"][0]["prompt"]["filename"] == "prompt.onnx"
        assert patch["pipeline"][1]["token"]["filename"] == "token.onnx"

    def test_no_pipeline_keeps_role_level_filename(self):
        from olive.cli.model_package import _lift_role_overlay_body

        role_body = {"filename": "decoder/model.onnx"}
        patch = _lift_role_overlay_body(role_body, ["model.onnx"])
        assert patch["filename"] == "model.onnx"
        assert "pipeline" not in patch


class TestVLMMultiRoleOverlay:
    """Multi-role (vision + embedding + decoder) VLM packaging.

    A flat VLM source dir packs >1 ONNX file referenced by >1 role in the
    same ``genai_config.json``. Each role becomes its own component
    (``models/vision/``, ``models/embedding/``, ``models/decoder/``); each
    component's overlay restores only that role's ``filename`` /
    ``session_options``. The base genai_config strips every role's
    filename / session_options and injects a ``component=<role>`` marker
    so the loader can map each role back to its on-disk directory.
    """

    def test_each_role_becomes_its_own_component(self, tmp_path):
        src = _create_vlm_source(tmp_path, "cpu_and_mobile")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models_dir = out.with_suffix(".ortpackage") / "models"
        assert (models_dir / "vision" / "metadata.json").is_file()
        assert (models_dir / "embedding" / "metadata.json").is_file()
        assert (models_dir / "decoder" / "metadata.json").is_file()

    def test_each_components_overlay_lifts_only_its_role(self, tmp_path):
        """Each per-role overlay carries exactly that role's filename, no others.

        The base config strips every role's filename, but each per-role
        overlay should only restore its own — duplicating the lift across
        all overlays would corrupt the loader's view (and trigger array
        append-merge problems for list-valued scalars).
        """
        src = _create_vlm_source(tmp_path, "cpu_and_mobile")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        decoder_overlay = json.loads((models / "decoder" / "cpu_and_mobile" / "genai_config_overlay.json").read_text())
        vision_overlay = json.loads((models / "vision" / "cpu_and_mobile" / "genai_config_overlay.json").read_text())
        embedding_overlay = json.loads(
            (models / "embedding" / "cpu_and_mobile" / "genai_config_overlay.json").read_text()
        )

        assert "decoder" in decoder_overlay["model"]
        assert decoder_overlay["model"]["decoder"]["filename"] == "text.onnx"
        assert "vision" not in decoder_overlay["model"]
        assert "embedding" not in decoder_overlay["model"]

        assert "vision" in vision_overlay["model"]
        assert vision_overlay["model"]["vision"]["filename"] == "vision.onnx"
        assert "decoder" not in vision_overlay["model"]
        assert "embedding" not in vision_overlay["model"]

        assert "embedding" in embedding_overlay["model"]
        assert embedding_overlay["model"]["embedding"]["filename"] == "embedding.onnx"
        assert "decoder" not in embedding_overlay["model"]
        assert "vision" not in embedding_overlay["model"]

    def test_variant_dirs_are_flat_one_onnx_per_role(self, tmp_path):
        """Each per-role variant dir holds exactly its own ONNX, flat.

        With one role per component there's no sibling-role disambiguation
        to do, so the writer drops any subdir prefixes from the source
        layout and writes each ONNX at the variant root.
        """
        src = _create_vlm_source(tmp_path, "cpu_and_mobile")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        assert (models / "decoder" / "cpu_and_mobile" / "text.onnx").is_file()
        assert not (models / "decoder" / "cpu_and_mobile" / "vision.onnx").exists()
        assert (models / "vision" / "cpu_and_mobile" / "vision.onnx").is_file()
        assert (models / "embedding" / "cpu_and_mobile" / "embedding.onnx").is_file()

    def test_base_genai_injects_component_marker_for_every_role(self, tmp_path):
        """Every role gets a ``component=<role>`` marker in the base config.

        The merged config the loader sees must know which component
        directory each role lives in. With per-role components the
        component name equals the role name.
        """
        src = _create_vlm_source(tmp_path, "cpu_and_mobile")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        base = json.loads((out.with_suffix(".ortpackage") / "configs" / "genai_config.json").read_text())
        model = base["model"]
        for role in ("vision", "embedding", "decoder"):
            assert model[role]["component"] == role, f"role {role} missing self-named component marker"


# ---------------------------------------------------------------------------
# Hierarchical multi-component sources (Mobius-style VLMs)
# ---------------------------------------------------------------------------


def _create_mobius_vlm_source(
    tmp_path: Path,
    name: str,
    *,
    ep: str = "CPUExecutionProvider",
    provider_options: dict | None = None,
    with_external_data: bool = True,
) -> Path:
    """Build a Mobius-style multi-component VLM source directory.

    Real ``olive capture-onnx-graph --use_mobius_builder`` output for a VLM
    nests each role's ONNX inside its own subdirectory: ``decoder/``,
    ``embedding/``, and ``vision_encoder/`` each contain ``model.onnx``
    (and a ``model.onnx.data`` external-data blob). The ``genai_config.json``
    references each role by its full subdirectory-prefixed path
    (``"filename": "decoder/model.onnx"``).

    With per-role components each role gets its own
    ``models/<role>/<variant>/`` directory, so the packager safely
    flattens the packaged filename to the basename (``model.onnx``) and
    rewrites the overlay to match the on-disk layout — no sibling-role
    disambiguation is needed inside any one variant dir. The source-side
    subdirectory prefix is used only to locate the file on disk in the
    source and does not propagate into the package.
    """
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)

    roles = {
        "decoder": "decoder/model.onnx",
        "embedding": "embedding/model.onnx",
        "vision": "vision_encoder/model.onnx",
    }
    for rel in roles.values():
        onnx_path = source_dir / rel
        if with_external_data:
            # Each subdir gets its own external-data blob whose ``location``
            # is recorded relative to the ONNX's own directory (basename).
            # If the writer routed external-data to the variant root, all
            # three would collide at ``<variant>/model.onnx.data``.
            _make_onnx_with_external(onnx_path, "model.onnx.data", f"role-{rel.split('/')[0]}".encode() * 16)
        else:
            _make_onnx_inline(onnx_path)

    ep_to_alias = {
        "CPUExecutionProvider": "CPU",
        "CUDAExecutionProvider": "cuda",
        "QNNExecutionProvider": "qnn",
        "DmlExecutionProvider": "DML",
    }
    alias = ep_to_alias.get(ep, "CPU")
    if alias == "CPU":
        session_options = {"provider_options": []}
    else:
        session_options = {"provider_options": [{alias: provider_options or {}}]}

    genai = {
        "model": {
            "type": "qwen2_5_vl",
            "vocab_size": 248320,
            "context_length": 262144,
            "decoder": {
                "filename": roles["decoder"],
                "session_options": dict(session_options),
                "head_size": 256,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
            },
            "embedding": {
                "filename": roles["embedding"],
                "session_options": dict(session_options),
            },
            "vision": {
                "filename": roles["vision"],
                "session_options": dict(session_options),
                "spatial_merge_size": 2,
            },
        }
    }
    (source_dir / "genai_config.json").write_text(json.dumps(genai))
    # Seed a couple of consumer-shared config files alongside the model
    # subdirs to verify the config-file sweep doesn't slurp up the model
    # directories themselves.
    (source_dir / "tokenizer_config.json").write_text(json.dumps({"vocab_size": 248320}))
    (source_dir / "model_config.json").write_text(json.dumps({"architectures": ["Qwen25VL"]}))
    return source_dir


class TestMobiusHierarchicalLayout:
    """End-to-end packaging of Mobius-style multi-component VLM sources.

    Each Mobius role (``decoder``/``embedding``/``vision``) becomes its
    own component in the package (``models/decoder/``,
    ``models/embedding/``, ``models/vision/``). Per the ORT model-package
    proposal, ``models/<component>/`` is the top-level grouping where one
    component == one inference session. Each component's variant
    directory is flat: the source-side subdir prefix is dropped because
    there's no longer a sibling role to disambiguate against.
    """

    def test_each_role_becomes_top_level_component(self, tmp_path):
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        for role in ("decoder", "embedding", "vision"):
            assert (models / role / "metadata.json").is_file(), f"missing component dir for role {role}"

    def test_variant_dir_is_flat_under_each_component(self, tmp_path):
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        # Each per-role variant dir holds the ONNX flat (basename only).
        for role in ("decoder", "embedding", "vision"):
            assert (models / role / "cpu" / "model.onnx").is_file(), (
                f"missing flat model.onnx under models/{role}/cpu (writer kept subdir?)"
            )
            # The source-side subdir should NOT propagate into the
            # variant dir — there's only one role here, no sibling to
            # disambiguate against.
            assert not (models / role / "cpu" / role / "model.onnx").exists()

    def test_external_data_lands_next_to_its_onnx_in_flat_layout(self, tmp_path):
        """Each role's external-data blob lives flat next to its ONNX in that role's variant dir.

        The ONNX file references ``model.onnx.data`` relative to its own
        directory; the loader resolves the same way. With one role per
        component each role's blob has its own dedicated directory, so
        no collision is possible even when sibling source-side roles
        shared the same basename.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu", with_external_data=True)
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        for role in ("decoder", "embedding", "vision"):
            blob = models / role / "cpu" / "model.onnx.data"
            assert blob.is_file(), f"external-data blob missing under models/{role}/cpu/"

    def test_overlay_filename_is_basename(self, tmp_path):
        """The overlay's ``filename`` is the basename, not the source subdir-prefixed path.

        Per-role variant dirs are flat; the overlay must match the on-disk
        layout. The source-side ``decoder/model.onnx`` prefix is purely a
        sibling-disambiguation device that no longer applies once the
        roles are separated into top-level components.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        decoder = json.loads((models / "decoder" / "cpu" / "genai_config_overlay.json").read_text())
        embedding = json.loads((models / "embedding" / "cpu" / "genai_config_overlay.json").read_text())
        vision = json.loads((models / "vision" / "cpu" / "genai_config_overlay.json").read_text())
        assert decoder["model"]["decoder"]["filename"] == "model.onnx"
        assert embedding["model"]["embedding"]["filename"] == "model.onnx"
        assert vision["model"]["vision"]["filename"] == "model.onnx"

    def test_configs_dir_excludes_model_artifact_subdirs(self, tmp_path):
        """``decoder/``/``embedding/``/``vision_encoder/`` must not leak into ``configs/``.

        Without explicit exclusion the config-file sweep would copy every
        source-root directory (including the model-artifact subdirs), so the
        package would carry duplicate ONNXs under ``configs/`` and bloat
        the deliverable. The sweep recognizes model-artifact subdirs via
        the genai_config's role filenames and skips them.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        configs_dir = out.with_suffix(".ortpackage") / "configs"
        for excluded in ("decoder", "embedding", "vision_encoder"):
            assert not (configs_dir / excluded).exists(), f"{excluded}/ leaked into configs/"
        assert (configs_dir / "tokenizer_config.json").is_file()
        assert (configs_dir / "model_config.json").is_file()
        assert (configs_dir / "genai_config.json").is_file()

    def test_base_genai_strips_filename_and_marks_self_named_components(self, tmp_path):
        """Base genai_config strips per-role ``filename`` and injects a self-named ``component`` marker.

        With per-role components the role name equals the component name,
        so every role's ``component`` field is its own name.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        base = json.loads((out.with_suffix(".ortpackage") / "configs" / "genai_config.json").read_text())
        model = base["model"]
        for role in ("decoder", "embedding", "vision"):
            assert "filename" not in model[role], f"{role}.filename should be stripped from base"
            assert model[role]["component"] == role, f"{role} component marker should equal role name"

    def test_two_sources_each_produce_per_role_variants(self, tmp_path):
        """CPU + GPU Mobius sources both contribute one variant per role to each component.

        This is the user-reported scenario:
        ``olive generate-model-package -s cpu -s gpu -o cpu_gpu``. Each
        role component (``decoder``/``embedding``/``vision``) ends up
        with two variants — ``cpu`` (CPUExecutionProvider) and ``gpu``
        (CUDAExecutionProvider).
        """
        cpu = _create_mobius_vlm_source(tmp_path, "cpu", ep="CPUExecutionProvider")
        gpu = _create_mobius_vlm_source(tmp_path, "gpu", ep="CUDAExecutionProvider")
        out = tmp_path / "cpu_gpu"
        cmd = _make_command(["generate-model-package", "-s", str(cpu), "-s", str(gpu), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        for role in ("decoder", "embedding", "vision"):
            for variant in ("cpu", "gpu"):
                assert (models / role / variant / "model.onnx").is_file(), f"missing models/{role}/{variant}/model.onnx"
            metadata = json.loads((models / role / "metadata.json").read_text())
            assert metadata["variants"]["cpu"]["ep"] == "CPUExecutionProvider"
            assert metadata["variants"]["gpu"]["ep"] == "CUDAExecutionProvider"

    def test_variant_level_scalars_lift_only_into_primary_role_overlay(self, tmp_path):
        """Variant-level scalars (eos_token_id, context_length, ...) appear in exactly one overlay.

        GenAI's overlay parser append-merges arrays. If
        ``eos_token_id`` (often a list) ended up in three different
        per-role overlays the merged config would triple every entry.
        Only the primary role per source (``_pick_primary_role`` —
        ``decoder`` here) carries these scalars.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        decoder_model = json.loads((models / "decoder" / "cpu" / "genai_config_overlay.json").read_text())["model"]
        embedding_model = json.loads((models / "embedding" / "cpu" / "genai_config_overlay.json").read_text())["model"]
        vision_model = json.loads((models / "vision" / "cpu" / "genai_config_overlay.json").read_text())["model"]
        # context_length and type are seeded on the Mobius fixture under
        # ``model``; they belong only to the primary role's overlay.
        assert "context_length" in decoder_model
        assert "type" in decoder_model
        for non_primary in (embedding_model, vision_model):
            assert "context_length" not in non_primary
            assert "type" not in non_primary

    def test_explicit_cpu_role_in_gpu_source_kept_as_cpu(self, tmp_path):
        """A role with explicit CPU ``provider_options`` keeps CPU even when the source dir is named like a GPU build.

        Variant-name heuristics must not override a producer's explicit
        per-role provider choice — Mobius outputs sometimes mark a
        helper role (e.g. ``embedding``) as CPU even inside a
        predominantly-GPU build.
        """
        src = tmp_path / "gpu"
        src.mkdir()
        # Build the GPU source manually so we can mix EPs per role.
        for _role, fname in (
            ("decoder", "decoder/model.onnx"),
            ("embedding", "embedding/model.onnx"),
            ("vision", "vision_encoder/model.onnx"),
        ):
            _make_onnx_inline(src / fname)
        genai = {
            "model": {
                "type": "qwen2_5_vl",
                "decoder": {
                    "filename": "decoder/model.onnx",
                    "session_options": {"provider_options": [{"cuda": {}}]},
                },
                "embedding": {
                    # Explicit CPU even though the source dir is named "gpu".
                    "filename": "embedding/model.onnx",
                    "session_options": {"provider_options": []},
                },
                "vision": {
                    "filename": "vision_encoder/model.onnx",
                    "session_options": {"provider_options": [{"cuda": {}}]},
                },
            }
        }
        (src / "genai_config.json").write_text(json.dumps(genai))

        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])
        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        assert (
            json.loads((models / "decoder" / "metadata.json").read_text())["variants"]["gpu"]["ep"]
            == "CUDAExecutionProvider"
        )
        assert (
            json.loads((models / "vision" / "metadata.json").read_text())["variants"]["gpu"]["ep"]
            == "CUDAExecutionProvider"
        )
        # Critical: explicit CPU role must NOT be promoted to CUDA via
        # the variant-name "gpu" heuristic.
        assert (
            json.loads((models / "embedding" / "metadata.json").read_text())["variants"]["gpu"]["ep"]
            == "CPUExecutionProvider"
        )

    def test_base_config_source_picks_richest_role_set(self, tmp_path):
        """When sources expose different role sets, the base config is taken from the source with the most roles.

        Otherwise the package's base ``configs/genai_config.json`` could
        miss role blocks that downstream components rely on. Example:
        gpu source only has decoder; cpu source has all three. The base
        must come from cpu so embedding/vision components have role
        markers in the base config.
        """
        # cpu source: full three-role VLM.
        cpu = _create_mobius_vlm_source(tmp_path, "cpu")
        # gpu source: decoder-only.
        gpu_dir = tmp_path / "gpu"
        gpu_dir.mkdir()
        _make_onnx_inline(gpu_dir / "decoder" / "model.onnx")
        gpu_genai = {
            "model": {
                "type": "qwen2_5_vl",
                "decoder": {
                    "filename": "decoder/model.onnx",
                    "session_options": {"provider_options": [{"cuda": {}}]},
                },
            }
        }
        (gpu_dir / "genai_config.json").write_text(json.dumps(gpu_genai))
        # Drop the cpu-only role markers into gpu so the source is otherwise
        # comparable; the difference is only in number of roles declared.

        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(gpu_dir), "-s", str(cpu), "-o", str(out)])
        cmd.run()

        # Base must carry all three role blocks (so the embedding/vision
        # components are findable). If first-source-wins ran, only
        # decoder would appear.
        base = json.loads((out.with_suffix(".ortpackage") / "configs" / "genai_config.json").read_text())
        for role in ("decoder", "embedding", "vision"):
            assert role in base["model"], f"base config missing {role} block; wrong source selected"


class TestUnsafeGenaiFilenamesRejected:
    """Path-safety: reject absolute filenames and parent-traversal in genai_config."""

    def test_rejects_absolute_filename(self, tmp_path):
        src = tmp_path / "bad_abs"
        src.mkdir()
        _make_onnx_inline(src / "model.onnx")
        genai = {
            "model": {
                "decoder": {"filename": "/etc/passwd", "session_options": {"provider_options": []}},
            }
        }
        (src / "genai_config.json").write_text(json.dumps(genai))
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(tmp_path / "out")])

        with pytest.raises(ValueError, match=r"safe relative path"):
            cmd.run()

    def test_rejects_parent_traversal_in_filename(self, tmp_path):
        src = tmp_path / "bad_traverse"
        src.mkdir()
        _make_onnx_inline(src / "model.onnx")
        genai = {
            "model": {
                "decoder": {
                    "filename": "../../../escape/model.onnx",
                    "session_options": {"provider_options": []},
                },
            }
        }
        (src / "genai_config.json").write_text(json.dumps(genai))
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(tmp_path / "out")])

        with pytest.raises(ValueError, match=r"safe relative path"):
            cmd.run()

    def test_rejects_unsafe_pipeline_stage_filename(self, tmp_path):
        src = tmp_path / "bad_pipeline"
        src.mkdir()
        _make_onnx_inline(src / "stage1.onnx")
        genai = {
            "model": {
                "decoder": {
                    "pipeline": [
                        {"first": {"filename": "stage1.onnx"}},
                        {"second": {"filename": "../escape.onnx"}},
                    ],
                    "session_options": {"provider_options": []},
                },
            }
        }
        (src / "genai_config.json").write_text(json.dumps(genai))
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(tmp_path / "out")])

        with pytest.raises(ValueError, match=r"safe relative path"):
            cmd.run()


class TestIsSafeRelativeLocationCrossPlatform:
    """Path-safety helper rejects unsafe inputs on both POSIX and Windows hosts.

    ``Path("/etc/passwd").is_absolute()`` returns ``False`` on Windows
    because there is no drive letter, and ``Path("C:/foo").is_absolute()``
    returns ``False`` on POSIX. A naive single-flavor check would let
    an attacker (or a malformed genai_config produced on a different
    platform) slip through. The helper must reject paths that look
    absolute under EITHER flavor, and must treat backslashes as
    separators on POSIX too so Windows-style traversal is caught.
    """

    @pytest.mark.parametrize(
        "candidate",
        [
            "/etc/passwd",
            "\\etc\\passwd",
            "C:/foo/bar",
            "C:\\foo\\bar",
            "C:foo",
            "D:\\etc\\passwd",
            "..\\..\\escape",
            "../escape.onnx",
            "..",
            "",
            "//server/share/file",
        ],
    )
    def test_rejects_unsafe_path(self, candidate):
        from olive.cli.model_package import _is_safe_relative_location

        assert not _is_safe_relative_location(candidate), f"unsafe path {candidate!r} was incorrectly accepted"

    @pytest.mark.parametrize(
        "candidate",
        [
            "model.onnx",
            "decoder/model.onnx",
            "decoder\\model.onnx",
            "a/b/c.onnx",
            "nested.dir/file.onnx",
        ],
    )
    def test_accepts_safe_relative_path(self, candidate):
        from olive.cli.model_package import _is_safe_relative_location

        assert _is_safe_relative_location(candidate), f"safe relative path {candidate!r} was incorrectly rejected"


class TestCopyWithCollisionCheck:
    """Writer collision-detection: same content dedupes, different content raises."""

    def test_skips_when_destination_is_identical_copy(self, tmp_path):
        from olive.cli.model_package import _copy_with_collision_check

        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.bin"
        src.write_bytes(b"identical-content" * 32)
        dst.write_bytes(b"identical-content" * 32)
        # Should be a no-op (does not raise, does not modify dst).
        _copy_with_collision_check(src, dst)
        assert dst.read_bytes() == b"identical-content" * 32

    def test_raises_when_destination_differs(self, tmp_path):
        from olive.cli.model_package import _copy_with_collision_check

        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.bin"
        src.write_bytes(b"one")
        dst.write_bytes(b"two")
        with pytest.raises(FileExistsError, match="content differs"):
            _copy_with_collision_check(src, dst)

    def test_copies_when_destination_missing(self, tmp_path):
        from olive.cli.model_package import _copy_with_collision_check

        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.bin"
        src.write_bytes(b"hello")
        _copy_with_collision_check(src, dst)
        assert dst.read_bytes() == b"hello"


class TestRoleToComponentConflictDetection:
    """Two variants mapping the same role to different components must raise.

    Per the per-role-component layout, each genai_config role belongs to
    exactly one package component. A direct caller that constructs
    variants by hand could violate this invariant (e.g. by reusing the
    same source_genai under two component names); ``write_model_package``
    detects the conflict at the role_to_component build step and raises
    rather than silently keep one mapping and drop the other.
    """

    def test_same_role_mapped_to_two_components_raises(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "pkg"

        shared_genai = {
            "model": {
                "decoder": {
                    "filename": "model.onnx",
                    "session_options": {"provider_options": []},
                }
            }
        }
        variants = [
            VariantSpec(
                component_name="comp_a",
                variant_name="cpu",
                role_name="decoder",
                onnx_files=[onnx_path],
                onnx_rel_paths=["model.onnx"],
                ep="CPUExecutionProvider",
                source_genai=shared_genai,
            ),
            VariantSpec(
                component_name="comp_b",
                variant_name="cpu",
                role_name="decoder",
                onnx_files=[onnx_path],
                onnx_rel_paths=["model.onnx"],
                ep="CPUExecutionProvider",
                source_genai=shared_genai,
            ),
        ]

        with pytest.raises(ValueError, match="mapped to two different components"):
            write_model_package(
                output_dir=out,
                variants=variants,
                producer_info={"tool": "olive-ai", "model_name": "demo"},
            )
