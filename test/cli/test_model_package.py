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


def _shared_asset_dir(package_root: Path) -> Path:
    """Return the package's single ``shared_assets/sha256-<hex>/`` directory."""
    assets = sorted((package_root / "shared_assets").glob("sha256-*"))
    assert len(assets) == 1, f"expected exactly one shared asset, found {[p.name for p in assets]}"
    return assets[0]


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
        assert manifest["schema_version"] == "1.0"
        # One component holds every genai_config role, because ORT-GenAI
        # selects a single component and loads one complete config from it.
        assert manifest["components"] == {"model": "models/model"}
        assert manifest["additional_metadata"]["producer"]["model_name"] == "test_model"
        assert manifest["additional_metadata"]["producer"]["model_version"] == "2.0"

        # metadata uses inline EP
        metadata = json.loads((out / "models" / "model" / "component.json").read_text())
        assert "schema_version" not in metadata
        assert metadata["component_name"] == "model"
        assert set(metadata["variants"]) == {"soc_60", "soc_73"}
        for variant_payload in metadata["variants"].values():
            assert variant_payload == {"ep": "QNNExecutionProvider"}

        # No variant.json is emitted; the ONNX file lands in the variant
        # directory.
        for v in ("soc_60", "soc_73"):
            assert not (out / "models" / "model" / v / "variant.json").exists()
            assert (out / "models" / "model" / v / "model.onnx").is_file()


class TestGeneratePackageSingleSource:
    def test_single_source_is_valid_package(self, tmp_path):
        src = _create_source_dir(tmp_path, "cpu_x64", ep="CPUExecutionProvider")
        out = tmp_path / "out.ortpackage"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["components"] == {"model": "models/model"}
        metadata = json.loads((out / "models" / "model" / "component.json").read_text())
        assert "cpu_x64" in metadata["variants"]
        assert metadata["variants"]["cpu_x64"] == {"ep": "CPUExecutionProvider"}
        # No shared_weights because nothing to dedup.
        assert not (out / "models" / "model" / "shared_weights").exists()


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
        assert (out / "models" / "decoder" / "component.json").is_file()
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
        assert manifest["schema_version"] == "1.0"
        assert manifest["components"] == {"decoder": "models/decoder"}
        assert manifest["package_name"] == "package"
        assert manifest["package_version"] == "1.0"
        assert manifest["additional_metadata"]["producer"] == {
            "tool": "olive-ai",
            "tool_version": "1.2.3",
            "model_name": "demo",
        }
        # The ORT schema rejects unknown top-level manifest keys, so nothing
        # outside its vocabulary may be emitted.
        assert set(manifest) <= {
            "schema_version",
            "package_name",
            "package_version",
            "description",
            "layout",
            "components",
            "shared_assets",
            "additional_metadata",
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

        metadata = json.loads((out / "models" / "decoder" / "component.json").read_text())
        assert "schema_version" not in metadata
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

        metadata = json.loads((out / "models" / "decoder" / "component.json").read_text())
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

        # Runtime fields live in the variant's own genai_config.json.
        assert not (out / "models" / "decoder" / "cuda" / "variant.json").exists()
        config = json.loads((out / "models" / "decoder" / "cuda" / "genai_config.json").read_text())
        assert config == {
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

        config = json.loads((out / "models" / "decoder" / "qnn" / "genai_config.json").read_text())
        assert config["model"]["decoder"]["session_options"]["provider_options"] == [
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

        config = json.loads((out / "models" / "decoder" / "cpu" / "genai_config.json").read_text())
        assert config == {
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

        overlay = json.loads((out / "models" / "decoder" / "npu" / "genai_config.json").read_text())
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

    def test_variant_config_is_complete_and_self_contained(self, tmp_path):
        """Each variant carries a full ``genai_config.json``, not a merge patch.

        ORT-GenAI loads ``<selected_variant_dir>/genai_config.json`` directly
        and never merges a package-level base, so every field it needs —
        structural (``vocab_size``) and per-variant (``context_length``,
        ``eos_token_id``, ``filename``) alike — must be present in that one
        file.
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

        config = json.loads((out / "models" / "decoder" / "cpu" / "genai_config.json").read_text())
        model = config["model"]
        # The variant config is complete: variant-level keys come from this
        # variant's own source rather than leaking in from a shared base.
        assert model["context_length"] == 131072
        assert model["pad_token_id"] == 199999
        assert model["eos_token_id"] == [200020, 199999]
        assert model["bos_token_id"] == 199999
        assert model["type"] == "phi3"
        assert model["decoder"]["filename"] == "model.onnx"
        # Structural shared fields survive the base strip.
        assert model["vocab_size"] == 200064
        assert model["decoder"]["head_size"] == 128
        # No package-level base config is emitted; ORT-GenAI only ever reads
        # the selected variant's genai_config.json.
        assert not (out / "configs").exists()
        assert not (out / "genai_config.json").exists()


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
    def test_shares_tokenizer_assets_by_content_address(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        cfg_a = tmp_path / "configs_src" / "tokenizer.json"
        cfg_a.parent.mkdir(parents=True)
        cfg_a.write_text("{}")
        cfg_b = tmp_path / "configs_src" / "genai_config.json"
        cfg_b.write_text("{}")
        cfg_c = tmp_path / "configs_src" / "tokenizer_config.json"
        cfg_c.write_text("{}")
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
            config_files={"tokenizer.json": cfg_a, "tokenizer_config.json": cfg_c, "genai_config.json": cfg_b},
        )

        asset_dir = _shared_asset_dir(out)
        assert (asset_dir / "tokenizer.json").is_file()
        assert (asset_dir / "tokenizer_config.json").is_file()
        # genai_config.json is the base each variant config is derived from,
        # never a shared asset of its own.
        assert not (asset_dir / "genai_config.json").exists()
        assert not (out / "configs").exists()

        config = json.loads((out / "models" / "decoder" / "cpu" / "genai_config.json").read_text())
        assert config["model"]["tokenizer_dir"] == f"sha256:{asset_dir.name.removeprefix('sha256-')}"

    def test_no_tokenizer_dir_without_tokenizer_config(self, tmp_path):
        """Shared assets that hold no tokenizer must not be advertised as one.

        onnxruntime-extensions unconditionally opens
        ``<tokenizer_dir>/tokenizer_config.json``, so pointing ``tokenizer_dir``
        at an asset without it cannot help and only misreports where the
        tokenizer was expected.
        """
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        stray = tmp_path / "src" / "notes.txt"
        stray.write_text("not a tokenizer")
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
            config_files={"notes.txt": stray},
        )

        # The file is still staged; only the tokenizer_dir pointer is withheld.
        assert (_shared_asset_dir(out) / "notes.txt").is_file()
        config = json.loads((out / "models" / "decoder" / "cpu" / "genai_config.json").read_text())
        assert "tokenizer_dir" not in config.get("model", {})

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
        asset_dir = _shared_asset_dir(out)
        assert not (out.parent / "escape.txt").exists()
        assert not (asset_dir / "subdir").exists()
        assert (asset_dir / "ok.txt").exists()
        # the shared asset should contain only the one safe entry
        assert sorted(p.name for p in asset_dir.iterdir()) == ["ok.txt"]


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
        metadata = json.loads((out / "models" / "model" / "component.json").read_text())
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

        variant_dir = out.with_suffix(".ortpackage") / "models" / "model" / "qnn_npu"
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

        overlay_path = out.with_suffix(".ortpackage") / "models" / "model" / "qnn_npu" / "genai_config.json"
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

    def test_variant_config_carries_pipeline_exactly_once(self, tmp_path):
        """The variant config holds the pipeline array exactly once.

        The shared base strips ``pipeline`` and each variant re-applies its
        own, so the stage list can never be duplicated by a merge.
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

        pkg = out.with_suffix(".ortpackage")
        config = json.loads((pkg / "models" / "model" / "qnn_npu" / "genai_config.json").read_text())
        decoder = config["model"]["decoder"]
        stage_files = [next(iter(stage.values()))["filename"] for stage in decoder["pipeline"]]
        assert stage_files == ["e.onnx", "c.onnx", "i.onnx", "h.onnx"]
        assert not (pkg / "configs").exists()

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

        metadata = json.loads((out.with_suffix(".ortpackage") / "models" / "model" / "component.json").read_text())
        assert metadata["variants"]["vitia_npu"]["ep"] == "VitisAIExecutionProvider"
        overlay = json.loads(
            (out.with_suffix(".ortpackage") / "models" / "model" / "vitia_npu" / "genai_config.json").read_text()
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
    same ``genai_config.json``. All of those roles land in ONE component
    (``models/model/``) because ORT-GenAI selects exactly one component
    per package and then resolves every role's ``filename`` against that
    single selected variant directory (``src/models/multi_modal.cpp``
    builds the vision / embedding / decoder sessions from one
    ``config_path``). Splitting roles across components would produce a
    package ORT-GenAI refuses to open ("declares N components;
    onnxruntime-genai requires exactly one").
    """

    def test_all_roles_share_one_component(self, tmp_path):
        src = _create_vlm_source(tmp_path, "cpu_and_mobile")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        pkg = out.with_suffix(".ortpackage")
        manifest = json.loads((pkg / "manifest.json").read_text())
        assert manifest["components"] == {"model": "models/model"}
        assert (pkg / "models" / "model" / "component.json").is_file()
        # No per-role components: those would make the package unloadable.
        for role in ("vision", "embedding", "decoder"):
            assert not (pkg / "models" / role).exists(), f"role {role} leaked into its own component"

    def test_single_config_restores_every_role_filename(self, tmp_path):
        """The one variant config restores EVERY role's filename.

        The shared base strips per-role ``filename`` / ``session_options``;
        the variant re-applies all of them, because the single component
        owns every role and GenAI reads one complete config from it.
        """
        src = _create_vlm_source(tmp_path, "cpu_and_mobile")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        models = out.with_suffix(".ortpackage") / "models"
        cfg = json.loads((models / "model" / "cpu_and_mobile" / "genai_config.json").read_text())

        with_filename = {name for name, body in cfg["model"].items() if isinstance(body, dict) and "filename" in body}
        assert with_filename == {"vision", "embedding", "decoder"}

        assert cfg["model"]["decoder"]["filename"] == "text.onnx"
        assert cfg["model"]["vision"]["filename"] == "vision.onnx"
        assert cfg["model"]["embedding"]["filename"] == "embedding.onnx"

    def test_variant_dir_holds_every_roles_onnx(self, tmp_path):
        """The single variant dir holds all three roles' ONNX files, flat.

        GenAI resolves each role's ``filename`` relative to the selected
        variant directory, so every role's graph must be reachable from
        that one directory.
        """
        src = _create_vlm_source(tmp_path, "cpu_and_mobile")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        variant = out.with_suffix(".ortpackage") / "models" / "model" / "cpu_and_mobile"
        assert (variant / "text.onnx").is_file()
        assert (variant / "vision.onnx").is_file()
        assert (variant / "embedding.onnx").is_file()

    def test_variant_config_omits_unknown_component_marker(self, tmp_path):
        """No ``component`` marker is emitted into any role block.

        ORT-GenAI's config parser rejects unknown fields
        (``src/config.cpp`` throws ``JSON::unknown_value_error``) and has no
        ``component`` field, so emitting one would make the package
        unloadable.
        """
        src = _create_vlm_source(tmp_path, "cpu_and_mobile")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        config = json.loads(
            (out.with_suffix(".ortpackage") / "models" / "model" / "cpu_and_mobile" / "genai_config.json").read_text()
        )
        for name, body in config["model"].items():
            if isinstance(body, dict):
                assert "component" not in body, f"config leaked a component marker under {name!r}"

    def test_conflicting_non_cpu_eps_across_roles_raises(self, tmp_path):
        """Two roles demanding different non-CPU EPs is rejected up front.

        All roles share one component and therefore one variant EP.
        ORT-GenAI additionally has a single model-level device
        (``src/models/model.cpp``: "Running a model with multiple
        providers is not supported"), so such a package could never load.
        Failing during packaging gives a far clearer diagnostic.
        """
        src = tmp_path / "mixed"
        src.mkdir()
        for fname in ("vision.onnx", "text.onnx"):
            _make_onnx_inline(src / fname)
        (src / "genai_config.json").write_text(
            json.dumps(
                {
                    "model": {
                        "type": "qwen3vl",
                        "vision": {
                            "filename": "vision.onnx",
                            "session_options": {"provider_options": [{"qnn": {}}]},
                        },
                        "decoder": {
                            "head_size": 128,
                            "filename": "text.onnx",
                            "session_options": {"provider_options": [{"cuda": {}}]},
                        },
                    }
                }
            )
        )
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(tmp_path / "out")])

        with pytest.raises(ValueError, match="more than one non-CPU execution provider"):
            cmd.run()


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

    All roles land in ONE package component and therefore share one
    variant directory, so the packager preserves each role's source-side
    subdirectory prefix verbatim. Flattening to the basename would make
    all three roles collide at ``<variant>/model.onnx``; keeping the
    prefix is inherently collision-free (the files already coexisted in
    the source) and means the genai_config ``filename`` values need no
    rewriting at all.
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
    """End-to-end packaging of Mobius-style multi-role VLM sources.

    Every Mobius role (``decoder``/``embedding``/``vision``) goes into the
    single package component ``models/model/``, because ORT-GenAI opens
    exactly one component and resolves all of a model's role filenames
    against that component's selected variant directory. The variant
    directory therefore mirrors the source layout, keeping each role's
    subdirectory (``decoder/model.onnx`` etc.) so same-named graphs don't
    collide.
    """

    def test_all_roles_share_one_component(self, tmp_path):
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        pkg = out.with_suffix(".ortpackage")
        manifest = json.loads((pkg / "manifest.json").read_text())
        assert manifest["components"] == {"model": "models/model"}
        assert (pkg / "models" / "model" / "component.json").is_file()
        for role in ("decoder", "embedding", "vision"):
            assert not (pkg / "models" / role).exists(), f"role {role} leaked into its own component"

    def test_variant_dir_preserves_source_role_subdirs(self, tmp_path):
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        variant = out.with_suffix(".ortpackage") / "models" / "model" / "cpu"
        # Sibling roles share one variant dir, so the source-side subdir
        # must survive — flattening to the basename would collide.
        for subdir in ("decoder", "embedding", "vision_encoder"):
            assert (variant / subdir / "model.onnx").is_file(), f"missing {subdir}/model.onnx under the variant dir"

    def test_external_data_lands_next_to_its_onnx_in_role_subdir(self, tmp_path):
        """Each role's external-data blob lives next to its own ONNX.

        The ONNX file references ``model.onnx.data`` relative to its own
        directory and the loader resolves the same way, so the blob must
        follow the ONNX into its role subdirectory. Routing blobs to the
        variant root would collide all three at
        ``<variant>/model.onnx.data``.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu", with_external_data=True)
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        variant = out.with_suffix(".ortpackage") / "models" / "model" / "cpu"
        seen = set()
        for subdir, role in (("decoder", "decoder"), ("embedding", "embedding"), ("vision_encoder", "vision_encoder")):
            blob = variant / subdir / "model.onnx.data"
            assert blob.is_file(), f"external-data blob missing under {subdir}/"
            payload = blob.read_bytes()
            assert payload == f"role-{role}".encode() * 16, f"{subdir} blob holds another role's weights"
            seen.add(payload)
        assert len(seen) == 3, "role blobs were deduped/overwritten into each other"

    def test_overlay_filename_keeps_role_subdir_prefix(self, tmp_path):
        """The overlay's ``filename`` keeps the source's subdir-prefixed path.

        All roles share one variant dir, whose layout mirrors the source,
        so the packaged filenames are byte-identical to the source's. That
        also means GenAI's ``config_path / filename`` resolution finds each
        graph without any rewriting.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        model = json.loads(
            (out.with_suffix(".ortpackage") / "models" / "model" / "cpu" / "genai_config.json").read_text()
        )["model"]
        assert model["decoder"]["filename"] == "decoder/model.onnx"
        assert model["embedding"]["filename"] == "embedding/model.onnx"
        assert model["vision"]["filename"] == "vision_encoder/model.onnx"

    def test_shared_assets_exclude_model_artifact_subdirs(self, tmp_path):
        """``decoder/``/``embedding/``/``vision_encoder/`` must not leak into shared assets.

        Without explicit exclusion the config-file sweep would copy every
        source-root directory (including the model-artifact subdirs), so the
        package would carry duplicate ONNXs in the shared asset and bloat
        the deliverable. The sweep recognizes model-artifact subdirs via
        the genai_config's role filenames and skips them.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        pkg = out.with_suffix(".ortpackage")
        asset_dir = _shared_asset_dir(pkg)
        for excluded in ("decoder", "embedding", "vision_encoder"):
            assert not (asset_dir / excluded).exists(), f"{excluded}/ leaked into the shared asset"
        assert (asset_dir / "tokenizer_config.json").is_file()
        assert (asset_dir / "model_config.json").is_file()
        # genai_config.json is never a shared asset: each variant owns a copy.
        assert not (asset_dir / "genai_config.json").exists()
        assert not (pkg / "configs").exists()

    def test_variant_config_restores_every_filename_and_omits_component_marker(self, tmp_path):
        """The variant config restores every role's ``filename`` and adds no marker.

        The shared base strips per-role ``filename``; the single variant
        re-applies all of them. No ``component`` field is emitted because
        ORT-GenAI's config parser rejects unknown fields.
        """
        src = _create_mobius_vlm_source(tmp_path, "cpu")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        model = json.loads(
            (out.with_suffix(".ortpackage") / "models" / "model" / "cpu" / "genai_config.json").read_text()
        )["model"]
        with_filename = {name for name, body in model.items() if isinstance(body, dict) and "filename" in body}
        assert with_filename == {"decoder", "embedding", "vision"}
        for name, body in model.items():
            if isinstance(body, dict):
                assert "component" not in body, f"config leaked a component marker under {name!r}"

    def test_two_sources_become_two_variants_of_one_component(self, tmp_path):
        """CPU + GPU Mobius sources contribute one variant each to the single component.

        This is the user-reported scenario:
        ``olive generate-model-package -s cpu -s gpu -o cpu_gpu``. The
        package holds one component with two variants — ``cpu``
        (CPUExecutionProvider) and ``gpu`` (CUDAExecutionProvider) — each
        carrying the full three-role model.
        """
        cpu = _create_mobius_vlm_source(tmp_path, "cpu", ep="CPUExecutionProvider")
        gpu = _create_mobius_vlm_source(tmp_path, "gpu", ep="CUDAExecutionProvider")
        out = tmp_path / "cpu_gpu"
        cmd = _make_command(["generate-model-package", "-s", str(cpu), "-s", str(gpu), "-o", str(out)])

        cmd.run()

        component = out.with_suffix(".ortpackage") / "models" / "model"
        for variant in ("cpu", "gpu"):
            for subdir in ("decoder", "embedding", "vision_encoder"):
                assert (component / variant / subdir / "model.onnx").is_file(), (
                    f"missing models/model/{variant}/{subdir}/model.onnx"
                )
        metadata = json.loads((component / "component.json").read_text())
        assert metadata["variants"]["cpu"]["ep"] == "CPUExecutionProvider"
        assert metadata["variants"]["gpu"]["ep"] == "CUDAExecutionProvider"

    def test_variant_level_scalars_present_in_every_variant_config(self, tmp_path):
        """Variant-level scalars appear in every variant config, at their exact value.

        Each variant config is standalone, so it must carry the model-level
        scalars in full. Because the writer replaces rather than appends,
        a list-valued ``eos_token_id`` keeps exactly the source's entries
        instead of accumulating copies.
        """
        cpu = _create_mobius_vlm_source(tmp_path, "cpu", ep="CPUExecutionProvider")
        gpu = _create_mobius_vlm_source(tmp_path, "gpu", ep="CUDAExecutionProvider")
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(cpu), "-s", str(gpu), "-o", str(out)])

        cmd.run()

        component = out.with_suffix(".ortpackage") / "models" / "model"
        source_model = json.loads((cpu / "genai_config.json").read_text())["model"]
        for variant in ("cpu", "gpu"):
            model = json.loads((component / variant / "genai_config.json").read_text())["model"]
            for key in ("context_length", "type", "eos_token_id", "pad_token_id", "bos_token_id"):
                if key in source_model:
                    assert model[key] == source_model[key], f"{variant}/{key} diverged from the source"

    def test_explicit_cpu_role_keeps_cpu_provider_options(self, tmp_path):
        """A role with explicit CPU ``provider_options`` keeps them, and doesn't sway the variant EP.

        Mobius outputs sometimes mark a helper role (e.g. ``embedding``) as
        CPU even inside a predominantly-GPU build. The variant EP is the
        single non-CPU EP across roles (CUDA here) — CPU roles are exempt
        because ORT-GenAI registers CPU implicitly and a CPU role never
        claims the model's device.
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

        component = out.with_suffix(".ortpackage") / "models" / "model"
        assert json.loads((component / "component.json").read_text())["variants"]["gpu"]["ep"] == (
            "CUDAExecutionProvider"
        )
        model = json.loads((component / "gpu" / "genai_config.json").read_text())["model"]
        # Critical: the CPU role's explicit (empty) provider_options survive
        # so the loader doesn't put the embedding on CUDA.
        assert model["embedding"]["session_options"]["provider_options"] == []
        assert model["decoder"]["session_options"]["provider_options"] == [{"cuda": {}}]
        assert model["vision"]["session_options"]["provider_options"] == [{"cuda": {}}]

    def test_base_config_source_picks_richest_role_set(self, tmp_path):
        """When sources expose different role sets, the base config is taken from the source with the most roles.

        Every variant config is derived from that base, so a base missing
        role blocks would produce variant configs that cannot describe the
        whole model. Example: gpu source only has decoder; cpu source has
        all three. The base must come from cpu so each variant config
        still carries embedding/vision.
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

        # Every variant config must carry all three role blocks (inherited
        # from the richest base). If first-source-wins ran, only decoder
        # would appear.
        config = json.loads(
            (out.with_suffix(".ortpackage") / "models" / "model" / "cpu" / "genai_config.json").read_text()
        )
        for role in ("decoder", "embedding", "vision"):
            assert role in config["model"], f"variant config missing {role} block; wrong source selected"


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

    A direct ``write_model_package`` caller may split roles across
    components (the plain ORT model-package spec allows it), but each
    genai_config role must still belong to exactly one component. A caller
    could violate that by hand (e.g. by reusing the same source_genai under
    two component names); ``write_model_package`` detects the conflict at
    the role_to_component build step and raises rather than silently keep
    one mapping and drop the other.
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
