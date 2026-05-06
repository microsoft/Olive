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
    model_attributes: dict,
    *,
    onnx_metadata: dict[str, str] | None = None,
    inference_settings: dict | None = None,
) -> Path:
    """Create a fake Olive output directory with model_config.json and a real ONNX file."""
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)
    onnx_path = source_dir / "model.onnx"
    _make_onnx_inline(onnx_path, metadata_props=onnx_metadata)
    cfg: dict = {"model_path": str(onnx_path), "model_attributes": model_attributes}
    if inference_settings is not None:
        cfg["inference_settings"] = inference_settings
    model_config = {"type": "ONNXModel", "config": cfg}
    (source_dir / "model_config.json").write_text(json.dumps(model_config))
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
        src = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(tmp_path / "out")])

        sources = cmd._parse_sources()

        assert sources == [("soc_60", src)]

    def test_rejects_missing_model_config(self, tmp_path):
        no_config = tmp_path / "no_config"
        no_config.mkdir()
        valid = _create_source_dir(tmp_path, "valid", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(
            ["generate-model-package", "-s", str(no_config), "-s", str(valid), "-o", str(tmp_path / "out")]
        )

        with pytest.raises(ValueError, match=r"model_config\.json"):
            cmd._parse_sources()

    def test_rejects_nonexistent_path(self, tmp_path):
        valid = _create_source_dir(tmp_path, "valid", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(
            ["generate-model-package", "-s", "/nonexistent/path", "-s", str(valid), "-o", str(tmp_path / "out")]
        )

        with pytest.raises(ValueError, match="does not exist"):
            cmd._parse_sources()

    def test_rejects_duplicate_source_basenames(self, tmp_path):
        # Two source dirs share basename "soc_60" — variant names would collide.
        src_a = _create_source_dir(tmp_path / "a", "soc_60", {"ep": "QNNExecutionProvider"})
        src_b = _create_source_dir(tmp_path / "b", "soc_60", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(["generate-model-package", "-s", str(src_a), "-s", str(src_b), "-o", str(tmp_path / "out")])

        with pytest.raises(ValueError, match="share the directory name"):
            cmd._parse_sources()

    def test_parses_two_valid_sources(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider"})
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
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider", "device": "NPU"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider", "device": "NPU"})
        out = tmp_path / "out"
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

        # assert: top-level layout (no models/ wrapper)
        assert (out / "manifest.json").is_file()
        assert not (out / "models").exists()

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["schema_version"] == 1
        assert manifest["components"] == ["model"]
        assert manifest["producer"]["model_name"] == "test_model"
        assert manifest["producer"]["model_version"] == "2.0"

        # metadata uses ep_compatibility[]
        metadata = json.loads((out / "model" / "metadata.json").read_text())
        assert set(metadata["variants"]) == {"soc_60", "soc_73"}
        for variant_payload in metadata["variants"].values():
            ep_compat = variant_payload["ep_compatibility"]
            assert ep_compat == [{"ep": "QNNExecutionProvider", "device": "NPU"}]

        # variant.json contains files[] with filename
        for v in ("soc_60", "soc_73"):
            variant_json = json.loads((out / "model" / v / "variant.json").read_text())
            assert variant_json["files"][0]["filename"] == "model.onnx"
            assert (out / "model" / v / "model.onnx").is_file()


class TestGeneratePackageSingleSource:
    def test_single_source_is_valid_package(self, tmp_path):
        src = _create_source_dir(tmp_path, "cpu_x64", {"ep": "CPUExecutionProvider"})
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        cmd.run()

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["components"] == ["model"]
        metadata = json.loads((out / "model" / "metadata.json").read_text())
        assert "cpu_x64" in metadata["variants"]
        assert metadata["variants"]["cpu_x64"]["ep_compatibility"] == [{"ep": "CPUExecutionProvider"}]
        # No shared_weights because nothing to dedup.
        assert not (out / "model" / "shared_weights").exists()


# ---------------------------------------------------------------------------
# Writer: layout + manifest + metadata + variant.json
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
        assert (out / "decoder" / "metadata.json").is_file()
        assert (out / "decoder" / "cpu" / "variant.json").is_file()
        assert (out / "decoder" / "cpu" / "model.onnx").is_file()
        assert not (out / "models").exists()

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
        assert manifest["producer"] == {
            "tool": "olive-ai",
            "tool_version": "1.2.3",
            "model_name": "demo",
        }
        # No legacy fields
        assert "name" not in manifest
        assert "component_models" not in manifest
        assert "model_version" not in manifest

    def test_metadata_uses_ep_compatibility_array(self, tmp_path):
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
                    compatibility=["soc_60", "soc_69"],
                )
            ],
        )

        metadata = json.loads((out / "decoder" / "metadata.json").read_text())
        ep_compat = metadata["variants"]["qnn-npu"]["ep_compatibility"]
        assert ep_compat == [{"ep": "QNNExecutionProvider", "device": "NPU", "compatibility": ["soc_60", "soc_69"]}]
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

        metadata = json.loads((out / "decoder" / "metadata.json").read_text())
        ep_compat = metadata["variants"]["cpu"]["ep_compatibility"][0]
        assert ep_compat == {"ep": "CPUExecutionProvider"}

    def test_variant_json_carries_session_and_provider_options(self, tmp_path):
        onnx_path = _make_onnx_inline(tmp_path / "src" / "model.onnx")
        out = tmp_path / "package"

        inference = {
            "session_options": {"graph_optimization_level": 3},
            "execution_provider": ["CPUExecutionProvider"],
            "provider_options": [{"intra_op_num_threads": 4}],
        }

        write_model_package(
            output_dir=out,
            variants=[
                VariantSpec(
                    component_name="decoder",
                    variant_name="cpu",
                    onnx_files=[onnx_path],
                    ep="CPUExecutionProvider",
                    inference_settings=inference,
                )
            ],
        )

        variant = json.loads((out / "decoder" / "cpu" / "variant.json").read_text())
        assert variant["files"] == [
            {
                "filename": "model.onnx",
                "session_options": {"graph_optimization_level": 3},
                "provider_options": {"intra_op_num_threads": 4},
            }
        ]

    def test_provider_options_match_ep_by_name(self, tmp_path):
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

        variant = json.loads((out / "decoder" / "qnn" / "variant.json").read_text())
        assert variant["files"][0].get("provider_options") == {"backend_path": "QnnHtp.so"}
        assert "session_options" not in variant["files"][0]


# ---------------------------------------------------------------------------
# Writer: shared_weights / external-data dedup
# ---------------------------------------------------------------------------


class TestSharedWeightsDedup:
    def test_dedups_identical_external_data_across_variants(self, tmp_path):
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

        shared_root = out / "decoder" / "shared_weights"
        assert shared_root.is_dir()
        sha_dirs = list(shared_root.iterdir())
        assert len(sha_dirs) == 1
        sha = sha_dirs[0].name
        assert (shared_root / sha / "model.onnx.data").is_file()
        assert not (out / "decoder" / "v1" / "model.onnx.data").exists()
        assert not (out / "decoder" / "v2" / "model.onnx.data").exists()

        for v in ("v1", "v2"):
            variant = json.loads((out / "decoder" / v / "variant.json").read_text())
            entry = variant["files"][0]
            assert entry["filename"] == "model.onnx"
            assert entry["shared_files"] == {"model.onnx.data": sha}

    def test_keeps_external_data_inline_when_unique(self, tmp_path):
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

        assert not (out / "decoder" / "shared_weights").exists()
        assert (out / "decoder" / "v1" / "model.onnx.data").is_file()
        assert (out / "decoder" / "v2" / "model.onnx.data").is_file()

        for v in ("v1", "v2"):
            variant = json.loads((out / "decoder" / v / "variant.json").read_text())
            assert "shared_files" not in variant["files"][0]

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

        assert (out / "decoder" / "cpu" / "model.onnx.data").is_file()
        assert not (out / "decoder" / "shared_weights").exists()
        variant = json.loads((out / "decoder" / "cpu" / "variant.json").read_text())
        assert "shared_files" not in variant["files"][0]


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
# CLI: mixed source types
# ---------------------------------------------------------------------------


class TestMixedSourceTypes:
    def test_rejects_mixed_onnx_and_composite(self, tmp_path):
        # setup: one ONNXModel source, one CompositeModel source
        onnx_src = _create_source_dir(tmp_path, "onnx_src", {"ep": "CPUExecutionProvider"})
        comp_src = tmp_path / "comp_src"
        comp_src.mkdir()
        comp_onnx = _make_onnx_inline(comp_src / "comp.onnx")
        (comp_src / "model_config.json").write_text(
            json.dumps(
                {
                    "type": "CompositeModel",
                    "config": {
                        "model_components": [{"type": "ONNXModel", "config": {"model_path": str(comp_onnx)}}],
                        "component_names": ["decoder"],
                    },
                }
            )
        )
        cmd = _make_command(
            ["generate-model-package", "-s", str(onnx_src), "-s", str(comp_src), "-o", str(tmp_path / "out")]
        )

        # execute + assert
        with pytest.raises(ValueError, match="mix model types"):
            cmd.run()


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
    def test_splits_comma_delimited_metadata(self, tmp_path):
        # setup: source with QNNExecutionProvider compat info in ONNX metadata_props
        src = _create_source_dir(
            tmp_path,
            "soc_60",
            {"ep": "QNNExecutionProvider", "device": "NPU"},
            onnx_metadata={"ep_compatibility_info.QNNExecutionProvider": "soc_60,soc_69,soc_73"},
        )
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        # execute
        cmd.run()

        # assert: compatibility array reflects the comma-split list
        metadata = json.loads((out / "model" / "metadata.json").read_text())
        ep_compat = metadata["variants"]["soc_60"]["ep_compatibility"][0]
        assert ep_compat["ep"] == "QNNExecutionProvider"
        assert ep_compat["compatibility"] == ["soc_60", "soc_69", "soc_73"]


# ---------------------------------------------------------------------------
# CLI: composite (per-component inference_settings precedence)
# ---------------------------------------------------------------------------


def _create_composite_source(
    tmp_path: Path,
    name: str,
    components: list[dict],
    component_names: list[str],
    *,
    target_inference: dict | None = None,
    target_attrs: dict | None = None,
) -> Path:
    """Create an Olive-style composite source dir."""
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)
    cfg = {"model_components": components, "component_names": component_names}
    if target_inference is not None:
        cfg["inference_settings"] = target_inference
    if target_attrs is not None:
        cfg["model_attributes"] = target_attrs
    (source_dir / "model_config.json").write_text(json.dumps({"type": "CompositeModel", "config": cfg}))
    return source_dir


class TestCompositeBuild:
    def test_per_component_inference_settings_wins(self, tmp_path):
        # setup: component-level inference_settings should override target-level
        comp_a_onnx = _make_onnx_inline(tmp_path / "comp_a" / "model.onnx")
        comp_b_onnx = _make_onnx_inline(tmp_path / "comp_b" / "model.onnx")

        target_inference = {
            "session_options": {"graph_optimization_level": 1},
            "execution_provider": ["CPUExecutionProvider"],
            "provider_options": [{}],
        }
        comp_b_inference = {
            "session_options": {"graph_optimization_level": 99},
            "execution_provider": ["CPUExecutionProvider"],
            "provider_options": [{}],
        }
        components = [
            {"type": "ONNXModel", "config": {"model_path": str(comp_a_onnx)}},
            {
                "type": "ONNXModel",
                "config": {"model_path": str(comp_b_onnx), "inference_settings": comp_b_inference},
            },
        ]
        src = _create_composite_source(
            tmp_path,
            "soc_60",
            components,
            ["encoder", "decoder"],
            target_inference=target_inference,
            target_attrs={"ep": "CPUExecutionProvider"},
        )
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(out)])

        # execute
        cmd.run()

        # assert: encoder uses target-level, decoder uses component-level
        encoder_v = json.loads((out / "encoder" / "soc_60" / "variant.json").read_text())
        assert encoder_v["files"][0]["session_options"] == {"graph_optimization_level": 1}

        decoder_v = json.loads((out / "decoder" / "soc_60" / "variant.json").read_text())
        assert decoder_v["files"][0]["session_options"] == {"graph_optimization_level": 99}


# ---------------------------------------------------------------------------
# CLI: unsupported model type
# ---------------------------------------------------------------------------


class TestUnsupportedModelType:
    def test_rejects_pytorch_model(self, tmp_path):
        # setup: a source whose model_config declares an unsupported type
        source_dir = tmp_path / "pytorch_src"
        source_dir.mkdir()
        (source_dir / "model_config.json").write_text(
            json.dumps({"type": "PyTorchModel", "config": {"model_path": "pt"}})
        )
        out = tmp_path / "out"
        cmd = _make_command(["generate-model-package", "-s", str(source_dir), "-o", str(out)])

        # execute + assert
        with pytest.raises(ValueError, match="Unsupported source model type"):
            cmd.run()
