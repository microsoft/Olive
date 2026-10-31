# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import pytest

from olive.cache import CacheConfig
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.genai_runtime_profiles import GenAIModelRuntimeProfiles


def _make_model(path: Path):
    graph = onnx.helper.make_graph([], "test", [], [])
    onnx.save(onnx.helper.make_model(graph), path)


def test_genai_runtime_profiles_creates_variant_and_runtime_overlay(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for filename in ("model.onnx", "dflash2.onnx", "embedding.onnx"):
        _make_model(source_dir / filename)
    (source_dir / "model_32gib.onnx").write_bytes(b"existing variant")
    (source_dir / "model.onnx.data").write_bytes(b"shared weights")
    (source_dir / "embedding.onnx.data").hardlink_to(source_dir / "model.onnx.data")
    (source_dir / "genai_config.json").write_text(
        json.dumps(
            {
                "engine": {"dynamic_batching": {}},
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "shared_initializers": [{"name": "weight", "data_file": "model.onnx.data"}],
                    },
                    "embedding": {"filename": "embedding.onnx"},
                },
            }
        ),
        encoding="utf-8",
    )

    calls = []

    class FakeKVCacheVariant:
        def __init__(self, scheme):
            self.scheme = scheme

        def create(self, source, output, scale_file):
            calls.append((Path(source), Path(output), self.scheme, scale_file))
            Path(output).write_bytes(Path(source).read_bytes())

    variant_module = types.ModuleType("onnxruntime_genai.models.kv_cache_variant")
    variant_module.KVCacheVariant = FakeKVCacheVariant
    monkeypatch.setitem(sys.modules, "onnxruntime_genai.models.kv_cache_variant", variant_module)

    component_names = ["model.onnx", "dflash2.onnx", "embedding.onnx"]
    model = CompositeModelHandler(
        [ONNXModelHandler(source_dir, onnx_file_name=name) for name in component_names],
        component_names,
        model_path=source_dir,
    )
    output_dir = tmp_path / "output"
    profile = {
        "id": "32gib",
        "kv_cache": {"scheme": "int8_per_channel", "scale_file": "scales.json"},
        "eligibility": {"minimum_total_device_memory_bytes": 34359738368},
        "overlay": {
            "engine": {"dynamic_batching": {"max_batch_size": 8, "num_blocks": 928}},
            "search": {"chunk_size": 512},
        },
    }

    output_model = create_pass_from_dict(
        GenAIModelRuntimeProfiles,
        {"runtime_profiles": [profile]},
        disable_search=True,
    ).run(model, output_dir)

    assert isinstance(output_model, CompositeModelHandler)
    assert "model_32gib.onnx" in dict(output_model.get_model_components())
    assert calls == [(output_dir / "model.onnx", output_dir / "model_32gib.onnx", "int8_per_channel", "scales.json")]
    assert (output_dir / "model.onnx.data").read_bytes() == b"shared weights"
    generated_config = json.loads((output_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert generated_config["model"]["decoder"]["shared_initializers"][0]["data_file"] == "embedding.onnx.data"
    assert generated_config["runtime_profiles"] == [
        {
            "id": "32gib",
            "eligibility": {"minimum_total_device_memory_bytes": 34359738368},
            "overlay": {
                "engine": {"dynamic_batching": {"max_batch_size": 8, "num_blocks": 928}},
                "search": {"chunk_size": 512},
                "model": {"decoder": {"filename": "model_32gib.onnx"}},
            },
        }
    ]
    assert (source_dir / "model_32gib.onnx").read_bytes() == b"existing variant"
    assert "runtime_profiles" not in json.loads((source_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert "kv_cache" in profile

    config_only_output = tmp_path / "config_only_output"
    (source_dir / "embedding.onnx.data").unlink()
    (source_dir / "embedding.onnx.data").write_bytes(b"different weights")
    config_only_profile = {
        "id": "32gib",
        "eligibility": {"minimum_total_device_memory_bytes": 34359738368},
        "overlay": {"engine": {"dynamic_batching": {"max_batch_size": 8, "num_blocks": 928}}},
    }
    create_pass_from_dict(
        GenAIModelRuntimeProfiles,
        {"runtime_profiles": [config_only_profile]},
        disable_search=True,
    ).run(model, config_only_output)

    assert len(calls) == 1
    config_only = json.loads((config_only_output / "genai_config.json").read_text(encoding="utf-8"))
    assert config_only["runtime_profiles"] == [config_only_profile]
    assert config_only["model"]["decoder"]["shared_initializers"][0]["data_file"] == "model.onnx.data"
    assert (config_only_output / "model_32gib.onnx").read_bytes() == b"existing variant"


@pytest.mark.parametrize("source_filename", ["../outside.onnx", None])
def test_genai_runtime_profiles_rejects_unsafe_decoder_filename(tmp_path, source_filename):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _make_model(source_dir / "model.onnx")
    outside_model = tmp_path / "outside.onnx"
    _make_model(outside_model)
    if source_filename is None:
        source_filename = str(outside_model)
    (source_dir / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"filename": source_filename}}}), encoding="utf-8"
    )
    model = CompositeModelHandler(
        [ONNXModelHandler(source_dir, onnx_file_name="model.onnx")], ["model.onnx"], model_path=source_dir
    )
    profile = {
        "id": "default",
        "eligibility": {"minimum_total_device_memory_bytes": 0},
        "overlay": {},
    }

    with pytest.raises(ValueError, match="relative path within the model directory"):
        create_pass_from_dict(GenAIModelRuntimeProfiles, {"runtime_profiles": [profile]}, disable_search=True).run(
            model, tmp_path / "output"
        )


@pytest.mark.parametrize("profile_filename", ["../outside.onnx", None])
def test_genai_runtime_profiles_rejects_unsafe_profile_decoder_filename(tmp_path, profile_filename):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _make_model(source_dir / "model.onnx")
    outside_model = tmp_path / "outside.onnx"
    _make_model(outside_model)
    if profile_filename is None:
        profile_filename = str(outside_model)
    (source_dir / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"filename": "model.onnx"}}}), encoding="utf-8"
    )
    model = CompositeModelHandler(
        [ONNXModelHandler(source_dir, onnx_file_name="model.onnx")], ["model.onnx"], model_path=source_dir
    )
    profile = {
        "id": "default",
        "eligibility": {"minimum_total_device_memory_bytes": 0},
        "overlay": {"model": {"decoder": {"filename": profile_filename}}},
    }

    with pytest.raises(ValueError, match="relative path within the model directory"):
        create_pass_from_dict(GenAIModelRuntimeProfiles, {"runtime_profiles": [profile]}, disable_search=True).run(
            model, tmp_path / "output"
        )


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (
            {
                "id": "invalid-range",
                "eligibility": {
                    "minimum_total_device_memory_bytes": 2,
                    "maximum_total_device_memory_bytes": 1,
                },
                "overlay": {},
            },
            "maximum_total_device_memory_bytes",
        ),
        (
            {
                "id": "invalid-overlay",
                "eligibility": {"minimum_total_device_memory_bytes": 0},
                "overlay": [],
            },
            "overlay must be a mapping",
        ),
        (
            {
                "id": "empty-overlay",
                "eligibility": {"minimum_total_device_memory_bytes": 0},
                "overlay": {},
            },
            "overlay must contain at least one overlay field",
        ),
        (
            {
                "id": "invalid-chunk-size",
                "eligibility": {"minimum_total_device_memory_bytes": 0},
                "overlay": {"search": {"chunk_size": 0}},
            },
            "chunk_size",
        ),
    ],
)
def test_genai_runtime_profiles_rejects_invalid_profile(tmp_path, profile, message):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _make_model(source_dir / "model.onnx")
    (source_dir / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"filename": "model.onnx"}}}), encoding="utf-8"
    )
    model = CompositeModelHandler(
        [ONNXModelHandler(source_dir, onnx_file_name="model.onnx")], ["model.onnx"], model_path=source_dir
    )

    with pytest.raises(ValueError, match=message):
        create_pass_from_dict(GenAIModelRuntimeProfiles, {"runtime_profiles": [profile]}, disable_search=True).run(
            model, tmp_path / "output"
        )


def test_genai_runtime_profiles_rejects_overlapping_eligibility(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _make_model(source_dir / "model.onnx")
    (source_dir / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"filename": "model.onnx"}}}), encoding="utf-8"
    )
    model = CompositeModelHandler(
        [ONNXModelHandler(source_dir, onnx_file_name="model.onnx")], ["model.onnx"], model_path=source_dir
    )
    profiles = [
        {
            "id": "small",
            "eligibility": {
                "minimum_total_device_memory_bytes": 0,
                "maximum_total_device_memory_bytes": 10,
            },
            "overlay": {},
        },
        {
            "id": "large",
            "eligibility": {"minimum_total_device_memory_bytes": 10},
            "overlay": {},
        },
    ]

    with pytest.raises(ValueError, match="eligibility overlaps"):
        create_pass_from_dict(GenAIModelRuntimeProfiles, {"runtime_profiles": profiles}, disable_search=True).run(
            model, tmp_path / "output"
        )


def test_genai_runtime_profiles_preserves_shared_external_data_in_final_output(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weight = onnx.numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "weight")
    graph = onnx.helper.make_graph([], "shared", [], [], [weight])
    onnx.save_model(
        onnx.helper.make_model(graph),
        source_dir / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    (source_dir / "embedding.onnx").write_bytes((source_dir / "model.onnx").read_bytes())
    (source_dir / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "shared_initializers": [{"name": "weight", "data_file": "model.onnx.data"}],
                    },
                    "embedding": {"filename": "embedding.onnx"},
                }
            }
        )
    )
    component_names = ["embedding.onnx", "model.onnx"]
    model = CompositeModelHandler(
        [ONNXModelHandler(source_dir, onnx_file_name=name) for name in component_names],
        component_names,
        model_path=source_dir,
        model_attributes={"additional_files": [str(source_dir / "genai_config.json")]},
    )
    profile = {
        "id": "32gib",
        "eligibility": {"minimum_total_device_memory_bytes": 34359738368},
        "overlay": {"search": {"chunk_size": 512}},
    }
    output_model = create_pass_from_dict(
        GenAIModelRuntimeProfiles, {"runtime_profiles": [profile]}, disable_search=True
    ).run(model, tmp_path / "profiled")
    cache = CacheConfig(cache_dir=tmp_path / "cache").create_cache()
    cache.cache_model("profiled", output_model.to_json())
    final_dir = tmp_path / "final"
    cache.save_model("profiled", final_dir)

    assert (final_dir / "model.onnx.data").is_file()
    assert not (final_dir / "embedding.onnx.data").exists()
    for name in component_names:
        final_graph = onnx.load(final_dir / name, load_external_data=False)
        assert {
            entry.value
            for tensor in final_graph.graph.initializer
            for entry in tensor.external_data
            if entry.key == "location"
        } == {"model.onnx.data"}
    final_config = json.loads((final_dir / "genai_config.json").read_text())
    assert final_config["model"]["decoder"]["shared_initializers"][0]["data_file"] == "model.onnx.data"
