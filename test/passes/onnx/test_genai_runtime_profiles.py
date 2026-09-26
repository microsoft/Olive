# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
import types
from pathlib import Path

import onnx

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
    (source_dir / "model.onnx.data").write_bytes(b"shared weights")
    (source_dir / "embedding.onnx.data").hardlink_to(source_dir / "model.onnx.data")
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
    assert not (config_only_output / "model_32gib.onnx").exists()
