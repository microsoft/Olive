# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import json
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from olive.workflows.run import hf_component_assembly as assembly_module
from olive.workflows.run.hf_component_assembly import (
    _assembly_lock,
    _Checkpoint,
    _merge_quantization_config,
    _validate_build_compatibility,
    try_assemble_hf_component_builds,
)


def _quantization_config(
    *,
    group_size,
    symmetric,
    quantize_vision,
    skips,
    overrides=None,
    tie_word_embeddings=False,
    lm_head=False,
    embeds=False,
):
    return {
        "quant_method": "olive",
        "bits": 4,
        "group_size": group_size,
        "symmetric": symmetric,
        "lm_head": lm_head,
        "embeds": embeds,
        "moe": False,
        "quantize_vision": quantize_vision,
        "modules_to_not_convert": skips,
        "overrides": overrides,
        "tie_word_embeddings": tie_word_embeddings,
    }


def _write_checkpoint(path: Path, tensors: dict, quantization_config: dict, model_config: dict | None = None):
    path.mkdir(parents=True)
    config = {
        "model_type": "llama",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "quantization_config": quantization_config,
        **(model_config or {}),
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    save_file(tensors, path / "model.safetensors")


def _run_config(output_dir: Path, component: str, source_path: str, pass_type: str):
    return SimpleNamespace(
        input_model=SimpleNamespace(
            type="hfmodel",
            config={
                "model_attributes": {
                    "component_name": component,
                    "component_source_paths": [source_path],
                }
            },
        ),
        engine=SimpleNamespace(output_dir=output_dir),
        passes={"pass": [SimpleNamespace(type=pass_type)]},
    )


def _result(model_dir: Path, device="cpu", execution_provider="CPUExecutionProvider"):
    class Output:
        model_type = "hfmodel"

        def __init__(self):
            self.model_path = str(model_dir)
            self.olive_model_config = {
                "type": "hfmodel",
                "config": {
                    "model_path": str(model_dir),
                    "model_attributes": {"component_name": "stale"},
                },
            }

        def _update_with_model_config(self, model_config):
            self.olive_model_config = model_config
            self.model_path = model_config["config"]["model_path"]

        def from_device(self):
            return device

        def from_execution_provider(self):
            return execution_provider

    output = Output()
    return SimpleNamespace(get_best_candidate=lambda: output)


def _checkpoint_keys(root: Path) -> set[str]:
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    keys = set()
    for filename in set(index["weight_map"].values()):
        with safe_open(root / filename, framework="pt") as handle:
            keys.update(handle.keys())
    return keys


class _MetadataCheckpoint:
    def __init__(self, metadata):
        self._metadata = metadata

    @property
    def keys(self):
        return set(self._metadata)

    def metadata(self, key):
        return self._metadata[key]


def _metadata_artifact(name, source_path, quantization_config, metadata, model_config=None):
    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "quantization_config": quantization_config,
        **(model_config or {}),
    }
    return SimpleNamespace(
        name=name,
        source_paths=[source_path],
        checkpoint=_MetadataCheckpoint(metadata),
        config=config,
        components=[name],
        pass_types=["rtn"],
    )


def test_rejects_safetensors_shards_outside_checkpoint_root(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with ExitStack() as stack, pytest.raises(ValueError, match="Unsafe safetensors shard path"):
        _Checkpoint(checkpoint, stack)


def test_assembles_disjoint_hf_components_and_preserves_unbuilt_weights(tmp_path):
    parent = tmp_path / "assembled"
    decoder_output = parent / "decoder-int4"
    vision_output = parent / "vision-rtn"
    decoder_model = decoder_output / "model"
    vision_model = vision_output / "model"
    decoder_output.mkdir(parents=True)
    vision_output.mkdir(parents=True)
    (decoder_output / "model_config.json").write_text("{}", encoding="utf-8")
    (vision_output / "model_config.json").write_text("{}", encoding="utf-8")
    tied_config = {
        "tie_word_embeddings": True,
        "text_config": {"tie_word_embeddings": True},
    }

    _write_checkpoint(
        decoder_model,
        {
            "model.decoder.weight_qweight": torch.zeros(8, 4, dtype=torch.uint8),
            "model.decoder.weight_scales": torch.ones(8, 2),
            "model.vision.weight": torch.ones(8, 8),
            "model.audio.weight": torch.full((8, 8), 2.0),
        },
        _quantization_config(
            group_size=32,
            symmetric=False,
            quantize_vision=False,
            skips=["model.vision", "model.audio"],
        ),
        model_config=tied_config,
    )
    _write_checkpoint(
        vision_model,
        {
            "model.decoder.weight": torch.ones(8, 8),
            "model.vision.weight_qweight": torch.zeros(8, 4, dtype=torch.uint8),
            "model.vision.weight_scales": torch.ones(8, 1),
            "model.audio.weight": torch.full((8, 8), 2.0),
        },
        _quantization_config(
            group_size=128,
            symmetric=True,
            quantize_vision=True,
            skips=["model.decoder", "model.audio"],
        ),
        model_config=tied_config,
    )

    build_configs = OrderedDict(
        [
            ("decoder-build", _run_config(decoder_output, "decoder", "model.decoder", "kquant")),
            ("vision-build", _run_config(vision_output, "vision_encoder", "model.vision", "rtn")),
        ]
    )
    results = OrderedDict(
        [
            ("decoder-build", _result(decoder_model)),
            ("vision-build", _result(vision_model)),
        ]
    )

    assembled = try_assemble_hf_component_builds(build_configs, results, parent)

    assert assembled == parent
    assert _checkpoint_keys(parent) == {
        "model.decoder.weight_qweight",
        "model.decoder.weight_scales",
        "model.vision.weight_qweight",
        "model.vision.weight_scales",
        "model.audio.weight",
    }
    config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"]["group_size"] == 32
    assert config["quantization_config"]["symmetric"] is False
    assert config["quantization_config"]["tie_word_embeddings"] is False
    assert config["tie_word_embeddings"] is True
    assert config["text_config"]["tie_word_embeddings"] is True
    assert config["quantization_config"]["overrides"]["model.vision"] == {
        "symmetric": True,
        "group_size": 128,
    }
    assert config["quantization_config"]["modules_to_not_convert"] == [r"re:^model\.audio$"]
    assert config["component_quantization"]["decoder"]["group_size"] == 32
    assert config["component_quantization"]["vision_encoder"]["group_size"] == 128
    assert config["component_quantization"]["decoder"]["modules_to_not_convert"] == [
        "model.vision",
        "model.audio",
    ]
    assert config["component_quantization"]["vision_encoder"]["modules_to_not_convert"] == [
        "model.decoder",
        "model.audio",
    ]
    assert config["olive_component_quantization"]["decoder-build"]["passes"] == ["kquant"]
    assert config["olive_component_quantization"]["vision-build"]["passes"] == ["rtn"]
    assert AutoConfig.from_pretrained(parent).model_type == "llama"
    parent_model_config = json.loads((parent / "model_config.json").read_text(encoding="utf-8"))
    assert parent_model_config["config"]["model_path"] == str(parent)
    assert parent_model_config["config"]["model_attributes"] == {"assembled_components": ["decoder", "vision_encoder"]}
    assert results["decoder-build"].get_best_candidate().model_path == str(parent)
    assert results["vision-build"].get_best_candidate().model_path == str(parent)
    assert not decoder_model.exists()
    assert not vision_model.exists()
    assert json.loads((decoder_output / "model_config.json").read_text(encoding="utf-8"))["config"][
        "model_path"
    ] == str(parent)
    assert json.loads((vision_output / "model_config.json").read_text(encoding="utf-8"))["config"]["model_path"] == str(
        parent
    )
    assert (decoder_output / "component.json").is_file()
    assert (vision_output / "component.json").is_file()
    index = json.loads((parent / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(index["weight_map"].values()) == {
        "model-unoptimized-00001.safetensors",
        "decoder-build/model-00001.safetensors",
        "vision-build/model-00001.safetensors",
    }
    decoder_manifest = json.loads((decoder_output / "component.json").read_text(encoding="utf-8"))
    assert decoder_manifest["quantization_config"]["group_size"] == 32


def test_quantization_merge_resolves_effective_overrides_and_float_skips():
    decoder_config = _quantization_config(
        group_size=32,
        symmetric=False,
        quantize_vision=False,
        skips=["model"],
        overrides={r"re:^model\.vision$": {"group_size": 16}},
        tie_word_embeddings=True,
        lm_head=True,
        embeds=True,
    )
    vision_config = _quantization_config(
        group_size=128,
        symmetric=True,
        quantize_vision=True,
        skips=["model.decoder", "model.audio"],
    )
    decoder = _metadata_artifact(
        "decoder",
        "model.decoder",
        decoder_config,
        {
            "model.decoder.weight_qweight": ((8, 4), "U8"),
            "model.decoder.weight_scales": ((8, 2), "F32"),
            "model.decoder.embed_tokens.weight_qweight": ((8, 4), "U8"),
            "model.decoder.embed_tokens.weight_scales": ((8, 2), "F32"),
            "model.audio.weight": ((8, 8), "F32"),
        },
    )
    vision = _metadata_artifact(
        "vision",
        "model.vision",
        vision_config,
        {
            "model.vision.weight_qweight": ((8, 4), "U8"),
            "model.vision.weight_scales": ((8, 1), "F32"),
            "model.audio.weight": ((8, 8), "F32"),
        },
    )

    merged, _ = _merge_quantization_config([decoder, vision])
    qconfig = assembly_module.OliveHfQuantizationConfig(**merged)

    assert qconfig.get_qlinear_init_args("model.decoder") == {
        "bits": 4,
        "symmetric": False,
        "group_size": 32,
    }
    assert qconfig.get_qlinear_init_args("model.vision") == {
        "bits": 4,
        "symmetric": True,
        "group_size": 128,
    }
    assert qconfig.modules_to_not_convert == [r"re:^model\.audio$"]
    assert qconfig.tie_word_embeddings is True
    model_config = {
        "tie_word_embeddings": False,
        "text_config": {"tie_word_embeddings": False},
    }
    assembly_module._set_tie_word_embeddings(model_config, qconfig.tie_word_embeddings)
    assert model_config["tie_word_embeddings"] is True
    assert model_config["text_config"]["tie_word_embeddings"] is True

    reversed_merged, _ = _merge_quantization_config([vision, decoder])
    reversed_qconfig = assembly_module.OliveHfQuantizationConfig(**reversed_merged)
    assert reversed_qconfig.get_qlinear_init_args("model.decoder") == qconfig.get_qlinear_init_args("model.decoder")
    assert reversed_qconfig.get_qlinear_init_args("model.vision") == qconfig.get_qlinear_init_args("model.vision")
    assert reversed_qconfig.tie_word_embeddings is True


def test_build_compatibility_rejects_config_and_unoptimized_tensor_mismatches():
    qconfig = _quantization_config(
        group_size=32,
        symmetric=False,
        quantize_vision=False,
        skips=[],
    )
    base = _metadata_artifact(
        "decoder",
        "model.decoder",
        qconfig,
        {
            "model.decoder.weight_qweight": ((8, 4), "U8"),
            "model.shared.weight": ((8, 8), "F32"),
        },
    )
    incompatible_config = _metadata_artifact(
        "vision",
        "model.vision",
        qconfig,
        {
            "model.vision.weight_qweight": ((8, 4), "U8"),
            "model.shared.weight": ((8, 8), "F32"),
        },
        model_config={"hidden_size": 16},
    )
    with pytest.raises(ValueError, match="incompatible model config"):
        _validate_build_compatibility([base, incompatible_config])

    incompatible_tensor = _metadata_artifact(
        "vision",
        "model.vision",
        qconfig,
        {
            "model.vision.weight_qweight": ((8, 4), "U8"),
            "model.shared.weight": ((16, 8), "F32"),
        },
    )
    with pytest.raises(ValueError, match="shape_or_dtype"):
        _validate_build_compatibility([base, incompatible_tensor])

    tied = _metadata_artifact(
        "decoder",
        "model.decoder",
        qconfig,
        {
            "model.decoder.weight_qweight": ((8, 4), "U8"),
            "model.shared.weight": ((8, 8), "F32"),
        },
        model_config={"text_config": {"tie_word_embeddings": True}},
    )
    untied = _metadata_artifact(
        "vision",
        "model.vision",
        qconfig,
        {
            "model.vision.weight_qweight": ((8, 4), "U8"),
            "model.shared.weight": ((8, 8), "F32"),
        },
        model_config={"text_config": {"tie_word_embeddings": False}},
    )
    with pytest.raises(ValueError, match="tied word embeddings"):
        _validate_build_compatibility([tied, untied])


def test_rejects_nonempty_workflow_output(tmp_path):
    parent = tmp_path / "assembled"
    output_dir = parent / "decoder"
    model_dir = output_dir / "model"
    output_dir.mkdir(parents=True)
    (output_dir / "model_config.json").write_text("{}", encoding="utf-8")
    _write_checkpoint(
        model_dir,
        {
            "model.decoder.weight_qweight": torch.zeros(8, 4, dtype=torch.uint8),
            "model.decoder.weight_scales": torch.ones(8, 2),
            "model.shared.weight": torch.ones(8, 8),
        },
        _quantization_config(
            group_size=32,
            symmetric=False,
            quantize_vision=False,
            skips=["model.shared"],
        ),
    )
    existing_config = b"user-config"
    (parent / "config.json").write_bytes(existing_config)
    build_configs = OrderedDict([("decoder", _run_config(output_dir, "decoder", "model.decoder", "kquant"))])
    results = OrderedDict([("decoder", _result(model_dir))])

    with pytest.raises(ValueError, match="already contains files"):
        try_assemble_hf_component_builds(build_configs, results, parent)

    assert (parent / "config.json").read_bytes() == existing_config
    assert model_dir.is_dir()


def test_ineligible_workflow_does_not_create_assembly_lock(tmp_path):
    parent = tmp_path / "assembled"
    parent.mkdir()
    build_configs = OrderedDict(
        [
            (
                "onnx",
                SimpleNamespace(input_model=SimpleNamespace(type="onnxmodel")),
            )
        ]
    )

    assert try_assemble_hf_component_builds(build_configs, OrderedDict(), parent) is None
    assert not (tmp_path / ".assembled.olive-hf-assembly.lock").exists()


def test_assembly_lock_rejects_concurrent_writer(tmp_path):
    parent = tmp_path / "assembled"
    parent.mkdir()

    with (
        _assembly_lock(parent),
        pytest.raises(RuntimeError, match="Another HF component assembly"),
        _assembly_lock(parent),
    ):
        pass


@pytest.mark.parametrize(
    "component_names",
    [
        (None, None),
        ("decoder", "decoder"),
    ],
)
def test_does_not_assemble_whole_model_or_overlapping_component_builds(tmp_path, component_names):
    parent = tmp_path / "output"
    build_configs = OrderedDict()
    results = OrderedDict()
    for index, component_name in enumerate(component_names):
        output_dir = parent / f"build-{index}"
        model_dir = output_dir / "model"
        _write_checkpoint(
            model_dir,
            {"model.weight": torch.ones(2, 2)},
            _quantization_config(
                group_size=32,
                symmetric=False,
                quantize_vision=False,
                skips=[],
            ),
        )
        attributes = (
            {
                "component_name": component_name,
                "component_source_paths": ["model"],
            }
            if component_name
            else {}
        )
        build_configs[str(index)] = SimpleNamespace(
            input_model=SimpleNamespace(type="hfmodel", config={"model_attributes": attributes}),
            engine=SimpleNamespace(output_dir=output_dir),
            passes={},
        )
        results[str(index)] = _result(model_dir)

    assert try_assemble_hf_component_builds(build_configs, results, parent) is None
    assert not (parent / "model.safetensors.index.json").exists()


def test_does_not_assemble_components_for_different_hardware_targets(tmp_path):
    parent = tmp_path / "output"
    build_configs = OrderedDict()
    results = OrderedDict()
    for component, ep in (("decoder", "CPUExecutionProvider"), ("vision", "CUDAExecutionProvider")):
        output_dir = parent / component
        model_dir = output_dir / "model"
        _write_checkpoint(
            model_dir,
            {f"model.{component}.weight": torch.ones(2, 2)},
            _quantization_config(
                group_size=32,
                symmetric=False,
                quantize_vision=False,
                skips=[],
            ),
        )
        build_configs[component] = _run_config(
            output_dir,
            component,
            f"model.{component}",
            "rtn",
        )
        results[component] = _result(model_dir, execution_provider=ep)

    assert try_assemble_hf_component_builds(build_configs, results, parent) is None


def test_assembles_component_build_outside_workflow_output(tmp_path):
    parent = tmp_path / "output"
    output_dir = tmp_path / "custom" / "decoder"
    model_dir = output_dir / "model"
    _write_checkpoint(
        model_dir,
        {"model.decoder.weight": torch.ones(2, 2)},
        _quantization_config(
            group_size=32,
            symmetric=False,
            quantize_vision=False,
            skips=[],
        ),
    )
    build_configs = OrderedDict([("decoder", _run_config(output_dir, "decoder", "model.decoder", "rtn"))])
    results = OrderedDict([("decoder", _result(model_dir))])

    assert try_assemble_hf_component_builds(build_configs, results, parent) == parent
    assert (parent / "model.safetensors.index.json").is_file()
    assert (parent / "decoder" / "model-00001.safetensors").is_file()
    assert (output_dir / "model-00001.safetensors").is_file()


def test_component_assembly_requires_workflow_output(tmp_path):
    output_dir = tmp_path / "decoder"
    model_dir = output_dir / "model"
    _write_checkpoint(
        model_dir,
        {"model.decoder.weight": torch.ones(2, 2)},
        _quantization_config(
            group_size=32,
            symmetric=False,
            quantize_vision=False,
            skips=[],
        ),
    )
    build_configs = OrderedDict([("decoder", _run_config(output_dir, "decoder", "model.decoder", "rtn"))])
    results = OrderedDict([("decoder", _result(model_dir))])

    with pytest.raises(ValueError, match=r"top-level `engine\.output_dir`"):
        try_assemble_hf_component_builds(build_configs, results, None)
