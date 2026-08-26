# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Assemble component-scoped HfModel builds into one standard HF checkpoint."""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safetensors import safe_open
from safetensors.torch import save_file

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.patterns import match_skip

if TYPE_CHECKING:
    from collections import OrderedDict

    import torch

    from olive.engine.output import WorkflowOutput
    from olive.workflows.run.config import RunConfig


logger = logging.getLogger(__name__)

_INDEX_NAME = "model.safetensors.index.json"
_SHARD_LIMIT = 1024**3


@dataclass
class _BuildArtifact:
    name: str
    components: list[str]
    source_paths: list[str]
    output_dir: Path
    model_dir: Path
    pass_types: list[str]
    checkpoint: _Checkpoint
    config: dict[str, Any]
    model_output: Any


class _Checkpoint:
    """Header-indexed safetensors checkpoint with lazily opened shards."""

    def __init__(self, directory: Path, stack: ExitStack):
        self.directory = directory
        index_path = directory / _INDEX_NAME
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
            if not weight_map:
                raise ValueError(f"Empty Hugging Face weight map: {index_path}")
            root = directory.resolve()
            key_to_path = {}
            for key, filename in weight_map.items():
                path = (directory / filename).resolve()
                if path != root and root not in path.parents:
                    raise ValueError(f"Unsafe safetensors shard path {filename!r} in {index_path}")
                key_to_path[key] = path
        else:
            files = sorted(directory.glob("*.safetensors"))
            if not files:
                raise ValueError(f"HF component assembly requires safetensors weights: {directory}")
            key_to_path = {}
            for path in files:
                with safe_open(path, framework="pt") as handle:
                    for key in handle.keys():  # noqa: SIM118
                        if key in key_to_path:
                            raise ValueError(f"Duplicate checkpoint tensor {key!r} in {directory}")
                        key_to_path[key] = path

        missing = sorted({path for path in key_to_path.values() if not path.is_file()})
        if missing:
            raise FileNotFoundError(f"Missing safetensors shard(s): {missing}")

        self.key_to_path = key_to_path
        self._handles = {
            path: stack.enter_context(safe_open(path, framework="pt")) for path in sorted(set(key_to_path.values()))
        }

    @property
    def keys(self) -> set[str]:
        return set(self.key_to_path)

    def tensor(self, key: str) -> torch.Tensor:
        return self._handles[self.key_to_path[key]].get_tensor(key)


def _matches_source_path(key: str, source_paths: list[str]) -> bool:
    return any(key == path or key.startswith(f"{path}.") for path in source_paths)


def _component_names(model_attributes: dict[str, Any]) -> list[str]:
    names = model_attributes.get("component_names")
    if names:
        return list(names)
    name = model_attributes.get("component_name")
    return [name] if name and name != "model" else []


def _collect_build_artifacts(
    build_configs: dict[str, RunConfig],
    results: OrderedDict[str, WorkflowOutput],
    stack: ExitStack,
    assembly_output_dir: Path | None,
) -> tuple[list[_BuildArtifact], Path] | None:
    if assembly_output_dir is None:
        return None
    artifacts = []
    hardware_targets = set()
    owned_components = set()
    parent = assembly_output_dir.resolve()

    for build_name, run_config in build_configs.items():
        if run_config.input_model.type.lower() != "hfmodel":
            return None
        attributes = run_config.input_model.config.get("model_attributes") or {}
        components = _component_names(attributes)
        source_paths = list(attributes.get("component_source_paths") or [])
        if not components or not source_paths:
            return None
        overlap = owned_components.intersection(components)
        if overlap:
            return None
        owned_components.update(components)

        output = results[build_name].get_best_candidate()
        if output is None or output.model_type.lower() != "hfmodel" or not output.model_path:
            return None
        model_dir = Path(output.model_path)
        output_dir = Path(run_config.engine.output_dir).resolve()
        if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
            return None
        if output_dir == parent or parent not in output_dir.parents:
            return None
        if hasattr(output, "from_device") and hasattr(output, "from_execution_provider"):
            hardware_targets.add((output.from_device(), output.from_execution_provider()))
        artifacts.append(
            _BuildArtifact(
                name=build_name,
                components=components,
                source_paths=source_paths,
                output_dir=output_dir,
                model_dir=model_dir,
                pass_types=[pass_config.type for configs in run_config.passes.values() for pass_config in configs],
                checkpoint=_Checkpoint(model_dir, stack),
                config=json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
                model_output=output,
            )
        )

    if len(hardware_targets) > 1:
        return None
    return artifacts, parent


def _qweight_module_name(key: str) -> str | None:
    if not key.endswith("_qweight"):
        return None
    stem = key.removesuffix("_qweight")
    return stem.removesuffix(".weight")


def _merge_quantization_config(artifacts: list[_BuildArtifact]) -> tuple[dict | None, dict[str, Any]]:
    configs = {
        artifact.name: artifact.config.get("quantization_config")
        for artifact in artifacts
        if artifact.config.get("quantization_config")
    }
    if not configs:
        return None, {}

    parsed = {name: OliveHfQuantizationConfig(**config) for name, config in configs.items()}
    first_name = next(iter(parsed))
    first = parsed[first_name]
    merged = deepcopy(configs[first_name])
    default_args = {
        "bits": first.bits,
        "symmetric": first.symmetric,
        "group_size": first.group_size,
    }

    overrides = deepcopy(configs[first_name].get("overrides") or {})
    quantized_modules = set()
    for artifact in artifacts:
        quant_config = parsed.get(artifact.name)
        if quant_config is None:
            continue
        for key in sorted(artifact.checkpoint.keys):
            module_name = _qweight_module_name(key)
            if module_name is None or not _matches_source_path(module_name, artifact.source_paths):
                continue
            quantized_modules.add(module_name)
            qargs = quant_config.get_qlinear_init_args(module_name)
            override = {name: value for name, value in qargs.items() if default_args[name] != value}
            if override:
                overrides[module_name] = override

    skip_patterns = list(
        dict.fromkeys(pattern for config in parsed.values() for pattern in (config.modules_to_not_convert or []))
    )
    final_skips = [
        pattern
        for pattern in skip_patterns
        if not any(match_skip(module_name, [pattern]) for module_name in quantized_modules)
    ]
    merged.update(
        {
            "lm_head": any(config.lm_head for config in parsed.values()),
            "embeds": any(config.embeds for config in parsed.values()),
            "moe": any(config.moe for config in parsed.values()),
            "quantize_vision": any(config.quantize_vision for config in parsed.values()),
            "modules_to_not_convert": final_skips or None,
            "overrides": overrides or None,
        }
    )
    component_configs = {
        artifact.name: {
            "components": artifact.components,
            "passes": artifact.pass_types,
            "quantization_config": configs.get(artifact.name),
        }
        for artifact in artifacts
    }
    return merged, component_configs


def _write_shards(
    entries: list[tuple[str, _Checkpoint]],
    output_dir: Path,
    prefix: str,
    relative_dir: Path,
) -> tuple[dict[str, str], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    total_size = 0
    batch = {}
    batch_size = 0
    shard_index = 0

    def flush() -> None:
        nonlocal batch, batch_size, shard_index
        if not batch:
            return
        shard_index += 1
        filename = f"{prefix}-{shard_index:05d}.safetensors"
        save_file(batch, output_dir / filename)
        relative_path = (relative_dir / filename).as_posix()
        weight_map.update(dict.fromkeys(batch, relative_path))
        batch = {}
        batch_size = 0

    for key, checkpoint in sorted(entries, key=lambda item: item[0]):
        tensor = checkpoint.tensor(key)
        tensor_size = tensor.numel() * tensor.element_size()
        if batch and batch_size + tensor_size > _SHARD_LIMIT:
            flush()
        batch[key] = tensor
        batch_size += tensor_size
        total_size += tensor_size
    flush()
    return weight_map, total_size


def _copy_non_weight_files(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix in {".safetensors", ".bin"} or path.name in {
            _INDEX_NAME,
            "pytorch_model.bin.index.json",
        }:
            continue
        shutil.copy2(path, destination / path.name)


def _materialize_component_artifacts(
    artifacts: list[_BuildArtifact],
    parent: Path,
    temporary: Path,
) -> tuple[dict[str, str], int, dict[str, list[str]]]:
    weight_map = {}
    total_size = 0
    artifact_files = {}
    owned_keys = set()

    for artifact in artifacts:
        entries = [
            (key, artifact.checkpoint)
            for key in artifact.checkpoint.keys
            if _matches_source_path(key, artifact.source_paths)
        ]
        if not entries:
            raise ValueError(
                f"Build {artifact.name!r} source paths matched no checkpoint tensors: {artifact.source_paths}"
            )
        keys = {key for key, _ in entries}
        overlap = owned_keys.intersection(keys)
        if overlap:
            raise ValueError(f"HF component builds own overlapping checkpoint tensors: {sorted(overlap)[:5]}")
        owned_keys.update(keys)

        artifact_dir = temporary / artifact.name
        component_map, component_size = _write_shards(
            entries,
            artifact_dir,
            "model",
            artifact.output_dir.resolve().relative_to(parent.resolve()),
        )
        weight_map.update(component_map)
        total_size += component_size
        artifact_files[artifact.name] = sorted(Path(path).name for path in component_map.values())
        manifest = {
            "type": "hf_component",
            "components": artifact.components,
            "source_paths": artifact.source_paths,
            "passes": artifact.pass_types,
            "quantization_config": artifact.config.get("quantization_config"),
            "weight_files": artifact_files[artifact.name],
        }
        (artifact_dir / "component.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    base = artifacts[0].checkpoint
    unoptimized_entries = [
        (key, base)
        for key in base.keys
        if not any(_matches_source_path(key, artifact.source_paths) for artifact in artifacts)
    ]
    base_map, base_size = _write_shards(
        unoptimized_entries,
        temporary,
        "model-unoptimized",
        Path(),
    )
    weight_map.update(base_map)
    total_size += base_size
    return weight_map, total_size, artifact_files


def _commit_assembly(
    artifacts: list[_BuildArtifact],
    parent: Path,
    temporary: Path,
    artifact_files: dict[str, list[str]],
) -> None:
    root_files = {path.name for path in temporary.iterdir() if path.is_file()}
    for path in temporary.iterdir():
        if path.is_file():
            shutil.copy2(path, parent / path.name)

    for artifact in artifacts:
        target = artifact.output_dir
        source = temporary / artifact.name
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)
        if not artifact_files[artifact.name]:
            raise ValueError(f"Build {artifact.name!r} produced no component shard")

    for path in parent.glob("model-*.safetensors"):
        if path.name not in root_files:
            path.unlink()
    for artifact in artifacts:
        target = artifact.output_dir
        model_dir = target / "model"
        if model_dir.is_dir():
            shutil.rmtree(model_dir)
        model_config = target / "model_config.json"
        if model_config.is_file():
            model_config.unlink()
        current_files = set(artifact_files[artifact.name])
        for path in target.glob("model-*.safetensors"):
            if path.name not in current_files:
                path.unlink()

    model_config = deepcopy(artifacts[0].model_output.olive_model_config)
    model_config["config"]["model_path"] = str(parent)
    attributes = dict(model_config["config"].get("model_attributes") or {})
    for name in ("component_name", "component_names", "component_role", "component_source_paths"):
        attributes.pop(name, None)
    attributes["assembled_components"] = [component for artifact in artifacts for component in artifact.components]
    model_config["config"]["model_attributes"] = attributes
    (parent / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    for artifact in artifacts:
        artifact.model_output._update_with_model_config(model_config)  # pylint: disable=protected-access


def try_assemble_hf_component_builds(
    build_configs: dict[str, RunConfig],
    results: OrderedDict[str, WorkflowOutput],
    assembly_output_dir: Path | None = None,
) -> Path | None:
    """Assemble disjoint component-scoped HfModel builds at their common parent."""
    with ExitStack() as stack:
        collected = _collect_build_artifacts(build_configs, results, stack, assembly_output_dir)
        if collected is None:
            return None
        artifacts, parent = collected
        logger.info(
            "Assembling HF component builds %s into %s",
            [artifact.name for artifact in artifacts],
            parent,
        )
        temporary = parent / ".olive-hf-assembly"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        try:
            _copy_non_weight_files(artifacts[0].model_dir, temporary)
            merged_quantization, component_quantization = _merge_quantization_config(artifacts)
            config_path = temporary / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if merged_quantization is None:
                config.pop("quantization_config", None)
            else:
                config["quantization_config"] = merged_quantization
            if component_quantization:
                config["olive_component_quantization"] = component_quantization
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

            weight_map, total_size, artifact_files = _materialize_component_artifacts(
                artifacts,
                parent,
                temporary,
            )
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": dict(sorted(weight_map.items())),
            }
            (temporary / _INDEX_NAME).write_text(json.dumps(index, indent=2), encoding="utf-8")
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    try:
        _commit_assembly(artifacts, parent, temporary, artifact_files)
    except Exception:
        logger.exception("HF component assembly commit failed; staged files remain at %s", temporary)
        raise
    else:
        shutil.rmtree(temporary)

    logger.info("Assembled standard Hugging Face checkpoint at %s", parent)
    return parent
