# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Assemble component-scoped HfModel builds into one standard HF checkpoint."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from safetensors import safe_open
from safetensors.torch import save_file

from olive.common.quant.hf_utils import OliveHfQuantizationConfig

if TYPE_CHECKING:
    from collections import OrderedDict

    import torch

    from olive.engine.output import WorkflowOutput
    from olive.workflows.run.config import RunConfig


logger = logging.getLogger(__name__)

_INDEX_NAME = "model.safetensors.index.json"
_SHARD_LIMIT = 1024**3
_QUANTIZATION_METADATA_KEYS = {
    "component_quantization",
    "olive_component_quantization",
    "quantization_config",
}
_PROVENANCE_CONFIG_KEYS = {"_name_or_path", "transformers_version"}
_CONFLICTING_CHECKPOINT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


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
    workflow_output: Any


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

    def metadata(self, key: str) -> tuple[tuple[int, ...], str]:
        tensor_slice = self._handles[self.key_to_path[key]].get_slice(key)
        return tuple(tensor_slice.get_shape()), tensor_slice.get_dtype()


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
                workflow_output=results[build_name],
            )
        )

    if len(hardware_targets) > 1:
        return None
    return artifacts, parent


def _normalized_model_config(value):
    if isinstance(value, dict):
        return {
            key: _normalized_model_config(item)
            for key, item in value.items()
            if key not in _QUANTIZATION_METADATA_KEYS and key not in _PROVENANCE_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_normalized_model_config(item) for item in value]
    return value


def _validate_build_compatibility(artifacts: list[_BuildArtifact]) -> None:
    base_config = _normalized_model_config(artifacts[0].config)
    for artifact in artifacts[1:]:
        config = _normalized_model_config(artifact.config)
        if config != base_config:
            differing_keys = sorted(
                key for key in set(base_config) | set(config) if base_config.get(key) != config.get(key)
            )
            raise ValueError(
                f"HF component build {artifact.name!r} has an incompatible model config; "
                f"differing fields: {differing_keys[:10]}"
            )

    component_paths = [path for artifact in artifacts for path in artifact.source_paths]

    def unowned_metadata(artifact: _BuildArtifact) -> dict[str, tuple[tuple[int, ...], str]]:
        return {
            key: artifact.checkpoint.metadata(key)
            for key in artifact.checkpoint.keys
            if not _matches_source_path(key, component_paths)
        }

    expected = unowned_metadata(artifacts[0])
    for artifact in artifacts[1:]:
        actual = unowned_metadata(artifact)
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key])
        if missing or extra or mismatched:
            raise ValueError(
                f"HF component build {artifact.name!r} has incompatible unoptimized tensors: "
                f"missing={missing[:5]}, extra={extra[:5]}, shape_or_dtype={mismatched[:5]}"
            )


def _final_checkpoint_entries(artifacts: list[_BuildArtifact]) -> list[tuple[str, _BuildArtifact]]:
    entries = []
    for artifact in artifacts:
        entries.extend(
            (key, artifact) for key in artifact.checkpoint.keys if _matches_source_path(key, artifact.source_paths)
        )

    component_paths = [path for artifact in artifacts for path in artifact.source_paths]
    entries.extend(
        (key, artifacts[0]) for key in artifacts[0].checkpoint.keys if not _matches_source_path(key, component_paths)
    )
    return entries


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
    entries = _final_checkpoint_entries(artifacts)

    observed_args: dict[str, set[tuple[int, bool, int]]] = {}
    for artifact in artifacts:
        quant_config = parsed.get(artifact.name)
        for key in artifact.checkpoint.keys:
            module_name = _qweight_module_name(key)
            if module_name is None:
                continue
            if quant_config is None:
                raise ValueError(
                    f"HF component build {artifact.name!r} contains quantized tensor {key!r} "
                    "without an Olive quantization_config."
                )
            qargs = quant_config.get_qlinear_init_args(module_name)
            observed_args.setdefault(module_name, set()).add((qargs["bits"], qargs["symmetric"], qargs["group_size"]))
    conflicts = sorted(module for module, values in observed_args.items() if len(values) > 1)
    if conflicts:
        raise ValueError(f"HF component builds disagree on quantization settings for: {conflicts[:10]}")

    module_args = {}
    for key, artifact in entries:
        module_name = _qweight_module_name(key)
        if module_name is None:
            continue
        quant_config = parsed.get(artifact.name)
        if quant_config is None:
            raise ValueError(
                f"Final quantized tensor {key!r} has no Olive quantization_config in build {artifact.name!r}."
            )
        module_args[module_name] = quant_config.get_qlinear_init_args(module_name)

    if not module_args:
        return None, {}

    arg_counts = Counter((args["bits"], args["symmetric"], args["group_size"]) for args in module_args.values())
    bits, symmetric, group_size = min(
        arg_counts,
        key=lambda values: (-arg_counts[values], values[0], values[2], values[1]),
    )
    default_args = {"bits": bits, "symmetric": symmetric, "group_size": group_size}
    overrides = {
        module_name: {name: value for name, value in qargs.items() if default_args[name] != value}
        for module_name, qargs in sorted(module_args.items())
    }
    overrides = {module_name: override for module_name, override in overrides.items() if override}

    quantized_modules = set(module_args)
    float_modules = set()
    for key, artifact in entries:
        if _qweight_module_name(key) is not None or key.endswith(("_scales", "_qzeros")):
            continue
        shape, _ = artifact.checkpoint.metadata(key)
        if len(shape) < 2:
            continue
        module_name = key.removesuffix(".weight")
        if module_name not in quantized_modules:
            float_modules.add(module_name)
    final_skips = [f"re:^{re.escape(module_name)}$" for module_name in sorted(float_modules)]

    merged = OliveHfQuantizationConfig(
        **default_args,
        lm_head=any(config.lm_head for config in parsed.values()),
        embeds=any(config.embeds for config in parsed.values()),
        moe=any(config.moe for config in parsed.values()),
        quantize_vision=any(config.quantize_vision for config in parsed.values()),
        modules_to_not_convert=final_skips or None,
        overrides=overrides or None,
        tie_word_embeddings=any(config.tie_word_embeddings for config in parsed.values()),
    ).to_dict()
    component_configs = {
        artifact.name: {
            "components": artifact.components,
            "passes": artifact.pass_types,
            "quantization_config": configs.get(artifact.name),
        }
        for artifact in artifacts
    }
    return merged, component_configs


def _component_quantization_mapping(artifacts: list[_BuildArtifact]) -> dict[str, dict[str, Any]]:
    mapping = {}
    for artifact in artifacts:
        quantization = artifact.config.get("quantization_config")
        if not quantization:
            continue
        quantization = deepcopy(quantization)
        for component in artifact.components:
            mapping[component] = quantization
    return mapping


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
    generation: str,
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
            f"model-{generation}",
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
        f"model-unoptimized-{generation}",
        Path(),
    )
    weight_map.update(base_map)
    total_size += base_size
    return weight_map, total_size, artifact_files


@contextmanager
def _assembly_lock(parent: Path):
    lock_dir = parent / ".olive-hf-assembly.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Another HF component assembly is using {parent}: {lock_dir}") from exc
    try:
        (lock_dir / "owner").write_text(str(os.getpid()), encoding="utf-8")
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rewrite_persisted_footprints(artifact: _BuildArtifact) -> None:
    footprint = getattr(artifact.workflow_output, "footprint", None)
    if footprint is None or not hasattr(footprint, "to_json"):
        return
    serialized = footprint.to_json()
    if not isinstance(serialized, str):
        return
    for name in ("footprint.json", "output_footprint.json"):
        path = artifact.output_dir / name
        if not path.is_file():
            continue
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _commit_assembly(
    artifacts: list[_BuildArtifact],
    parent: Path,
    temporary: Path,
    artifact_files: dict[str, list[str]],
    weight_map: dict[str, str],
) -> None:
    for artifact in artifacts:
        if not artifact_files[artifact.name]:
            raise ValueError(f"Build {artifact.name!r} produced no component shard")

    copied_shards = []
    moved_conflicts = []
    rollback_dir = temporary / "rollback"
    rollback_dir.mkdir()
    old_config = parent / "config.json"
    config_backup = rollback_dir / "config.json"
    had_config = old_config.is_file()
    index_published = False

    root_shards = [path for path in temporary.iterdir() if path.suffix == ".safetensors"]
    component_shards = [
        (path, artifact.output_dir / path.name)
        for artifact in artifacts
        for path in (temporary / artifact.name).glob("*.safetensors")
    ]
    try:
        for source in root_shards:
            destination = parent / source.name
            shutil.copy2(source, destination)
            copied_shards.append(destination)
        for source, destination in component_shards:
            shutil.copy2(source, destination)
            copied_shards.append(destination)

        conflicts = [
            path
            for path in parent.iterdir()
            if path.is_file() and (path.name in _CONFLICTING_CHECKPOINT_NAMES or path.match("pytorch_model-*.bin"))
        ]
        for path in conflicts:
            backup = rollback_dir / path.name
            path.replace(backup)
            moved_conflicts.append((path, backup))

        if had_config:
            shutil.copy2(old_config, config_backup)
        _atomic_copy(temporary / "config.json", old_config)
        _atomic_copy(temporary / _INDEX_NAME, parent / _INDEX_NAME)
        index_published = True
    except Exception:
        if not index_published:
            if had_config and config_backup.is_file():
                config_backup.replace(old_config)
            elif not had_config:
                old_config.unlink(missing_ok=True)
            for destination, backup in moved_conflicts:
                if backup.exists():
                    backup.replace(destination)
            for path in copied_shards:
                path.unlink(missing_ok=True)
        raise

    for path in temporary.iterdir():
        if not path.is_file() or path.name in {"config.json", _INDEX_NAME} or path.suffix == ".safetensors":
            continue
        _atomic_copy(path, parent / path.name)
    for artifact in artifacts:
        target = artifact.output_dir
        source = temporary / artifact.name
        component_manifest = source / "component.json"
        if component_manifest.is_file():
            _atomic_copy(component_manifest, target / component_manifest.name)

    root_files = {Path(filename).name for filename in weight_map.values() if "/" not in filename}
    for path in parent.glob("model-*.safetensors"):
        if path.name not in root_files:
            path.unlink()

    model_config = deepcopy(artifacts[0].model_output.olive_model_config)
    model_config["config"]["model_path"] = str(parent)
    attributes = dict(model_config["config"].get("model_attributes") or {})
    for name in ("component_name", "component_names", "component_role", "component_source_paths"):
        attributes.pop(name, None)
    attributes["assembled_components"] = [component for artifact in artifacts for component in artifact.components]
    model_config["config"]["model_attributes"] = attributes
    _atomic_write_json(parent / "model_config.json", model_config)

    for artifact in artifacts:
        target = artifact.output_dir
        artifact.model_output._update_with_model_config(model_config)  # pylint: disable=protected-access
        _atomic_write_json(target / "model_config.json", model_config)
        _rewrite_persisted_footprints(artifact)

        model_dir = target / "model"
        if model_dir.is_dir():
            shutil.rmtree(model_dir)
        current_files = set(artifact_files[artifact.name])
        for path in target.glob("model-*.safetensors"):
            if path.name not in current_files:
                path.unlink()


def try_assemble_hf_component_builds(
    build_configs: dict[str, RunConfig],
    results: OrderedDict[str, WorkflowOutput],
    assembly_output_dir: Path | None = None,
) -> Path | None:
    """Assemble disjoint component-scoped HfModel builds at their common parent."""
    if assembly_output_dir is None:
        return None
    assembly_output_dir.mkdir(parents=True, exist_ok=True)
    with _assembly_lock(assembly_output_dir.resolve()), ExitStack() as stack:
        collected = _collect_build_artifacts(build_configs, results, stack, assembly_output_dir)
        if collected is None:
            return None
        artifacts, parent = collected
        _validate_build_compatibility(artifacts)
        logger.info(
            "Assembling HF component builds %s into %s",
            [artifact.name for artifact in artifacts],
            parent,
        )
        generation = uuid4().hex[:12]
        temporary = Path(tempfile.mkdtemp(prefix=f".olive-hf-assembly-{generation}-", dir=parent))

        try:
            _copy_non_weight_files(artifacts[0].model_dir, temporary)
            merged_quantization, build_quantization = _merge_quantization_config(artifacts)
            config_path = temporary / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if merged_quantization is None:
                config.pop("quantization_config", None)
            else:
                config["quantization_config"] = merged_quantization
            component_quantization = _component_quantization_mapping(artifacts)
            if component_quantization:
                config["component_quantization"] = component_quantization
            if build_quantization:
                config["olive_component_quantization"] = build_quantization
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

            weight_map, total_size, artifact_files = _materialize_component_artifacts(
                artifacts,
                parent,
                temporary,
                generation,
            )
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": dict(sorted(weight_map.items())),
            }
            (temporary / _INDEX_NAME).write_text(json.dumps(index, indent=2), encoding="utf-8")
            _commit_assembly(artifacts, parent, temporary, artifact_files, weight_map)
        except Exception:
            logger.exception("HF component assembly failed in staging directory %s", temporary)
            raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    logger.info("Assembled standard Hugging Face checkpoint at %s", parent)
    return parent
