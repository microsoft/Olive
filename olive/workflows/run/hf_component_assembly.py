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
_PASS_MUTATED_CONFIG_KEYS = {"tie_word_embeddings"}
_WORD_EMBEDDING_MODULE_NAMES = {
    "codec_embedding",
    "codec_head",
    "embed_tokens",
    "lm_head",
    "output_projection",
    "proj_out",
    "shared",
    "text_embedding",
    "tok_embeddings",
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
) -> list[_BuildArtifact] | None:
    artifacts = []
    hardware_targets = set()
    owned_components = set()

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

    if not artifacts or len(hardware_targets) > 1:
        return None
    return artifacts


def _normalized_model_config(value):
    if isinstance(value, dict):
        return {
            key: _normalized_model_config(item)
            for key, item in value.items()
            if key not in _QUANTIZATION_METADATA_KEYS
            and key not in _PROVENANCE_CONFIG_KEYS
            and key not in _PASS_MUTATED_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_normalized_model_config(item) for item in value]
    return value


def _changes_word_embedding_storage(artifacts: list[_BuildArtifact]) -> bool:
    for artifact in artifacts:
        quantization = artifact.config.get("quantization_config") or {}
        if quantization.get("lm_head") or quantization.get("embeds"):
            return True
    return False


def _effective_tie_word_embeddings(config: dict[str, Any]) -> bool:
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "tie_word_embeddings" in text_config:
        return bool(text_config["tie_word_embeddings"])
    return bool(config.get("tie_word_embeddings", False))


def _set_tie_word_embeddings(config: dict[str, Any], value: bool) -> None:
    config["tie_word_embeddings"] = value
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "tie_word_embeddings" in text_config:
        text_config["tie_word_embeddings"] = value


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

    if not _changes_word_embedding_storage(artifacts):
        tying_values = {_effective_tie_word_embeddings(artifact.config) for artifact in artifacts}
        if len(tying_values) > 1:
            raise ValueError("HF component builds disagree on the source model's tied word embeddings.")

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

    tying_configs = [config for config in parsed.values() if config.lm_head or config.embeds]
    tying_values = {config.tie_word_embeddings for config in tying_configs}
    if len(tying_values) > 1:
        raise ValueError("HF component builds disagree on tied word-embedding storage.")
    tie_word_embeddings = tying_values.pop() if tying_values else False
    if tie_word_embeddings and not all(config.lm_head and config.embeds for config in tying_configs):
        raise ValueError("Tied quantized word embeddings require both embeds and lm_head in the same build.")
    if tie_word_embeddings and not any(
        module_name.rsplit(".", 1)[-1] in _WORD_EMBEDDING_MODULE_NAMES for module_name in quantized_modules
    ):
        raise ValueError("Tied word-embedding metadata does not match the assembled quantized tensors.")

    merged = OliveHfQuantizationConfig(
        **default_args,
        lm_head=any(config.lm_head for config in parsed.values()),
        embeds=any(config.embeds for config in parsed.values()),
        moe=any(config.moe for config in parsed.values()),
        quantize_vision=any(config.quantize_vision for config in parsed.values()),
        modules_to_not_convert=final_skips or None,
        overrides=overrides or None,
        tie_word_embeddings=tie_word_embeddings,
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
            Path(artifact.name),
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


@contextmanager
def _assembly_lock(output_dir: Path):
    lock_path = output_dir.parent / f".{output_dir.name}.olive-hf-assembly.lock"
    with lock_path.open("a+b") as lock_file:
        if lock_path.stat().st_size == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another HF component assembly is using {output_dir}") from exc
        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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


def _ensure_clean_assembly_root(output_dir: Path) -> None:
    existing_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if existing_files:
        raise ValueError(
            f"Cannot automatically assemble HF component builds because the workflow output directory {output_dir} "
            f"already contains files: {existing_files[:10]}. Use a clean output directory."
        )


def _commit_assembly(
    artifacts: list[_BuildArtifact],
    output_dir: Path,
    temporary: Path,
    artifact_files: dict[str, list[str]],
    model_config: dict,
) -> None:
    for artifact in artifacts:
        if not artifact_files[artifact.name]:
            raise ValueError(f"Build {artifact.name!r} produced no component shard")

    publications = [
        (path, output_dir / path.name) for path in temporary.iterdir() if path.is_file() and path.name != _INDEX_NAME
    ]
    final_component_publications = [
        (path, output_dir / artifact.name / path.name)
        for artifact in artifacts
        for path in (temporary / artifact.name).iterdir()
        if path.is_file()
    ]
    publications.extend(final_component_publications)
    publications.append((temporary / _INDEX_NAME, output_dir / _INDEX_NAME))

    artifact_copies = [
        (source, artifact.output_dir / source.name)
        for artifact in artifacts
        for source in (temporary / artifact.name).iterdir()
        if source.is_file() and artifact.output_dir.resolve() != (output_dir / artifact.name).resolve()
    ]
    destinations = [destination for _, destination in publications + artifact_copies]
    destination_counts = Counter(destination.resolve() for destination in destinations)
    duplicate_destinations = sorted(str(path) for path, count in destination_counts.items() if count > 1)
    if duplicate_destinations:
        raise ValueError(f"HF component assembly destinations overlap: {duplicate_destinations[:10]}")
    conflicts = sorted(str(destination) for destination in destinations if destination.exists())
    if conflicts:
        raise ValueError(f"HF component assembly destinations already exist: {conflicts[:10]}")

    published = []
    try:
        for source, destination in artifact_copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_copy(source, destination)
            published.append(destination)
        for source, destination in publications:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            published.append(destination)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise

    for artifact in artifacts:
        target = artifact.output_dir
        artifact.model_output._update_with_model_config(model_config)  # pylint: disable=protected-access
        try:
            _atomic_write_json(target / "model_config.json", model_config)
            _rewrite_persisted_footprints(artifact)
            model_dir = target / "model"
            if model_dir.is_dir():
                shutil.rmtree(model_dir)
        except (OSError, TypeError, ValueError):
            logger.warning("Could not finalize component artifact %s", target, exc_info=True)


def try_assemble_hf_component_builds(
    build_configs: dict[str, RunConfig],
    results: OrderedDict[str, WorkflowOutput],
    output_dir: Path | None,
) -> Path | None:
    """Automatically assemble compatible component-scoped HfModel builds into the workflow output directory."""
    with ExitStack() as stack:
        artifacts = _collect_build_artifacts(build_configs, results, stack)
        if artifacts is None:
            return None
        if output_dir is None:
            raise ValueError("Automatic HF component assembly requires a top-level `engine.output_dir`.")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with _assembly_lock(output_dir):
            _ensure_clean_assembly_root(output_dir)
            _validate_build_compatibility(artifacts)
            logger.info(
                "Assembling HF component builds %s into %s",
                [artifact.name for artifact in artifacts],
                output_dir,
            )
            with tempfile.TemporaryDirectory(prefix=".olive-hf-assembly-", dir=output_dir) as temporary_dir:
                temporary = Path(temporary_dir)
                _copy_non_weight_files(artifacts[0].model_dir, temporary)
                merged_quantization, build_quantization = _merge_quantization_config(artifacts)
                config_path = temporary / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if merged_quantization is None:
                    config.pop("quantization_config", None)
                else:
                    config["quantization_config"] = merged_quantization
                    tie_word_embeddings = (
                        merged_quantization["tie_word_embeddings"]
                        if _changes_word_embedding_storage(artifacts)
                        else _effective_tie_word_embeddings(config)
                    )
                    _set_tie_word_embeddings(config, tie_word_embeddings)
                component_quantization = _component_quantization_mapping(artifacts)
                if component_quantization:
                    config["component_quantization"] = component_quantization
                if build_quantization:
                    config["olive_component_quantization"] = build_quantization
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

                weight_map, total_size, artifact_files = _materialize_component_artifacts(
                    artifacts,
                    temporary,
                )
                index = {
                    "metadata": {"total_size": total_size},
                    "weight_map": dict(sorted(weight_map.items())),
                }
                (temporary / _INDEX_NAME).write_text(json.dumps(index, indent=2), encoding="utf-8")

                model_config = deepcopy(artifacts[0].model_output.olive_model_config)
                model_config["config"]["model_path"] = str(output_dir)
                attributes = dict(model_config["config"].get("model_attributes") or {})
                for name in ("component_name", "component_names", "component_role", "component_source_paths"):
                    attributes.pop(name, None)
                attributes["assembled_components"] = [
                    component for artifact in artifacts for component in artifact.components
                ]
                model_config["config"]["model_attributes"] = attributes
                (temporary / "model_config.json").write_text(
                    json.dumps(model_config, indent=2),
                    encoding="utf-8",
                )

                stack.close()
                _commit_assembly(
                    artifacts,
                    output_dir,
                    temporary,
                    artifact_files,
                    model_config,
                )

    logger.info("Assembled standard Hugging Face checkpoint at %s", output_dir)
    return output_dir
