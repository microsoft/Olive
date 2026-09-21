# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Assemble component-scoped ONNX builds into a complete CompositeModel package."""

from __future__ import annotations

import json
import logging
import shutil
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from olive.common.utils import hardlink_copy_dir, hardlink_copy_file
from olive.model import ModelConfig
from olive.model.handler import CompositeModelHandler, ONNXModelHandler
from olive.passes.onnx.common import get_external_data_file_names, resave_model

if TYPE_CHECKING:
    from olive.engine.output import WorkflowOutput
    from olive.workflows.run.config import RunConfig

logger = logging.getLogger(__name__)


def _collect_optimized_components(
    build_components: OrderedDict[str, list[str]],
    results: OrderedDict[str, WorkflowOutput],
) -> OrderedDict[str, ONNXModelHandler] | None:
    if not build_components or any(not components for components in build_components.values()):
        return None

    optimized = OrderedDict()
    for build_name, component_names in build_components.items():
        overlap = set(optimized).intersection(component_names)
        if overlap:
            raise ValueError(f"CompositeModel builds select overlapping components: {sorted(overlap)}")

        model_output = results[build_name].get_best_candidate()
        output_model = ModelConfig.model_validate(model_output.olive_model_config).create_model()
        if isinstance(output_model, ONNXModelHandler):
            if len(component_names) != 1:
                raise ValueError(
                    f"Build {build_name!r} selected components {component_names} but produced one ONNX model."
                )
            optimized[component_names[0]] = output_model
            continue
        if not isinstance(output_model, CompositeModelHandler):
            raise ValueError(
                f"Build {build_name!r} selected CompositeModel components but produced {type(output_model).__name__}."
            )

        output_components = dict(output_model.get_model_components())
        if set(output_components) != set(component_names):
            raise ValueError(
                f"Build {build_name!r} produced components {list(output_components)}; expected {component_names}."
            )
        if not all(isinstance(component, ONNXModelHandler) for component in output_components.values()):
            raise ValueError(f"Build {build_name!r} produced non-ONNX CompositeModel components.")
        optimized.update((name, output_components[name]) for name in component_names)

    return optimized


def _rebase_additional_files(
    attributes: dict,
    source_root: Path,
    output_root: Path,
    component_dir: Path,
    temporary_root: Path,
    package_file_names: set[str],
) -> dict:
    attributes = deepcopy(attributes)
    rebased = []
    for file_name in attributes.get("additional_files") or []:
        path = Path(file_name)
        try:
            relative = path.resolve().relative_to(source_root)
            destination = output_root / relative
        except ValueError:
            if path.name in package_file_names:
                destination = output_root / path.name
            elif path.is_file():
                temporary_destination = temporary_root / component_dir / path.name
                temporary_destination.parent.mkdir(parents=True, exist_ok=True)
                hardlink_copy_file(path, temporary_destination)
                destination = output_root / component_dir / path.name
            else:
                raise FileNotFoundError(f"CompositeModel additional file does not exist: {path}") from None
        if str(destination) not in rebased:
            rebased.append(str(destination))

    if rebased:
        attributes["additional_files"] = rebased
    else:
        attributes.pop("additional_files", None)
    return attributes


def _replace_component(
    source_component: ONNXModelHandler,
    optimized_component: ONNXModelHandler,
    source_root: Path,
    temporary_root: Path,
) -> Path:
    source_model_path = Path(source_component.model_path).resolve()
    try:
        relative_model_path = source_model_path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"CompositeModel component {source_model_path} is outside package root {source_root}."
        ) from exc

    destination = temporary_root / relative_model_path
    for external_file_name in get_external_data_file_names(destination):
        external_path = (destination.parent / external_file_name).resolve()
        if external_path.is_relative_to(destination.parent.resolve()):
            external_path.unlink(missing_ok=True)

    destination.unlink(missing_ok=True)
    resave_model(optimized_component.model_path, destination)
    for additional_path in (
        optimized_component.external_initializers_path,
        optimized_component.constant_inputs_path,
    ):
        if additional_path:
            hardlink_copy_file(additional_path, destination.parent / Path(additional_path).name)
    return relative_model_path


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_assembly(temporary: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_root = output_dir.parent / f".{output_dir.name}.{uuid4().hex}.bak"
    backup_root.mkdir()
    published = []
    backups = []
    try:
        for source in sorted(temporary.iterdir(), key=lambda path: path.name):
            destination = output_dir / source.name
            if destination.exists():
                backup = backup_root / source.name
                destination.replace(backup)
                backups.append((backup, destination))
            source.replace(destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            _remove_path(destination)
        for backup, destination in reversed(backups):
            backup.replace(destination)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def _cleanup_build_outputs(
    build_configs: dict[str, RunConfig],
    output_dir: Path,
    component_relative_paths: dict[str, Path],
) -> None:
    component_dirs = {(output_dir / relative.parent).resolve() for relative in component_relative_paths.values()}
    for run_config in build_configs.values():
        build_output = Path(run_config.engine.output_dir).resolve()
        if output_dir not in build_output.parents:
            continue
        if any(
            build_output == component_dir or build_output in component_dir.parents for component_dir in component_dirs
        ):
            continue
        if build_output.exists():
            _remove_path(build_output)
        parent = build_output.parent
        while parent != output_dir and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent


def try_assemble_composite_model_builds(
    input_model: ModelConfig | None,
    build_components: OrderedDict[str, list[str]],
    build_configs: dict[str, RunConfig],
    results: OrderedDict[str, WorkflowOutput],
    output_dir: Path | None,
) -> Path | None:
    """Assemble component-scoped ONNX builds and untouched source components into one package."""
    if input_model is None or input_model.type != "compositemodel":
        return None

    source_root_value = input_model.config.get("model_path")
    if not source_root_value:
        return None
    source_root = Path(source_root_value).resolve()
    if not source_root.is_dir():
        return None

    source_model = input_model.create_model()
    if not isinstance(source_model, CompositeModelHandler) or len(source_model.model_component_names) < 2:
        return None
    source_components = OrderedDict(source_model.get_model_components())
    if not all(isinstance(component, ONNXModelHandler) for component in source_components.values()):
        return None
    if output_dir is None:
        return None

    optimized_components = _collect_optimized_components(build_components, results)
    if optimized_components is None:
        return None
    unknown_components = set(optimized_components) - set(source_components)
    if unknown_components:
        raise ValueError(f"CompositeModel builds produced unknown components: {sorted(unknown_components)}")
    output_dir = Path(output_dir).resolve()
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError("CompositeModel workflow output directory must not be inside the input package.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid4().hex}.tmp")
    package_files = {path.name for path in source_root.iterdir() if path.is_file() and path.name != "model_config.json"}
    component_relative_paths = {}
    try:
        hardlink_copy_dir(source_root, temporary)
        for name, source_component in source_components.items():
            source_model_path = Path(source_component.model_path).resolve()
            try:
                relative_model_path = source_model_path.relative_to(source_root)
            except ValueError as exc:
                raise ValueError(
                    f"CompositeModel component {source_model_path} is outside package root {source_root}."
                ) from exc
            component_relative_paths[name] = relative_model_path
            if name in optimized_components:
                component_relative_paths[name] = _replace_component(
                    source_component,
                    optimized_components[name],
                    source_root,
                    temporary,
                )

        component_configs = []
        for name, source_component in source_components.items():
            component = optimized_components.get(name, source_component)
            component_config = deepcopy(component.to_json())
            relative_model_path = component_relative_paths[name]
            component_dir = relative_model_path.parent
            component_config["config"]["model_path"] = str(output_dir / component_dir)
            component_config["config"]["onnx_file_name"] = relative_model_path.name
            component_config["config"]["model_attributes"] = _rebase_additional_files(
                component_config["config"].get("model_attributes") or {},
                source_root,
                output_dir,
                component_dir,
                temporary,
                package_files,
            )
            component_configs.append(component_config)

        parent_attributes = deepcopy(source_model.model_attributes or {})
        parent_additional_files = {
            str(output_dir / path.name)
            for path in source_root.iterdir()
            if path.is_file() and path.name != "model_config.json"
        }
        parent_additional_files.update(
            _rebase_additional_files(
                parent_attributes,
                source_root,
                output_dir,
                Path(),
                temporary,
                package_files,
            ).get("additional_files", [])
        )
        parent_attributes["no_flatten"] = True
        parent_attributes["assembled_components"] = list(optimized_components)
        if parent_additional_files:
            parent_attributes["additional_files"] = sorted(parent_additional_files)

        model_config = {
            "type": "compositemodel",
            "config": {
                "model_path": str(output_dir),
                "model_components": component_configs,
                "model_component_names": list(source_components),
                "model_attributes": parent_attributes,
            },
        }
        model_config_path = temporary / "model_config.json"
        model_config_path.unlink(missing_ok=True)
        model_config_path.write_text(json.dumps(model_config, indent=4), encoding="utf-8")
        _publish_assembly(temporary, output_dir)
        _cleanup_build_outputs(build_configs, output_dir, component_relative_paths)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    for result in results.values():
        result.get_best_candidate()._update_with_model_config(deepcopy(model_config))  # pylint: disable=protected-access

    logger.info("Assembled complete CompositeModel package at %s", output_dir)
    return output_dir
