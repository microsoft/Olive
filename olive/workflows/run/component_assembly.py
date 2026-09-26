# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Assemble component-scoped ONNX builds into a complete CompositeModel package."""

from __future__ import annotations

import filecmp
import json
import logging
import os
import shutil
from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import onnx

from olive.common.utils import copy_dir
from olive.model import ModelConfig
from olive.model.handler import CompositeModelHandler, ONNXModelHandler
from olive.passes.onnx.common import get_context_bin_file_names, get_external_data_file_names, resave_model
from olive.workflows.run._package_paths import (
    confined_artifact_file,
    confined_artifact_path,
    destination_path,
    reject_links,
    validate_source_tree,
)

if TYPE_CHECKING:
    from olive.engine.output import WorkflowOutput
    from olive.workflows.run.builds import ComponentBuildContext
    from olive.workflows.run.config import RunConfig

logger = logging.getLogger(__name__)


def try_assemble_component_builds(
    context: ComponentBuildContext,
    build_configs: dict[str, RunConfig],
    results: OrderedDict[str, WorkflowOutput],
) -> Path | None:
    """Assemble a component-scoped workflow using the input model's package format."""
    if context.input_model.type == "hfmodel":
        from olive.workflows.run.hf_component_assembly import try_assemble_hf_component_builds

        return try_assemble_hf_component_builds(build_configs, results, context.output_dir)
    if context.input_model.type == "compositemodel":
        return _try_assemble_onnx_package(context, build_configs, results)
    return None


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
    protected_files: set[Path],
    artifact_root: Path | None,
    asset_sources: dict[Path, Path],
) -> dict:
    """Copy new component assets while preserving updated package-level files."""
    attributes = deepcopy(attributes)
    rebased = []
    for file_name in attributes.get("additional_files") or []:
        path = Path(file_name)
        reject_links(path)
        if not path.exists():
            raise FileNotFoundError(f"CompositeModel additional file does not exist: {path}")
        resolved = path.resolve()
        if source_root in resolved.parents:
            relative = resolved.relative_to(source_root)
            destination = destination_path(output_root, relative)
        else:
            if artifact_root is None:
                raise ValueError(f"Additional file {path} is outside the input package.")
            resolved = confined_artifact_path(artifact_root, path)
            if resolved.is_dir():
                validate_source_tree(resolved)
            package_file = resolved.name in package_file_names
            if package_file and not resolved.is_file():
                raise ValueError(f"Package-level file {resolved.name!r} must be a regular file.")
            relative = Path(resolved.name) if package_file else component_dir / resolved.name
            if relative in protected_files:
                raise ValueError(f"Optimized asset {resolved} conflicts with a referenced ONNX file {relative}.")
            if not package_file:
                index = 0
                while (temporary_root / relative).exists():
                    index += 1
                    relative = component_dir / f"{resolved.stem}.optimized-{index}{resolved.suffix}"
            destination = destination_path(output_root, relative)
            staged = destination_path(temporary_root, relative)
            previous = asset_sources.get(staged)
            if previous is not None and previous != resolved:
                if not (resolved.is_file() and previous.is_file() and filecmp.cmp(staged, resolved, shallow=False)):
                    raise ValueError(f"Conflicting component assets target {destination}: {previous} and {resolved}.")
            elif previous is None:
                staged.parent.mkdir(parents=True, exist_ok=True)
                if resolved.is_dir():
                    shutil.copytree(resolved, staged)
                else:
                    shutil.copy2(resolved, staged)
                asset_sources[staged] = resolved
        if str(destination) not in rebased:
            rebased.append(str(destination))

    if rebased:
        attributes["additional_files"] = rebased
    else:
        attributes.pop("additional_files", None)
    return attributes


def _rename_context_reference(model_path: Path, old_name: str, new_name: str) -> None:
    """Update EPContext references without modifying a staged model hardlinked to the build."""
    model = onnx.load(model_path, load_external_data=False)
    updated = False
    for node in model.graph.node:
        if node.op_type == "EPContext":
            for attribute in node.attribute:
                if attribute.name == "ep_cache_context" and attribute.s == old_name.encode("utf-8"):
                    attribute.s = new_name.encode("utf-8")
                    updated = True
    if not updated:
        raise ValueError(f"EPContext binary {old_name!r} is not referenced by {model_path}")
    rewritten = model_path.with_name(f"{model_path.name}.rewritten")
    try:
        onnx.save_model(model, rewritten)
        rewritten.replace(model_path)
    finally:
        rewritten.unlink(missing_ok=True)


def _replace_component(
    source_component: ONNXModelHandler,
    optimized_component: ONNXModelHandler,
    source_root: Path,
    temporary_root: Path,
    artifact_root: Path,
    staging_root: Path,
) -> tuple[Path, dict[str, str]]:
    """Stage an optimized model and its owned assets before touching the package copy."""
    source_model_path = Path(source_component.model_path).resolve()
    try:
        relative_model_path = source_model_path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"CompositeModel component {source_model_path} is outside package root {source_root}."
        ) from exc

    optimized_model_path = confined_artifact_file(artifact_root, Path(optimized_component.model_path))
    context_names = get_context_bin_file_names(optimized_model_path)
    for location in (*get_external_data_file_names(optimized_model_path), *context_names):
        if Path(location).is_absolute():
            raise ValueError(f"ONNX artifact location must be relative: {location}")
        if location in context_names and ".." in Path(location).parts:
            raise ValueError(f"ONNX context binary must stay in the model directory: {location}")
        confined_artifact_file(artifact_root, optimized_model_path.parent / location)
    destination = destination_path(temporary_root, relative_model_path)
    index = 0
    while True:
        suffix = "" if index == 0 else f"-{index}"
        staged_name = f"{destination.stem}.optimized{suffix}{destination.suffix}"
        if (
            not (destination.parent / staged_name).exists()
            and not (destination.parent / f"{staged_name}.data").exists()
            and staged_name not in context_names
            and f"{staged_name}.data" not in context_names
        ):
            break
        index += 1

    staging_root.mkdir()
    for location in context_names:
        destination_path(staging_root, Path(location)).parent.mkdir(parents=True, exist_ok=True)
    staged_model = staging_root / staged_name
    resave_model(optimized_model_path, staged_model)
    context_companions: dict[Path, Path] = {}
    for location in context_names:
        reference = Path(location)
        if reference.suffix.lower() != ".xml":
            continue
        companion = reference.with_suffix(".bin")
        companion_source = optimized_model_path.parent / companion
        if not companion_source.exists():
            continue
        companion_source = confined_artifact_file(artifact_root, companion_source)
        companion_staged = destination_path(staging_root, companion)
        if companion_staged.exists():
            if not filecmp.cmp(companion_source, companion_staged, shallow=False):
                raise ValueError(f"ONNX context companion {companion_source} conflicts with a staged asset.")
        else:
            companion_staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(companion_source, companion_staged)
        context_companions[reference] = companion

    staged_files = sorted(path for path in staging_root.rglob("*") if path.is_file())
    staged_names = {path.relative_to(staging_root) for path in staged_files}
    context_references = {Path(name): name for name in get_context_bin_file_names(staged_model)}
    external_names = {Path(name) for name in get_external_data_file_names(staged_model)}
    staged_destinations = {staged_model: destination}
    for staged in staged_files:
        if staged == staged_model or staged.relative_to(staging_root) in context_companions.values():
            continue
        relative = staged.relative_to(staging_root)
        companion = context_companions.get(relative)
        staged_asset_names = (relative, companion) if companion else (relative,)
        targets = tuple(
            destination_path(temporary_root, relative_model_path.parent / asset_name)
            for asset_name in staged_asset_names
        )
        if any(target.exists() or target == destination for target in targets):
            if relative not in context_references or any(name in external_names for name in staged_asset_names):
                raise ValueError(f"Optimized ONNX asset {staged} collides with package file {targets[0]}.")
            index = 0
            while True:
                index += 1
                renamed = relative.with_name(f"{relative.stem}.optimized-{index}{relative.suffix}")
                renamed_names = (renamed, renamed.with_suffix(".bin")) if companion else (renamed,)
                targets = tuple(
                    destination_path(temporary_root, relative_model_path.parent / name) for name in renamed_names
                )
                if all(
                    not target.exists() and name not in staged_names and target not in staged_destinations.values()
                    for name, target in zip(renamed_names, targets)
                ):
                    break
            _rename_context_reference(staged_model, context_references[relative], renamed.as_posix())
            if companion and companion in context_references:
                _rename_context_reference(
                    staged_model, context_references[companion], renamed.with_suffix(".bin").as_posix()
                )
        for name, target in zip(staged_asset_names, targets):
            if target in staged_destinations.values():
                raise ValueError(f"Optimized ONNX assets target the same package file: {target}")
            staged_destinations[staging_root / name] = target
    destination.unlink()
    for staged, target in staged_destinations.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged, target)

    asset_names: dict[str, str] = {}
    for field_name, additional_path in (
        ("external_initializers_file_name", optimized_component.external_initializers_path),
        ("constant_inputs_file_name", optimized_component.constant_inputs_path),
    ):
        if not additional_path:
            continue
        source_asset = confined_artifact_file(artifact_root, Path(additional_path))
        configured_name = Path(getattr(optimized_component, field_name))
        if configured_name.is_absolute() or ".." in configured_name.parts:
            raise ValueError(f"ONNX component asset name must be relative: {configured_name}")
        asset_name = configured_name
        asset_target = destination_path(temporary_root, relative_model_path.parent / asset_name)
        suffix = 0
        while asset_target.exists():
            suffix += 1
            asset_name = configured_name.with_name(f"{configured_name.stem}.optimized-{suffix}{configured_name.suffix}")
            asset_target = destination_path(temporary_root, relative_model_path.parent / asset_name)
        asset_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, asset_target)
        asset_names[field_name] = str(asset_name)
    return relative_model_path, asset_names


def _publish_assembly(temporary: Path, output_dir: Path) -> None:
    """Publish only new package paths, retaining unrelated files and build outputs."""
    validate_source_tree(temporary)
    reject_links(output_dir)
    source_directories = sorted(
        (path for path in temporary.rglob("*") if path.is_dir()),
        key=lambda path: (len(path.relative_to(temporary).parts), str(path)),
    )
    source_files = sorted(path for path in temporary.rglob("*") if path.is_file())
    for source in source_directories:
        destination = destination_path(output_dir, source.relative_to(temporary))
        if destination.exists() and not destination.is_dir():
            raise ValueError(f"ONNX package directory conflicts with output file {destination}")
    for source in source_files:
        destination = destination_path(output_dir, source.relative_to(temporary))
        if destination.exists():
            raise ValueError(f"ONNX package file already exists in output directory: {destination}")

    output_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    created_directories: list[Path] = []
    try:
        for source in source_directories:
            destination = destination_path(output_dir, source.relative_to(temporary))
            if not destination.exists():
                destination.mkdir()
                created_directories.append(destination)

        for source in source_files:
            destination = destination_path(output_dir, source.relative_to(temporary))
            os.link(source, destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            destination.unlink()
        for directory in reversed(created_directories):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        raise


def _try_assemble_onnx_package(
    context: ComponentBuildContext,
    build_configs: dict[str, RunConfig],
    results: OrderedDict[str, WorkflowOutput],
) -> Path | None:
    """Assemble component-scoped ONNX builds and untouched source components into one package."""
    source_root_value = context.input_model.config.get("model_path")
    if not source_root_value:
        return None
    source_path = Path(source_root_value)
    reject_links(source_path)
    source_root = source_path.resolve()
    if not source_root.is_dir():
        return None
    validate_source_tree(source_path)
    reject_links(context.output_dir)
    output_dir = context.output_dir.resolve()
    if output_dir == source_root or source_root in output_dir.parents or output_dir in source_root.parents:
        raise ValueError("CompositeModel workflow output directory must not overlap the input package.")
    artifact_roots = {}
    for build_name, run_config in build_configs.items():
        artifact_root = Path(run_config.engine.output_dir)
        reject_links(artifact_root)
        if not artifact_root.is_dir():
            raise ValueError(f"CompositeModel build output directory does not exist: {artifact_root}")
        validate_source_tree(artifact_root)
        artifact_roots[build_name] = artifact_root.resolve()

    source_model = context.input_model.create_model()
    if not isinstance(source_model, CompositeModelHandler):
        return None
    source_components = OrderedDict(source_model.get_model_components())
    if not all(isinstance(component, ONNXModelHandler) for component in source_components.values()):
        return None
    optimized_components = _collect_optimized_components(context.components, results)
    if optimized_components is None:
        return None
    unknown_components = set(optimized_components) - set(source_components)
    if unknown_components:
        raise ValueError(f"CompositeModel builds produced unknown components: {sorted(unknown_components)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid4().hex}.tmp")
    staging = output_dir.with_name(f".{output_dir.name}.{uuid4().hex}.assets.tmp")
    package_files = {path.name for path in source_root.iterdir() if path.is_file() and path.name != "model_config.json"}
    component_relative_paths: dict[str, Path] = {}
    component_asset_names: dict[str, dict[str, str]] = {}
    build_artifacts = {
        component: artifact_roots[build_name]
        for build_name, selected in context.components.items()
        for component in selected
    }
    asset_sources: dict[Path, Path] = {}
    try:
        copy_dir(source_root, temporary, symlinks=True)
        validate_source_tree(temporary)
        staging.mkdir()

        source_model_paths = [Path(component.model_path).resolve() for component in source_components.values()]
        duplicate_model_paths = [path for path, count in Counter(source_model_paths).items() if count > 1]
        if duplicate_model_paths:
            raise ValueError(f"CompositeModel components share an ONNX model file: {duplicate_model_paths}")

        protected_files: set[Path] = set()
        for name, source_component in source_components.items():
            model_path = confined_artifact_file(source_root, Path(source_component.model_path))
            protected_files.add(model_path.relative_to(source_root))
            if name in optimized_components:
                continue
            context_locations = get_context_bin_file_names(model_path)
            for location in (*get_external_data_file_names(model_path), *context_locations):
                if Path(location).is_absolute():
                    raise ValueError(f"ONNX package asset location must be relative: {location}")
                asset = confined_artifact_file(source_root, model_path.parent / location)
                protected_files.add(asset.relative_to(source_root))
                if location in context_locations and asset.suffix.lower() == ".xml":
                    companion = asset.with_suffix(".bin")
                    if companion.exists():
                        protected_files.add(confined_artifact_file(source_root, companion).relative_to(source_root))
            for asset_path in (source_component.external_initializers_path, source_component.constant_inputs_path):
                if asset_path:
                    asset = confined_artifact_file(source_root, Path(asset_path))
                    protected_files.add(asset.relative_to(source_root))

        for index, (name, source_component) in enumerate(source_components.items()):
            source_model_path = Path(source_component.model_path).resolve()
            try:
                relative_model_path = source_model_path.relative_to(source_root)
            except ValueError as exc:
                raise ValueError(
                    f"CompositeModel component {source_model_path} is outside package root {source_root}."
                ) from exc
            component_relative_paths[name] = relative_model_path
            if name in optimized_components:
                component_relative_paths[name], component_asset_names[name] = _replace_component(
                    source_component,
                    optimized_components[name],
                    source_root,
                    temporary,
                    build_artifacts[name],
                    staging / f"component-{index}",
                )

        component_configs = []
        for name, source_component in source_components.items():
            component = optimized_components.get(name, source_component)
            component_config = deepcopy(component.to_json())
            relative_model_path = component_relative_paths[name]
            component_dir = relative_model_path.parent
            component_config["config"]["model_path"] = str(output_dir / component_dir)
            component_config["config"]["onnx_file_name"] = relative_model_path.name
            for field_name, property_name in (
                ("external_initializers_file_name", "external_initializers_path"),
                ("constant_inputs_file_name", "constant_inputs_path"),
            ):
                if name in optimized_components:
                    if field_name in component_asset_names[name]:
                        component_config["config"][field_name] = component_asset_names[name][field_name]
                elif source_path := getattr(source_component, property_name):
                    source_asset = Path(source_path).resolve()
                    if source_root not in source_asset.parents:
                        raise ValueError(f"CompositeModel component asset {source_asset} is outside {source_root}")
                    component_config["config"][field_name] = os.path.relpath(source_asset, source_root / component_dir)
            component_config["config"]["model_attributes"] = _rebase_additional_files(
                component_config["config"].get("model_attributes") or {},
                source_root,
                output_dir,
                component_dir,
                temporary,
                package_files,
                protected_files,
                build_artifacts[name] if name in optimized_components else None,
                asset_sources,
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
                protected_files,
                None,
                asset_sources,
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
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if staging.exists():
            shutil.rmtree(staging)

    for result in results.values():
        result.get_best_candidate()._update_with_model_config(deepcopy(model_config))  # pylint: disable=protected-access

    logger.info("Assembled complete CompositeModel package at %s", output_dir)
    return output_dir
