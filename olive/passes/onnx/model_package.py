# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX Runtime model-package pass.

Assembles pre-compiled EPContext ONNX models into an ONNX Runtime model package using ORT's
``onnxruntime.ModelPackage`` authoring API. Each input model becomes a variant of a single component. The
original model and its external-data files can be stored once as a content-addressed shared asset that
weightless EPContext variants read their weights from.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional, Union

from packaging import version

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

_MINIMUM_ORT_VERSION = version.Version("1.32.0")
_DEFAULT_VARIANT = "default"


class ORTModelPackage(Pass):
    """Assemble EPContext ONNX models into an ONNX Runtime model package."""

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "package_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Package/component name. Defaults to the shared-asset model's file stem, else the first"
                    " variant's file stem."
                ),
            ),
            "package_version": PassConfigParam(
                type_=str,
                default_value="1.0.0",
                description="Version recorded under manifest.package_version.",
            ),
            "package_suffix": PassConfigParam(
                type_=str,
                default_value=".ortpackage",
                description="Directory suffix for the package root.",
            ),
            "device": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Device class recorded on each variant and used in the variant name. Defaults to the"
                    " accelerator's device type."
                ),
            ),
            "variant_name_template": PassConfigParam(
                type_=str,
                default_value="{package_name}.{device}_{variant}",
                description=(
                    "Template for variant names. Available fields: package_name, device, variant. 'variant' is the"
                    f" component name for a composite input model and '{_DEFAULT_VARIANT}' for a single model."
                ),
            ),
            "session_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Runtime ONNX Runtime session options written verbatim into each variant's"
                    " executor_info.ort.session_options (e.g. {'ep.context_file_path': 'model_ctx.onnx'}). The"
                    " shared-asset weights folder is added automatically."
                ),
            ),
            "executor_namespace": PassConfigParam(
                type_=str,
                default_value="ort",
                description="executor_info namespace key written for each variant.",
            ),
            "external_initializers_option": PassConfigParam(
                type_=str,
                default_value="session.model_external_initializers_file_folder_path",
                description="Session-option key set to the shared-asset URI so weightless variants find their weights.",
            ),
            "create_shared_asset": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether to stage the original model + external-data files as a content-addressed asset. Requires"
                    " shared_asset_model_path or a 'source_model_path' model attribute (set by"
                    " EPContextBinaryGenerator)."
                ),
            ),
            "shared_asset_model_path": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "ONNX model path used for shared-asset staging. Defaults to the input model's 'source_model_path'"
                    " attribute."
                ),
            ),
        }

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> CompositeModelHandler:
        import onnxruntime as ort

        self._check_ort_version(ort.__version__)
        if not hasattr(ort, "ModelPackage"):
            raise RuntimeError(
                "ORTModelPackage requires an ONNX Runtime build exposing onnxruntime.ModelPackage. "
                "Rebuild/install ONNX Runtime with the model package Python bindings enabled."
            )

        sources = self._get_variant_sources(model)
        ep_name = str(self.accelerator_spec.execution_provider)
        device = config.device or str(self.accelerator_spec.accelerator_type).lower()

        asset_model_path = self._get_asset_model_path(model, config)
        package_name = config.package_name or (asset_model_path or next(iter(sources.values()))).stem

        variant_names = {
            variant_id: config.variant_name_template.format(
                package_name=package_name, device=device, variant=variant_id
            )
            for variant_id in sources
        }
        if len(set(variant_names.values())) != len(variant_names):
            raise ValueError(f"variant_name_template produced duplicate variant names: {list(variant_names.values())}.")
        for variant_name in variant_names.values():
            normalized_name = variant_name.replace("\\", "/")
            if (
                not variant_name
                or normalized_name in {".", ".."}
                or PurePosixPath(normalized_name).name != normalized_name
                or PureWindowsPath(variant_name).name != variant_name
            ):
                raise ValueError(f"Generated variant name must be a single safe path component: {variant_name!r}.")

        session_options = dict(config.session_options or {})
        weights_model_path = asset_model_path or (model.model_attributes or {}).get("source_model_path")
        if (
            not config.create_shared_asset
            and weights_model_path
            and self._external_data_files(Path(weights_model_path))
            and not session_options.get(config.external_initializers_option)
        ):
            raise ValueError(
                "create_shared_asset=False requires session_options to provide a non-empty "
                f"'{config.external_initializers_option}' for models with external weights."
            )

        logger.info("Building package '%s' for variants %s", package_name, list(variant_names.values()))

        output_root = Path(output_model_path)
        package_dir = output_root / f"{package_name}{config.package_suffix}"
        if package_dir.exists() and any(package_dir.iterdir()):
            raise FileExistsError(
                f"Package directory {package_dir} already exists and is not empty. Refusing to mix stale variants or"
                " shared assets with a new manifest; remove it or choose a different output location."
            )
        output_root.mkdir(parents=True, exist_ok=True)
        # Establish the package root through ORT before adding relative variant paths.
        # Work in a temporary directory so a failed commit never publishes a partial package.
        with tempfile.TemporaryDirectory(prefix=".ortpackage-", dir=output_root) as staging:
            staging_root = Path(staging)
            staged_package = staging_root / "package"
            with ort.ModelPackage() as package:
                package.set_metadata(name=package_name, version=config.package_version)
                package.commit(str(staged_package))
                package.set_component_inline(package_name, json.dumps({"variants": {}}))
                shared_asset_uri = None
                if config.create_shared_asset:
                    shared_asset_uri = self._add_shared_asset(asset_model_path, package, staging_root / "asset")
                    logger.info("Shared asset: %s", shared_asset_uri)

                variants = []
                for variant_id, source_path in sources.items():
                    variant_name = variant_names[variant_id]
                    variant_dir = staged_package / package_name / variant_name
                    variant_dir.mkdir(parents=True, exist_ok=True)
                    self._copy_variant(source_path, variant_dir / source_path.name)
                    variants.append(
                        {
                            "variant_name": variant_name,
                            "model_file": source_path.name,
                            "compat_string": self._read_compat(source_path, ep_name),
                        }
                    )
                self._add_variants(
                    package=package,
                    package_name=package_name,
                    ep_name=ep_name,
                    device=device,
                    variants=variants,
                    executor_namespace=config.executor_namespace,
                    session_options=session_options,
                    external_initializers_option=config.external_initializers_option,
                    shared_asset_uri=shared_asset_uri,
                )
                package.commit()
            if package_dir.exists():
                package_dir.rmdir()
            staged_package.rename(package_dir)
        logger.info("Package created: %s", package_dir)

        components = [
            ONNXModelHandler(
                model_path=str(package_dir / package_name / v["variant_name"]), onnx_file_name=v["model_file"]
            )
            for v in variants
        ]
        return CompositeModelHandler(
            components,
            [v["variant_name"] for v in variants],
            model_path=str(output_root),
            model_attributes={
                **(model.model_attributes or {}),
                "no_flatten": True,
                "ep": ep_name,
                "package_name": package_name,
                "package_version": config.package_version,
            },
        )

    @staticmethod
    def _get_variant_sources(model: Union[ONNXModelHandler, CompositeModelHandler]) -> dict[str, Path]:
        """Return {variant id: EPContext model file} for the input model."""
        if isinstance(model, CompositeModelHandler):
            components = list(model.get_model_components())
        else:
            components = [(_DEFAULT_VARIANT, model)]

        sources = {}
        for name, component in components:
            if not isinstance(component, ONNXModelHandler):
                raise NotImplementedError(f"ORTModelPackage expects ONNX models, got {type(component).__name__}.")
            source_path = Path(component.model_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"Model file not found: {source_path}")
            sources[name] = source_path
        if not sources:
            raise ValueError("ORTModelPackage requires at least one model.")
        return sources

    @staticmethod
    def _get_asset_model_path(model, config) -> Optional[Path]:
        if not config.create_shared_asset:
            return Path(config.shared_asset_model_path) if config.shared_asset_model_path else None

        asset_model = config.shared_asset_model_path or (model.model_attributes or {}).get("source_model_path")
        if not asset_model:
            raise ValueError(
                "create_shared_asset requires shared_asset_model_path or a 'source_model_path' model attribute."
                " Set create_shared_asset=False if the EPContext models embed their weights."
            )
        asset_model_path = Path(asset_model)
        if not asset_model_path.is_file():
            raise FileNotFoundError(f"Shared-asset model file not found: {asset_model_path}")
        return asset_model_path

    @staticmethod
    def _check_ort_version(ort_version: str) -> None:
        # Compare the release segment so dev/rc builds of the minimum version are accepted.
        if version.parse(ort_version).release < _MINIMUM_ORT_VERSION.release:
            raise RuntimeError(f"ORTModelPackage requires ONNX Runtime >= {_MINIMUM_ORT_VERSION}; found {ort_version}.")

    @classmethod
    def _copy_variant(cls, source_ctx: Path, output_ctx: Path) -> None:
        cls._copy_preserving_location(source_ctx, output_ctx)
        for rel, src in cls._compiled_context_auxiliary_files(source_ctx):
            cls._copy_preserving_location(src, output_ctx.parent / rel)

    @classmethod
    def _compiled_context_auxiliary_files(cls, ctx_model_path: Path) -> list[tuple[str, Path]]:
        # Preserve conventional sidecars while also honoring explicit EPContext references.
        files: dict[str, Path] = {}
        for path in ctx_model_path.parent.glob(f"{ctx_model_path.stem}*.bin"):
            location = cls._safe_relative_location(path.name)
            source = cls._resolve_source_within_directory(ctx_model_path.parent, location, ctx_model_path)
            cls._add_asset_entry(files, location, source)
        for location, src in cls._ep_context_binary_files(ctx_model_path):
            cls._add_asset_entry(files, location, src)
        for location, src in cls._external_data_files(ctx_model_path):
            cls._add_asset_entry(files, location, src)
        return list(files.items())

    @classmethod
    def _ep_context_binary_files(cls, ctx_model_path: Path) -> list[tuple[str, Path]]:
        """Return validated non-embedded EPContext binary references from an ONNX model."""
        import onnx

        model = onnx.load(str(ctx_model_path), load_external_data=False)
        files: dict[str, Path] = {}
        for node in cls._all_nodes(model.graph, onnx):
            if node.op_type != "EPContext":
                continue
            attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
            location = attributes.get("ep_cache_context")
            if not location or attributes.get("embed_mode", 0):
                continue
            if isinstance(location, bytes):
                location = location.decode("utf-8")
            safe_location = cls._safe_relative_location(location)
            source = cls._resolve_source_within_directory(ctx_model_path.parent, safe_location, ctx_model_path)
            cls._add_asset_entry(files, safe_location, source)
        return list(files.items())

    @classmethod
    def _all_nodes(cls, graph, onnx):
        for node in graph.node:
            yield node
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from cls._all_nodes(attribute.g, onnx)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        yield from cls._all_nodes(subgraph, onnx)

    @staticmethod
    def _read_compat(ctx_model_path: Path, ep_name: str) -> str:
        import onnxruntime as ort

        return ort.get_compatibility_info_from_model(str(ctx_model_path), ep_name) or ""

    @classmethod
    def _add_shared_asset(cls, model_path: Path, package, asset_dir: Path) -> str:
        # Keep the model filename and each external-data file's recorded relative location.
        entries: dict[str, Path] = {model_path.name: model_path}
        for location, src in cls._external_data_files(model_path):
            cls._add_asset_entry(entries, location, src)
        for rel, src in entries.items():
            cls._link_or_copy(src, asset_dir / rel)
        # ORT owns hashing, URI generation, validation, and materialization at commit.
        # The staged source directory must remain alive until package.commit().
        return package.add_shared_asset(str(asset_dir))

    @staticmethod
    def _external_data_files(model_path: Path) -> list[tuple[str, Path]]:
        """Return (recorded_relative_location, resolved_source_path) for each external-data file."""
        import onnx

        model = onnx.load(str(model_path), load_external_data=False)
        from onnx import external_data_helper

        seen: set[str] = set()
        files: list[tuple[str, Path]] = []
        for tensor in external_data_helper._get_all_tensors(model):  # pylint: disable=W0212
            if not external_data_helper.uses_external_data(tensor):
                continue
            location = external_data_helper.ExternalDataInfo(tensor).location
            if location in seen:
                continue
            seen.add(location)
            safe_location = ORTModelPackage._safe_relative_location(location)
            src = ORTModelPackage._resolve_source_within_directory(model_path.parent, safe_location, model_path)
            files.append((safe_location, src))
        return files

    @staticmethod
    def _safe_relative_location(location: str) -> str:
        """Validate an ONNX-recorded path using both POSIX and Windows path rules."""
        if not isinstance(location, str):
            raise ValueError(f"External-data or EPContext location must be a string, got {type(location).__name__}.")
        normalized = location.replace("\\", "/")
        posix_path = PurePosixPath(normalized)
        windows_path = PureWindowsPath(location)
        if (
            not normalized
            or "\x00" in normalized
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError(f"External-data or EPContext location must be a safe relative path: {location!r}.")
        return posix_path.as_posix()

    @staticmethod
    def _resolve_source_within_directory(directory: Path, location: str, model_path: Path) -> Path:
        root = directory.resolve()
        source = (root / location).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"Referenced file {location!r} resolves outside the model directory {root}.")
        if not source.is_file():
            raise FileNotFoundError(f"Referenced file {location!r} for {model_path} not found: {source}")
        return source

    @staticmethod
    def _add_asset_entry(entries: dict[str, Path], location: str, src: Path) -> None:
        """Add {location: src} to entries, rejecting distinct files that collide on the same location."""
        existing = entries.get(location)
        if existing is not None and existing.resolve() != src.resolve():
            raise ValueError(f"Two distinct files map to the same package location '{location}': {existing}, {src}")
        entries[location] = src

    @staticmethod
    def _link_or_copy(src: Path, dest: Path) -> None:
        """Hard-link src to dest to avoid duplicating large weights, falling back to a copy across volumes."""
        if src.resolve() == dest.resolve():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(str(src), str(dest))

    @staticmethod
    def _copy_preserving_location(src: Path, dest: Path) -> None:
        if src.resolve() == dest.resolve():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))

    @staticmethod
    def _add_variants(
        package,
        package_name: str,
        ep_name: str,
        device: str,
        variants: list[dict],
        executor_namespace: str,
        session_options: dict,
        external_initializers_option: str,
        shared_asset_uri: Optional[str],
    ) -> None:
        for v in variants:
            variant_body: dict = {
                "variant_directory": f"{package_name}/{v['variant_name']}",
                "ep": ep_name,
                "device": device,
            }
            if v["compat_string"]:
                variant_body["compatibility_string"] = v["compat_string"]

            ort_info: dict = {"model_file": v["model_file"]}
            variant_session_options = dict(session_options)
            if shared_asset_uri:
                variant_session_options[external_initializers_option] = shared_asset_uri
            if variant_session_options:
                ort_info["session_options"] = variant_session_options
            package.set_variant(package_name, v["variant_name"], json.dumps(variant_body))
            package.set_variant_executor_info_inline(
                package_name, v["variant_name"], executor_namespace, json.dumps(ort_info)
            )
