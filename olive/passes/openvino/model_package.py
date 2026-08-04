# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""OpenVINO model-package pass.

Compiles an ONNX model into per-device OpenVINO EPContext variants and assembles
an ONNX Runtime model package. The pass can also package existing EPContext models
without compiling. Each requested virtual NPU target becomes a variant under a single
component; the original model and its external-data files are stored once as a
content-addressed shared asset that the (weightless) compiled variants read their
weights from.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

_OPENVINO_EP_PLUGIN_PACKAGE = "onnxruntime_ep_openvino"
_EP_NAME = "OpenVINOExecutionProvider"
_COMPAT_METADATA_KEY = "ep_compatibility_info.OpenVINOExecutionProvider"


class OpenVINOModelPackage(Pass):
    """Compile per-device OpenVINO variants and assemble an ONNX Runtime model package."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "package_name": PassConfigParam(
                type_=str,
                default_value=None,
                description="Package/component name. Defaults to the input model's file stem.",
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
            "target_devices": PassConfigParam(
                type_=list,
                default_value=None,
                description=(
                    "Device ids (uppercase hex, e.g. ['643E', 'B03E']) to compile variants for. Matches both"
                    " virtual (cross-compile) and physical devices. Empty or unset auto-discovers every device the"
                    " plugin exposes."
                ),
            ),
            "device": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Device class recorded on each variant and used in the variant name (<package>.<device>_<id>)."
                    " Defaults to the accelerator's device type."
                ),
            ),
            "variant_name_template": PassConfigParam(
                type_=str,
                default_value="{package_name}.{device}_{device_id}",
                description="Template for variant names. Available fields: package_name, device, device_id.",
            ),
            "output_ctx_name": PassConfigParam(
                type_=str,
                default_value="model_ctx.onnx",
                description="File name of the compiled EPContext model written into each variant directory.",
            ),
            "package_only": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to skip compilation and package existing EPContext model(s).",
            ),
            "compiled_model_paths": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Mapping from device id to existing EPContext model path for package_only mode. Values may be"
                    " files or directories containing output_ctx_name. If unset, the input model is used and exactly"
                    " one target_devices entry must be provided."
                ),
            ),
            "compile_flow": PassConfigParam(
                type_=str,
                default_value="aot",
                description="'aot' compiles via ORT's ModelCompiler; 'jit' compiles via an InferenceSession.",
            ),
            "compile_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "ONNX Runtime session config entries applied verbatim at compile time (e.g."
                    " 'ep.context_embed_mode', 'ep.enable_weightless_ep_context_nodes')."
                ),
            ),
            "session_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Runtime session options written verbatim into each variant's executor_info.ort.session_options"
                    " (e.g. {'ep.context_file_path': 'model_ctx.onnx'}). The shared-asset weights folder is added"
                    " automatically when a shared asset is created."
                ),
            ),
            "provider_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Provider options passed verbatim to the OpenVINO plugin EP for every compile.",
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
                description="Whether to stage the original model + external-data files as a content-addressed asset.",
            ),
            "allow_partial_package": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "If False (default), the pass fails when any requested target device fails to compile. If True,"
                    " failed devices are skipped (logged) and a package is produced from the ones that succeeded."
                ),
            ),
            "shared_asset_model_path": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "ONNX model path used for shared-asset staging. Defaults to the input model path. In package_only"
                    " mode, set this to the original ONNX model when the input model is an existing EPContext model."
                ),
            ),
            "ep_registration_name": PassConfigParam(
                type_=str,
                default_value="ovep_registration.virtual",
                description=(
                    "Name used to register the plugin EP library. A '.virtual' suffix makes the plugin enumerate"
                    " virtual (cross-compile) devices."
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
        if isinstance(model, CompositeModelHandler):
            raise NotImplementedError("OpenVINOModelPackage expects a single ONNX model.")

        model_path = Path(model.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        package_name = config.package_name or model_path.stem
        device = config.device or str(self.accelerator_spec.accelerator_type).lower()
        target_device_ids = [str(d).upper() for d in (config.target_devices or [])]

        if config.package_only:
            targets = self._get_package_only_targets(
                config.compiled_model_paths, target_device_ids, model_path, config.output_ctx_name
            )
        else:
            targets = self._discover_targets(config.ep_registration_name, target_device_ids)
        if not targets:
            raise ValueError(f"No OpenVINO target devices found for {target_device_ids or 'auto-discovery'}.")

        logger.info("Building package '%s' for devices %s", package_name, [t["device_id_hex"] for t in targets])

        output_root = Path(output_model_path)
        package_dir = output_root / f"{package_name}{config.package_suffix}"
        if package_dir.exists() and any(package_dir.iterdir()):
            raise FileExistsError(
                f"Package directory {package_dir} already exists and is not empty. Refusing to mix stale variants or"
                " shared assets with a new manifest; remove it or choose a different output location."
            )
        package_dir.mkdir(parents=True, exist_ok=True)

        shared_asset_uri = self._maybe_add_shared_asset(config, model_path, package_dir)

        variants, failures = [], {}
        for target in targets:
            try:
                variants.append(self._produce_variant(config, package_dir, package_name, device, model_path, target))
            except Exception as e:
                failures[target["device_id_hex"]] = str(e)
                shutil.rmtree(package_dir / package_name / self._variant_name(config, package_name, device, target), ignore_errors=True)
                logger.warning("Variant failed for device %s: %s", target["device_id_hex"], e)

        if failures and not config.allow_partial_package:
            raise RuntimeError(
                f"Failed to produce variants for devices {sorted(failures)}: {failures}. "
                "Set allow_partial_package=true to package only the successful devices."
            )
        if not variants:
            raise RuntimeError(f"No variants produced successfully. Failures: {failures}")
        if failures:
            logger.warning("Producing partial package; skipped devices: %s", sorted(failures))

        self._write_manifest(
            package_dir=package_dir,
            package_name=package_name,
            package_version=config.package_version,
            device=device,
            variants=variants,
            output_ctx_name=config.output_ctx_name,
            executor_namespace=config.executor_namespace,
            session_options=dict(config.session_options or {}),
            external_initializers_option=config.external_initializers_option,
            shared_asset_uri=shared_asset_uri,
        )
        logger.info("Package created: %s", package_dir)

        components = [
            ONNXModelHandler(model_path=str(v["variant_dir"]), onnx_file_name=config.output_ctx_name) for v in variants
        ]
        names = [v["variant_name"] for v in variants]
        return CompositeModelHandler(
            components,
            names,
            model_path=str(output_root),
            model_attributes={
                **(model.model_attributes or {}),
                "no_flatten": True,
                "ep": _EP_NAME,
                "package_name": package_name,
                "package_version": config.package_version,
            },
        )

    @classmethod
    def _maybe_add_shared_asset(cls, config, model_path: Path, package_dir: Path) -> Optional[str]:
        """Stage the original model + external-data files as a content-addressed shared asset."""
        if not config.create_shared_asset:
            return None
        asset_model_path = Path(config.shared_asset_model_path) if config.shared_asset_model_path else model_path
        if not asset_model_path.exists():
            raise FileNotFoundError(f"Shared-asset model file not found: {asset_model_path}")
        uri = cls._add_shared_asset(asset_model_path, package_dir)
        logger.info("Shared asset: %s", uri)
        return uri

    @staticmethod
    def _variant_name(config, package_name: str, device: str, target: dict) -> str:
        return config.variant_name_template.format(
            package_name=package_name, device=device, device_id=target["device_id_hex"]
        )

    def _produce_variant(self, config, package_dir: Path, package_name: str, device: str, model_path: Path, target: dict) -> dict:
        """Compile (or copy, in package_only mode) one variant's EPContext model into the package."""
        dev_id = target["device_id_hex"]
        variant_name = self._variant_name(config, package_name, device, target)
        variant_dir = package_dir / package_name / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        output_ctx = variant_dir / config.output_ctx_name

        logger.info("%s variant=%s device=%s", "Packaging" if config.package_only else "Compiling", variant_name, dev_id)
        if config.package_only:
            self._copy_existing_variant(target["compiled_model_path"], output_ctx)
        else:
            self._compile_variant(config, model_path, output_ctx, target)

        return {
            "variant_name": variant_name,
            "device_id_hex": dev_id,
            "variant_dir": variant_dir,
            "compat_string": self._read_compat(output_ctx),
        }

    def _discover_targets(self, ep_registration_name: str, target_device_ids: list[str]) -> list[dict]:
        import importlib

        import onnxruntime as ort

        from olive.passes.onnx.context_binary import registered_ep_library

        try:
            ep_package = importlib.import_module(_OPENVINO_EP_PLUGIN_PACKAGE)
        except ImportError as e:
            raise ImportError(
                f"The plugin EP package '{_OPENVINO_EP_PLUGIN_PACKAGE}' is required. Install it with"
                f" 'pip install {_OPENVINO_EP_PLUGIN_PACKAGE.replace('_', '-')}'."
            ) from e

        # Accept virtual and physical devices keyed by device_id; prefer the npu_platform entry on collision.
        wanted = set(target_device_ids)
        targets: dict[str, dict] = {}
        with registered_ep_library(ort, ep_registration_name, ep_package.get_library_path()):
            for dev in ort.get_ep_devices():
                if dev.ep_name != _EP_NAME:
                    continue
                dev_id_hex = f"{dev.device.device_id:X}"
                if wanted and dev_id_hex not in wanted:
                    continue
                npu_platform = dict(dev.ep_metadata).get("npu_platform")
                if dev_id_hex not in targets or (npu_platform and not targets[dev_id_hex]["npu_platform"]):
                    targets[dev_id_hex] = {"device_id_hex": dev_id_hex, "npu_platform": npu_platform}
        return list(targets.values())

    @classmethod
    def _get_package_only_targets(
        cls,
        compiled_model_paths: Optional[dict],
        target_device_ids: list[str],
        model_path: Path,
        output_ctx_name: str,
    ) -> list[dict]:
        if compiled_model_paths:
            compiled_by_device = {
                str(device_id).upper(): cls._resolve_existing_ctx_path(path, output_ctx_name)
                for device_id, path in compiled_model_paths.items()
            }
            selected_device_ids = target_device_ids or list(compiled_by_device)
            missing_device_ids = [device_id for device_id in selected_device_ids if device_id not in compiled_by_device]
            if missing_device_ids:
                raise ValueError(
                    "target_devices contains device ids without compiled_model_paths entries: "
                    f"{missing_device_ids}. Available: {list(compiled_by_device)}"
                )
            return [
                {"device_id_hex": device_id, "compiled_model_path": compiled_by_device[device_id]}
                for device_id in selected_device_ids
            ]

        if len(target_device_ids) != 1:
            raise ValueError(
                "package_only without compiled_model_paths requires exactly one target_devices entry so the existing"
                " EPContext model can be named as a package variant."
            )
        return [
            {
                "device_id_hex": target_device_ids[0],
                "compiled_model_path": cls._resolve_existing_ctx_path(model_path, output_ctx_name),
            }
        ]

    @staticmethod
    def _resolve_existing_ctx_path(path: Union[str, Path], output_ctx_name: str) -> Path:
        ctx_path = Path(path)
        if ctx_path.is_dir():
            ctx_path = ctx_path / output_ctx_name
        if not ctx_path.is_file():
            raise FileNotFoundError(f"Compiled EPContext model file not found: {ctx_path}")
        return ctx_path

    @staticmethod
    def _compile_variant(config, model_path: Path, output_ctx: Path, target: dict) -> None:
        from olive.passes.onnx.context_binary import EPContextBinaryGenerator

        # npu_platform pins the exact virtual device; fall back to device_id for physical devices.
        platform = target.get("npu_platform")
        ep_device_filters = {"npu_platform": platform} if platform else {"device_id": target["device_id_hex"]}

        EPContextBinaryGenerator._generate_plugin_ep_context_binary(
            ep_package_name=_OPENVINO_EP_PLUGIN_PACKAGE,
            model_path=str(model_path),
            output_model_path=output_ctx,
            compile_flow=config.compile_flow,
            provider_options=config.provider_options,
            session_options=dict(config.compile_options or {}),
            ep_device_filters=ep_device_filters,
            ep_registration_name=config.ep_registration_name,
        )

    @classmethod
    def _copy_existing_variant(cls, source_ctx: Path, output_ctx: Path) -> None:
        cls._copy_preserving_location(source_ctx, output_ctx)
        for rel, src in cls._compiled_context_auxiliary_files(source_ctx):
            cls._copy_preserving_location(src, output_ctx.parent / rel)

    @classmethod
    def _compiled_context_auxiliary_files(cls, ctx_model_path: Path) -> list[tuple[str, Path]]:
        # .bin sidecars keyed by basename; external-data files keep their ONNX-recorded relative location.
        files: dict[str, Path] = {p.name: p for p in ctx_model_path.parent.glob(f"{ctx_model_path.stem}*.bin")}
        for location, src in cls._external_data_files(ctx_model_path):
            cls._add_asset_entry(files, location, src)
        return list(files.items())

    @staticmethod
    def _read_compat(ctx_model_path: Path) -> str:
        import onnx

        model = onnx.load(str(ctx_model_path), load_external_data=False)
        for prop in model.metadata_props:
            if prop.key == _COMPAT_METADATA_KEY:
                return prop.value
        return ""

    @classmethod
    def _add_shared_asset(cls, model_path: Path, package_dir: Path) -> str:
        # Keep the model filename and each external-data file's recorded relative location.
        entries: dict[str, Path] = {model_path.name: model_path}
        for location, src in cls._external_data_files(model_path):
            cls._add_asset_entry(entries, location, src)
        digest = cls._compute_directory_hash(entries)
        asset_dir = package_dir / "shared_assets" / f"sha256-{digest}"
        for rel, src in entries.items():
            cls._copy_preserving_location(src, asset_dir / rel)
        return f"sha256:{digest}"

    @staticmethod
    def _external_data_files(model_path: Path) -> list[tuple[str, Path]]:
        """Return (recorded_relative_location, resolved_source_path) for each external-data file."""
        import onnx

        model = onnx.load(str(model_path), load_external_data=False)
        seen: set[str] = set()
        files: list[tuple[str, Path]] = []
        for init in model.graph.initializer:
            if init.data_location != onnx.TensorProto.EXTERNAL:
                continue
            for entry in init.external_data:
                if entry.key != "location" or entry.value in seen:
                    continue
                seen.add(entry.value)
                src = model_path.parent / entry.value
                if not src.is_file():
                    raise FileNotFoundError(f"External-data file referenced by {model_path} not found: {src}")
                files.append((entry.value, src))
        return files

    @staticmethod
    def _add_asset_entry(entries: dict[str, Path], location: str, src: Path) -> None:
        """Add {location: src} to entries, rejecting distinct files that collide on the same location."""
        existing = entries.get(location)
        if existing is not None and existing.resolve() != src.resolve():
            raise ValueError(f"Two distinct files map to the same package location '{location}': {existing}, {src}")
        entries[location] = src

    @staticmethod
    def _copy_preserving_location(src: Path, dest: Path) -> None:
        if src.resolve() == dest.resolve():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))

    @staticmethod
    def _compute_directory_hash(entries: dict[str, Path]) -> str:
        def sha256_file(path: Path) -> str:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        text = "".join(f"{sha256_file(entries[rel])}  {rel}\n" for rel in sorted(entries))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _write_manifest(
        package_dir: Path,
        package_name: str,
        package_version: str,
        device: str,
        variants: list[dict],
        output_ctx_name: str,
        executor_namespace: str,
        session_options: dict,
        external_initializers_option: str,
        shared_asset_uri: Optional[str],
    ) -> None:
        variant_bodies: dict[str, dict] = {}
        for v in variants:
            variant_body: dict = {
                "variant_directory": f"{package_name}/{v['variant_name']}",
                "ep": _EP_NAME,
                "device": device,
            }
            if v["compat_string"]:
                variant_body["compatibility_string"] = v["compat_string"]

            ort_info: dict = {"model_file": output_ctx_name}
            variant_session_options = dict(session_options)
            if shared_asset_uri:
                variant_session_options[external_initializers_option] = shared_asset_uri
            if variant_session_options:
                ort_info["session_options"] = variant_session_options
            variant_body["executor_info"] = {executor_namespace: ort_info}
            variant_bodies[v["variant_name"]] = variant_body

        manifest: dict = {
            "schema_version": "1.0",
            "components": {package_name: {"variants": variant_bodies}},
            "package_name": package_name,
            "package_version": package_version,
        }
        if shared_asset_uri is not None:
            manifest["shared_assets"] = {}

        with (package_dir / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
