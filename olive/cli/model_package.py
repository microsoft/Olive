# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""``olive generate-model-package`` CLI command.

Assemble one or more Olive output directories into a proposal-shaped ORT
model package.

Each ``--source`` directory is one Olive output (an ``ONNXModel`` or a
``CompositeModel`` with ONNX components). Single-source packages are
allowed: a single variant under one component is a normal, valid package.

Output layout (per the ORT model-package proposal)::

    <output>/
    ├── manifest.json
    ├── configs/
    │   └── <consumer-shared assets>           # tokenizer, genai_config, ...
    └── models/
        └── <component>/
            ├── metadata.json
            └── <variant>/
                ├── genai_config_overlay.json      # optional: per-variant runtime fields
                ├── model.onnx
                └── ...                            # external-data blobs (inline)

Notes:
- ``metadata.json`` is selection-only. Each variant declares a single
  execution provider inline (``ep``) plus optional ``device`` and opaque
  ``compatibility_string``.
- Each variant directory is self-contained: the ONNX file and any external-data
  blobs it references are copied inline so stock ORT can load it directly.
- ``genai_config.json`` is canonicalized into ``<output>/configs/``: variant-
  specific runtime fields (``filename``, ``session_options``) are stripped from
  the base and each role gets a ``component`` pointer so ORT GenAI can map
  roles to ``models/<component>/`` at load time. The stripped fields are
  re-injected per variant as a ``genai_config_overlay.json`` (an RFC 7386 JSON
  Merge Patch applied on top of ``configs/genai_config.json``).

"""

import hashlib
import json
import logging
import re
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from olive.cli.base import (
    BaseOliveCLICommand,
    add_logging_options,
    add_telemetry_options,
)
from olive.telemetry import action

logger = logging.getLogger(__name__)

# Files inside an Olive output dir that always belong next to the ONNX model
# rather than under <package>/configs/.
_MODEL_SUFFIXES = {".onnx", ".bin", ".data", ".xml"}

# Schema versions emitted in the package JSON files. Keep in sync with the
# ORT model-package schema.
_MANIFEST_SCHEMA_VERSION = 1
_METADATA_SCHEMA_VERSION = 1

# Directory under the package root that holds consumer-shared config assets
# (genai_config base, tokenizer, processor configs, chat templates).
_CONFIGS_DIR = "configs"

# Directory under the package root that holds per-component subdirectories.
# Required by the ORT model-package schema; ORT's model-package loader
# discovers components via ``<package>/models/<component>/metadata.json``.
_MODELS_DIR = "models"

# Conventional directory suffix for an ORT model package. Not enforced by
# ORT/ORT-GenAI loaders (they probe structure, not filenames), but matches
# the canonical naming used in ORT's model-package documentation and the
# reference ``build_packages.py`` examples.
_PACKAGE_SUFFIX = ".ortpackage"

# Map canonical ONNX Runtime EP names to the short provider aliases used inside
# genai_config.json's ``session_options.provider_options`` list. Matches the
# accepted aliases reported by ORT-GenAI when it parses an unknown provider name
# ("Currently supported values are 'DML'/'DmlExecutionProvider', ...").
_EP_TO_GENAI: dict[str, str] = {
    "CPUExecutionProvider": "CPU",
    "CUDAExecutionProvider": "cuda",
    "DmlExecutionProvider": "DML",
    "WebGpuExecutionProvider": "WebGPU",
    "JsExecutionProvider": "JS",
    "QNNExecutionProvider": "qnn",
    "OpenVINOExecutionProvider": "OpenVINO",
    "ROCMExecutionProvider": "rocm",
    "TensorrtExecutionProvider": "tensorrt",
    "NvTensorRTRTXExecutionProvider": "NvTensorRtRtx",
    "XnnpackExecutionProvider": "XNNPACK",
    "WebNNExecutionProvider": "WEBNN",
    "AzureExecutionProvider": "AZURE",
    "VitisAIExecutionProvider": "VitisAI",
    "CoreMLExecutionProvider": "CoreML",
    "MIGraphXExecutionProvider": "MIGraphX",
    "SNPEExecutionProvider": "SNPE",
}

# Hash chunk size for SHA-256 over external-data blobs.
_HASH_CHUNK = 1024 * 1024

# Disallow path separators / traversal in component and variant names so a
# producer can't write files outside the package directory.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-][A-Za-z0-9._\- ]*$")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class ModelPackageCommand(BaseOliveCLICommand):
    """Merge one or more Olive output directories into a model package."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "generate-model-package",
            help="Merge one or more Olive output directories into a model package",
        )

        sub_parser.add_argument(
            "-s",
            "--source",
            type=str,
            action="append",
            required=True,
            help=(
                "Source Olive output directory. Repeat to add multiple variants. "
                "A single source is allowed (single-variant package)."
            ),
        )

        sub_parser.add_argument(
            "-o",
            "--output_path",
            type=str,
            required=True,
            help=(
                "Output directory for the model package. The ``.ortpackage`` "
                "suffix is appended automatically if missing. Must be empty "
                "or non-existent."
            ),
        )

        sub_parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="Optional model name recorded under manifest.producer.",
        )

        sub_parser.add_argument(
            "--model_version",
            type=str,
            default="1.0",
            help="Optional model version recorded under manifest.producer. Default: 1.0",
        )

        add_logging_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=ModelPackageCommand)

    @action
    def run(self):
        sources = self._parse_sources()
        output_dir = Path(self.args.output_path)
        if output_dir.suffix != _PACKAGE_SUFFIX:
            output_dir = output_dir.with_name(output_dir.name + _PACKAGE_SUFFIX)
        package_default_name = output_dir.stem

        targets = []
        for target_name, source_path in sources:
            model_config = self._read_model_config(source_path)
            targets.append((target_name, source_path, model_config))

        types = {(targets[i][2].get("type") or "").lower() for i in range(len(targets))}
        supported = {"onnxmodel", "compositemodel"}
        if types - supported:
            unsupported = sorted(types - supported)
            raise ValueError(
                f"Unsupported source model type(s) {unsupported!r}. "
                "generate-model-package supports ONNXModel and CompositeModel only."
            )
        if len(types) > 1:
            raise ValueError(
                f"Sources mix model types {sorted(types)!r}. All sources must share the same type "
                "(all ONNXModel or all CompositeModel)."
            )
        is_composite = next(iter(types)) == "compositemodel"

        if is_composite:
            variants = self._build_composite_variants(targets)
        else:
            variants = self._build_single_variants(targets)

        config_files = self._collect_config_files(targets)

        task = self._extract_task(targets)
        producer_info: dict[str, str] = {"tool": "olive-ai"}
        try:
            from olive import __version__ as _olive_version

            producer_info["tool_version"] = _olive_version
        except Exception:
            logger.debug("Could not read olive.__version__", exc_info=True)
        producer_info["model_name"] = self.args.model_name or package_default_name
        producer_info["model_version"] = self.args.model_version
        if task:
            producer_info["task"] = task

        write_model_package(
            output_dir=output_dir,
            variants=variants,
            config_files=config_files,
            producer_info=producer_info,
            package_name=self.args.model_name or package_default_name,
            package_version=self.args.model_version,
        )

        logger.info("Model package generated at %s", output_dir)
        print(f"Model package generated at {output_dir}")

    # ------------------------------------------------------------------
    # VariantSpec construction
    # ------------------------------------------------------------------

    def _build_single_variants(self, targets: list[tuple[str, Path, dict]]) -> list["VariantSpec"]:
        task = self._extract_task(targets)
        component_name = _task_to_component_name(task)
        variants: list[VariantSpec] = []
        for target_name, _src, model_config in targets:
            attrs = _get_model_attributes(model_config)
            onnx_path = _resolve_onnx_path(model_config)
            ep, device, compatibility_string = _ep_device_compatibility(attrs, onnx_path, target_name)
            variants.append(
                VariantSpec(
                    component_name=component_name,
                    variant_name=target_name,
                    onnx_files=[onnx_path],
                    ep=ep,
                    device=device,
                    compatibility_string=compatibility_string,
                    inference_settings=model_config.get("config", {}).get("inference_settings") or {},
                )
            )
        return variants

    def _build_composite_variants(self, targets: list[tuple[str, Path, dict]]) -> list["VariantSpec"]:
        from collections import OrderedDict

        # Track per-component variants in source insertion order.
        component_variants: dict[str, list[VariantSpec]] = OrderedDict()

        for target_name, _src, model_config in targets:
            target_attrs = _get_model_attributes(model_config)
            target_inference = model_config.get("config", {}).get("inference_settings") or {}
            components = model_config["config"].get("model_components", [])
            component_names = model_config["config"].get("model_component_names", [])

            if not components:
                raise ValueError(f"Composite source {target_name!r} declares no model_components.")
            if len(components) != len(component_names):
                raise ValueError(
                    f"Composite source {target_name!r} has {len(components)} model_components but "
                    f"{len(component_names)} model_component_names; counts must match."
                )

            for comp_config, comp_name in zip(components, component_names):
                # Component-level inference_settings overrides target-level if present.
                comp_inference = comp_config.get("config", {}).get("inference_settings") or target_inference
                # Component-level model_attributes overlay target-level.
                comp_attrs = dict(target_attrs)
                comp_attrs.update(_get_model_attributes(comp_config))

                onnx_path = _resolve_onnx_path(comp_config)
                ep, device, compatibility_string = _ep_device_compatibility(comp_attrs, onnx_path, target_name)

                spec = VariantSpec(
                    component_name=comp_name,
                    variant_name=target_name,
                    onnx_files=[onnx_path],
                    ep=ep,
                    device=device,
                    compatibility_string=compatibility_string,
                    inference_settings=comp_inference,
                )
                component_variants.setdefault(comp_name, []).append(spec)

        flat: list[VariantSpec] = []
        for comp_specs in component_variants.values():
            flat.extend(comp_specs)
        return flat

    # ------------------------------------------------------------------
    # Config file handling
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_config_files(targets: list[tuple[str, Path, dict]]) -> dict[str, Path]:
        """Pick consumer-shared config files (genai_config, tokenizer, ...).

        Source-of-truth order:
        1. ``model_attributes.additional_files`` of any source that has it.
        2. Otherwise, the first source's non-model files.
        """
        config_entries: dict[str, Path] = {}

        for _target_name, _source_path, model_config in targets:
            attrs = _get_model_attributes(model_config)
            for fp in attrs.get("additional_files", []):
                p = Path(fp)
                if (p.is_file() or p.is_dir()) and p.name not in config_entries:
                    config_entries[p.name] = p
            if config_entries:
                break

        if not config_entries:
            for _target_name, source_path, _model_config in targets:
                for f in sorted(source_path.iterdir()):
                    if f.name == "model_config.json":
                        continue
                    if (f.is_file() and f.suffix not in _MODEL_SUFFIXES) or f.is_dir():
                        config_entries[f.name] = f
                if config_entries:
                    break

        return config_entries

    # ------------------------------------------------------------------
    # Source validation / reading
    # ------------------------------------------------------------------

    def _parse_sources(self) -> list[tuple[str, Path]]:
        sources: list[tuple[str, Path]] = []
        seen_names: set[str] = set()
        for source in self.args.source:
            path = Path(source)
            if not path.is_dir():
                raise ValueError(f"Source path does not exist or is not a directory: {path}")
            if not (path / "model_config.json").exists():
                raise ValueError(
                    f"No model_config.json found in {path}. "
                    "Source must be an Olive output directory with model_config.json."
                )
            name = path.name
            if name in seen_names:
                raise ValueError(
                    f"Two sources share the directory name {name!r}. Variant names are derived from "
                    "the source directory name; please rename so each source is unique."
                )
            seen_names.add(name)
            sources.append((name, path))
        if not sources:
            raise ValueError("At least one --source directory is required.")
        return sources

    @staticmethod
    def _read_model_config(source_path: Path) -> dict:
        with (source_path / "model_config.json").open() as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Task extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_task(targets: list[tuple[str, Path, dict]]) -> str:
        model_name_or_path = ""
        for _target_name, _source_path, model_config in targets:
            attrs = _get_model_attributes(model_config)
            model_name_or_path = attrs.get("_name_or_path", "")
            if model_name_or_path:
                break

        if not model_name_or_path:
            return ""

        try:
            from huggingface_hub import model_info

            info = model_info(model_name_or_path)
            tag = info.pipeline_tag or ""
            return tag.replace("-", "_")
        except Exception:
            logger.debug("Could not fetch task from HuggingFace Hub for %s", model_name_or_path, exc_info=True)
            return ""


# ---------------------------------------------------------------------------
# Writer (CLI-private; kept here because only this command produces packages)
# ---------------------------------------------------------------------------


@dataclass
class VariantSpec:
    """One variant of one component, ready to be packaged."""

    component_name: str
    variant_name: str
    onnx_files: list[Path]
    ep: str
    device: Optional[str] = None
    compatibility_string: Optional[str] = None
    inference_settings: dict[str, Any] = field(default_factory=dict)
    consumer_metadata: Optional[dict[str, Any]] = None


def write_model_package(
    output_dir: Path,
    variants: list[VariantSpec],
    config_files: Optional[dict[str, Path]] = None,
    producer_info: Optional[dict[str, Any]] = None,
    package_name: Optional[str] = None,
    package_version: str = "1.0",
) -> None:
    """Materialize a model package on disk.

    :param output_dir: Target directory. Must be empty (or non-existent) so a
        partial overwrite cannot mix the new layout with stale files from a
        previous run.
    :param variants: Ordered list of variants. Component insertion order is
        the order each component first appears in this list.
    :param config_files: Map from filename (basename) to source path; copied
        into ``<output_dir>/configs/``. Same-named files contributed by
        different sources should be byte-identical; the first wins on
        conflict and a warning is logged.
    :param producer_info: Olive-specific provenance recorded under
        ``manifest.producer``. Schema-tolerated extra field; producers may add
        namespaced extras.
    :param package_name: Name recorded under ``manifest.package_name``.
        Defaults to the output directory name.
    :param package_version: Version recorded under ``manifest.package_version``.
    """
    if not variants:
        raise ValueError("write_model_package requires at least one variant.")

    output_dir = Path(output_dir)
    _ensure_empty_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map each package component to the genai_config role that references it, so
    # per-variant overlays patch the right ``model.<role>`` block. Roles can be
    # named differently from components, so we resolve via the base config's
    # ``model.<role>.component`` pointers and fall back to the component name.
    component_to_role = _resolve_component_roles(config_files)

    # Group by component while preserving insertion order.
    components: dict[str, list[VariantSpec]] = {}
    for v in variants:
        _validate_name(v.component_name, "component")
        _validate_name(v.variant_name, "variant")
        components.setdefault(v.component_name, []).append(v)

    # Per component, fail fast on duplicate variant names. The caller is
    # expected to disambiguate (e.g. with a rank suffix) before calling.
    for comp_name, comp_variants in components.items():
        seen: set[str] = set()
        for v in comp_variants:
            if v.variant_name in seen:
                raise ValueError(
                    f"Duplicate variant name '{v.variant_name}' under component "
                    f"'{comp_name}'. Variant names must be unique per component."
                )
            seen.add(v.variant_name)

    for comp_name, comp_variants in components.items():
        _write_component(output_dir, comp_name, comp_variants, component_to_role.get(comp_name, comp_name))

    # Build the role -> component map needed by _copy_config_files so it can
    # inject ``model.<role>.component`` markers into the base genai_config. ORT
    # requires every role-block to declare which package component it loads;
    # without those markers ORT-GenAI's variant auto-selection fails with
    # "the genai config does not reference any package components".
    role_to_component: dict[str, str] = {}
    for comp_name in components:
        role = component_to_role.get(comp_name, comp_name)
        role_to_component.setdefault(role, comp_name)

    if config_files:
        _copy_config_files(output_dir, config_files, role_to_component)

    _write_manifest(
        output_dir, list(components.keys()), producer_info, package_name or output_dir.name, package_version
    )


def _write_component(
    output_dir: Path,
    component_name: str,
    comp_variants: list[VariantSpec],
    component_role: str,
) -> None:
    component_dir = output_dir / _MODELS_DIR / component_name
    component_dir.mkdir(parents=True, exist_ok=True)

    # Copy each variant's ONNX file(s) along with any external-data blobs they
    # reference, keeping everything inline in the variant directory so each
    # variant is self-contained and loadable by stock ORT.
    for v in comp_variants:
        if not v.onnx_files:
            raise ValueError(f"Variant '{v.variant_name}' under component '{component_name}' has no ONNX files.")

        variant_dir = component_dir / v.variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        source_dirs: set[Path] = set()
        for onnx_src in v.onnx_files:
            onnx_src_path = Path(onnx_src)
            if not onnx_src_path.is_file():
                raise FileNotFoundError(f"ONNX file not found: {onnx_src_path}")

            onnx_dst = variant_dir / onnx_src_path.name
            shutil.copy2(str(onnx_src_path), str(onnx_dst))
            source_dirs.add(onnx_src_path.parent.resolve())

            ext_refs = _discover_external_data(onnx_src_path)
            external_root = onnx_src_path.parent.resolve()
            for graph_location in ext_refs:
                blob_src = (onnx_src_path.parent / graph_location).resolve()
                if not blob_src.is_relative_to(external_root):
                    logger.warning(
                        "External-data file referenced by %s resolves outside its source directory "
                        "(symlink escape?); skipping: %s",
                        onnx_src_path,
                        blob_src,
                    )
                    continue
                if not blob_src.is_file():
                    logger.warning(
                        "External-data file referenced by %s but missing: %s",
                        onnx_src_path,
                        blob_src,
                    )
                    continue

                blob_dst = variant_dir / graph_location
                blob_dst.parent.mkdir(parents=True, exist_ok=True)
                if not blob_dst.exists():
                    shutil.copy2(str(blob_src), str(blob_dst))

        # Sweep each source directory for remaining model-suffix sidecar files
        # (e.g. an EPContext stub ``.onnx`` typically points at a same-stem
        # ``.xml``/``.bin`` pair for OpenVINO or a ``.bin`` context blob for
        # QNN; these sidecars don't appear in the ONNX initializer
        # ``external_data`` table so the standard external-data copy above
        # misses them). Each Olive source directory holds the artifacts for a
        # single variant, so any file with a model suffix is part of this
        # variant and belongs next to the ONNX. Duplicates already copied as
        # external-data are skipped.
        for src_dir in sorted(source_dirs):
            for entry in sorted(src_dir.iterdir()):
                if not entry.is_file() or entry.suffix not in _MODEL_SUFFIXES:
                    continue
                dst = variant_dir / entry.name
                if dst.exists():
                    continue
                shutil.copy2(str(entry), str(dst))

        # Per-variant runtime fields flow through genai_config_overlay.json.
        _write_genai_config_overlay(variant_dir, component_role, v)

    _write_metadata(component_dir, component_name, comp_variants)


def _write_metadata(component_dir: Path, component_name: str, comp_variants: list[VariantSpec]) -> None:
    variants_payload: dict[str, Any] = {}
    for v in comp_variants:
        # EP fields are inline on the variant object; a variant targets a
        # single execution provider.
        variant_obj: dict[str, Any] = {"ep": v.ep}
        if v.device:
            variant_obj["device"] = v.device
        if v.compatibility_string:
            variant_obj["compatibility_string"] = v.compatibility_string
        variants_payload[v.variant_name] = variant_obj
    _write_json(
        component_dir / "metadata.json",
        {
            "schema_version": _METADATA_SCHEMA_VERSION,
            "component_name": component_name,
            "variants": variants_payload,
        },
    )


def _genai_provider_name(ep: str) -> str:
    """Map a canonical ORT EP name to the genai_config provider alias."""
    if ep in _EP_TO_GENAI:
        return _EP_TO_GENAI[ep]
    # Best-effort fallback: strip the ExecutionProvider suffix and lowercase.
    return ep[: -len("ExecutionProvider")].lower() if ep.endswith("ExecutionProvider") else ep


def _write_genai_config_overlay(variant_dir: Path, component_role: str, v: VariantSpec) -> None:
    """Emit a per-variant ``genai_config_overlay.json`` (RFC 7386 merge patch).

    Per-variant runtime fields flow through a JSON Merge Patch applied on top of
    the package's base ``configs/genai_config.json``. We express the variant's
    ``filename`` (the variant-local ONNX file basename), ``session_options`` and
    EP-scoped ``provider_options`` under the role that references this component
    (``model.<role>``). The base config has those keys stripped (see
    ``_strip_variant_specific``); each variant overlay puts them back so ORT
    resolves files inside the chosen variant directory.
    """
    inference = v.inference_settings or {}
    session_options: dict[str, Any] = dict(inference.get("session_options") or {})
    provider_options = _provider_options_for_ep(inference, v.ep)
    genai_ep = _genai_provider_name(v.ep)

    # ORT-GenAI's FinalizeConfig builds session_options.providers from
    # provider_options[*].name (src/config.cpp:1643-1645), and
    # SetProviderSessionOptions then registers each named provider. CPU is not
    # in the dispatch table (src/models/session_options.cpp:150-159); it has no
    # configurable options, and ORT InferenceSession adds it implicitly when no
    # other EP is registered (onnxruntime/core/session/inference_session.cc:
    # SetCpuProviderWasImplicitlyAdded). For CPU variants we therefore emit an
    # empty list rather than a sentinel ``[{"CPU": {}}]`` entry. For every
    # other EP we name it explicitly (NormalizeProviderName canonicalises the
    # case for QNN/DML/OpenVINO/etc., and "cuda" is already lowercase in the
    # dispatch table). This matches the convention used by reference ORT model
    # packages and avoids registering CPU through the V1 no-op path.
    if genai_ep == "CPU":
        session_options["provider_options"] = []
    else:
        session_options["provider_options"] = [{genai_ep: provider_options}]

    role_patch: dict[str, Any] = {"session_options": session_options}
    if v.onnx_files:
        # The base config strips ``filename`` (it was a variant-specific path
        # like ``decoder/model.onnx``); the loader resolves the variant ONNX as
        # ``<variant_dir>/<filename>``, so emit the basename here.
        role_patch["filename"] = Path(v.onnx_files[0]).name

    overlay = {"model": {component_role: role_patch}}
    _write_json(variant_dir / "genai_config_overlay.json", overlay)


def _strip_variant_specific(node: Any, keys: tuple[str, ...] = ("filename", "session_options")) -> Any:
    """Recursively drop variant-specific keys from a genai_config-shaped dict.

    ``filename`` and ``session_options`` are intrinsically variant-specific and
    must not live in the package's base ``configs/genai_config.json``; per-variant
    ``genai_config_overlay.json`` files patch them back in. Returns a deep copy.
    """
    if isinstance(node, dict):
        return {k: _strip_variant_specific(v, keys) for k, v in node.items() if k not in keys}
    if isinstance(node, list):
        return [_strip_variant_specific(v, keys) for v in node]
    return node


def _resolve_component_roles(config_files: Optional[dict[str, Path]]) -> dict[str, str]:
    """Map each package component to the genai_config role that references it.

    The base ``genai_config.json`` declares roles under ``model.<role>``. The
    role name and component directory name are not always the same (e.g.
    Mobius emits role ``vision`` for a component dir named ``vision_encoder``),
    so per-variant overlays need a role lookup. We try two signals in order:

    1. ``model.<role>.component`` (explicit pointer, ORT spec).
    2. The first path segment of ``model.<role>.filename`` — Mobius and other
       flat-dir producers write paths like ``vision_encoder/model.onnx`` so the
       directory naturally names the component.

    Returns an empty map when no base config is available; callers fall back
    to the component name as the role.
    """
    if not config_files:
        return {}
    src = config_files.get("genai_config.json")
    if src is None:
        return {}
    try:
        with Path(src).open(encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception:
        logger.debug("Could not read genai_config.json from %s for role mapping.", src, exc_info=True)
        return {}

    model_block = config.get("model")
    if not isinstance(model_block, dict):
        return {}

    component_to_role: dict[str, str] = {}
    for role, role_block in model_block.items():
        if not isinstance(role_block, dict):
            continue
        component = role_block.get("component")
        if not (isinstance(component, str) and component):
            filename = role_block.get("filename")
            if isinstance(filename, str) and filename:
                parts = Path(filename).parts
                if len(parts) >= 2:
                    component = parts[0]
        if isinstance(component, str) and component and component not in component_to_role:
            component_to_role[component] = role
    return component_to_role


def _write_manifest(
    output_dir: Path,
    components: list[str],
    producer_info: Optional[dict[str, Any]],
    package_name: str,
    package_version: str,
) -> None:
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "package_name": package_name,
        "package_version": package_version,
        "components": components,
        "configs_dir": _CONFIGS_DIR,
    }
    if producer_info:
        # Olive-specific provenance under a namespaced key so future schema
        # evolution can't collide with it.
        manifest["producer"] = producer_info
    _write_json(output_dir / "manifest.json", manifest)


# ---------------------------------------------------------------------------
# configs/ handling
# ---------------------------------------------------------------------------


def _copy_config_files(
    output_dir: Path,
    config_files: dict[str, Path],
    role_to_component: Optional[dict[str, str]] = None,
) -> None:
    configs_dir = output_dir / _CONFIGS_DIR
    configs_dir.mkdir(parents=True, exist_ok=True)
    configs_root = configs_dir.resolve()
    for name, src in config_files.items():
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            logger.warning("Skipping config file with unsafe name %r.", name)
            continue
        src_path = Path(src)
        dest = configs_dir / name
        # Belt-and-suspenders: even with the name check above, refuse a dest
        # that doesn't land directly under configs/.
        if dest.resolve().parent != configs_root:
            logger.warning("Skipping config file %r: resolved path escapes configs/.", name)
            continue
        if dest.exists():
            if not _paths_equal(src_path, dest):
                logger.warning(
                    "configs/%s already present and differs from %s; keeping the existing copy. "
                    "Per-variant config differences belong in genai_config_overlay.json, "
                    "not in the shared configs/ directory.",
                    name,
                    src_path,
                )
            continue
        if name == "genai_config.json" and src_path.is_file():
            # Strip variant-specific keys from the base genai_config and inject
            # ``model.<role>.component`` markers so ORT-GenAI can resolve each
            # role to a package component (and apply the right per-variant
            # overlay). Each variant's genai_config_overlay.json patches the
            # stripped keys back in.
            try:
                with src_path.open(encoding="utf-8") as fh:
                    base_genai = json.load(fh)
                stripped = _strip_variant_specific(base_genai)
                if role_to_component:
                    _inject_role_components(stripped, role_to_component)
                _write_json(dest, stripped)
                continue
            except Exception:
                logger.debug(
                    "Failed to strip variant-specific keys from %s; falling back to verbatim copy.",
                    src_path,
                    exc_info=True,
                )
        if src_path.is_dir():
            shutil.copytree(str(src_path), str(dest))
        elif src_path.is_file():
            shutil.copy2(str(src_path), str(dest))
        else:
            logger.warning("Config source %s does not exist; skipping.", src_path)


def _inject_role_components(genai: dict, role_to_component: dict[str, str]) -> None:
    """Inject ``model.<role>.component = <component>`` markers in-place.

    ORT-GenAI's model-package variant selection requires every role block in
    the base ``configs/genai_config.json`` to declare which package component
    serves it. Olive-generated source ``genai_config.json`` typically lacks
    these markers because the source is a flat-directory build, not a package.
    """
    model_block = genai.get("model")
    if not isinstance(model_block, dict):
        return
    for role, component in role_to_component.items():
        role_block = model_block.get(role)
        if isinstance(role_block, dict):
            role_block["component"] = component


def _paths_equal(a: Path, b: Path) -> bool:
    """Return True if a and b have identical content (file or directory)."""
    if a.is_file() and b.is_file():
        if a.stat().st_size != b.stat().st_size:
            return False
        return _sha256_file(a) == _sha256_file(b)
    if a.is_dir() and b.is_dir():
        a_entries = sorted(p.name for p in a.iterdir())
        b_entries = sorted(p.name for p in b.iterdir())
        if a_entries != b_entries:
            return False
        return all(_paths_equal(a / name, b / name) for name in a_entries)
    return False


# ---------------------------------------------------------------------------
# ONNX external-data discovery
# ---------------------------------------------------------------------------


def _discover_external_data(onnx_path: Path) -> list[str]:
    """Return the relative ``location`` strings of every external-data blob.

    Locations are validated as safe relative paths (no absolute paths, no
    upward traversal). Unsafe references are dropped with a warning rather
    than failing — better to package a slightly broken model than to refuse
    progress on something the user can fix downstream.
    """
    try:
        import onnx
    except ImportError:
        logger.warning("onnx package not available; external-data discovery skipped.")
        return []

    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
    except Exception:
        logger.debug("Failed to parse %s; skipping external-data discovery.", onnx_path, exc_info=True)
        return []

    locations: list[str] = []
    seen: set[str] = set()
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in init.external_data:
            if entry.key != "location":
                continue
            location = entry.value
            if not _is_safe_relative_location(location):
                logger.warning(
                    "Skipping unsafe external-data location %r in %s.",
                    location,
                    onnx_path,
                )
                continue
            if location not in seen:
                locations.append(location)
                seen.add(location)
    return locations


def _is_safe_relative_location(location: str) -> bool:
    if not location:
        return False
    p = Path(location)
    if p.is_absolute():
        return False
    parts = p.parts
    if any(part in ("..", "") for part in parts):
        return False
    # Reject Windows-drive style paths that slip through is_absolute on POSIX.
    return not (len(location) >= 2 and location[1] == ":")


# ---------------------------------------------------------------------------
# Helpers (module-level so tests can exercise them directly)
# ---------------------------------------------------------------------------


def _provider_options_for_ep(inference_settings: dict[str, Any], ep: str) -> dict[str, Any]:
    """Return the provider_options dict that matches ``ep`` by name.

    Olive's inference_settings has ``execution_provider`` (list of EP names)
    and ``provider_options`` (parallel list). Match by EP name; do not rely on
    positional indexing.
    """
    eps = inference_settings.get("execution_provider") or []
    pos = inference_settings.get("provider_options") or []
    for name, opts in zip(eps, pos):
        if name == ep:
            return opts or {}
    return {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _validate_name(name: str, kind: str) -> None:
    if not name or not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid {kind} name {name!r}: must be non-empty and contain only "
            "alphanumerics, dot, underscore, hyphen, and space."
        )
    if name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"Invalid {kind} name {name!r}: path separators and traversal are not allowed.")


def _ensure_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path {output_dir} exists and is not a directory.")
        if any(output_dir.iterdir()):
            raise ValueError(
                f"Output directory {output_dir} is not empty. Refusing to mix stale files with a new "
                "package; please point at an empty (or non-existent) directory."
            )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    logger.info("Wrote %s", path)


def parse_compatibility_strings(raw: Optional[str]) -> list[str]:
    """Split Olive's ``ep_compatibility_info.<EP>`` ONNX metadata string.

    Producers store comma-delimited lists today (e.g. ``"sm_80,sm_86,sm_90"``);
    the proposal expects a JSON list of opaque strings. Splitting here keeps
    consumers from having to know Olive's convention.
    """
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def disambiguate_variant_names(candidates: list[tuple[str, str]]) -> list[str]:
    """Return per-candidate variant names with rank suffixes on collision.

    ``candidates`` is a list of ``(component_name, base_variant_name)``
    tuples; the function returns a parallel list of disambiguated variant
    names (suffixing ``_rank{N}`` deterministically when two candidates land
    on the same ``(component, base_variant)``).
    """
    counts: dict[tuple[str, str], int] = {}
    for key in candidates:
        counts[key] = counts.get(key, 0) + 1

    used: dict[tuple[str, str], int] = {}
    result: list[str] = []
    for comp, base in candidates:
        if counts[(comp, base)] == 1:
            result.append(base)
            continue
        used[(comp, base)] = used.get((comp, base), 0) + 1
        result.append(f"{base}_rank{used[(comp, base)]}")
    return result


# ---------------------------------------------------------------------------
# Olive model-config helpers
# ---------------------------------------------------------------------------


def _get_model_attributes(model_config: dict) -> dict:
    return model_config.get("config", {}).get("model_attributes") or {}


def _resolve_onnx_path(model_config: dict) -> Path:
    """Resolve the ONNX file path from an Olive model config.

    The config's ``model_path`` may be either:
    - the ONNX file itself (a ``LocalFile`` resource),
    - a directory containing the ONNX file (a ``LocalFolder`` resource),
      in which case ``onnx_file_name`` (or a single ``.onnx`` in the dir)
      identifies the actual file.
    """
    cfg = model_config.get("config", {}) or {}
    raw = cfg.get("model_path")
    if not raw:
        raise ValueError("Model config has no model_path.")
    p = Path(raw)
    if p.is_file():
        return p
    if p.is_dir():
        onnx_name = cfg.get("onnx_file_name")
        if onnx_name:
            candidate = p / onnx_name
            if candidate.is_file():
                return candidate
        onnx_files = list(p.glob("*.onnx"))
        if len(onnx_files) == 1:
            return onnx_files[0]
        raise ValueError(
            f"Cannot resolve a unique ONNX file under {p}; "
            "set onnx_file_name in the model config or pass the file path directly."
        )
    raise FileNotFoundError(f"model_path does not exist: {p}")


def _ep_device_compatibility(
    attrs: dict, onnx_path: Path, variant_name: Optional[str] = None
) -> tuple[str, Optional[str], Optional[str]]:
    """Extract (ep, device, compatibility_string) for one variant from Olive metadata.

    Each variant declares a single opaque ``compatibility_string``. Olive stores
    the EP-side preference as a comma-delimited string in the ONNX metadata prop
    ``ep_compatibility_info.<EP>``; it is passed through verbatim (ORT does not
    interpret the encoding).

    When ``model_attributes.ep`` is absent, fall back to a common-variant-name
    heuristic (``gpu``/``cuda`` → CUDA, ``qnn`` → QNN, etc.) so users who don't
    manually annotate their Olive outputs still get distinct EP entries in each
    component's metadata.json. Final fallback is CPU.
    """
    ep = attrs.get("ep") or _guess_ep_from_variant_name(variant_name) or "CPUExecutionProvider"
    device = attrs.get("device") or None
    raw = _extract_ep_compatibility_from_onnx(onnx_path, ep)
    compatibility_string = raw.strip() if raw and raw.strip() else None
    return ep, device, compatibility_string


# Best-effort mapping from common Olive output / EP-build directory names to
# canonical ORT EP strings. Used only as a fallback when model_attributes.ep is
# not set. Keep substrings short and lowercased; matched via ``in``.
_VARIANT_NAME_EP_HINTS: tuple[tuple[str, str], ...] = (
    ("cuda", "CUDAExecutionProvider"),
    ("gpu", "CUDAExecutionProvider"),
    ("trt", "TensorrtExecutionProvider"),
    ("tensorrt", "TensorrtExecutionProvider"),
    ("rocm", "ROCMExecutionProvider"),
    ("dml", "DmlExecutionProvider"),
    ("directml", "DmlExecutionProvider"),
    ("qnn", "QNNExecutionProvider"),
    ("npu", "QNNExecutionProvider"),
    ("openvino", "OpenVINOExecutionProvider"),
    ("ovep", "OpenVINOExecutionProvider"),
    ("webgpu", "WebGpuExecutionProvider"),
    ("xnnpack", "XnnpackExecutionProvider"),
    ("coreml", "CoreMLExecutionProvider"),
    ("cpu", "CPUExecutionProvider"),
)


def _guess_ep_from_variant_name(variant_name: Optional[str]) -> Optional[str]:
    if not variant_name:
        return None
    name = variant_name.lower()
    for hint, ep in _VARIANT_NAME_EP_HINTS:
        if hint in name:
            return ep
    return None


def _extract_ep_compatibility_from_onnx(model_path: Path, ep: str = "") -> Optional[str]:
    """Read ``ep_compatibility_info.<EP>`` from the ONNX model's metadata_props."""
    if not model_path.is_file():
        return None
    try:
        import onnx

        onnx_model = onnx.load(str(model_path), load_external_data=False)
        prefix = "ep_compatibility_info."
        ep_compat_map = {
            entry.key[len(prefix) :]: entry.value for entry in onnx_model.metadata_props if entry.key.startswith(prefix)
        }
    except Exception:
        logger.debug("Could not read ONNX metadata from %s", model_path, exc_info=True)
        return None

    if not ep_compat_map:
        return None
    if ep and ep in ep_compat_map:
        return ep_compat_map[ep]
    if len(ep_compat_map) == 1:
        return next(iter(ep_compat_map.values()))
    return None


def _task_to_component_name(task: str) -> str:
    task_component_map = {
        "text_generation": "decoder",
        "text2text_generation": "encoder_decoder",
        "text_classification": "classifier",
        "token_classification": "token_classifier",
        "question_answering": "qa_model",
        "image_generation": "image_generator",
        "image_classification": "image_classifier",
        "object_detection": "object_detector",
        "automatic_speech_recognition": "speech_recognizer",
    }
    return task_component_map.get(task, "model")
