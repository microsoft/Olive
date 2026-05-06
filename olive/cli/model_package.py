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
    └── <component>/
        ├── metadata.json
        ├── shared_weights/
        │   └── <sha256>/<blob>                # opt-in cross-variant dedup
        └── <variant>/
            ├── variant.json
            ├── model.onnx
            └── ...

Notes:
- ``shared_weights`` is opt-in per blob. A blob whose SHA-256 appears in only
  one variant stays inline next to its ONNX file in the variant directory,
  keeping the single-variant case loadable by stock ORT.
- Cross-variant dedup moves a duplicated blob to
  ``<component>/shared_weights/<sha256>/<basename>`` and records the mapping
  in the per-file ``shared_files`` map of the variant's ``variant.json``.
  Loading such a variant requires a model-package-aware consumer.
- ``genai_config.json`` is copied verbatim into ``<output>/configs/``;
  per-variant overlays are ORT-GenAI's responsibility, not Olive's.

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

# Schema version emitted in manifest.json. Keep in sync with the proposal.
_MANIFEST_SCHEMA_VERSION = 1

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
            help="Output directory for the model package. Must be empty or non-existent.",
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

        targets = []
        for target_name, source_path in sources:
            model_config = self._read_model_config(source_path)
            targets.append((target_name, source_path, model_config))

        types = {targets[i][2].get("type") for i in range(len(targets))}
        if types - {"ONNXModel", "CompositeModel"}:
            unsupported = sorted(types - {"ONNXModel", "CompositeModel"})
            raise ValueError(
                f"Unsupported source model type(s) {unsupported!r}. "
                "generate-model-package supports ONNXModel and CompositeModel only."
            )
        if len(types) > 1:
            raise ValueError(
                f"Sources mix model types {sorted(types)!r}. All sources must share the same type "
                "(all ONNXModel or all CompositeModel)."
            )
        is_composite = next(iter(types)) == "CompositeModel"

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
        producer_info["model_name"] = self.args.model_name or output_dir.name
        producer_info["model_version"] = self.args.model_version
        if task:
            producer_info["task"] = task

        write_model_package(
            output_dir=output_dir,
            variants=variants,
            config_files=config_files,
            producer_info=producer_info,
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
            ep, device, compatibility = _ep_device_compatibility(attrs, onnx_path)
            variants.append(
                VariantSpec(
                    component_name=component_name,
                    variant_name=target_name,
                    onnx_files=[onnx_path],
                    ep=ep,
                    device=device,
                    compatibility=compatibility,
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
            component_names = model_config["config"].get("component_names", [])

            if not components:
                raise ValueError(f"Composite source {target_name!r} declares no model_components.")

            for comp_config, comp_name in zip(components, component_names):
                # Component-level inference_settings overrides target-level if present.
                comp_inference = comp_config.get("config", {}).get("inference_settings") or target_inference
                # Component-level model_attributes overlay target-level.
                comp_attrs = dict(target_attrs)
                comp_attrs.update(_get_model_attributes(comp_config))

                onnx_path = _resolve_onnx_path(comp_config)
                ep, device, compatibility = _ep_device_compatibility(comp_attrs, onnx_path)

                spec = VariantSpec(
                    component_name=comp_name,
                    variant_name=target_name,
                    onnx_files=[onnx_path],
                    ep=ep,
                    device=device,
                    compatibility=compatibility,
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
    compatibility: list[str] = field(default_factory=list)
    inference_settings: dict[str, Any] = field(default_factory=dict)
    consumer_metadata: Optional[dict[str, Any]] = None


def write_model_package(
    output_dir: Path,
    variants: list[VariantSpec],
    config_files: Optional[dict[str, Path]] = None,
    producer_info: Optional[dict[str, Any]] = None,
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
        ``manifest.producer``. Schema-tolerated extra field (the proposal
        defines only ``schema_version``, ``components``, and
        ``merge_provenance``; producers may add namespaced extras).
    """
    if not variants:
        raise ValueError("write_model_package requires at least one variant.")

    output_dir = Path(output_dir)
    _ensure_empty_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        _write_component(output_dir, comp_name, comp_variants)

    if config_files:
        _copy_config_files(output_dir, config_files)

    _write_manifest(output_dir, list(components.keys()), producer_info)


def _write_component(output_dir: Path, component_name: str, comp_variants: list[VariantSpec]) -> None:
    component_dir = output_dir / component_name
    component_dir.mkdir(parents=True, exist_ok=True)

    # First pass: copy each variant's ONNX file(s) and discover external-data
    # references. We hash blobs as we copy so multi-variant packages don't
    # re-read the data later.
    blob_index: dict[str, dict[str, Any]] = {}
    variant_files: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {}

    for v in comp_variants:
        if not v.onnx_files:
            raise ValueError(f"Variant '{v.variant_name}' under component '{component_name}' has no ONNX files.")

        variant_dir = component_dir / v.variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        files_for_variant: list[tuple[str, list[tuple[str, str]]]] = []

        for onnx_src in v.onnx_files:
            onnx_src_path = Path(onnx_src)
            if not onnx_src_path.is_file():
                raise FileNotFoundError(f"ONNX file not found: {onnx_src_path}")

            onnx_dst = variant_dir / onnx_src_path.name
            shutil.copy2(str(onnx_src_path), str(onnx_dst))

            ext_refs = _discover_external_data(onnx_src_path)
            external_root = onnx_src_path.parent.resolve()
            blob_records: list[tuple[str, str]] = []
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

                sha = _sha256_file(blob_dst)
                blob_records.append((graph_location, sha))

                entry = blob_index.setdefault(
                    sha, {"first_path": blob_dst, "occurrences": 0, "basename": Path(graph_location).name}
                )
                entry["occurrences"] += 1

            files_for_variant.append((onnx_dst.name, blob_records))

        variant_files[v.variant_name] = files_for_variant

    # Second pass: dedup any blob that appears in 2+ variants of this
    # component into <component>/shared_weights/<sha>/<basename>. Single-
    # occurrence blobs stay inline so single-variant packages remain
    # loadable without the package API.
    shared_weights_dir = component_dir / "shared_weights"
    shared_blob_paths: dict[str, Path] = {}
    for sha, entry in blob_index.items():
        if entry["occurrences"] < 2:
            continue
        sha_dir = shared_weights_dir / sha
        sha_dir.mkdir(parents=True, exist_ok=True)
        target = sha_dir / entry["basename"]
        if not target.exists():
            shutil.copy2(str(entry["first_path"]), str(target))
        shared_blob_paths[sha] = target

    # Third pass: for each variant, remove deduped blobs from the variant
    # directory and emit variant.json with the right shared_files map per
    # files[i]. Then emit metadata.json for the component.
    for v in comp_variants:
        variant_dir = component_dir / v.variant_name
        files_payload: list[dict[str, Any]] = []
        for onnx_filename, blob_records in variant_files[v.variant_name]:
            shared_files: dict[str, str] = {}
            for graph_location, sha in blob_records:
                if sha in shared_blob_paths:
                    inline = variant_dir / graph_location
                    if inline.exists():
                        inline.unlink()
                        # Clean up any now-empty parent directories created for
                        # nested graph_location paths, but stop at variant_dir.
                        parent = inline.parent
                        while parent != variant_dir and parent.is_dir() and not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                    shared_files[graph_location] = sha

            file_entry: dict[str, Any] = {"filename": onnx_filename}
            so = (v.inference_settings or {}).get("session_options") or {}
            po = _provider_options_for_ep(v.inference_settings or {}, v.ep)
            if so:
                file_entry["session_options"] = so
            if po:
                file_entry["provider_options"] = po
            if shared_files:
                file_entry["shared_files"] = shared_files
            files_payload.append(file_entry)

        variant_payload: dict[str, Any] = {"files": files_payload}
        if v.consumer_metadata is not None:
            variant_payload["consumer_metadata"] = v.consumer_metadata
        _write_json(variant_dir / "variant.json", variant_payload)

    _write_metadata(component_dir, comp_variants)


def _write_metadata(component_dir: Path, comp_variants: list[VariantSpec]) -> None:
    variants_payload: dict[str, Any] = {}
    for v in comp_variants:
        ep_entry: dict[str, Any] = {"ep": v.ep}
        if v.device:
            ep_entry["device"] = v.device
        if v.compatibility:
            ep_entry["compatibility"] = list(v.compatibility)
        variants_payload[v.variant_name] = {"ep_compatibility": [ep_entry]}
    _write_json(component_dir / "metadata.json", {"variants": variants_payload})


def _write_manifest(
    output_dir: Path,
    components: list[str],
    producer_info: Optional[dict[str, Any]],
) -> None:
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "components": components,
    }
    if producer_info:
        # Olive-specific provenance under a namespaced key so future schema
        # evolution can't collide with it.
        manifest["producer"] = producer_info
    _write_json(output_dir / "manifest.json", manifest)


# ---------------------------------------------------------------------------
# configs/ handling
# ---------------------------------------------------------------------------


def _copy_config_files(output_dir: Path, config_files: dict[str, Path]) -> None:
    configs_dir = output_dir / "configs"
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
                    "Per-variant config differences belong in variant.json's consumer_metadata, "
                    "which is consumer-defined and out of Olive's scope.",
                    name,
                    src_path,
                )
            continue
        if src_path.is_dir():
            shutil.copytree(str(src_path), str(dest))
        elif src_path.is_file():
            shutil.copy2(str(src_path), str(dest))
        else:
            logger.warning("Config source %s does not exist; skipping.", src_path)


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


def _ep_device_compatibility(attrs: dict, onnx_path: Path) -> tuple[str, Optional[str], list[str]]:
    """Extract (ep, device, compatibility[]) for one variant from Olive metadata."""
    ep = attrs.get("ep") or "CPUExecutionProvider"
    device = attrs.get("device") or None
    compatibility = parse_compatibility_strings(_extract_ep_compatibility_from_onnx(onnx_path, ep))
    return ep, device, compatibility


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
