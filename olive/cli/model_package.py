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
from pathlib import Path, PurePosixPath, PureWindowsPath
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

# Reverse lookup: ``genai_config`` provider alias (case-insensitive) → canonical
# ORT EP name. Used when reading a source's ``genai_config.json`` to derive the
# EP from a ``provider_options`` entry without requiring the source to also
# carry an Olive ``model_config.json``.
_GENAI_TO_EP: dict[str, str] = {alias.lower(): ep for ep, alias in _EP_TO_GENAI.items()}

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

        targets: list[tuple[str, Path, dict]] = []
        for target_name, source_path in sources:
            source_genai = _load_source_genai(source_path)
            if not isinstance(source_genai, dict):
                raise ValueError(
                    f"Source {source_path} has an unreadable genai_config.json. "
                    "The packager is genai_config-driven; the file must be valid JSON describing "
                    "the model layout (role filenames, session_options, etc.)."
                )
            targets.append((target_name, source_path, source_genai))

        variants = self._build_variants(targets)

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

    def _build_variants(self, targets: list[tuple[str, Path, dict]]) -> list["VariantSpec"]:
        variants: list[VariantSpec] = []
        for target_name, source_path, source_genai in targets:
            # Each role under ``genai_config.model`` is an independent ORT
            # inference session at runtime, so each role becomes its own
            # package component. A text-only model has one role (``decoder``)
            # → one component. A VLM has three roles (``vision``,
            # ``embedding``, ``decoder``) → three components. A QNN
            # pipeline-shaped role becomes ONE component whose variant
            # directory holds every stage's ONNX flat.
            artifacts_by_role = _collect_artifacts_per_role(source_path, source_genai)
            for role_name, role_artifacts in artifacts_by_role.items():
                onnx_files = [a.source_path for a in role_artifacts]
                onnx_rel_paths = [a.package_rel_path for a in role_artifacts]

                ep = _resolve_ep_for_role(source_genai, role_name)
                # ``ep_compatibility_info`` metadata is conventionally
                # written on the role's first ONNX (the primary stage for
                # a pipeline role); probe that file for the EP-scoped
                # compatibility string.
                raw_compat = _extract_ep_compatibility_from_onnx(onnx_files[0], ep) if onnx_files else None
                compatibility_string = raw_compat.strip() if raw_compat and raw_compat.strip() else None

                variants.append(
                    VariantSpec(
                        component_name=role_name,
                        variant_name=target_name,
                        role_name=role_name,
                        onnx_files=onnx_files,
                        onnx_rel_paths=onnx_rel_paths,
                        ep=ep,
                        compatibility_string=compatibility_string,
                        source_genai=source_genai,
                    )
                )
        return variants

    # ------------------------------------------------------------------
    # Config file handling
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_config_files(targets: list[tuple[str, Path, dict]]) -> dict[str, Path]:
        """Pick consumer-shared config files (genai_config, tokenizer, ...).

        Sweeps one source's directory for any non-ONNX/binary files
        (tokenizer assets, genai_config.json, chat_template, processor_config,
        etc.). The chosen source is the one with the most genai_config roles
        — when sources don't all expose the same role set (e.g. cpu has
        vision/embedding/decoder but gpu has only decoder), picking the
        first source could otherwise emit a base ``genai_config.json``
        missing role blocks that downstream components rely on. Other
        sources don't contribute files; the package emits one shared base
        config set.

        Subdirectories that hold model artifacts (e.g. Mobius VLM's
        ``decoder/``, ``embedding/``, ``vision_encoder/``, recognized via
        the ``model.<role>.filename`` references in the source's
        ``genai_config.json``) are excluded from this sweep — they're
        copied per-variant into ``models/<component>/<variant>/`` and have
        no business duplicating large ONNX/data blobs into the shared
        ``configs/`` tree.
        """
        if not targets:
            return {}
        _target_name, source_path, source_genai = _select_base_config_source(targets)
        model_dirs = _model_artifact_dirs(source_genai)
        config_entries: dict[str, Path] = {}
        for f in sorted(source_path.iterdir()):
            if f.is_dir() and f.name in model_dirs:
                continue
            if (f.is_file() and f.suffix not in _MODEL_SUFFIXES) or f.is_dir():
                config_entries[f.name] = f
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
            # ``genai_config.json`` is the single source of truth: it tells
            # us which ONNX files are roles vs pipeline stages, the EP for
            # each role (via session_options.provider_options), and the
            # variant-specific model scalars to lift into the overlay.
            if not (path / "genai_config.json").is_file():
                raise ValueError(
                    f"Source {path} has no genai_config.json. Each source must be a "
                    "GenAI-shaped directory containing genai_config.json plus the ONNX "
                    "file(s) it references."
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

    # ------------------------------------------------------------------
    # Task extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_task(targets: list[tuple[str, Path, dict]]) -> str:
        # Inspect each source's genai_config.json roles to infer the task.
        # GenAI roles map cleanly to tasks (``decoder`` → text generation;
        # both ``encoder`` and ``decoder`` → text2text generation). Used
        # only for the ``producer.task`` manifest hint; component names
        # are derived per-role from the genai_config, not from the task.
        for _target_name, _source_path, source_genai in targets:
            if not isinstance(source_genai, dict):
                continue
            model_block = source_genai.get("model")
            if not isinstance(model_block, dict):
                continue
            roles = {k for k, v in model_block.items() if isinstance(v, dict)}
            if "encoder" in roles and "decoder" in roles:
                return "text2text_generation"
            if "decoder" in roles:
                return "text_generation"
        return ""


# ---------------------------------------------------------------------------
# Writer (CLI-private; kept here because only this command produces packages)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnnxArtifact:
    """An ONNX file the writer must place inside a variant directory.

    ``source_path`` points at the absolute path the writer reads from;
    ``package_rel_path`` is the location inside the variant directory the
    writer must write to. In the per-role-component layout
    ``_collect_artifacts_per_role`` emits this as the source filename's
    basename (e.g. ``model.onnx``) because each role gets its own
    ``models/<role>/<variant>/`` directory and there is no sibling role
    to disambiguate against. Direct callers that construct a VariantSpec
    by hand may supply a nested subpath when the variant truly needs a
    multi-file layout under one component. The rel path is the same
    string the variant's ``genai_config_overlay.json`` emits under
    ``model.<role>.filename``, so the on-disk layout and the loader's
    view stay aligned.
    """

    source_path: Path
    package_rel_path: str


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
    # The variant's source ``genai_config.json`` (parsed). Used to lift
    # per-variant fields (context_length, pad_token_id, decoder.inputs, ...)
    # into the variant overlay. Kept as a deep object rather than a path so
    # callers can synthesize it without touching disk.
    source_genai: Optional[dict[str, Any]] = None
    # Optional per-ONNX target paths inside the variant directory. When
    # supplied, must be index-aligned with ``onnx_files`` and contain a
    # safe relative path for every ONNX. When empty, the writer falls back
    # to the legacy flat layout (each ONNX placed at
    # ``<variant_dir>/<basename>``) — kept for direct callers and tests that
    # predate multi-component sources.
    onnx_rel_paths: list[str] = field(default_factory=list)
    # The genai_config role this variant represents (e.g. ``decoder``,
    # ``vision``, ``embedding``). When set, the overlay writer scopes its
    # lift to just this role rather than the whole ``model`` block, so a
    # multi-role source (Mobius VLM) produces one VariantSpec per role
    # under one component per role. When unset, the writer falls back to
    # the legacy multi-role lift for direct callers / tests that predate
    # per-role components.
    role_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.onnx_rel_paths and len(self.onnx_rel_paths) != len(self.onnx_files):
            raise ValueError(
                f"VariantSpec '{self.variant_name}': onnx_rel_paths length "
                f"({len(self.onnx_rel_paths)}) must match onnx_files length ({len(self.onnx_files)})."
            )
        for rel in self.onnx_rel_paths:
            if not isinstance(rel, str) or not _is_safe_relative_location(rel):
                raise ValueError(
                    f"VariantSpec '{self.variant_name}': unsafe ONNX package_rel_path {rel!r}. "
                    "Must be a non-empty relative path without absolute prefixes or '..' segments."
                )


def _variant_artifacts(v: VariantSpec) -> list[OnnxArtifact]:
    """Return the variant's ONNX files paired with their in-package rel paths.

    When ``onnx_rel_paths`` is empty (legacy callers), the rel path defaults
    to each source file's basename, preserving the original flat-layout
    behavior. When supplied (the CLI's genai_config-driven path), the rel
    path is used verbatim so multi-component sources land in matching
    subdirectories under the variant dir.
    """
    if v.onnx_rel_paths:
        return [
            OnnxArtifact(source_path=Path(src), package_rel_path=rel)
            for src, rel in zip(v.onnx_files, v.onnx_rel_paths)
        ]
    return [OnnxArtifact(source_path=Path(src), package_rel_path=Path(src).name) for src in v.onnx_files]


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

    def _assign_role(role: str, component: str) -> None:
        existing = role_to_component.get(role)
        if existing is None:
            role_to_component[role] = component
        elif existing != component:
            raise ValueError(
                f"Role '{role}' is mapped to two different components: "
                f"'{existing}' and '{component}'. Each genai_config role must "
                "belong to exactly one package component."
            )

    # Preferred path: per-role variants (each VariantSpec carries the role
    # it represents). One variant = one role = one component, so the
    # mapping is direct and conflicts are easy to surface.
    for v in variants:
        if v.role_name:
            _assign_role(v.role_name, v.component_name)
    # Legacy / multi-role path: variants without ``role_name`` aggregate
    # several roles under one component. Seed from each variant's source
    # genai_config so every role that appears under ``model.<role>``
    # (vision, embedding, decoder, ...) still points back at that
    # variant's component_name. Preserves backward compatibility for
    # direct ``write_model_package`` callers and tests.
    for v in variants:
        if v.role_name:
            continue
        src_genai = getattr(v, "source_genai", None) or {}
        model_block = src_genai.get("model") if isinstance(src_genai, dict) else None
        if isinstance(model_block, dict):
            for role_name, role_body in model_block.items():
                if isinstance(role_body, dict):
                    _assign_role(role_name, v.component_name)
    # Fallback for components whose variants carried no usable source_genai:
    # map the component name to itself as the role, matching the legacy
    # writer behavior for direct ``write_model_package`` callers.
    for comp_name in components:
        explicit_role = component_to_role.get(comp_name, comp_name)
        if explicit_role not in role_to_component:
            role_to_component[explicit_role] = comp_name

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

        artifacts = _variant_artifacts(v)

        # Build the set of "our" ONNX stems for each source dir so the
        # sidecar sweep below can avoid scooping up sibling roles' ONNX
        # files when multiple roles share a source directory (flat VLM
        # source: ``vision.onnx`` / ``embedding.onnx`` / ``text.onnx``
        # all live alongside one another in the source root; under the
        # per-role-component layout each role gets its own component, so
        # we must NOT propagate sibling roles' files into a given role's
        # variant dir). Mobius-style sources have one ONNX per subdir
        # so this dict typically maps each src_dir to a single stem.
        source_to_dst_dir: dict[Path, Path] = {}
        source_to_onnx_stems: dict[Path, set[str]] = {}

        for artifact in artifacts:
            onnx_src_path = artifact.source_path
            if not onnx_src_path.is_file():
                raise FileNotFoundError(f"ONNX file not found: {onnx_src_path}")

            onnx_dst = variant_dir / artifact.package_rel_path
            onnx_dst.parent.mkdir(parents=True, exist_ok=True)
            _copy_with_collision_check(onnx_src_path, onnx_dst)
            src_dir_resolved = onnx_src_path.parent.resolve()
            source_to_dst_dir.setdefault(src_dir_resolved, onnx_dst.parent)
            # ``Path.stem`` strips one suffix: ``model.onnx`` → ``model``,
            # ``model.onnx.data`` → ``model.onnx``. We want the bare
            # base-name so the prefix check below catches both the .onnx
            # and its companion .data / .bin / .xml files.
            source_to_onnx_stems.setdefault(src_dir_resolved, set()).add(onnx_src_path.stem)

            ext_refs = _discover_external_data(onnx_src_path)
            external_root = src_dir_resolved
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

                # External-data ``location`` is recorded in the ONNX file
                # relative to the ONNX's own directory; the loader resolves
                # it the same way. So the blob's destination is relative to
                # the ONNX's destination directory, NOT the variant root.
                blob_dst = onnx_dst.parent / graph_location
                blob_dst.parent.mkdir(parents=True, exist_ok=True)
                _copy_with_collision_check(blob_src, blob_dst)

        # Sweep each source ONNX directory for remaining model-suffix sidecar
        # files (e.g. an EPContext stub ``.onnx`` typically points at a
        # same-stem ``.xml``/``.bin`` pair for OpenVINO or a ``.bin`` context
        # blob for QNN; these sidecars don't appear in the ONNX initializer
        # ``external_data`` table so the standard external-data copy above
        # misses them). The sweep filters by ONNX stem prefix so it only
        # picks up legitimate companion files (``decoder.bin`` for
        # ``decoder.onnx``, ``decoder.onnx.data`` for ``decoder.onnx``)
        # and ignores sibling roles' ONNXes that happen to share the same
        # source directory (a flat VLM source has ``vision.onnx`` and
        # ``embedding.onnx`` next to ``text.onnx``; without the prefix
        # filter the decoder variant would pull in every sibling ONNX).
        # Duplicates already copied as external-data are skipped because
        # their content matches.
        for src_dir, dst_dir in sorted(source_to_dst_dir.items()):
            stems = source_to_onnx_stems.get(src_dir, set())
            for entry in sorted(src_dir.iterdir()):
                if not entry.is_file() or entry.suffix not in _MODEL_SUFFIXES:
                    continue
                # Only accept files whose name starts with one of our
                # ONNX stems followed by a separator (``.`` for
                # ``decoder.onnx.data``, ``_`` for ``decoder_init.bin``)
                # or whose name is exactly the stem (rare; OpenVINO can
                # produce ``model``-named blobs).
                entry_name = entry.name
                if not any(entry_name == stem or entry_name.startswith((f"{stem}.", f"{stem}_")) for stem in stems):
                    continue
                dst = dst_dir / entry_name
                _copy_with_collision_check(entry, dst, skip_if_identical=True)

        # Per-variant runtime fields flow through genai_config_overlay.json.
        _write_genai_config_overlay(variant_dir, component_role, v)

    _write_metadata(component_dir, component_name, comp_variants)


def _copy_with_collision_check(src: Path, dst: Path, *, skip_if_identical: bool = False) -> None:
    """Copy ``src`` to ``dst`` while refusing to silently overwrite mismatched content.

    If ``dst`` already exists with content identical to ``src`` (by SHA-256),
    the copy is skipped — a deduplication path the writer relies on when a
    sidecar file has already been brought in by the external-data sweep.
    If ``dst`` exists with *different* content, the writer raises rather
    than choose a winner: silently keeping either copy could leave the
    package referencing a stale or wrong blob. ``skip_if_identical`` is
    accepted as an explicit-intent flag at the call site (the dedupe
    behavior is always on; the flag exists so the sidecar sweep reads
    self-documenting).
    """
    del skip_if_identical  # behavior is always content-aware; flag is documentation
    if dst.exists():
        if not dst.is_file():
            raise FileExistsError(f"Cannot copy {src} to {dst}: destination exists and is not a regular file.")
        if _sha256_file(src) == _sha256_file(dst):
            return
        raise FileExistsError(
            f"Refusing to overwrite {dst} (existing content differs from {src}). "
            "Two source artifacts map to the same package destination; rename one or fix the source."
        )
    shutil.copy2(str(src), str(dst))


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

    Per-variant runtime fields flow through a JSON Merge Patch applied on top
    of the package's base ``configs/genai_config.json``. The base has every
    role's ``filename`` / ``session_options`` / ``pipeline`` stripped (see
    ``_strip_variant_specific``); this overlay restores them.

    When the variant was built per-role (``v.role_name`` is set, the default
    CLI path) the overlay lifts only that role's body — each role gets its
    own component / variant directory, so each overlay scopes to exactly the
    role it represents. The model-level scalars in
    ``_VARIANT_LEVEL_MODEL_KEYS`` (``context_length``, ``eos_token_id``,
    ``pad_token_id``, ``bos_token_id``, ``type``) are written into the
    overlay of only the source's primary role (``_pick_primary_role``).
    Writing them on every per-role overlay would corrupt the merged config
    because GenAI's overlay parser appends arrays rather than replacing
    them — ``eos_token_id`` is commonly a list, and a three-role VLM
    overlay set would triple every entry.

    When the variant carries every role at once (no ``role_name``, legacy
    multi-role-per-component callers) every role's per-variant body is
    lifted into the overlay together with the variant-level scalars.

    Pipeline-shaped roles (multi-stage exports, e.g. QNN) are covered by
    the same lift: ``pipeline`` is in the strip set so the base loses it,
    the overlay restores it.

    Direct ``write_model_package`` callers that don't pass ``source_genai``
    fall back to the legacy ``inference_settings``-driven shape so existing
    programmatic tests keep working.
    """
    src_genai = v.source_genai or {}
    src_model = src_genai.get("model") if isinstance(src_genai, dict) else None

    model_patch: dict[str, Any] = {}

    if isinstance(src_model, dict):
        if v.role_name:
            # Preferred path: per-role variant. Only lift this role's body.
            # Other roles end up in their own components (one component
            # per role), each with their own overlay.
            role_body = src_model.get(v.role_name)
            if isinstance(role_body, dict):
                role_patch = _lift_role_overlay_body(role_body, v.onnx_rel_paths)
                if role_patch:
                    model_patch[v.role_name] = role_patch
        else:
            # Legacy multi-role-per-component path: lift every role's
            # per-variant fields. Used by direct ``write_model_package``
            # callers / tests that predate per-role components.
            for role_name, role_body in src_model.items():
                if not isinstance(role_body, dict):
                    continue
                role_patch = _lift_role_overlay_body(role_body)
                if role_patch:
                    model_patch[role_name] = role_patch
    else:
        # Legacy path: callers that don't pass source_genai (writer-only
        # tests) construct a single-role overlay from VariantSpec's
        # inference_settings.
        inference = v.inference_settings or {}
        session_options: dict[str, Any] = dict(inference.get("session_options") or {})
        provider_options = _provider_options_for_ep(inference, v.ep)
        genai_ep = _genai_provider_name(v.ep)

        # ORT-GenAI's FinalizeConfig builds session_options.providers from
        # provider_options[*].name (src/config.cpp:1643-1645), and
        # SetProviderSessionOptions then registers each named provider. CPU
        # is not in the dispatch table (src/models/session_options.cpp:
        # 150-159); it has no configurable options, and ORT InferenceSession
        # adds it implicitly when no other EP is registered. We therefore
        # emit ``provider_options: []`` for CPU variants and an explicit
        # named entry for every other EP.
        if genai_ep == "CPU":
            session_options["provider_options"] = []
        else:
            session_options["provider_options"] = [{genai_ep: provider_options}]

        legacy_patch: dict[str, Any] = {"session_options": session_options}
        if v.onnx_files:
            # The base strips ``filename``; the loader resolves the variant
            # ONNX as ``<variant_dir>/<filename>``. Prefer the writer-known
            # package-relative path; fall back to basename for callers that
            # supply only ``onnx_files``.
            if v.onnx_rel_paths:
                legacy_patch["filename"] = v.onnx_rel_paths[0]
            else:
                legacy_patch["filename"] = Path(v.onnx_files[0]).name
        model_patch[component_role] = legacy_patch

    # Lift per-variant model-level scalars from the variant's own
    # genai_config.json. The base config strips these because they
    # legitimately differ across variants (e.g. NPU runtime caps
    # ``context_length`` at 4224 while CPU/CUDA use the full 131072;
    # pad_token_id can differ when one exporter uses the EOS as PAD and
    # another uses the sentinel). Without this lift the merged config
    # would silently use whichever variant happened to win the base
    # selection. For per-role variants only the primary role's overlay
    # carries these so the same scalar isn't append-merged once per
    # component (critical for list-valued ``eos_token_id``).
    if isinstance(src_model, dict):
        is_primary = (not v.role_name) or (v.role_name == _pick_primary_role(src_genai))
        if is_primary:
            for k in _VARIANT_LEVEL_MODEL_KEYS:
                if k in src_model:
                    # Deep-copy via JSON round-trip so we never share refs with
                    # the caller's dict; arrays in particular must be
                    # independent because GenAI's overlay parser treats arrays
                    # as append-merge.
                    model_patch[k] = json.loads(json.dumps(src_model[k]))

    overlay = {"model": model_patch}
    _write_json(variant_dir / "genai_config_overlay.json", overlay)


def _lift_role_overlay_body(role_body: dict, onnx_rel_paths: Optional[list[str]] = None) -> dict:
    """Lift the per-variant fields from a single source genai_config role body.

    Each role body may carry ``filename`` (flat-variant primary file),
    ``pipeline`` (multi-stage), and ``session_options`` (provider_options +
    EP knobs). All three are stripped from the base genai_config; this
    helper recovers them as the role's overlay patch.

    Filenames are normalised to their basenames. In the per-role-component
    layout each role gets its own variant directory under
    ``models/<role>/<variant>/`` and the writer places the ONNX(s) there
    flat — the original source-side ``decoder/`` / ``vision_encoder/`` /
    ``embedding/`` subdirectory prefixes (Mobius VLM convention) are no
    longer needed to disambiguate sibling roles inside one variant dir.
    When ``onnx_rel_paths`` is supplied (preferred), the role's
    ``filename`` is replaced by the writer-known package-relative path so
    the overlay matches the on-disk layout exactly even if the source
    diverged. Pipeline-stage filenames are likewise rewritten to their
    basenames so the per-stage references resolve inside the flat variant
    directory.

    Every filename — top-level role and pipeline stages alike — is
    validated as a safe relative path before basename normalisation;
    absolute paths or upward traversal raise rather than silently
    propagate into a generated overlay. Pipeline and session_options are
    deep-copied to avoid aliasing with the caller's dict.
    """
    patch: dict[str, Any] = {}
    pipeline = role_body.get("pipeline")
    has_pipeline = isinstance(pipeline, list) and pipeline
    filename = role_body.get("filename")
    # ``pipeline`` takes precedence when both are present (mirrors the
    # behavior of ``_collect_artifacts_per_role``, which only emits
    # pipeline stage artifacts in that case). Lifting both would produce
    # an overlay with both ``filename`` and ``pipeline`` (malformed for the
    # loader) and reuse ``onnx_rel_paths[0]`` — which is stage 0's
    # basename — for the spurious role-level filename, silently aliasing
    # the two filename fields.
    if not has_pipeline and isinstance(filename, str) and filename:
        if not _is_safe_relative_location(filename):
            raise ValueError(
                f"Unsafe genai_config filename {filename!r}: must be a relative path "
                "without absolute prefixes or '..' segments."
            )
        # Prefer the writer-known rel path when available so the overlay's
        # filename matches the on-disk layout under the variant dir; fall
        # back to the source filename's basename otherwise.
        if onnx_rel_paths:
            patch["filename"] = onnx_rel_paths[0]
        else:
            patch["filename"] = Path(filename).name
    so = role_body.get("session_options")
    if isinstance(so, dict):
        patch["session_options"] = json.loads(json.dumps(so))
    if has_pipeline:
        new_pipeline: list[Any] = []
        stage_idx = 0
        for stage in pipeline:
            if not isinstance(stage, dict):
                new_pipeline.append(json.loads(json.dumps(stage)))
                continue
            new_stage: dict[str, Any] = {}
            for stage_name, stage_body in stage.items():
                if not isinstance(stage_body, dict):
                    new_stage[stage_name] = json.loads(json.dumps(stage_body))
                    continue
                new_stage_body = json.loads(json.dumps(stage_body))
                stage_fn = stage_body.get("filename")
                if isinstance(stage_fn, str) and stage_fn:
                    if not _is_safe_relative_location(stage_fn):
                        raise ValueError(
                            f"Unsafe genai_config pipeline stage filename {stage_fn!r}: "
                            "must be a relative path without absolute prefixes or '..' segments."
                        )
                    if onnx_rel_paths and stage_idx < len(onnx_rel_paths):
                        new_stage_body["filename"] = onnx_rel_paths[stage_idx]
                    else:
                        new_stage_body["filename"] = Path(stage_fn).name
                    stage_idx += 1
                new_stage[stage_name] = new_stage_body
            new_pipeline.append(new_stage)
        patch["pipeline"] = new_pipeline
    return patch


# Per-variant model-level keys that we strip from the package's base
# genai_config.json and re-supply from each variant's source. These appear
# directly under ``model`` (not nested under ``model.<role>``) and we have
# observed them to legitimately diverge across variants of the same model
# (NPU context truncation, exporter-specific pad token encoding, etc.). Kept
# minimal: only add a key here when we have evidence it varies AND its base
# value would be wrong for some variant.
_VARIANT_LEVEL_MODEL_KEYS: tuple[str, ...] = (
    "context_length",
    "pad_token_id",
    "eos_token_id",
    "bos_token_id",
    "type",
)


def _strip_variant_specific(
    node: Any,
    keys: tuple[str, ...] = ("filename", "session_options", "pipeline", *_VARIANT_LEVEL_MODEL_KEYS),
) -> Any:
    """Recursively drop variant-specific keys from a genai_config-shaped dict.

    ``filename`` and ``session_options`` are intrinsically variant-specific and
    must not live in the package's base ``configs/genai_config.json``; per-variant
    ``genai_config_overlay.json`` files patch them back in. ``pipeline`` is
    also stripped because GenAI's overlay parser appends arrays rather than
    replacing them — a pipeline present in both base and overlay would
    duplicate every stage on merge. The same logic applies to per-variant
    model-level scalars listed in ``_VARIANT_LEVEL_MODEL_KEYS`` (e.g.
    ``context_length`` differs between NPU and GPU variants of the same
    model). Returns a deep copy.
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
    r"""Reject any path that wouldn't sit safely under a single variant dir.

    The check is intentionally OS-independent: a genai_config can be
    authored on one platform and packaged on another, so a path that
    looks "relative" on the packaging host must still be rejected if it
    would be interpreted as absolute, drive-rooted, or upward-traversing
    on either POSIX or Windows. We therefore evaluate the candidate with
    both ``PurePosixPath`` and ``PureWindowsPath`` and reject if either
    flags it. Backslashes in the input are normalized to forward slashes
    so a string like ``"..\..\escape"`` is caught on POSIX too.
    """
    if not isinstance(location, str) or not location:
        return False
    normalized = location.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(normalized)
    if posix.is_absolute() or windows.is_absolute():
        return False
    parts = posix.parts
    if not parts or any(part in ("..", "") for part in parts):
        return False
    # PureWindowsPath strips a leading drive letter into its own anchor
    # (caught above), but a bare ``C:foo`` (drive-relative) still slips
    # through is_absolute on both pure paths. Reject any segment that
    # contains a drive-letter colon as the second character.
    return all(not (len(part) >= 2 and part[1] == ":") for part in PureWindowsPath(normalized).parts)


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
# genai_config helpers
# ---------------------------------------------------------------------------


def _load_source_genai(source_path: Path) -> Optional[dict]:
    """Return the parsed ``<source>/genai_config.json`` if present.

    Each variant's source directory carries its own genai_config; the writer
    lifts per-variant model-level fields from it into the variant overlay.
    Missing or unparseable files yield ``None`` rather than failing so a
    source without genai_config (e.g. a pure-ONNX export not destined for
    GenAI) can still be packaged.
    """
    path = Path(source_path) / "genai_config.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.debug("Could not parse %s; skipping per-variant model-field lift.", path, exc_info=True)
        return None


def _pick_primary_role(source_genai: Optional[dict]) -> Optional[str]:
    """Pick the genai_config role that names the model's primary component.

    A genai_config's ``model`` block keys mix per-role objects (``decoder``,
    ``embedding``, ...) with model-level scalars (``vocab_size``,
    ``context_length``, ...). The primary role is the first key whose value
    is an object carrying either a ``filename`` (flat variant) or a
    ``pipeline`` (multi-stage variant). Returns ``None`` when no such role
    is found (e.g. genai_config missing or malformed).
    """
    if not isinstance(source_genai, dict):
        return None
    model_block = source_genai.get("model")
    if not isinstance(model_block, dict):
        return None
    for role, body in model_block.items():
        if not isinstance(body, dict):
            continue
        if "pipeline" in body or "filename" in body:
            return role
    return None


def _model_artifact_dirs(source_genai: Optional[dict]) -> set[str]:
    """Return source-root subdirectory names that hold ONNX model artifacts.

    Each entry is the first path segment of some genai_config role
    ``filename`` (or pipeline stage filename) — e.g. ``decoder`` for
    ``decoder/model.onnx``. Used by config-file collection to avoid copying
    multi-component model directories (``decoder/``, ``embedding/``,
    ``vision_encoder/``) into the package's shared ``configs/`` tree, which
    would duplicate large model artifacts that already live under
    ``models/<component>/<variant>/``.
    """
    dirs: set[str] = set()
    if not isinstance(source_genai, dict):
        return dirs
    model_block = source_genai.get("model")
    if not isinstance(model_block, dict):
        return dirs

    def _first_segment(filename: str) -> Optional[str]:
        if not isinstance(filename, str) or not filename:
            return None
        parts = Path(filename).parts
        if not parts or parts[0] in (".", "..", ""):
            return None
        # A flat filename like ``model.onnx`` has only one part — no
        # subdirectory to exclude.
        if len(parts) == 1:
            return None
        return parts[0]

    for role_body in model_block.values():
        if not isinstance(role_body, dict):
            continue
        seg = _first_segment(role_body.get("filename", ""))
        if seg:
            dirs.add(seg)
        pipeline = role_body.get("pipeline")
        if isinstance(pipeline, list):
            for stage in pipeline:
                if not isinstance(stage, dict):
                    continue
                for stage_body in stage.values():
                    if not isinstance(stage_body, dict):
                        continue
                    seg = _first_segment(stage_body.get("filename", ""))
                    if seg:
                        dirs.add(seg)
    return dirs


def _collect_artifacts_per_role(source_path: Path, source_genai: Optional[dict]) -> dict[str, list[OnnxArtifact]]:
    """Group ONNX artifacts by genai_config role.

    Each role under ``model`` that declares a ``filename`` (flat role) or a
    ``pipeline`` (multi-stage role) becomes one key in the returned dict.
    The role's value is the ordered list of artifacts that belong to it —
    one for a flat role, one per stage for a pipeline role.

    Every artifact's ``package_rel_path`` is the source filename's basename:
    in the per-role-component layout, each role gets its own variant
    directory under ``models/<role>/<variant>/``, so files don't need
    subdirectory prefixes to disambiguate from sibling roles' ONNXes
    (they no longer share a directory). The source filename's subdirectory
    is used only to locate the file on disk in the source — it does not
    propagate into the package.

    Filenames are validated as safe relative paths; absolute or
    upward-traversing entries raise ``ValueError``. A source with no
    role declaring any usable ONNX also raises.
    """
    if not isinstance(source_genai, dict):
        raise ValueError(f"Source {source_path} has no parseable genai_config.json.")
    model_block = source_genai.get("model")
    if not isinstance(model_block, dict):
        raise ValueError(f"Source {source_path} genai_config.json has no ``model`` block.")

    by_role: dict[str, list[OnnxArtifact]] = {}

    def _validated_artifact(filename: str, kind: str, role: str) -> OnnxArtifact:
        if not _is_safe_relative_location(filename):
            raise ValueError(
                f"Source {source_path} role {role!r} {kind} {filename!r} is not a safe "
                "relative path (absolute paths and '..' segments are rejected)."
            )
        return OnnxArtifact(source_path=source_path / filename, package_rel_path=Path(filename).name)

    for role_name, role_body in model_block.items():
        if not isinstance(role_body, dict):
            continue
        # Pipeline takes precedence: a role with both fields is malformed,
        # but pipeline is the multi-stage shape so prefer that when present.
        artifacts: list[OnnxArtifact] = []
        pipeline = role_body.get("pipeline")
        if isinstance(pipeline, list) and pipeline:
            for stage in pipeline:
                if not isinstance(stage, dict):
                    continue
                for stage_body in stage.values():
                    if not isinstance(stage_body, dict):
                        continue
                    stage_fn = stage_body.get("filename")
                    if isinstance(stage_fn, str) and stage_fn:
                        artifacts.append(_validated_artifact(stage_fn, "pipeline stage filename", role_name))
        else:
            filename = role_body.get("filename")
            if isinstance(filename, str) and filename:
                artifacts.append(_validated_artifact(filename, "filename", role_name))
        if artifacts:
            by_role[role_name] = artifacts

    if not by_role:
        raise ValueError(
            f"Source {source_path} has no role in genai_config.json with a usable "
            "``filename`` or ``pipeline``; cannot determine which ONNX file(s) to package."
        )
    return by_role


def _resolve_ep_for_role(source_genai: Optional[dict], role_name: str) -> str:
    """Pick the role's ORT EP from its genai_config ``provider_options``.

    Returns the first non-CPU EP alias found under the role's
    ``session_options.provider_options`` (or any pipeline stage's).
    Returns ``CPUExecutionProvider`` for a CPU role (empty
    ``provider_options`` list, CPU-only aliases, or no provider_options
    at all). ``genai_config.json`` is the single source of truth for the
    role's EP — variant directory names are NOT consulted, since the
    producer's explicit declaration must win even when a CPU helper role
    lives inside a source dir colloquially named ``gpu``.
    """
    body = ((source_genai or {}).get("model") or {}).get(role_name) or {}

    so_blocks: list[dict] = []
    so = body.get("session_options")
    if isinstance(so, dict):
        so_blocks.append(so)
    so_blocks.extend(
        stage_body["session_options"]
        for stage in body.get("pipeline") or []
        if isinstance(stage, dict)
        for stage_body in stage.values()
        if isinstance(stage_body, dict) and isinstance(stage_body.get("session_options"), dict)
    )

    for so_block in so_blocks:
        for entry in so_block.get("provider_options") or []:
            if not isinstance(entry, dict):
                continue
            for alias in entry:
                ep = _GENAI_TO_EP.get(alias.lower())
                if ep and ep != "CPUExecutionProvider":
                    return ep
    return "CPUExecutionProvider"


def _select_base_config_source(
    targets: list[tuple[str, Path, dict]],
) -> tuple[str, Path, dict]:
    """Pick the source with the most complete role set for the base genai_config.

    When sources don't all expose the same role set (e.g. cpu has
    vision/embedding/decoder but gpu has only decoder), the per-source
    sweep that builds ``configs/genai_config.json`` could otherwise pick
    the smallest source and leave the base missing role blocks that the
    package's components rely on. Choose the source declaring the most
    roles with ``filename`` or ``pipeline``; ties resolve to the first
    target so the choice is deterministic.
    """

    def _role_count(source_genai: dict) -> int:
        if not isinstance(source_genai, dict):
            return 0
        model_block = source_genai.get("model")
        if not isinstance(model_block, dict):
            return 0
        return sum(
            1 for body in model_block.values() if isinstance(body, dict) and ("filename" in body or "pipeline" in body)
        )

    best_idx = 0
    best_count = _role_count(targets[0][2]) if targets else 0
    for idx in range(1, len(targets)):
        count = _role_count(targets[idx][2])
        if count > best_count:
            best_count = count
            best_idx = idx
    return targets[best_idx]


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
