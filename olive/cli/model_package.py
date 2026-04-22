# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from olive.cli.base import (
    BaseOliveCLICommand,
    add_logging_options,
    add_telemetry_options,
)
from olive.telemetry import action

logger = logging.getLogger(__name__)

# Model file suffixes that belong in the models/ directory, not configs/
_MODEL_SUFFIXES = {".onnx", ".bin", ".data", ".xml"}


class ModelPackageCommand(BaseOliveCLICommand):
    """Merge multiple Olive output directories into a model package with manifest."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "generate-model-package",
            help="Merge multiple model outputs into a model package with manifest",
        )

        sub_parser.add_argument(
            "-s",
            "--source",
            type=str,
            action="append",
            required=True,
            help="Source Olive output directory. Can be specified multiple times.",
        )

        sub_parser.add_argument(
            "-o",
            "--output_path",
            type=str,
            required=True,
            help="Output directory for the merged model package.",
        )

        sub_parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="Model name for the manifest. If not set, derived from the output directory name.",
        )

        sub_parser.add_argument(
            "--model_version",
            type=str,
            default="1.0",
            help="Model version string for the manifest. Default: 1.0",
        )

        add_logging_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=ModelPackageCommand)

    @action
    def run(self):
        sources = self._parse_sources()
        output_dir = Path(self.args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.args.model_name or output_dir.name
        model_version = self.args.model_version

        # Read model configs from each source
        targets = []
        for target_name, source_path in sources:
            model_config = self._read_model_config(source_path)
            targets.append((target_name, source_path, model_config))

        is_composite = targets[0][2].get("type") == "CompositeModel"
        if is_composite:
            self._package_composite(targets, output_dir, model_name, model_version)
        else:
            self._package_single(targets, output_dir, model_name, model_version)

        logger.info("Model package generated at %s", output_dir)
        # ruff: noqa: T201
        print(f"Model package generated at {output_dir}")

    # ------------------------------------------------------------------
    # Single-component packaging
    # ------------------------------------------------------------------

    def _package_single(
        self,
        targets: list[tuple[str, Path, dict]],
        output_dir: Path,
        model_name: str,
        model_version: str,
    ) -> None:
        """Package non-composite models (single ONNX per target)."""
        config_file_names = self._copy_config_files(targets, output_dir)
        task = self._extract_task(targets)
        component_name = _task_to_component_name(task)

        component_dir = output_dir / "models" / component_name
        component_dir.mkdir(parents=True, exist_ok=True)

        model_variants = {}
        for target_name, _source_path, model_config in targets:
            attrs = _get_model_attributes(model_config)
            model_path = Path(model_config["config"]["model_path"])

            target_dir = component_dir / target_name
            _copy_model_files_single(model_path, target_dir)

            constraints = _build_constraints(attrs, model_path)
            model_variants[target_name] = {"file": model_path.name, "constraints": constraints}

        _remove_config_files(component_dir, config_file_names)

        metadata = {"name": component_name, "model_variants": model_variants}
        _write_json(component_dir / "metadata.json", metadata)

        manifest = {
            "name": model_name,
            "model_version": model_version,
            "task": task,
            "component_models": [component_name],
        }
        _write_json(output_dir / "manifest.json", manifest)

    # ------------------------------------------------------------------
    # Composite-model packaging
    # ------------------------------------------------------------------

    def _package_composite(
        self,
        targets: list[tuple[str, Path, dict]],
        output_dir: Path,
        model_name: str,
        model_version: str,
    ) -> None:
        """Package composite models with per-component directory layout."""
        config_file_names = self._copy_config_files(targets, output_dir)

        # Collect component info: component_data[comp_name][target_name] = (comp_config, target_attrs)
        from collections import OrderedDict

        component_data: dict[str, dict] = OrderedDict()

        for target_name, _source_path, model_config in targets:
            target_attrs = _get_model_attributes(model_config)
            components = model_config["config"].get("model_components", [])
            component_names = model_config["config"].get("component_names", [])

            for comp_config, comp_name in zip(components, component_names):
                if comp_name not in component_data:
                    component_data[comp_name] = OrderedDict()
                component_data[comp_name][target_name] = (comp_config, target_attrs)

        models_dir = output_dir / "models"
        comp_names_list = list(component_data.keys())

        for comp_name in comp_names_list:
            comp_dir = models_dir / comp_name
            comp_dir.mkdir(parents=True, exist_ok=True)

            model_variants = {}
            for target_name, (comp_config, target_attrs) in component_data[comp_name].items():
                comp_model_path = Path(comp_config["config"]["model_path"])
                target_dir = comp_dir / target_name
                _copy_component_files(comp_model_path, target_dir)

                constraints = _build_constraints(target_attrs, comp_model_path)
                model_variants[target_name] = {"file": comp_model_path.name, "constraints": constraints}

            _remove_config_files(comp_dir, config_file_names)

            metadata = {"name": comp_name, "model_variants": model_variants}
            _write_json(comp_dir / "metadata.json", metadata)

        task = self._extract_task(targets)
        manifest = {
            "name": model_name,
            "model_version": model_version,
            "task": task,
            "component_models": comp_names_list,
        }
        _write_json(output_dir / "manifest.json", manifest)

    # ------------------------------------------------------------------
    # Config file handling
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_config_files(
        targets: list[tuple[str, Path, dict]],
        output_dir: Path,
    ) -> set[str]:
        """Copy non-model config files (genai_config, tokenizer, etc.) to configs/."""
        config_entries: dict[str, Path] = {}

        # Collect from the first target's additional_files or source directory
        for _target_name, _source_path, model_config in targets:
            attrs = _get_model_attributes(model_config)
            for fp in attrs.get("additional_files", []):
                p = Path(fp)
                if (p.is_file() or p.is_dir()) and p.name not in config_entries:
                    config_entries[p.name] = p
            if config_entries:
                break

        # Fall back to scanning the source directory for non-model files
        if not config_entries:
            for _target_name, source_path, _model_config in targets:
                for f in sorted(source_path.iterdir()):
                    if f.name == "model_config.json":
                        continue
                    if (f.is_file() and f.suffix not in _MODEL_SUFFIXES) or f.is_dir():
                        config_entries[f.name] = f
                if config_entries:
                    break

        if not config_entries:
            return set()

        configs_dir = output_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        for name, src_path in config_entries.items():
            dest = configs_dir / name
            if src_path.is_dir():
                if not dest.exists():
                    shutil.copytree(str(src_path), str(dest))
            else:
                shutil.copy2(str(src_path), str(dest))
            logger.info("Copied %s to %s", name, configs_dir)

        return set(config_entries.keys())

    # ------------------------------------------------------------------
    # Source validation and reading
    # ------------------------------------------------------------------

    def _parse_sources(self) -> list[tuple[str, Path]]:
        sources = []
        for source in self.args.source:
            path = Path(source)
            if not path.is_dir():
                raise ValueError(f"Source path does not exist or is not a directory: {path}")

            if not (path / "model_config.json").exists():
                raise ValueError(
                    f"No model_config.json found in {path}. "
                    "Source must be an Olive output directory with model_config.json."
                )

            sources.append((path.name, path))

        if len(sources) < 2:
            raise ValueError("At least two --source directories are required to merge.")

        return sources

    @staticmethod
    def _read_model_config(source_path: Path) -> dict:
        config_path = source_path / "model_config.json"
        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def _extract_accelerator_info(target_models: list[dict]) -> tuple[str, str]:
        for model_config in target_models:
            attrs = model_config.get("config", {}).get("model_attributes") or {}
            ep = attrs.get("ep", "CPUExecutionProvider")
            device = attrs.get("device", "cpu")
            return ep, device.lower()
        return "CPUExecutionProvider", "cpu"

    # ------------------------------------------------------------------
    # Task extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_task(targets: list[tuple[str, Path, dict]]) -> str:
        """Extract the HuggingFace pipeline task for the model."""
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


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _get_model_attributes(model_config: dict) -> dict:
    return model_config.get("config", {}).get("model_attributes") or {}


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Generated %s", path)


def _build_constraints(attrs: dict, model_path: Path) -> dict:
    """Build variant constraints from model attributes and ONNX metadata."""
    constraints = {}
    ep = attrs.get("ep")
    if ep:
        constraints["ep"] = ep
    device = attrs.get("device")
    if device:
        constraints["device"] = device
    ep_compat = _extract_ep_compatibility_from_onnx(model_path, ep or "")
    constraints["ep_compatibility_info"] = ep_compat or ""
    return constraints


def _extract_ep_compatibility_from_onnx(model_path: Path, ep: str = "") -> Optional[str]:
    """Extract ep_compatibility_info from ONNX model custom metadata."""
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


def _copy_model_files_single(model_path: Path, dest_dir: Path) -> None:
    """Copy model files for a single ONNX model into dest_dir."""
    if dest_dir.exists():
        return

    src_dir = model_path.parent if model_path.is_file() else model_path
    if src_dir.is_dir():
        shutil.copytree(str(src_dir), str(dest_dir))
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(model_path), str(dest_dir))


def _copy_component_files(model_path: Path, dest_dir: Path) -> None:
    """Copy files for a single ONNX component to dest_dir.

    Copies the .onnx file and its associated context binary (.bin) files
    and external data files.
    """
    if dest_dir.exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    src_dir = model_path.parent

    # Copy the ONNX file itself
    shutil.copy2(str(model_path), str(dest_dir / model_path.name))

    # Find associated files
    associated_files: set[str] = set()
    try:
        from olive.passes.onnx.common import get_context_bin_file_names

        associated_files.update(get_context_bin_file_names(str(model_path)))
    except Exception:
        logger.debug("Could not read context binary file names from %s", model_path, exc_info=True)

    try:
        import onnx

        onnx_model = onnx.load(str(model_path), load_external_data=False)
        for init in onnx_model.graph.initializer:
            if init.data_location == onnx.TensorProto.EXTERNAL:
                for entry in init.external_data:
                    if entry.key == "location":
                        associated_files.add(entry.value)
    except Exception:
        logger.debug("Could not read ONNX external data from %s", model_path, exc_info=True)

    for file_name in associated_files:
        src = src_dir / file_name
        if src.is_file():
            shutil.copy2(str(src), str(dest_dir / file_name))


def _remove_config_files(component_dir: Path, config_file_names: set[str]) -> None:
    """Remove config files from variant subdirectories (they belong in configs/)."""
    for name in config_file_names:
        for p in component_dir.rglob(name):
            if p.is_dir():
                shutil.rmtree(str(p))
            else:
                p.unlink()
            logger.debug("Removed duplicate config entry %s from variant directory", p)


def _task_to_component_name(task: str) -> str:
    """Map a task string to a component name for single-component models."""
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
