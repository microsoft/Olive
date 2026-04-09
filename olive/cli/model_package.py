# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

from olive.cli.base import BaseOliveCLICommand, add_logging_options, add_telemetry_options
from olive.common.utils import hardlink_copy_dir
from olive.telemetry import action

logger = logging.getLogger(__name__)


@action
class ModelPackageCommand(BaseOliveCLICommand):
    """Merge multiple model outputs into a model package with manifest.json."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "generate-model-package",
            help="Merge multiple model outputs into a model package with manifest.json",
        )

        sub_parser.add_argument(
            "-s",
            "--source",
            type=str,
            action="append",
            required=True,
            help=("Source context binary output directory. Can be specified multiple times. "),
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

    def run(self):
        sources = self._parse_sources()
        output_dir = Path(self.args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.args.model_name or output_dir.name
        model_version = self.args.model_version

        # Copy config files (genai_config.json, chat_template) to configs/
        config_file_names = self._copy_config_files(sources, output_dir)

        # Check if sources contain composite models
        is_composite = self._is_composite_source(sources)

        if is_composite:
            self._run_composite(sources, output_dir, model_name, model_version, config_file_names)
        else:
            self._run_single_component(sources, output_dir, model_name, model_version, config_file_names)

    def _run_single_component(self, sources, output_dir, model_name, model_version, config_file_names):
        """Package non-composite models (single ONNX per target)."""
        # Create component model directory under models/
        component_dir = output_dir / "models" / model_name
        component_dir.mkdir(parents=True, exist_ok=True)

        model_variants = {}
        for target_name, source_path in sources:
            model_config = self._read_model_config(source_path)
            model_attrs = model_config.get("config", {}).get("model_attributes") or {}

            # Copy source directory into component_dir/{target_name}/
            target_dir = component_dir / target_name
            hardlink_copy_dir(source_path, target_dir)

            constraints = {}
            for key in ("ep", "device", "architecture"):
                if model_attrs.get(key) is not None:
                    constraints[key] = model_attrs[key]

            # Extract ep_compatibility_info from ONNX model metadata
            ep_compat = self._extract_ep_compatibility_from_onnx(source_path, constraints.get("ep", ""))
            if ep_compat:
                constraints["ep_compatibility_info"] = ep_compat

            model_variants[target_name] = {
                "file": model_config.get("config", {}).get("model_path", f"{target_name}/"),
                "constraints": constraints,
            }

        # Write metadata.json in component directory
        metadata = {"name": model_name, "model_variants": model_variants}
        with open(component_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Remove config files from variant directories (they belong in configs/)
        for name in config_file_names:
            for p in component_dir.rglob(name):
                if p.is_file():
                    p.unlink()

        # Extract task from HF config
        task = self._extract_task(output_dir)

        # Write manifest.json at package root
        manifest = {
            "name": model_name,
            "model_version": model_version,
            "task": task,
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Merged {len(sources)} targets into {output_dir}")
        print(f"Manifest written to {manifest_path}")

    def _run_composite(self, sources, output_dir, model_name, model_version, config_file_names):
        """Package composite models with per-component directory layout."""
        # Collect component info across all targets.
        # component_data[comp_name][target_name] = (comp_config, source_path, constraints)
        component_data: dict[str, dict] = OrderedDict()

        for target_name, source_path in sources:
            model_config = self._read_model_config(source_path)
            model_attrs = model_config.get("config", {}).get("model_attributes") or {}
            comp_names = model_config.get("config", {}).get("model_component_names", [])
            comp_configs = model_config.get("config", {}).get("model_components", [])

            constraints = {}
            for key in ("ep", "device", "architecture"):
                if model_attrs.get(key) is not None:
                    constraints[key] = model_attrs[key]

            ep_compat = self._extract_ep_compatibility_from_onnx(source_path, constraints.get("ep", ""))
            if ep_compat:
                constraints["ep_compatibility_info"] = ep_compat

            for comp_name, comp_config in zip(comp_names, comp_configs):
                if comp_name not in component_data:
                    component_data[comp_name] = OrderedDict()
                component_data[comp_name][target_name] = (comp_config, source_path, constraints)

        models_dir = output_dir / "models"
        component_names = list(component_data.keys())

        for comp_name in component_names:
            comp_dir = models_dir / comp_name
            comp_dir.mkdir(parents=True, exist_ok=True)

            model_variants = {}
            for target_name, (comp_config, source_path, constraints) in component_data[comp_name].items():
                target_dir = comp_dir / target_name
                self._copy_component_files_from_config(comp_config, source_path, target_dir)

                # Determine the ONNX file name for the file path
                onnx_file_name = comp_config.get("config", {}).get("onnx_file_name")
                if not onnx_file_name:
                    comp_model_path = comp_config.get("config", {}).get("model_path", "")
                    if comp_model_path and Path(comp_model_path).suffix == ".onnx":
                        onnx_file_name = Path(comp_model_path).name
                    else:
                        onnx_files = sorted(target_dir.glob("*.onnx")) if target_dir.exists() else []
                        onnx_file_name = onnx_files[0].name if onnx_files else f"{comp_name}.onnx"

                file_path = f"{target_name}/{onnx_file_name}"
                model_variants[target_name] = {"file": file_path, "constraints": constraints}

            # Remove config files from component variant directories
            for name in config_file_names:
                for p in comp_dir.rglob(name):
                    if p.is_file():
                        p.unlink()

            # Write per-component metadata.json
            metadata = {"name": comp_name, "model_variants": model_variants}
            with open(comp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        # Extract task from HF config
        task = self._extract_task(output_dir)

        # Write manifest.json at package root
        manifest = {
            "name": model_name,
            "model_version": model_version,
            "task": task,
            "component_models": component_names,
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Merged {len(sources)} targets ({len(component_names)} components) into {output_dir}")
        print(f"Manifest written to {manifest_path}")

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
    def _is_composite_source(sources: list[tuple[str, Path]]) -> bool:
        """Check if sources contain composite models by inspecting model_config.json."""
        for _, source_path in sources:
            model_config = ModelPackageCommand._read_model_config(source_path)
            return model_config.get("type", "").lower() == "compositemodel"
        return False

    @staticmethod
    def _read_model_config(source_path: Path) -> dict:
        """Read and return model_config.json from a source directory."""
        config_path = source_path / "model_config.json"
        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def _copy_component_files_from_config(comp_config: dict, source_path: Path, dest_dir: Path) -> None:
        """Copy files for a single component based on its serialized config."""
        if dest_dir.exists():
            return

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the ONNX file path from the component config
        config = comp_config.get("config", {})
        comp_model_path = config.get("model_path", "")
        onnx_file_name = config.get("onnx_file_name")

        if onnx_file_name and comp_model_path:
            model_path = Path(comp_model_path) / onnx_file_name
        elif comp_model_path:
            model_path = Path(comp_model_path)
        else:
            # Fall back: copy all files from source
            hardlink_copy_dir(source_path, dest_dir)
            return

        # If model_path is relative, resolve against source_path
        if not model_path.is_absolute():
            model_path = source_path / model_path
        if not model_path.is_file():
            # Try finding it in source_path by name
            model_path = source_path / model_path.name

        if not model_path.is_file():
            logger.warning("Component ONNX file not found: %s, copying entire source", model_path)
            hardlink_copy_dir(source_path, dest_dir)
            return

        src_dir = model_path.parent

        # Copy the ONNX file
        shutil.copy2(str(model_path), str(dest_dir / model_path.name))

        # Find associated context binary files
        associated_files = set()
        try:
            from olive.passes.onnx.common import get_context_bin_file_names

            associated_files.update(get_context_bin_file_names(str(model_path)))
        except Exception:
            logger.debug("Could not read context binary file names from %s", model_path, exc_info=True)

        # Also check for ONNX external data files
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

        # Copy all associated files
        for file_name in associated_files:
            src = src_dir / file_name
            if src.is_file():
                shutil.copy2(str(src), str(dest_dir / file_name))

    @staticmethod
    def _copy_config_files(sources, output_dir):
        """Copy non-model files (genai_config, tokenizer, chat_template, etc.) to configs/.

        Collects files listed in ``additional_files`` from the first source's
        model_config.json and copies them to ``configs/`` at the package root.
        Returns the set of copied file names so they can be removed from variant dirs.
        """
        config_files: dict[str, Path] = {}

        for _, source_path in sources:
            model_config = ModelPackageCommand._read_model_config(source_path)
            additional_files = model_config.get("config", {}).get("model_attributes", {}).get("additional_files", [])
            for fp in additional_files:
                p = Path(fp)
                # The file may be in the original cache dir; also check source_path
                if not p.is_file():
                    p = source_path / p.name
                if p.is_file() and p.name not in config_files:
                    config_files[p.name] = p
            if config_files:
                break

        if not config_files:
            return set()

        configs_dir = output_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        for name, src_path in config_files.items():
            shutil.copy2(str(src_path), str(configs_dir / name))

        return set(config_files.keys())

    @staticmethod
    def _extract_task(output_dir: Path) -> list[str]:
        """Extract task from HuggingFace config.json in the configs/ directory."""
        config_path = output_dir / "configs" / "config.json"
        if not config_path.is_file():
            return []

        try:
            with open(config_path) as f:
                hf_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        architectures = hf_config.get("architectures", [])
        tasks = set()
        for arch in architectures:
            arch_lower = arch.lower()
            if "causal" in arch_lower or "forgenerating" in arch_lower:
                tasks.add("text_generation")
            elif "seq2seq" in arch_lower or "conditional" in arch_lower:
                tasks.add("text2text_generation")
            elif "sequenceclassification" in arch_lower:
                tasks.add("text_classification")
            elif "tokenclassification" in arch_lower:
                tasks.add("token_classification")
            elif "questionanswering" in arch_lower:
                tasks.add("question_answering")
            elif "imagegeneration" in arch_lower or "diffusion" in arch_lower:
                tasks.add("image_generation")
            elif "imageclassification" in arch_lower:
                tasks.add("image_classification")
            elif "objectdetection" in arch_lower:
                tasks.add("object_detection")
            elif "speechseq2seq" in arch_lower or "ctc" in arch_lower:
                tasks.add("automatic_speech_recognition")

        return sorted(tasks) if tasks else []

    @staticmethod
    def _extract_ep_compatibility_from_onnx(source_path: Path, ep: str = "") -> "str | None":
        """Extract ep_compatibility_info from ONNX model files in *source_path*.

        Looks for metadata keys prefixed with ``ep_compatibility_info.`` in the
        first ``.onnx`` file found in the directory.
        """
        onnx_files = sorted(source_path.glob("*.onnx"))
        if not onnx_files:
            return None

        try:
            import onnx

            onnx_model = onnx.load(str(onnx_files[0]), load_external_data=False)
            prefix = "ep_compatibility_info."
            ep_compat_map = {
                entry.key[len(prefix) :]: entry.value
                for entry in onnx_model.metadata_props
                if entry.key.startswith(prefix)
            }
        except Exception:
            logger.debug("Could not read ONNX metadata from %s", onnx_files[0], exc_info=True)
            return None

        if not ep_compat_map:
            return None
        if ep and ep in ep_compat_map:
            return ep_compat_map[ep]
        if len(ep_compat_map) == 1:
            return next(iter(ep_compat_map.values()))
        return None
