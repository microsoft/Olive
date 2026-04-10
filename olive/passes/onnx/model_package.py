# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import model_info

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.handler.model_package import ModelPackageModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class ModelPackage(Pass):
    """Generate a model package with manifest.json and per-component metadata.json.

    This pass takes a ModelPackageModelHandler (containing model variants for different
    deployment targets) and generates a structured model package:

    - manifest.json at package root with model version, task, and component list
    - metadata.json per component with variant descriptors for each deployment target
    - configs/ directory for genai_config.json and chat_template files

    For composite models (where each target contains multiple ONNX components), the package
    is organized by component first, then by target:

        models/<component_name>/<target>/  (files)
        models/<component_name>/metadata.json

    Variant constraints include:
    - ep (required): execution provider name
    - device (optional): target device type (cpu, gpu, npu)
    - ep_compatibility_info (always present): EP-specific compatibility string, empty if unavailable
    """

    _accepts_composite_model = True
    _accepts_model_package_model = True
    _skip_additional_files_carry_forward = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                default_value=None,
                description="Model name for the manifest. If not set, derived from the output directory name.",
            ),
            "model_version": PassConfigParam(
                type_=str,
                default_value="1.0",
                description="Model version string for the manifest.",
            ),
        }

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self,
        model: ModelPackageModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ModelPackageModelHandler:
        assert isinstance(model, ModelPackageModelHandler), "ModelPackage requires a ModelPackageModelHandler as input."

        output_dir = Path(output_model_path).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = config.model_name or output_dir.name

        # Check if target models are composite (have multiple ONNX components)
        is_composite = model.is_composite

        if is_composite:
            return self._run_for_composite(model, config, output_dir, model_name)

        return self._run_for_single_component(model, config, output_dir, model_name)

    def _run_for_single_component(
        self,
        model: ModelPackageModelHandler,
        config: type[BasePassConfig],
        output_dir: Path,
        model_name: str,
    ) -> ModelPackageModelHandler:
        """Package a non-composite model (single ONNX per target)."""
        # Copy config files (genai_config.json, chat_template) to configs/
        config_file_names = self._copy_config_files(model, output_dir)

        # Extract task and derive component name
        task = self._extract_task(model)
        component_name = self._task_to_component_name(task)

        # Build model_variants dict and copy files into models/<component_name>/
        component_dir = output_dir / "models" / component_name
        component_dir.mkdir(parents=True, exist_ok=True)

        model_variants = {}
        for target_name, target_model in model.get_target_models():
            target_attrs = target_model.model_attributes or {}
            self._copy_target_model(target_name, target_model, component_dir)
            file_path = self._get_relative_model_path(target_name, target_model)
            constraints = self._build_constraints(target_attrs, target_model)
            model_variants[target_name] = {"file": file_path, "constraints": constraints}

        # Copy base model (pre-context-binary) into base/ subdirectory
        base_model_path = (model.model_attributes or {}).get("base_model_path")
        if base_model_path:
            self._copy_base_model(Path(base_model_path), component_dir, config_file_names)
            base_file = self._get_base_model_file(component_dir / "base")
            if base_file:
                model_variants["base"] = {"file": base_file, "constraints": {}}

        # Remove config files from variant directories (they belong in configs/)
        self._remove_config_files(component_dir, config_file_names)

        # Write metadata.json in the component directory
        metadata = {"name": component_name, "model_variants": model_variants}
        metadata_path = component_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Generated metadata at %s", metadata_path)

        # Write manifest.json at package root
        manifest = {
            "name": model_name,
            "model_version": config.model_version,
            "task": task,
            "component_models": [component_name],
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Generated manifest at %s", manifest_path)

        return self._build_result(model, output_dir, manifest_path)

    def _run_for_composite(
        self,
        model: ModelPackageModelHandler,
        config: type[BasePassConfig],
        output_dir: Path,
        model_name: str,
    ) -> ModelPackageModelHandler:
        """Package a composite model with per-component directory layout."""
        # Copy config files (genai_config.json, chat_template) to configs/
        config_file_names = self._copy_config_files(model, output_dir)

        # Collect component info across all targets.
        # component_data[comp_name][target_name] = (comp_handler, constraints)
        component_data: dict[str, dict] = OrderedDict()

        for target_name, target_model in model.get_target_models():
            assert isinstance(target_model, CompositeModelHandler), (
                "Expected CompositeModelHandler for composite packaging"
            )
            target_attrs = target_model.model_attributes or {}

            for comp_name, comp_handler in target_model.get_model_components():
                if comp_name not in component_data:
                    component_data[comp_name] = OrderedDict()

                constraints = self._build_constraints(target_attrs, comp_handler)
                component_data[comp_name][target_name] = (comp_handler, constraints)

        models_dir = output_dir / "models"
        component_names = list(component_data.keys())

        # Get base model path for copying pre-optimized files
        base_model_path = (model.model_attributes or {}).get("base_model_path")

        for comp_name in component_names:
            comp_dir = models_dir / comp_name
            comp_dir.mkdir(parents=True, exist_ok=True)

            model_variants = {}
            for target_name, (comp_handler, constraints) in component_data[comp_name].items():
                target_dir = comp_dir / target_name
                self._copy_component_files(comp_handler, target_dir)

                file_path = Path(comp_handler.model_path).name
                model_variants[target_name] = {"file": file_path, "constraints": constraints}

            # Copy base model for this component
            if base_model_path:
                self._copy_base_component(Path(base_model_path), comp_name, comp_dir, config_file_names)
                base_file = self._get_base_model_file(comp_dir / "base")
                if base_file:
                    model_variants["base"] = {"file": base_file, "constraints": {}}

            # Remove config files from component variant directories
            self._remove_config_files(comp_dir, config_file_names)

            # Write per-component metadata.json
            metadata = {"name": comp_name, "model_variants": model_variants}
            with open(comp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Generated metadata for component %s", comp_name)

        # Extract task
        task = self._extract_task(model)

        # Write manifest.json at package root
        manifest = {
            "name": model_name,
            "model_version": config.model_version,
            "task": task,
            "component_models": component_names,
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Generated manifest at %s", manifest_path)

        return self._build_result(model, output_dir, manifest_path)

    def _build_constraints(
        self,
        target_attrs: dict,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
    ) -> dict:
        """Build the constraints dict for a variant."""
        constraints = {"ep": self.accelerator_spec.execution_provider}
        device = target_attrs.get("device")
        if device:
            constraints["device"] = device
        ep_compat = self._extract_ep_compatibility_from_onnx(target_model, self.accelerator_spec.execution_provider)
        constraints["ep_compatibility_info"] = ep_compat or ""
        return constraints

    @staticmethod
    def _build_result(
        model: ModelPackageModelHandler,
        output_dir: Path,
        manifest_path: Path,
    ) -> ModelPackageModelHandler:
        """Build the result ModelPackageModelHandler with updated attributes."""
        new_model_attributes = dict(model.model_attributes or {})
        new_model_attributes["manifest_path"] = str(manifest_path)
        new_model_attributes.pop("additional_files", None)
        new_model_attributes.pop("base_model_path", None)

        return ModelPackageModelHandler(
            [target_model for _, target_model in model.get_target_models()],
            [target_name for target_name, _ in model.get_target_models()],
            model_path=output_dir,
            model_attributes=new_model_attributes,
        )

    @staticmethod
    def _copy_target_model(
        target_name: str,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
        output_dir: Path,
    ) -> None:
        dest_dir = output_dir / target_name
        if dest_dir.exists():
            return

        if isinstance(target_model, CompositeModelHandler):
            src_dir = Path(target_model.model_path)
        else:
            src_dir = Path(target_model.model_path).parent

        if src_dir.is_dir():
            shutil.copytree(str(src_dir), str(dest_dir))
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(target_model.model_path), str(dest_dir))

    @staticmethod
    def _get_relative_model_path(
        target_name: str,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
    ) -> str:
        if isinstance(target_model, ONNXModelHandler):
            return Path(target_model.model_path).name
        return ""

    @staticmethod
    def _copy_component_files(component: ONNXModelHandler, dest_dir: Path) -> None:
        """Copy files for a single ONNX component to dest_dir.

        Copies the .onnx file and its associated context binary (.bin) files
        by reading EPContext nodes in the ONNX model.
        """
        if dest_dir.exists():
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(component.model_path)
        src_dir = model_path.parent

        # Copy the ONNX file itself
        shutil.copy2(str(model_path), str(dest_dir / model_path.name))

        # Find associated context binary files from EPContext nodes
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

    @classmethod
    def _copy_config_files(cls, model, output_dir):
        """Copy non-model files (genai_config, tokenizer, chat_template, etc.) to configs/.

        Collects files and directories from target model ``additional_files`` and copies
        them to the ``configs/`` directory at the package root.  Returns the set of copied
        entry names so they can be removed from the variant directories later.
        """
        config_entries = cls._collect_config_files(model)
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

    @classmethod
    def _collect_config_files(cls, model):
        """Find config files from target model additional_files or model directories."""
        config_files: dict[str, Path] = {}

        # Collect from each target model's additional_files
        for _, target_model in model.get_target_models():
            for fp in (target_model.model_attributes or {}).get("additional_files", []):
                p = Path(fp)
                if (p.is_file() or p.is_dir()) and p.name not in config_files:
                    config_files[p.name] = p
            if config_files:
                break

        # Fall back to parent model's additional_files
        if not config_files:
            for fp in (model.model_attributes or {}).get("additional_files", []):
                p = Path(fp)
                if (p.is_file() or p.is_dir()) and p.name not in config_files:
                    config_files[p.name] = p

        return config_files

    @staticmethod
    def _get_model_dir(target_model):
        """Get the directory containing the target model."""
        if isinstance(target_model, CompositeModelHandler):
            return Path(target_model.model_path)
        p = Path(target_model.model_path)
        return p.parent if p.is_file() else p

    @staticmethod
    def _remove_config_files(component_dir, config_file_names):
        """Remove config files and directories from variant subdirectories.

        Skips the ``base/`` directory since base model files are copied separately.
        """
        for name in config_file_names:
            for p in component_dir.rglob(name):
                # Don't remove from base/ — base model is handled by _copy_base_model
                if "base" in p.relative_to(component_dir).parts:
                    continue
                if p.is_dir():
                    shutil.rmtree(str(p))
                    logger.debug("Removed duplicate config directory %s from variant directory", p)
                else:
                    p.unlink()
                    logger.debug("Removed duplicate config file %s from variant directory", p)

    @staticmethod
    def _copy_base_model(base_model_path, component_dir, config_file_names):
        """Copy the pre-optimized base model to the ``base/`` subdirectory.

        Only model files are copied — config files that belong in ``configs/`` are
        skipped.  Recognised model suffixes: ``.onnx``, ``.data``, ``.xml``, ``.bin``.
        """
        base_dir = component_dir / "base"
        if base_dir.exists():
            return

        base_model_path = Path(base_model_path)
        if not base_model_path.is_dir():
            logger.warning("Base model path %s not found, skipping base model copy", base_model_path)
            return

        base_dir.mkdir(parents=True, exist_ok=True)
        model_suffixes = {".onnx", ".data", ".xml", ".bin"}
        for f in sorted(base_model_path.iterdir()):
            if f.is_file() and f.name not in config_file_names and f.suffix in model_suffixes:
                shutil.copy2(str(f), str(base_dir / f.name))
                logger.info("Copied base model file %s to %s", f.name, base_dir)

    @staticmethod
    def _copy_base_component(base_model_path, comp_name, comp_dir, config_file_names):
        """Copy the base model files for a specific component to the ``base/`` subdirectory.

        Searches the base model directory for a subdirectory matching *comp_name*
        and copies model files from it.
        """
        base_dir = comp_dir / "base"
        if base_dir.exists():
            return

        base_model_path = Path(base_model_path)
        if not base_model_path.is_dir():
            logger.warning("Base model path %s not found, skipping base model copy for %s", base_model_path, comp_name)
            return

        # For composite models the base path is the parent directory containing component subdirs
        comp_src = base_model_path / comp_name
        if not comp_src.is_dir():
            logger.debug("No base directory found for component %s at %s", comp_name, comp_src)
            return

        base_dir.mkdir(parents=True, exist_ok=True)
        model_suffixes = {".onnx", ".data", ".xml", ".bin"}
        for f in sorted(comp_src.iterdir()):
            if f.is_file() and f.name not in config_file_names and f.suffix in model_suffixes:
                shutil.copy2(str(f), str(base_dir / f.name))
                logger.info("Copied base model file %s to %s for component %s", f.name, base_dir, comp_name)

    @staticmethod
    def _get_base_model_file(base_dir: Path) -> Optional[str]:
        """Find the primary model file in the base/ directory.

        Returns the filename of the first ``.onnx`` or ``.xml`` file found,
        or ``None`` if the directory does not exist or contains no model files.
        """
        if not base_dir.is_dir():
            return None
        for suffix in (".onnx", ".xml"):
            for f in sorted(base_dir.iterdir()):
                if f.is_file() and f.suffix == suffix:
                    return f.name
        return None

    @staticmethod
    def _task_to_component_name(task: str) -> str:
        """Map a task string to a component name for single-component models.

        Used when the model is not a composite pipeline but still needs
        a component directory name in the package structure.
        """
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

    @staticmethod
    def _extract_task(model: ModelPackageModelHandler) -> str:
        """Extract the task for this model using the HuggingFace Hub API.

        Reads ``_name_or_path`` from the model attributes and queries
        ``huggingface_hub.model_info`` for the ``pipeline_tag``.
        Returns an empty string if the task cannot be determined.
        """
        attrs = model.model_attributes or {}
        # Try the first target's attributes as fallback
        if "_name_or_path" not in attrs:
            for _, target_model in model.get_target_models():
                attrs = target_model.model_attributes or {}
                if "_name_or_path" in attrs:
                    break

        model_name_or_path = attrs.get("_name_or_path", "")
        if not model_name_or_path:
            return ""

        try:
            info = model_info(model_name_or_path)
            tag = info.pipeline_tag or ""
            # HF uses hyphens (e.g., "text-generation"); normalize to underscores
            return tag.replace("-", "_")
        except Exception:
            logger.debug("Could not fetch task from HuggingFace Hub for %s", model_name_or_path, exc_info=True)
            return ""

    @staticmethod
    def _extract_ep_compatibility_from_onnx(
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
        ep: str = "",
    ) -> Optional[str]:
        """Extract ep_compatibility_info from ONNX model custom metadata.

        Looks for metadata keys prefixed with ``ep_compatibility_info.`` in the
        ONNX model file.  If *ep* is given, the entry matching that EP name is
        preferred.  When only a single entry exists it is returned regardless of
        the EP name.
        """
        model_path = None
        if isinstance(target_model, ONNXModelHandler):
            model_path = Path(target_model.model_path)
        elif isinstance(target_model, CompositeModelHandler):
            for component in target_model.model_components:
                if isinstance(component, ONNXModelHandler):
                    model_path = Path(component.model_path)
                    break

        if model_path is None or not model_path.is_file():
            return None

        try:
            import onnx

            onnx_model = onnx.load(str(model_path), load_external_data=False)
            prefix = "ep_compatibility_info."
            ep_compat_map = {
                entry.key[len(prefix) :]: entry.value
                for entry in onnx_model.metadata_props
                if entry.key.startswith(prefix)
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
