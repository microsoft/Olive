# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import json
import re
from pathlib import Path

from olive.common.utils import hardlink_copy_dir
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

_PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class GenAIModelRuntimeProfiles(Pass):
    """Create alternate decoder graphs and runtime profiles for an ORT GenAI model directory."""

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "runtime_profiles": PassConfigParam(
                type_=list[dict],
                required=True,
                description=(
                    "Runtime profiles to add to genai_config.json. A profile may contain a build-only kv_cache "
                    "object with scheme and scale_file to generate an alternate decoder graph; this object is not "
                    "written to the runtime config. Without kv_cache, the pass only updates genai_config.json."
                ),
            )
        }

    def _run_for_config(
        self,
        model: CompositeModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> CompositeModelHandler:
        if not isinstance(model, CompositeModelHandler):
            raise ValueError("GenAIModelRuntimeProfiles requires the multi-file output of ModelBuilder.")
        if not all(isinstance(component, ONNXModelHandler) for component in model.model_components):
            raise ValueError("All GenAI model components must be ONNXModelHandler instances.")

        source_dir = Path(model.model_path)
        output_dir = Path(output_model_path)
        if output_dir.suffix:
            output_dir = output_dir.with_suffix("")
        if source_dir.resolve() == output_dir.resolve():
            raise ValueError("GenAIModelRuntimeProfiles requires a distinct output directory.")
        hardlink_copy_dir(source_dir, output_dir)

        config_path = output_dir / "genai_config.json"
        if not config_path.is_file():
            raise ValueError(f"Model directory does not contain {config_path.name}.")
        with config_path.open(encoding="utf-8") as stream:
            genai_config = json.load(stream)

        try:
            source_filename = genai_config["model"]["decoder"]["filename"]
        except (KeyError, TypeError) as error:
            raise ValueError("genai_config.json must define model.decoder.filename.") from error
        source_model_path = output_dir / source_filename
        if not source_model_path.is_file():
            raise ValueError(f"Decoder graph does not exist: {source_model_path}.")

        runtime_profiles = []
        profile_ids = set()
        for profile_config in config.runtime_profiles:
            profile = copy.deepcopy(profile_config)
            profile_id = profile.get("id")
            if not isinstance(profile_id, str) or not _PROFILE_ID_PATTERN.fullmatch(profile_id):
                raise ValueError(
                    "Runtime profile id must contain only ASCII letters, digits, underscores, and hyphens."
                )
            if profile_id in profile_ids:
                raise ValueError(f"Duplicate runtime profile id: {profile_id}.")
            profile_ids.add(profile_id)

            try:
                minimum_memory = profile["eligibility"]["minimum_total_device_memory_bytes"]
                overlay = profile["overlay"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Runtime profile {profile_id!r} must define "
                    "eligibility.minimum_total_device_memory_bytes and overlay."
                ) from error
            if not isinstance(minimum_memory, int) or isinstance(minimum_memory, bool) or minimum_memory < 0:
                raise ValueError("minimum_total_device_memory_bytes must be a non-negative integer.")

            kv_cache = profile.pop("kv_cache", None)
            if kv_cache is not None:
                try:
                    scheme = kv_cache["scheme"]
                    scale_file = kv_cache["scale_file"]
                except (KeyError, TypeError) as error:
                    raise ValueError(
                        f"Runtime profile {profile_id!r} kv_cache must define scheme and scale_file."
                    ) from error
                variant_filename = f"model_{profile_id}.onnx"
                model_overlay = overlay.setdefault("model", {}).setdefault("decoder", {})
                configured_filename = model_overlay.setdefault("filename", variant_filename)
                if configured_filename != variant_filename:
                    raise ValueError(
                        f"Runtime profile {profile_id!r} decoder filename must be {variant_filename!r}, "
                        f"got {configured_filename!r}."
                    )

                from onnxruntime_genai.models.kv_cache_variant import create_kv_cache_variant

                create_kv_cache_variant(source_model_path, output_dir / variant_filename, scheme, scale_file)
            runtime_profiles.append(profile)

        genai_config["runtime_profiles"] = runtime_profiles
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump(genai_config, stream, indent=4)

        components = []
        component_names = []
        for component_name, component in model.get_model_components():
            components.append(ONNXModelHandler(output_dir, onnx_file_name=Path(component.model_path).name))
            component_names.append(component_name)
        return CompositeModelHandler(
            components,
            component_names,
            model_path=output_dir,
            model_attributes=copy.deepcopy(model.model_attributes),
        )
