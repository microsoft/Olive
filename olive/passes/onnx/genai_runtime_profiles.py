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


def _resolve_decoder_path(model_dir: Path, filename: str, field_name: str) -> Path:
    if not isinstance(filename, str):
        raise ValueError(f"{field_name} must be a relative path within the model directory.")
    filename_path = Path(filename)
    model_path = model_dir / filename_path
    if filename_path.is_absolute() or not model_path.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError(f"{field_name} must be a relative path within the model directory.")
    return model_path


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
        source_model_path = _resolve_decoder_path(output_dir, source_filename, "model.decoder.filename")
        if not source_model_path.is_file():
            raise ValueError(f"Decoder graph does not exist: {source_model_path}.")

        runtime_profiles = []
        variant_filenames = []
        profile_ids = set()
        eligibility_ranges = []
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
            maximum_memory = profile["eligibility"].get("maximum_total_device_memory_bytes")
            if maximum_memory is not None and (
                not isinstance(maximum_memory, int)
                or isinstance(maximum_memory, bool)
                or maximum_memory < minimum_memory
            ):
                raise ValueError(
                    "maximum_total_device_memory_bytes must be an integer greater than or equal to "
                    "minimum_total_device_memory_bytes."
                )
            for existing_id, existing_minimum, existing_maximum in eligibility_ranges:
                if (maximum_memory is None or existing_minimum <= maximum_memory) and (
                    existing_maximum is None or minimum_memory <= existing_maximum
                ):
                    raise ValueError(
                        f"Runtime profile {profile_id!r} eligibility overlaps runtime profile {existing_id!r}."
                    )
            eligibility_ranges.append((profile_id, minimum_memory, maximum_memory))
            if not isinstance(overlay, dict):
                raise ValueError(f"Runtime profile {profile_id!r} overlay must be a mapping.")

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
                variant_path = output_dir / variant_filename
                if variant_path == source_model_path:
                    raise ValueError(f"Runtime profile {profile_id!r} variant cannot replace the source decoder graph.")
                variant_path.unlink(missing_ok=True)

                from onnxruntime_genai.models.kv_cache_variant import KVCacheVariant

                KVCacheVariant(scheme).create(source_model_path, variant_path, scale_file)
                variant_filenames.append(variant_filename)
            runtime_profiles.append(profile)

        from onnxruntime_genai.models import builder_config

        if validate_runtime_profiles := getattr(builder_config, "validate_runtime_profiles", None):
            validate_runtime_profiles(runtime_profiles, genai_config)
        else:
            for profile in runtime_profiles:
                overlay = profile["overlay"]
                if not overlay:
                    raise ValueError(
                        f"runtime_config.runtime_profiles[{profile['id']!r}].overlay "
                        "must contain at least one overlay field"
                    )
                runtime_overlay = {key: overlay[key] for key in ("engine", "search", "speculative") if key in overlay}
                builder_config.validate_runtime_config(runtime_overlay, genai_config)
        for profile in runtime_profiles:
            configured_filename = profile["overlay"].get("model", {}).get("decoder", {}).get("filename")
            if configured_filename is not None:
                field_name = f"Runtime profile {profile['id']!r} overlay.model.decoder.filename"
                configured_model_path = _resolve_decoder_path(output_dir, configured_filename, field_name)
                if not configured_model_path.is_file():
                    raise ValueError(f"Decoder graph does not exist: {configured_model_path}.")

        genai_config["runtime_profiles"] = runtime_profiles
        decoder_data = output_dir / f"{source_filename}.data"
        embedding_filename = genai_config["model"].get("embedding", {}).get("filename")
        embedding_data = output_dir / f"{embedding_filename}.data" if embedding_filename else None
        if (
            embedding_data
            and decoder_data.is_file()
            and embedding_data.is_file()
            and decoder_data.samefile(embedding_data)
        ):
            for section in genai_config["model"].values():
                if isinstance(section, dict):
                    for initializer in section.get("shared_initializers", []):
                        if initializer.get("data_file") == decoder_data.name:
                            initializer["data_file"] = embedding_data.name
        updated_config_path = config_path.with_suffix(".json.tmp")
        with updated_config_path.open("w", encoding="utf-8") as stream:
            json.dump(genai_config, stream, indent=4)
        updated_config_path.replace(config_path)

        components = []
        component_names = []
        for component_name, _ in model.get_model_components():
            components.append(ONNXModelHandler(output_dir, onnx_file_name=component_name))
            component_names.append(component_name)
        for filename in variant_filenames:
            components.append(ONNXModelHandler(output_dir, onnx_file_name=filename))
            component_names.append(filename)
        model_attributes = copy.deepcopy(model.model_attributes) or {}
        model_attributes["keep_shared_external_data_names"] = True
        return CompositeModelHandler(
            components,
            component_names,
            model_path=output_dir,
            model_attributes=model_attributes,
        )
