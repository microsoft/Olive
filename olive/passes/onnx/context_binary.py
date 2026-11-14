# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import json
import logging
import os
import platform
import re
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Any, Callable, Literal, Optional, Union

from packaging import version

from olive.common.ort_genai_config import register_ort_genai_config_update
from olive.common.ort_inference import get_ort_hardware_device_type
from olive.common.utils import hardlink_copy_file
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.hardware.constants import ExecutionProvider
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    get_context_bin_file_names,
    get_external_data_file_names,
    process_llm_pipeline,
    resave_model,
    update_llm_pipeline_genai_config_gpu_ctxbin,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

_OPENVINO_EP_PLUGIN_PACKAGE = "onnxruntime_ep_openvino"
# plugin EP registration, get_ep_devices, add_provider_for_devices and the compile API are all available from 1.23
_MINIMUM_PLUGIN_EP_ORT_VERSION = version.Version("1.23.0")
# 'ep.enable_weightless' replaced 'ep.enable_weightless_ep_context_nodes' in 1.30
_MINIMUM_WEIGHTLESS_ORT_VERSION = version.Version("1.30.0")
_WEIGHTLESS_SESSION_OPTION = "ep.enable_weightless"
_HEX_DEVICE_ID = re.compile(r"(0[xX])?[0-9a-fA-F]+")
_EP_LIBRARY_REGISTRATION_LOCK = RLock()
# SessionOptions attributes that are typed properties rather than string config entries
_TYPED_SESSION_OPTIONS = ("log_severity_level", "log_verbosity_level")


@contextmanager
def registered_ep_library(ort, registration_name: str, library_path: str, ep_name: Optional[str] = None):
    """Register an EP plugin library, unregistering on exit only if this call actually registered it.

    EP registration is process-global and keyed by registration_name; if that name is already
    registered by another caller, we reuse it and leave it in place rather than tearing down state
    we don't own. When ep_name is given, an EP that already exposes devices is reused if registration fails,
    except for '.virtual' registrations whose devices differ from the already exposed physical ones.
    """
    with _EP_LIBRARY_REGISTRATION_LOCK:
        owns_registration = True
        try:
            ort.register_execution_provider_library(registration_name, library_path)
        except Exception as e:
            ep_already_exposed = (
                ep_name
                and not registration_name.endswith(".virtual")
                and any(d.ep_name == ep_name for d in ort.get_ep_devices())
            )
            if "already registered" not in str(e) and not ep_already_exposed:
                raise
            owns_registration = False
            if "already registered" in str(e):
                logger.debug("Execution provider %s already registered, reusing it.", registration_name)
            else:
                logger.warning(
                    "Failed to register execution provider library '%s' from '%s': %s. Reusing the already exposed"
                    " '%s' execution provider instead.",
                    registration_name,
                    library_path,
                    e,
                    ep_name,
                )
        try:
            yield
        finally:
            if owns_registration:
                ort.unregister_execution_provider_library(registration_name)


class EPContextBinaryGenerator(Pass):
    """Generate EP specific context binary for the model."""

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "embed_context": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to embed context bin into the model.",
            ),
            "weight_sharing": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Enable weight sharing between QNN components or paired OpenVINO context/iterator models."
                    " OpenVINO requires compile_flow='jit' and embed_context=False."
                ),
            ),
            "provider_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Provider options for the EP.",
            ),
            "session_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Session options for the EP. Keys are added as session config entries (e.g."
                    " 'ep.context_embed_mode', 'ep.context_file_path'), except log_severity_level and"
                    " log_verbosity_level which are set as typed attributes."
                ),
            ),
            "disable_cpu_fallback": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to disable CPU fallback.",
            ),
            "update_genai_config": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Update an ORT GenAI role or OpenVINO LLM pipeline with generated context model and provider settings."
                ),
            ),
            "genai_config_role": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Top-level model role in genai_config.json to update when update_genai_config is enabled."
                ),
            ),
            "model_compiler_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Plugin EP only. Options forwarded to onnxruntime.ModelCompiler's constructor:"
                    " 'embed_compiled_data_into_model' (bool), 'external_initializers_file_path' (str),"
                    " 'external_initializers_size_threshold' (int), 'graph_optimization_level' (an"
                    " ort.GraphOptimizationLevel name, e.g. ORT_DISABLE_ALL), and 'flags' (an ort.OrtCompileApiFlags"
                    " name or '|'-joined names, e.g. ERROR_IF_NO_NODES_COMPILED)."
                ),
            ),
            "ep_device_filters": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Filters to select a specific OrtEpDevice when the plugin exposes multiple devices of the"
                    " accelerator's device type. Keys are matched against each device's ep_metadata, except"
                    " 'device_id', which is matched against the hardware device id (hex string or integer)."
                ),
            ),
            "target_devices": PassConfigParam(
                type_=list,
                default_value=None,
                description=(
                    "Device ids (hex strings such as 'B03E' or '0xB03E', or integers) to compile the model for, used"
                    " to compile for multiple devices in one run. Each id must be a device of the accelerator's"
                    " device type. Produces a composite model with one component per device, each in its own"
                    " '<DEVICE_ID>' directory (uppercase hex, no prefix), that can be fed to ORTModelPackage."
                    " Mutually exclusive with ep_device_filters."
                ),
            ),
            "ep_registration_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Plugin EP only. Name used to register the EP library with ONNX Runtime. Defaults to the"
                    " plugin's EP name. Some plugins change the exposed devices based on the name used for registration."
                ),
            ),
            "output_model_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Plugin EP only. File name for the generated EPContext model, written to the pass output"
                    " location. Defaults to '<input_model_stem>_ctx.onnx'."
                ),
            ),
            "compile_flow": PassConfigParam(
                type_=Literal["aot", "jit"],
                default_value="aot",
                description=(
                    "Plugin EP only. 'aot' compiles via ORT's ModelCompiler (Compile API, no session); 'jit'"
                    " compiles via an InferenceSession (the EPContext model is written when the session options"
                    " request it, e.g. ep.context_enable=1)."
                ),
            ),
        }

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Override this method to return False by using the accelerator spec information."""
        return False

    def _run_for_config(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        execution_provider = self.accelerator_spec.execution_provider
        if execution_provider == ExecutionProvider.OpenVINOExecutionProvider:
            if (
                isinstance(model, CompositeModelHandler)
                and (model.model_attributes or {}).get("llm_pipeline") is not None
            ):
                return self._run_openvino_llm_pipeline(model, config, output_model_path)
            return self._run_openvino(model, config, output_model_path)

        # TODO(jambayk): validate and support other NPU EPs
        assert execution_provider == ExecutionProvider.QNNExecutionProvider, (
            "Only QNNExecutionProvider and OpenVINOExecutionProvider are supported for now."
        )
        return self._run_qnn(model, config, output_model_path)

    def _run_qnn(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        """Generate EPContext binaries for the QNN EP."""
        # session created using providers argument so will use the ort.get_available_providers()
        # TODO(jambayk): consider switching to the new EP API for Windows
        from onnxruntime import __version__ as OrtVersion
        from onnxruntime import get_available_providers

        if version.parse(OrtVersion).release <= version.parse("1.23.2").release:
            assert self.accelerator_spec.execution_provider in get_available_providers(), (
                f"Execution provider {self.accelerator_spec.execution_provider} is not available. Available providers:"
                f" {get_available_providers()}"
            )

        result = self._run_single_target(model, config, output_model_path)

        # Populate model_attributes with context binary metadata so it persists in model_config.json
        result.model_attributes = {**(model.model_attributes or {}), **(result.model_attributes or {})}
        result.model_attributes["ep"] = self.accelerator_spec.execution_provider
        result.model_attributes["device"] = str(self.accelerator_spec.accelerator_type).upper()
        if config.provider_options:
            result.model_attributes["provider_options"] = config.provider_options
            result.model_attributes["architecture"] = config.provider_options.get("soc_model")
        if config.update_genai_config:
            if not config.genai_config_role:
                raise ValueError("genai_config_role is required when update_genai_config is enabled.")
            if not isinstance(result, ONNXModelHandler):
                raise ValueError("update_genai_config only supports single-model EPContext outputs.")
            self._update_single_model_genai_config(model, result, config)

        return result

    def _update_single_model_genai_config(
        self,
        source_model: Union[ONNXModelHandler, CompositeModelHandler],
        output_model: ONNXModelHandler,
        config: type[BasePassConfig],
    ) -> None:
        role = config.genai_config_role
        if "/" in role:
            raise ValueError(f"Invalid genai_config_role: {role!r}")

        config_paths = [
            Path(path)
            for path in (source_model.model_attributes or {}).get("additional_files") or []
            if Path(path).name == "genai_config.json"
        ]
        if len(config_paths) != 1:
            raise ValueError("update_genai_config requires exactly one genai_config.json additional file.")

        with config_paths[0].open(encoding="utf-8") as config_file:
            genai_config = json.load(config_file)
        role_config = genai_config.get("model", {}).get(role)
        if not isinstance(role_config, dict):
            raise ValueError(f"genai_config.json does not define model role {role!r}.")

        provider_options = deepcopy(config.provider_options or {})
        if self.accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider:
            backend_name = "libQnnHtp.so" if platform.system() == "Linux" else "QnnHtp.dll"
            provider_options.setdefault("backend_path", backend_name)
        provider_name = self.accelerator_spec.execution_provider.lower().replace("executionprovider", "")
        session_options = deepcopy(role_config.get("session_options") or {})
        session_options.update(deepcopy(config.session_options or {}))
        session_options["provider_options"] = [{provider_name: provider_options}]

        role_config["filename"] = Path(output_model.model_path).name
        role_config["session_options"] = session_options

        output_config_path = Path(output_model.model_path).parent / "genai_config.json"
        with output_config_path.open("w", encoding="utf-8") as config_file:
            json.dump(genai_config, config_file, indent=4)

        output_model.model_attributes = register_ort_genai_config_update(
            output_model.model_attributes,
            output_config_path.name,
            [f"/model/{role}"],
        )
        additional_files = set(output_model.model_attributes.get("additional_files") or [])
        additional_files.add(str(output_config_path))
        output_model.model_attributes["additional_files"] = sorted(additional_files)

    def _run_openvino(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
        update_genai_config: bool = True,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        """Generate EPContext ONNX model(s) for the OpenVINO plugin EP (aot: ModelCompiler, jit: InferenceSession)."""
        if isinstance(model, CompositeModelHandler):
            raise NotImplementedError("OpenVINO EPContext generation does not yet support composite models.")
        if config.weight_sharing:
            raise ValueError("weight_sharing is not supported for OpenVINO EPContext generation.")
        if config.genai_config_role and not config.update_genai_config:
            raise ValueError(
                "update_genai_config and genai_config_role are only supported for QNNExecutionProvider; they would"
                " be ignored for OpenVINO EPContext generation."
            )
        if config.update_genai_config and not config.genai_config_role:
            raise ValueError("genai_config_role is required when update_genai_config is enabled.")
        if config.compile_flow == "jit" and config.model_compiler_options:
            raise ValueError("model_compiler_options are only used with compile_flow 'aot'.")
        if config.target_devices and config.update_genai_config:
            raise ValueError("update_genai_config is not supported with target_devices.")

        # write the epctx model into the pass output location (matches the QNN path)
        input_model_path = Path(model.model_path)
        output_model_name = config.output_model_name or f"{input_model_path.stem}_ctx.onnx"
        session_options, model_compiler_options = self._get_openvino_compile_options(config)

        if config.target_devices:
            if config.ep_device_filters:
                raise ValueError("target_devices and ep_device_filters are mutually exclusive.")
            return self._run_openvino_targets(
                model, config, output_model_path, output_model_name, session_options, model_compiler_options
            )

        result = self._generate_plugin_ep_context_binary(
            ep_package_name=_OPENVINO_EP_PLUGIN_PACKAGE,
            model_path=model.model_path,
            output_model_path=resolve_onnx_path(output_model_path, output_model_name),
            compile_flow=config.compile_flow,
            provider_options=deepcopy(config.provider_options or {}),
            session_options=session_options,
            model_compiler_options=model_compiler_options,
            ep_device_filters=config.ep_device_filters,
            ep_registration_name=config.ep_registration_name,
            device=str(self.accelerator_spec.accelerator_type).lower(),
            prefer_ep_devices=self._prefer_openvino_virtual_device,
        )

        result.model_attributes = {**(model.model_attributes or {}), **(result.model_attributes or {})}
        result.model_attributes["ep"] = self.accelerator_spec.execution_provider
        result.model_attributes["device"] = str(self.accelerator_spec.accelerator_type).upper()
        # weightless EPContext models read their weights from the source model (consumed by ORTModelPackage)
        result.model_attributes["source_model_path"] = str(model.model_path)
        if config.provider_options:
            result.model_attributes["provider_options"] = config.provider_options
        if config.ep_device_filters:
            result.model_attributes["ep_device_filters"] = config.ep_device_filters
        if config.update_genai_config and update_genai_config:
            self._update_single_model_genai_config(model, result, config)

        return result

    @staticmethod
    def _openvino_llm_component_names(model: CompositeModelHandler) -> list[str]:
        pipeline = (model.model_attributes or {}).get("llm_pipeline")
        if not isinstance(pipeline, dict) or any(
            not isinstance(pipeline.get(group), list) or not pipeline[group] for group in ("context", "iterator")
        ):
            raise ValueError("llm_pipeline must define non-empty context and iterator component lists.")
        names = [*pipeline["context"], *pipeline["iterator"]]
        lm_head = pipeline.get("lm_head")
        embeddings = pipeline.get("embeddings")
        if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("llm_pipeline must use distinct context and iterator component names.")
        if not isinstance(lm_head, str) or lm_head not in model.model_component_names or lm_head in names:
            raise ValueError("llm_pipeline must define a distinct lm_head component.")
        if embeddings is not None and (
            not isinstance(embeddings, str)
            or embeddings not in model.model_component_names
            or embeddings in names
            or embeddings == lm_head
        ):
            raise ValueError("llm_pipeline must use distinct context, iterator, embeddings, and lm_head components.")
        return names

    def _run_openvino_llm_pipeline(
        self, model: CompositeModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> CompositeModelHandler:
        if config.target_devices:
            raise ValueError("target_devices are not supported for OpenVINO LLM pipeline composites.")
        if config.genai_config_role and not config.update_genai_config:
            raise ValueError(
                "update_genai_config and genai_config_role are only supported for QNNExecutionProvider; they would"
                " be ignored for OpenVINO EPContext generation."
            )

        selected_names = self._openvino_llm_component_names(model)
        pipeline = model.model_attributes["llm_pipeline"]
        session_options, _ = self._get_openvino_compile_options(config)
        if config.weight_sharing:
            if len(pipeline["context"]) != len(pipeline["iterator"]):
                raise ValueError("OpenVINO weight_sharing requires matching context and iterator component counts.")
            if config.compile_flow != "jit":
                raise ValueError("OpenVINO weight_sharing requires compile_flow='jit'.")
            if session_options["ep.context_embed_mode"] != "0":
                raise ValueError("OpenVINO weight_sharing requires embed_context=False.")
            if config.model_compiler_options:
                raise ValueError("model_compiler_options are only supported with compile_flow='aot'.")
            if str((config.session_options or {}).get("ep.enable_weightless", "0")).lower() in ("1", "true"):
                raise ValueError("OpenVINO weight_sharing cannot be combined with ep.enable_weightless.")
        components = dict(deepcopy(model).get_model_components())
        for name in selected_names:
            if name not in components:
                raise ValueError(f"Unknown component {name!r}. Available components: {list(components)}.")
        if not all(isinstance(component, ONNXModelHandler) for component in components.values()):
            raise ValueError("OpenVINO composite inputs must contain only ONNX models.")
        if config.update_genai_config:
            if not config.genai_config_role:
                raise ValueError("genai_config_role is required when update_genai_config is enabled.")
            if config.genai_config_role != "decoder":
                raise ValueError("LLM pipeline GenAI updates require genai_config_role='decoder'.")
            self._load_openvino_llm_genai_config(model)

        output_dir = Path(output_model_path).with_suffix("")
        if config.output_model_name:
            raise ValueError("output_model_name cannot be set when compiling multiple LLM pipeline components.")
        output_names = [
            f"{Path(component.model_path).stem}_ctx.onnx" if name in selected_names else Path(component.model_path).name
            for name, component in components.items()
        ]
        if len({name.casefold() for name in output_names}) != len(output_names):
            raise ValueError("Composite output ONNX filenames must be unique.")

        shared_models = (
            self._run_openvino_shared(model, components, config, output_dir) if config.weight_sharing else {}
        )
        for name, component in components.items():
            if name in selected_names:
                compiled = (
                    shared_models[name]
                    if config.weight_sharing
                    else self._run_openvino(component, config, str(output_dir), update_genai_config=False)
                )
                compiled.model_attributes = {**(component.model_attributes or {}), **(compiled.model_attributes or {})}
                if config.weight_sharing:
                    compiled.model_attributes["ep"] = self.accelerator_spec.execution_provider
                    compiled.model_attributes["device"] = str(self.accelerator_spec.accelerator_type).upper()
                    if config.provider_options:
                        compiled.model_attributes["provider_options"] = config.provider_options
                    if config.ep_device_filters:
                        compiled.model_attributes["ep_device_filters"] = config.ep_device_filters
                components[name] = compiled
            else:
                destination = output_dir / Path(component.model_path).name
                resave_model(component.model_path, destination)
                for sidecar in (component.external_initializers_path, component.constant_inputs_path):
                    if sidecar:
                        hardlink_copy_file(sidecar, output_dir / Path(sidecar).name)
                component.set_resource("model_path", str(output_dir))
                component.onnx_file_name = destination.name
            self._carry_forward_additional_files(component, components[name])
        result = CompositeModelHandler(
            list(components.values()),
            list(components),
            model_path=output_dir,
            model_attributes=deepcopy(model.model_attributes),
        )
        if config.update_genai_config:
            self._update_openvino_llm_genai_config(result, config)
        return result

    def _run_openvino_shared(
        self,
        model: CompositeModelHandler,
        components: dict[str, ONNXModelHandler],
        config: type[BasePassConfig],
        output_dir: Path,
    ) -> dict[str, ONNXModelHandler]:
        import onnxruntime as ort
        import onnxruntime_ep_openvino as ep_package

        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Shared EPContext output directory must be empty: {output_dir}.")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        pipeline = model.model_attributes["llm_pipeline"]
        selected_names = [*pipeline["context"], *pipeline["iterator"]]
        registration_name = config.ep_registration_name or ep_package.get_ep_name()
        reserved_names = {"genai_config.json"}
        for name, component in components.items():
            filename = Path(component.model_path).name
            if name in selected_names:
                reserved_names.add(f"{Path(filename).stem}_ctx.onnx".casefold())
            else:
                reserved_names.update((filename.casefold(), f"{filename}.data".casefold()))
        weight_sources = {}
        shared_models = {}
        session_options, _ = self._get_openvino_compile_options(config)
        with TemporaryDirectory(prefix="ov_shared_", dir=output_dir.parent) as temporary:
            staging_dir = Path(temporary)
            for name in selected_names:
                source_dir = Path(components[name].model_path).parent
                for location in get_external_data_file_names(components[name].model_path):
                    relative_path = Path(location)
                    if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
                        raise ValueError("Shared OpenVINO weights must use relative paths within the model directory.")
                    if relative_path.parts[0].casefold() in reserved_names:
                        raise ValueError(f"External weights conflict with a composite output filename: {location!r}.")
                    source = source_dir / relative_path
                    if relative_path in weight_sources:
                        if not source.samefile(weight_sources[relative_path]):
                            raise ValueError(f"Different external weights use the same relative path: {location!r}.")
                        continue
                    destination = staging_dir / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    hardlink_copy_file(source, destination)
                    weight_sources[relative_path] = source

            for context_name, iterator_name in zip(pipeline["context"], pipeline["iterator"]):
                sessions = []
                with registered_ep_library(ort, registration_name, ep_package.get_library_path()):
                    try:
                        pair_models = []
                        for index, name in enumerate((context_name, iterator_name)):
                            options = {
                                **session_options,
                                "ep.share_ep_contexts": "1",
                                "ep.stop_share_ep_contexts": str(index),
                            }
                            output_name = f"{Path(components[name].model_path).stem}_ctx.onnx"
                            compiled = self._generate_plugin_ep_context_binary(
                                ep_package_name=_OPENVINO_EP_PLUGIN_PACKAGE,
                                model_path=components[name].model_path,
                                output_model_path=staging_dir / output_name,
                                device=self.accelerator_spec.accelerator_type,
                                compile_flow="jit",
                                provider_options=config.provider_options,
                                session_options=options,
                                ep_device_filters=config.ep_device_filters,
                                ep_registration_name=config.ep_registration_name,
                                shared_sessions=sessions,
                                defer_binary_validation=True,
                            )
                            pair_models.append(compiled)
                            shared_models[name] = compiled
                        binaries = [
                            self._validate_plugin_ep_context(Path(compiled.model_path)) for compiled in pair_models
                        ]
                        if not binaries[0] or binaries[0] != binaries[1]:
                            raise RuntimeError(
                                "OpenVINO did not generate a common binary for the context/iterator pair."
                            )
                    finally:
                        sessions.clear()

            output_dir.mkdir(parents=True, exist_ok=True)
            staged_files = list(staging_dir.iterdir())
            if any((output_dir / path.name).exists() for path in staged_files):
                raise FileExistsError(f"Shared EPContext output already exists at {output_dir}.")
            for path in staged_files:
                path.replace(output_dir / path.name)
        weight_paths = sorted({str(output_dir / path.parts[0]) for path in weight_sources})
        for compiled in shared_models.values():
            compiled.set_resource("model_path", output_dir)
            compiled.model_attributes = {"additional_files": weight_paths}
        return shared_models

    @staticmethod
    def _load_openvino_llm_genai_config(model: CompositeModelHandler) -> tuple[Path, dict]:
        paths = [
            Path(path)
            for path in (model.model_attributes or {}).get("additional_files") or []
            if Path(path).name == "genai_config.json"
        ]
        if len(paths) != 1:
            raise ValueError("update_genai_config requires exactly one genai_config.json additional file.")
        with paths[0].open(encoding="utf-8") as config_file:
            genai_config = json.load(config_file)
        pipeline = genai_config.get("model", {}).get("decoder", {}).get("pipeline")
        if (
            not isinstance(pipeline, list)
            or len(pipeline) != 1
            or not isinstance(pipeline[0], dict)
            or any(not isinstance(pipeline[0].get(name), dict) for name in model.model_component_names)
        ):
            raise ValueError("genai_config.json must define decoder pipeline stages for all composite components.")
        return paths[0], genai_config

    def _update_openvino_llm_genai_config(self, model: CompositeModelHandler, config: type[BasePassConfig]) -> None:
        selected_names = self._openvino_llm_component_names(model)
        source_path, genai_config = self._load_openvino_llm_genai_config(model)
        stages = genai_config["model"]["decoder"]["pipeline"][0]
        for name, component in model.get_model_components():
            stage = stages[name]
            stage["filename"] = Path(component.model_path).name
            if name not in selected_names:
                continue
            io_config = component.io_config
            stage["inputs"] = io_config["input_names"]
            stage["outputs"] = io_config["output_names"]
            session_options = deepcopy(stage.get("session_options") or {})
            session_options.update(deepcopy(config.session_options or {}))
            session_options["provider_options"] = [{"OpenVINO": deepcopy(config.provider_options or {})}]
            if config.weight_sharing:
                session_options["ep.share_ep_contexts"] = "1"
                session_options.pop("ep.stop_share_ep_contexts", None)
            stage["session_options"] = session_options

        output_path = Path(model.model_path) / "genai_config.json"
        with output_path.open("w", encoding="utf-8") as config_file:
            json.dump(genai_config, config_file, indent=4)
        model.model_attributes = register_ort_genai_config_update(
            model.model_attributes, output_path.name, ["/model/decoder"]
        )
        additional_files = set(model.model_attributes.get("additional_files") or [])
        additional_files.discard(str(source_path))
        additional_files.add(str(output_path))
        model.model_attributes["additional_files"] = sorted(additional_files)

    @staticmethod
    def _get_openvino_compile_options(config: type[BasePassConfig]) -> tuple[dict, dict]:
        """Translate the pass-level EPContext options to OpenVINO plugin compilation options.

        ``embed_context`` / ``disable_cpu_fallback`` and the matching ``session_options`` / ``model_compiler_options``
        keys all default to off; a setting is on if any of them turns it on. The pass-level flag wins over an explicit
        "off" in the low-level options.
        """
        session_options = deepcopy(config.session_options or {})
        model_compiler_options = deepcopy(config.model_compiler_options or {})
        is_aot = config.compile_flow == "aot"

        embed_explicit = {}
        if "ep.context_embed_mode" in session_options:
            embed_explicit["session_options['ep.context_embed_mode']"] = session_options["ep.context_embed_mode"]
        if is_aot and "embed_compiled_data_into_model" in model_compiler_options:
            embed_explicit["model_compiler_options['embed_compiled_data_into_model']"] = model_compiler_options[
                "embed_compiled_data_into_model"
            ]
        embed = EPContextBinaryGenerator._resolve_bool_option("embed_context", config.embed_context, embed_explicit)

        fallback_explicit = {}
        if "session.disable_cpu_ep_fallback" in session_options:
            fallback_explicit["session_options['session.disable_cpu_ep_fallback']"] = session_options[
                "session.disable_cpu_ep_fallback"
            ]
        disable_cpu_fallback = EPContextBinaryGenerator._resolve_bool_option(
            "disable_cpu_fallback", config.disable_cpu_fallback, fallback_explicit
        )

        session_options["ep.context_enable"] = "1"
        session_options["ep.context_embed_mode"] = str(int(embed))
        session_options["session.disable_cpu_ep_fallback"] = str(int(disable_cpu_fallback))
        if is_aot:
            model_compiler_options["embed_compiled_data_into_model"] = embed
        return session_options, model_compiler_options

    @staticmethod
    def _resolve_bool_option(pass_option: str, pass_value: bool, explicit: dict[str, Any]) -> bool:
        """Resolve a boolean setting that can come from the pass flag or from low-level options.

        All sources default to off, so the setting is on if ANY of them turns it on. The pass-level flag
        (``pass_value``) is the highest-priority source: when it is on, it wins over an explicit low-level "off"
        and a warning is logged about the ignored value.
        """
        parsed = {}
        for key, value in explicit.items():
            normalized = str(value).strip().lower()
            if normalized not in ("0", "1", "true", "false"):
                raise ValueError(f"{key} must be a boolean or '0'/'1', got {value!r}.")
            parsed[key] = normalized in ("1", "true")

        resolved = pass_value or any(parsed.values())
        if resolved:
            ignored = [key for key, value in parsed.items() if not value]
            if ignored:
                logger.warning("%s is enabled, so the explicit setting(s) %s are overridden.", pass_option, ignored)
        return resolved

    def _run_openvino_targets(
        self,
        model: ONNXModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
        output_model_name: str,
        session_options: dict,
        model_compiler_options: dict,
    ) -> CompositeModelHandler:
        """Generate one EPContext model per target device, each in its own '<device_id>' directory."""
        target_device_ids = list(dict.fromkeys(self._normalize_device_id(d) for d in config.target_devices))
        filters_by_device = self._resolve_openvino_device_filters(
            config.ep_registration_name, target_device_ids, str(self.accelerator_spec.accelerator_type).lower()
        )

        output_root = Path(output_model_path).with_suffix("")
        device = str(self.accelerator_spec.accelerator_type).upper()
        components = []
        for device_id, ep_device_filters in filters_by_device.items():
            device_dir = output_root / device_id
            component = self._generate_plugin_ep_context_binary(
                ep_package_name=_OPENVINO_EP_PLUGIN_PACKAGE,
                model_path=model.model_path,
                output_model_path=device_dir / output_model_name,
                compile_flow=config.compile_flow,
                provider_options=deepcopy(config.provider_options or {}),
                session_options=deepcopy(session_options),
                model_compiler_options=deepcopy(model_compiler_options),
                ep_device_filters=ep_device_filters,
                ep_registration_name=config.ep_registration_name,
                device=device,
                prefer_ep_devices=self._prefer_openvino_virtual_device,
                ep_context_binary_information=(
                    (device_dir, Path(model.model_path).stem) if config.compile_flow == "aot" else None
                ),
            )
            # same compile metadata as the single-target path, plus the filter that selected this device
            component.model_attributes = {
                "ep": self.accelerator_spec.execution_provider,
                "device": device,
                "device_id": device_id,
                "ep_device_filters": ep_device_filters,
            }
            if config.provider_options:
                component.model_attributes["provider_options"] = deepcopy(config.provider_options)
            components.append(component)

        composite_attributes = {
            **(model.model_attributes or {}),
            "no_flatten": True,
            "ep": self.accelerator_spec.execution_provider,
            "device": device,
            # weightless EPContext models read their weights from the source model
            "source_model_path": str(model.model_path),
        }
        if config.provider_options:
            composite_attributes["provider_options"] = deepcopy(config.provider_options)
        return CompositeModelHandler(
            components, list(filters_by_device), model_path=output_root, model_attributes=composite_attributes
        )

    @classmethod
    def _resolve_openvino_device_filters(
        cls, ep_registration_name: Optional[str], target_device_ids: list[str], device: str
    ) -> dict[str, dict]:
        """Map each device id to ep_device_filters that select exactly one OrtEpDevice of the given device type.

        Virtual and physical devices can share a device id; the filter then also pins the 'npu_platform' of the virtual
        device so the physical one is not selected.
        """
        import onnxruntime as ort

        cls._check_plugin_ep_ort_version(ort.__version__)
        ep_package = cls._import_ep_package(_OPENVINO_EP_PLUGIN_PACKAGE)
        ep_name = ep_package.get_ep_name()
        device_type = get_ort_hardware_device_type(device)
        wanted = set(target_device_ids)
        discovered: dict[str, Optional[str]] = {}
        exposed_ids = set()
        with registered_ep_library(
            ort, ep_registration_name or ep_name, ep_package.get_library_path(), ep_name=ep_name
        ):
            for ep_device in ort.get_ep_devices():
                if ep_device.ep_name != ep_name:
                    continue
                device_id = cls._normalize_device_id(ep_device.device.device_id)
                exposed_ids.add(f"{device_id} ({ep_device.device.type})")
                if device_id not in wanted or ep_device.device.type != device_type:
                    continue
                npu_platform = dict(ep_device.ep_metadata).get("npu_platform")
                if device_id not in discovered or (npu_platform and not discovered[device_id]):
                    discovered[device_id] = npu_platform

        missing = [device_id for device_id in target_device_ids if device_id not in discovered]
        if missing:
            raise ValueError(
                f"Requested OpenVINO {device.upper()} target devices were not discovered: {missing}. Discovered:"
                f" {sorted(exposed_ids)}."
            )
        # always pin the device id; 'npu_platform' additionally separates a virtual device from a physical one
        return {
            device_id: (
                {"device_id": device_id, "npu_platform": discovered[device_id]}
                if discovered[device_id]
                else {"device_id": device_id}
            )
            for device_id in target_device_ids
        }

    @staticmethod
    def _check_plugin_ep_ort_version(ort_version: str) -> None:
        # Compare the release segment so dev/rc builds of the minimum version are accepted.
        if version.parse(ort_version).release < _MINIMUM_PLUGIN_EP_ORT_VERSION.release:
            raise RuntimeError(
                f"Plugin EP compilation requires ONNX Runtime >= {_MINIMUM_PLUGIN_EP_ORT_VERSION}; found {ort_version}."
            )

    @staticmethod
    def _check_weightless_ort_version(ort_version: str, session_options: dict) -> None:
        # Older releases silently ignore the unknown key, so weights would stay in every compiled model.
        if (
            _WEIGHTLESS_SESSION_OPTION in session_options
            and version.parse(ort_version).release < _MINIMUM_WEIGHTLESS_ORT_VERSION.release
        ):
            raise ValueError(
                f"'{_WEIGHTLESS_SESSION_OPTION}' requires ONNX Runtime >= {_MINIMUM_WEIGHTLESS_ORT_VERSION}; found"
                f" {ort_version}. Older releases use 'ep.enable_weightless_ep_context_nodes' instead."
            )

    @staticmethod
    def _normalize_device_id(device_id: Union[str, int]) -> str:
        """Return a hardware device id as uppercase hex without prefix or leading zeros, e.g. '0x00b03e' -> 'B03E'."""
        if isinstance(device_id, int) and not isinstance(device_id, bool) and device_id >= 0:
            return f"{device_id:X}"
        # int(..., 16) alone would also accept underscores and signs
        if isinstance(device_id, str) and _HEX_DEVICE_ID.fullmatch(device_id.strip()):
            return f"{int(device_id.strip(), 16):X}"
        raise ValueError(f"Invalid device id {device_id!r}; expected a hex string such as 'B03E' or an integer.")

    @staticmethod
    def _import_ep_package(ep_package_name: str):
        try:
            return importlib.import_module(ep_package_name)
        except ImportError as e:
            raise ImportError(
                f"The plugin EP package '{ep_package_name}' is required. Install it with"
                f" 'pip install {ep_package_name.replace('_', '-')}'."
            ) from e

    @classmethod
    def _ep_device_matches(cls, ep_device, filters: dict) -> bool:
        """Match an OrtEpDevice against filters, checking ep_metadata and the hardware device_id (hex or int)."""
        for key, value in filters.items():
            if key == "device_id":
                if cls._normalize_device_id(ep_device.device.device_id) != cls._normalize_device_id(value):
                    return False
            elif key not in ep_device.ep_metadata or str(ep_device.ep_metadata[key]) != str(value):
                return False
        return True

    @classmethod
    def _generate_plugin_ep_context_binary(
        cls,
        ep_package_name: str,
        model_path: str,
        output_model_path: Union[str, Path],
        compile_flow: str,
        provider_options: Optional[dict] = None,
        session_options: Optional[dict] = None,
        model_compiler_options: Optional[dict] = None,
        ep_device_filters: Optional[dict] = None,
        ep_registration_name: Optional[str] = None,
        ep_context_binary_information: Optional[tuple[Union[str, Path], str]] = None,
        device: Optional[str] = None,
        prefer_ep_devices: Optional[Callable[[list], list]] = None,
        shared_sessions: Optional[list] = None,
        defer_binary_validation: bool = False,
    ) -> ONNXModelHandler:
        """Generate an EPContext model with a plugin EP. EP-agnostic: a new EP only needs its plugin package and options.

        compile_flow 'aot' compiles via ORT's ModelCompiler (no inference); 'jit' creates an InferenceSession that
        compiles and writes the EPContext model. The given options are passed to ORT as-is; this method adds no
        EPContext keys itself. Callers map the pass-level flags beforehand (see ``_get_openvino_compile_options``).
        When ``device`` is given, only OrtEpDevices of that hardware type are considered. If compilation or validation
        fails, the files written by this call are removed so the same output path can be retried.
        """
        import onnxruntime as ort

        cls._check_plugin_ep_ort_version(ort.__version__)
        session_options = session_options or {}
        model_compiler_options = model_compiler_options or {}
        # validate the flow before registering the plugin or touching the output location
        if compile_flow not in ("aot", "jit"):
            raise ValueError(f"Unknown compile_flow '{compile_flow}'. Expected 'aot' or 'jit'.")
        if compile_flow == "jit" and model_compiler_options:
            raise ValueError("model_compiler_options are only used with compile_flow 'aot'.")
        if compile_flow == "jit" and ep_context_binary_information:
            raise ValueError("ep_context_binary_information is only used with compile_flow 'aot'.")
        cls._check_weightless_ort_version(ort.__version__, session_options)
        ep_package = cls._import_ep_package(ep_package_name)
        ep_name = ep_package.get_ep_name()
        registration_name = ep_registration_name or ep_name

        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)

        # refuse to overwrite an existing epctx model from a previous run
        # binaries are named after the output model, or after the binary name set via ep_context_binary_information
        bin_stems = {output_model_path.stem}
        if compile_flow == "aot" and ep_context_binary_information:
            bin_stems.add(ep_context_binary_information[1])
        existing = [output_model_path] if output_model_path.exists() else []
        for stem in sorted(bin_stems):
            existing += list(output_model_path.parent.glob(f"{stem}*.bin"))
        if existing:
            raise FileExistsError(
                f"EPContext file already exists: {output_model_path}. Remove it (and its .bin files) or choose a"
                " different output_model_name before re-running."
            )

        # set session options as ORT session config entries
        sess_options = ort.SessionOptions()
        cls._apply_session_options(sess_options, session_options)

        completed = False
        try:
            registration = (
                nullcontext()
                if shared_sessions is not None
                else registered_ep_library(ort, registration_name, ep_package.get_library_path(), ep_name=ep_name)
            )
            with registration:
                selected = cls._select_ep_device(ort, ep_name, ep_device_filters, device, prefer_ep_devices)
                ep_provider_options = {str(k): str(v) for k, v in (provider_options or {}).items()}
                sess_options.add_provider_for_devices([selected], ep_provider_options)

                if compile_flow == "aot":
                    compiler_kwargs = cls._build_model_compiler_kwargs(ort, model_compiler_options)
                    compiler = ort.ModelCompiler(sess_options, str(model_path), **compiler_kwargs)
                    if ep_context_binary_information:
                        cls._set_ep_context_binary_information(compiler, *ep_context_binary_information)
                    compiler.compile_to_file(str(output_model_path))
                else:
                    sess_options.add_session_config_entry("ep.context_file_path", str(output_model_path))
                    session = ort.InferenceSession(str(model_path), sess_options=sess_options)
                    if shared_sessions is not None:
                        shared_sessions.append(session)
                    del session

            if defer_binary_validation:
                cls._validate_plugin_ep_context(output_model_path, defer_binary_validation=True)
            else:
                cls._validate_ep_context_model(output_model_path)
            completed = True
        finally:
            if not completed:
                # remove the partial outputs of this call so the same output path can be retried; anything that
                # matches was written by this call, since the FileExistsError check above found nothing
                output_model_path.unlink(missing_ok=True)
                for stem in sorted(bin_stems):
                    for binary_path in output_model_path.parent.glob(f"{stem}*.bin"):
                        binary_path.unlink(missing_ok=True)

        # return the stub with dir + file name so only the epctx model and its .bin are saved
        return ONNXModelHandler(
            model_path=output_model_path.parent,
            onnx_file_name=output_model_path.name,
        )

    @staticmethod
    def _validate_plugin_ep_context(output_model_path: Path, defer_binary_validation: bool = False) -> set[str]:
        import onnx

        if not output_model_path.is_file():
            raise FileNotFoundError(f"EPContext model was not generated at {output_model_path}.")
        graphs = [onnx.load(output_model_path, load_external_data=False).graph]
        has_context = False
        binaries = set()
        while graphs:
            for node in graphs.pop().node:
                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.GRAPH:
                        graphs.append(attribute.g)
                    elif attribute.type == onnx.AttributeProto.GRAPHS:
                        graphs.extend(attribute.graphs)
                if node.op_type != "EPContext":
                    continue
                has_context = True
                attributes = {attribute.name: attribute for attribute in node.attribute}
                embed_mode = attributes.get("embed_mode")
                if embed_mode is None or embed_mode.i:
                    continue
                context = attributes.get("ep_cache_context")
                if context is None or not context.s:
                    continue
                context_file = context.s.decode("utf-8")
                binaries.add(context_file)
                if not defer_binary_validation and not (output_model_path.parent / context_file).is_file():
                    raise FileNotFoundError(
                        f"EPContext binary {context_file!r} was not generated for {output_model_path}."
                    )
        if not has_context:
            raise RuntimeError(f"No EPContext nodes were generated in {output_model_path}.")
        return binaries

    @staticmethod
    def _prefer_openvino_virtual_device(candidates: list) -> list:
        """Prefer the virtual NPU when it shares a device id with a physical one (only it has 'npu_platform')."""
        virtual = [d for d in candidates if "npu_platform" in d.ep_metadata]
        if len(virtual) != 1:
            return candidates
        logger.info(
            "Multiple OpenVINO devices matched; selecting the virtual device with npu_platform=%s. Set"
            " ep_device_filters to select a different device.",
            virtual[0].ep_metadata["npu_platform"],
        )
        return virtual

    @classmethod
    def _select_ep_device(
        cls,
        ort,
        ep_name: str,
        ep_device_filters: Optional[dict],
        device: Optional[str],
        prefer_ep_devices: Optional[Callable[[list], list]] = None,
    ):
        """Return the single OrtEpDevice of ep_name that matches the device type and ep_device_filters.

        ``prefer_ep_devices`` lets the calling EP narrow several matching devices with EP-specific knowledge; the
        generic selection itself makes no EP-specific assumptions.
        """
        available_devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
        if not available_devices:
            raise RuntimeError(f"Execution provider {ep_name} is not available after registering its plugin.")
        selected = available_devices
        if device:
            device_type = get_ort_hardware_device_type(device)
            selected = [d for d in selected if d.device.type == device_type]
        if ep_device_filters:
            selected = [d for d in selected if cls._ep_device_matches(d, ep_device_filters)]
        if len(selected) > 1 and prefer_ep_devices:
            selected = prefer_ep_devices(selected)
        if len(selected) != 1:
            available = [
                (cls._normalize_device_id(d.device.device_id), str(d.device.type), dict(d.ep_metadata))
                for d in available_devices
            ]
            device_label = f"{device.upper()} " if device else ""
            raise ValueError(
                f"Expected exactly one {device_label}OrtEpDevice for {ep_name} matching filters {ep_device_filters},"
                f" found {len(selected)}. Set ep_device_filters to select the target device. Available: {available}"
            )
        return selected[0]

    @staticmethod
    def _set_ep_context_binary_information(compiler, binary_dir: Union[str, Path], binary_name: str) -> None:
        # the Python binding is only available from ORT 1.32 (microsoft/onnxruntime#33640)
        if not hasattr(compiler, "set_ep_context_binary_information"):
            logger.info(
                "ModelCompiler.set_ep_context_binary_information is not available in this ONNX Runtime build; the"
                " EPContext binary keeps the default name next to the output model."
            )
            return
        # ORT treats an extensionless path without a trailing separator as a file, not a directory
        compiler.set_ep_context_binary_information(os.path.join(binary_dir, ""), binary_name)

    @staticmethod
    def _validate_ep_context_model(model_path: Path) -> None:
        """Check that the generated model has EPContext nodes and that their non-embedded binaries exist."""
        import onnx

        if not model_path.is_file():
            raise FileNotFoundError(f"EPContext model was not generated at {model_path}.")

        graphs = [onnx.load(model_path, load_external_data=False).graph]
        has_ep_context = False
        while graphs:
            for node in graphs.pop().node:
                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.GRAPH:
                        graphs.append(attribute.g)
                    elif attribute.type == onnx.AttributeProto.GRAPHS:
                        graphs.extend(attribute.graphs)
                if node.op_type != "EPContext":
                    continue
                has_ep_context = True
                attributes = {attribute.name: attribute for attribute in node.attribute}
                # embed_mode defaults to 1; only decode ep_cache_context when it holds a file path, not binary data
                embed_mode = attributes.get("embed_mode")
                context = attributes.get("ep_cache_context")
                if embed_mode is None or embed_mode.i or context is None or not context.s:
                    continue
                context_file = context.s.decode("utf-8")
                if not (model_path.parent / context_file).is_file():
                    raise FileNotFoundError(f"EPContext binary {context_file!r} was not generated for {model_path}.")
        if not has_ep_context:
            raise RuntimeError(
                f"No EPContext nodes were generated in {model_path}. The execution provider did not compile any node."
            )

    @staticmethod
    def _apply_session_options(sess_options, session_options: dict) -> None:
        """Apply session options to an ORT SessionOptions.

        Keys are added as session config entries, except ``log_severity_level`` and ``log_verbosity_level``, which are
        typed attributes on SessionOptions; adding them as config entries would silently have no effect.
        """
        for key, value in session_options.items():
            if key in _TYPED_SESSION_OPTIONS:
                setattr(sess_options, key, int(value))
            else:
                sess_options.add_session_config_entry(key, str(value))

    @staticmethod
    def _build_model_compiler_kwargs(ort, model_compiler_options: Optional[dict]) -> dict:
        """Coerce 'model_compiler_options' into onnxruntime.ModelCompiler constructor kwargs.

        Accepted keys and their coercion are derived from the ModelCompiler.__init__ signature, so the pass tracks the
        ORT API automatically: the value is coerced to the type of the parameter's default. Enum params (e.g.
        graph_optimization_level, flags) take a member name, a '|'-joined set of names, or the integer value; bool/int
        params are validated strictly (e.g. 2 is not a bool and 3.7 is not an int); everything else is passed verbatim.
        """
        import inspect

        from pydantic import TypeAdapter

        opts = dict(model_compiler_options or {})
        if not opts:
            return {}

        params = inspect.signature(ort.ModelCompiler.__init__).parameters
        # positional model/session args and callback hooks aren't config values
        settable = {
            name: p.default
            for name, p in params.items()
            if name not in ("self", "sess_options", "input_model_path_or_bytes")
            and not name.endswith("_func")
            and p.default is not inspect.Parameter.empty
        }

        unknown = set(opts) - set(settable)
        if unknown:
            raise ValueError(
                f"Unsupported model_compiler_options keys: {sorted(unknown)}. Supported: {sorted(settable)}"
            )

        def coerce(key, default, value):
            enum_type = type(default)
            if hasattr(enum_type, "__members__"):  # pybind enum (GraphOptimizationLevel, OrtCompileApiFlags)
                if not isinstance(value, str):
                    return coerce_enum_value(key, enum_type, value)
                result = None
                for name in str(value).split("|"):
                    member_name = name.strip()
                    if member_name not in enum_type.__members__:
                        raise ValueError(
                            f"Unsupported {enum_type.__name__} member {member_name!r}. "
                            f"Supported: {sorted(enum_type.__members__)}"
                        )
                    member = enum_type.__members__[member_name]
                    result = member if result is None else result | member
                return result
            if isinstance(default, (bool, int)):
                # validates against the exact default type (bool or int); pydantic.ValidationError is a ValueError
                return TypeAdapter(type(default)).validate_python(value)
            return value  # str / path-like — passed verbatim

        def coerce_enum_value(key, enum_type, value):
            # JSON configs may carry the integer value of an ORT enum instead of its member name
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be a {enum_type.__name__} member name or integer value, got {value!r}.")
            member_values = [int(member) for member in enum_type.__members__.values()]
            if key == "flags":
                all_bits = 0
                for member_value in member_values:
                    all_bits |= member_value
                is_valid = value >= 0 and not value & ~all_bits
            else:
                is_valid = value in member_values
            if not is_valid:
                raise ValueError(
                    f"Unsupported {enum_type.__name__} value {value}. Supported: "
                    f"{ {name: int(member) for name, member in enum_type.__members__.items()} }"
                )
            return enum_type(value)

        return {key: coerce(key, settable[key], value) for key, value in opts.items()}

    def _run_single_target(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        """Generate context binary for a single target. This is the original logic."""
        from onnxruntime import __version__ as OrtVersion

        generate_kwargs = {
            "execution_provider": self.accelerator_spec.execution_provider,
            "provider_options": config.provider_options,
            "session_options": config.session_options,
            "embed_context": config.embed_context,
            "disable_cpu_fallback": config.disable_cpu_fallback,
            "model": model,
        }

        if (
            config.weight_sharing
            and self.accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider
            and str(self.accelerator_spec.accelerator_type).lower() == "gpu"
        ):
            assert isinstance(model, CompositeModelHandler), (
                "weight_sharing for QNN GPU is only supported for composite prefill/decode models generated by "
                "StaticLLM (model must be a CompositeModelHandler)."
            )

        if isinstance(model, ONNXModelHandler):
            return self._generate_context_binary(
                model_path=model.model_path,
                output_model_path=resolve_onnx_path(output_model_path, f"{Path(model.model_path).stem}_ctx.onnx"),
                device=self.accelerator_spec.accelerator_type,
                **generate_kwargs,
            )

        if config.weight_sharing and (version.parse(OrtVersion).release < version.parse("1.22.0").release):
            raise ValueError("weight sharing is only supported in onnxruntime >= 1.22.0")

        output_model_path = Path(output_model_path).with_suffix("")

        # name: model
        component_map = dict(model.get_model_components())

        new_component_models = {}
        new_model_attributes = deepcopy(model.model_attributes) or {}
        if pipeline := (model.model_attributes or {}).get("llm_pipeline"):

            def process_context_iterator(component_models, llm_pipeline, output_dir):
                new_groups = {
                    "context": {},
                    "iterator": {},
                }
                for ctx_model_name, iter_model_name in zip(llm_pipeline["context"], llm_pipeline["iterator"]):
                    # potentially share context binary between corresponding context/iterator components
                    new_ctx_model_name = f"{ctx_model_name}_ctx"
                    new_iter_model_name = f"{iter_model_name}_ctx"
                    composite_binaries = self._generate_composite_binaries(
                        model_paths_map={
                            new_ctx_model_name: component_models[ctx_model_name].model_path,
                            new_iter_model_name: component_models[iter_model_name].model_path,
                        },
                        device=self.accelerator_spec.accelerator_type,
                        output_model_dir=output_dir,
                        generate_kwargs=generate_kwargs,
                        weight_sharing=config.weight_sharing,
                    )
                    new_groups["context"][new_ctx_model_name] = composite_binaries[new_ctx_model_name]
                    new_groups["iterator"][new_iter_model_name] = composite_binaries[new_iter_model_name]

                return new_groups

            group_session_options = config.session_options or {}
            provider_options = config.provider_options or {}
            if (
                version.parse(OrtVersion).release < version.parse("1.22.0").release
            ) and self.accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider:
                provider_options["backend_path"] = "QnnHtp.dll"
            group_session_options["provider_options"] = [
                {self.accelerator_spec.execution_provider.lower().replace("executionprovider", ""): provider_options}
            ]

            return process_llm_pipeline(
                model,
                pipeline,
                process_context_iterator,
                output_model_path,
                group_session_options=group_session_options,
            )

        new_component_models = self._generate_composite_binaries(
            model_paths_map={f"{name}_ctx": component.model_path for name, component in component_map.items()},
            output_model_dir=output_model_path,
            device=self.accelerator_spec.accelerator_type,
            generate_kwargs=generate_kwargs,
            weight_sharing=config.weight_sharing,
            shared_bin_stem=(
                "model_ctx"
                if config.weight_sharing and str(self.accelerator_spec.accelerator_type).lower() == "gpu"
                else None
            ),
        )
        return CompositeModelHandler(
            list(new_component_models.values()),
            list(new_component_models.keys()),
            model_path=output_model_path,
            model_attributes=new_model_attributes,
        )

    @classmethod
    def _generate_composite_binaries(
        cls,
        model_paths_map: dict[str, str],
        device: Union[Device, str],
        output_model_dir: str,
        generate_kwargs: dict[str, str],
        weight_sharing: bool = False,
        shared_bin_stem: Optional[str] = None,
    ) -> dict[str, ONNXModelHandler]:
        """Generate context binary for each model in the composite model.

        :param model_paths_map: Map of model names to model paths.
        :param output_model_dir: Directory to save the output model files.
        :param generate_kwargs: Additional arguments for the context binary generation.
        :param weight_sharing: Whether to enable weight sharing between the models.
        :param shared_bin_stem: weight_sharing bin is generated under this neutral stem instead of its own logical name for QNN GPU.
        :return: Map of model names to ONNXModelHandler for the generated context binaries.
        """
        output_model_dir = Path(output_model_dir)
        new_models = {}
        for idx, (model_name, model_path) in enumerate(model_paths_map.items()):
            generate_kwargs = deepcopy(generate_kwargs)
            generation_name = model_name
            if weight_sharing:
                generate_kwargs["share_ep_contexts"] = True
                generate_kwargs["stop_share_ep_contexts"] = idx == len(model_paths_map) - 1
                generate_kwargs["embed_context"] = False
                generate_kwargs["ignore_missing_cb_bin"] = idx != len(model_paths_map) - 1
                if idx == 0 and shared_bin_stem:
                    generation_name = shared_bin_stem

            handler = cls._generate_context_binary(
                model_path=model_path,
                output_model_path=output_model_dir / f"{generation_name}.onnx",
                device=device,
                logical_name=model_name,
                **generate_kwargs,
            )

            if generation_name != model_name:
                renamed_path = output_model_dir / f"{model_name}.onnx"
                (output_model_dir / handler.onnx_file_name).replace(renamed_path)
                handler = ONNXModelHandler(model_path=output_model_dir, onnx_file_name=renamed_path.name)

            new_models[model_name] = handler

        return new_models

    @staticmethod
    def _generate_context_binary(
        model_path: str,
        output_model_path: Union[str, Path],
        device: Union[Device, str],
        execution_provider: str,
        provider_options: Optional[dict] = None,
        session_options: Optional[dict] = None,
        embed_context: bool = False,
        disable_cpu_fallback: bool = False,
        share_ep_contexts: bool = False,
        stop_share_ep_contexts: bool = False,
        ignore_missing_cb_bin: bool = False,
        model: Optional[Union[ONNXModelHandler, CompositeModelHandler]] = None,
        logical_name: Optional[str] = None,
    ) -> ONNXModelHandler:
        """Generate context binary for the model.

        :param model_path: Path to the model file.
        :param output_model_path: Path to the output model file.
        :param execution_provider: Execution provider to use.
        :param provider_options: Provider options for the execution provider.
        :param embed_context: Whether to embed context bin into the model.
        :param disable_cpu_fallback: Whether to disable CPU fallback.
        :param share_ep_contexts: Whether to share EP contexts.
        :param stop_share_ep_contexts: Whether to stop sharing EP contexts.
        :param ignore_missing_cb_bin: Whether to ignore missing context binary files.
        :param logical_name: Component's logical name used to update genai_config.json.
        :return: ONNXModelHandler for the generated context binary.
        """
        import onnxruntime as ort
        from onnxruntime import __version__ as OrtVersion

        is_abi = (
            "QNNExecutionProvider" not in ort.get_available_providers()
            or version.parse(OrtVersion).release >= version.parse("1.25.0").release
        )
        logger.debug(" Using ABI EP: %s", str(is_abi))

        # prepare provider options
        provider_options = deepcopy(provider_options or {})
        if execution_provider == ExecutionProvider.QNNExecutionProvider:
            if str(device).lower() == "gpu":
                provider_options["backend_path"] = "libQnnGpu.so" if platform.system() == "Linux" else "QnnGpu.dll"
                ctxbin_path = (
                    Path(output_model_path).with_name(f"{logical_name}.onnx")
                    if logical_name
                    else Path(output_model_path)
                )
                update_llm_pipeline_genai_config_gpu_ctxbin(model, ctxbin_path)
            else:
                provider_options["backend_path"] = "libQnnHtp.so" if platform.system() == "Linux" else "QnnHtp.dll"
                if share_ep_contexts:
                    provider_options["enable_htp_weight_sharing"] = "1"

        # prepare session options
        session_options = deepcopy(session_options or {})
        session_options.update(
            {
                "ep.context_enable": "1",
                "ep.context_embed_mode": int(embed_context),
                "session.disable_cpu_ep_fallback": int(disable_cpu_fallback),
                "ep.share_ep_contexts": int(share_ep_contexts),
                "ep.stop_share_ep_contexts": int(stop_share_ep_contexts),
            }
        )
        sess_options = ort.SessionOptions()
        EPContextBinaryGenerator._apply_session_options(sess_options, session_options)

        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        # clean up files that may be present from previous failed runs
        if output_model_path.exists():
            logger.debug("Context binary onnx file %s already exists. Deleting it.", str(output_model_path))
            output_model_path.unlink()
        for file_name in output_model_path.parent.glob(f"{output_model_path.stem}*.bin"):
            logger.debug("Context binary bin file %s already exists. Deleting it.", str(file_name))
            file_name.unlink()
        # set the context file path
        sess_options.add_session_config_entry("ep.context_file_path", str(output_model_path))

        # create the inference session
        # requires regular onnxruntime package, not winml (not tested with winml)
        logger.debug("Creating context binary for model %s", str(model_path))

        if is_abi:
            try:
                import onnxruntime_qnn as qnn_ep

                ep_lib_path = qnn_ep.get_library_path()
                ep_registration_name = "QNNExecutionProvider"
                ort.register_execution_provider_library(ep_registration_name, ep_lib_path)
            except Exception as e:
                if "already registered" in str(e):
                    logger.debug(
                        "Execution provider %s is already registered, skipping registration.", ep_registration_name
                    )
                else:
                    raise
            session_created = False
            try:
                all_ep_devices = ort.get_ep_devices()
                selected_ep_devices = [
                    ep_device
                    for ep_device in all_ep_devices
                    if ep_device.ep_name == ExecutionProvider.QNNExecutionProvider
                ]

                sess_options.add_provider_for_devices(selected_ep_devices, provider_options)
                ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                )
                session_created = True
            finally:
                # Keep successful shared contexts alive until the final component, but do not
                # leave a registration behind when session creation fails.
                if not session_created or not share_ep_contexts or stop_share_ep_contexts:
                    ort.unregister_execution_provider_library(ep_registration_name)
        else:
            ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=[execution_provider],
                provider_options=[provider_options],
            )

        assert output_model_path.exists(), f"Context binary not found at {output_model_path}"

        if not embed_context:
            cb_file_names = get_context_bin_file_names(output_model_path)
            assert cb_file_names, f"Context binary files not found for model {output_model_path}"

            if not ignore_missing_cb_bin:
                for cb_file_name in cb_file_names:
                    if not (output_model_path.parent / cb_file_name).exists():
                        raise FileNotFoundError(f"Context binary file {cb_file_name} not found.")

        return ONNXModelHandler(
            model_path=output_model_path if embed_context else output_model_path.parent,
            onnx_file_name=None if embed_context else output_model_path.name,
        )
