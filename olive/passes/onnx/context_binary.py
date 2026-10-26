# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import json
import logging
import platform
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Literal, Optional, Union

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
    VISION_PIPELINE_KEY,
    get_context_bin_file_names,
    process_llm_pipeline,
    resave_model,
    update_llm_pipeline_genai_config_gpu_ctxbin,
    update_vision_pipeline_genai_config,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

_EP_LIBRARY_REGISTRATION_LOCK = RLock()


@contextmanager
def registered_ep_library(ort, registration_name: str, library_path: str):
    """Register a plugin EP for this scope without unregistering a registration owned by another caller."""
    with _EP_LIBRARY_REGISTRATION_LOCK:
        owns_registration = True
        try:
            ort.register_execution_provider_library(registration_name, library_path)
        except Exception as e:
            if "already registered" not in str(e):
                raise
            owns_registration = False
            logger.debug("Execution provider %s is already registered, reusing it.", registration_name)
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
            "components": PassConfigParam(
                type_=list[str],
                default_value=None,
                description=(
                    "OpenVINO composite inputs only. Select exactly one component to compile; other components"
                    " are preserved. Names refer to Olive components, not GenAI pipeline stages."
                ),
            ),
            "embed_context": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to embed context bin into the model.",
            ),
            "weight_sharing": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to enable weight sharing between QNN component models. Only applicable to QNN composite"
                    " models."
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
                    "Session options for the EP. Keys are added as session config entries, except"
                    " log_severity_level and log_verbosity_level which are set as typed attributes."
                ),
            ),
            "disable_cpu_fallback": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to disable CPU fallback.",
            ),
            "compile_flow": PassConfigParam(
                type_=Literal["aot", "jit"],
                default_value="aot",
                description=(
                    "OpenVINO only. 'aot' uses ORT ModelCompiler; 'jit' generates EPContext when creating an"
                    " InferenceSession. QNN continues to use its existing session-based compilation."
                ),
            ),
            "model_compiler_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "OpenVINO AOT only. Keyword arguments for ORT ModelCompiler, such as external_initializers_file_path,"
                    " external_initializers_size_threshold, graph_optimization_level, and flags. Enum values accept"
                    " member names; flags also accept '|'-joined names. embed_compiled_data_into_model overrides"
                    " embed_context when provided."
                ),
            ),
            "ep_device_filters": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "OpenVINO only. Filters on OrtEpDevice metadata, such as {'npu_platform': '5010'}, or on the"
                    " hexadecimal hardware device_id. The selected device must also match the accelerator type."
                ),
            ),
            "ep_registration_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "OpenVINO only. Plugin registration name, defaulting to the plugin's EP name. A '.virtual' suffix,"
                    " such as 'ovep_registration.virtual', enables virtual devices for cross-platform compilation."
                ),
            ),
            "output_model_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "OpenVINO only. EPContext ONNX filename in the pass output directory."
                    " Defaults to '<input_model_stem>_ctx.onnx'."
                ),
            ),
            "update_genai_config": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Update a single-model ORT GenAI role, or refresh a composite vision pipeline while preserving"
                    " its existing stage settings."
                ),
            ),
            "genai_config_role": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Top-level model role in genai_config.json to update when update_genai_config is enabled."
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
        is_openvino_composite = execution_provider == ExecutionProvider.OpenVINOExecutionProvider and isinstance(
            model, CompositeModelHandler
        )
        if config.components is not None and not is_openvino_composite:
            raise ValueError("components is only supported for OpenVINO composite inputs.")
        if execution_provider == ExecutionProvider.QNNExecutionProvider:
            result = self._run_qnn(model, config, output_model_path)
        elif is_openvino_composite:
            result = self._run_openvino_composite(model, config, output_model_path)
        elif execution_provider == ExecutionProvider.OpenVINOExecutionProvider:
            result = self._run_openvino(model, config, output_model_path)
        else:
            raise ValueError(f"EPContext generation is not supported for {execution_provider}.")

        # Populate model_attributes with context binary metadata so it persists in model_config.json
        result.model_attributes = {**(model.model_attributes or {}), **(result.model_attributes or {})}
        metadata_models = (
            [dict(result.get_model_components())[config.components[0]]] if is_openvino_composite else [result]
        )
        for metadata_model in metadata_models:
            metadata_model.model_attributes = dict(metadata_model.model_attributes or {})
            metadata_model.model_attributes["ep"] = execution_provider
            metadata_model.model_attributes["device"] = str(self.accelerator_spec.accelerator_type).upper()
            if config.provider_options:
                metadata_model.model_attributes["provider_options"] = config.provider_options
                if execution_provider == ExecutionProvider.QNNExecutionProvider:
                    metadata_model.model_attributes["architecture"] = config.provider_options.get("soc_model")
            if execution_provider == ExecutionProvider.OpenVINOExecutionProvider and config.ep_device_filters:
                metadata_model.model_attributes["ep_device_filters"] = config.ep_device_filters
        if config.update_genai_config:
            if not config.genai_config_role:
                raise ValueError("genai_config_role is required when update_genai_config is enabled.")
            if is_openvino_composite:
                result = update_vision_pipeline_genai_config(result)
            elif not isinstance(result, ONNXModelHandler):
                raise ValueError("update_genai_config only supports single-model EPContext outputs.")
            else:
                self._update_single_model_genai_config(model, result, config)

        return result

    def _run_openvino_composite(
        self, model: CompositeModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> CompositeModelHandler:
        if not config.components or len(config.components) != 1:
            raise ValueError("OpenVINO composite inputs require components to select exactly one component.")
        if config.weight_sharing:
            raise ValueError("weight_sharing is only supported for QNN composite models.")
        selected_name = config.components[0]
        components = dict(deepcopy(model).get_model_components())
        if selected_name not in components:
            raise ValueError(f"Unknown component {selected_name!r}. Available components: {list(components)}.")
        if not all(isinstance(component, ONNXModelHandler) for component in components.values()):
            raise ValueError("OpenVINO composite inputs must contain only ONNX models.")
        if config.update_genai_config:
            pipeline = (model.model_attributes or {}).get(VISION_PIPELINE_KEY)
            if config.genai_config_role != "vision" or not isinstance(pipeline, dict):
                raise ValueError(
                    "Composite GenAI updates require genai_config_role='vision' and vision_pipeline metadata."
                )
            if selected_name not in pipeline.values() or any(name not in components for name in pipeline.values()):
                raise ValueError(
                    "vision_pipeline must reference the selected component and valid composite components."
                )
            config_files = [
                path
                for path in (model.model_attributes or {}).get("additional_files") or []
                if Path(path).name == "genai_config.json"
            ]
            if len(config_files) != 1:
                raise ValueError("update_genai_config requires exactly one genai_config.json additional file.")

        output_dir = Path(output_model_path).with_suffix("")
        selected = components[selected_name]
        output_name = config.output_model_name or f"{Path(selected.model_path).stem}_ctx.onnx"
        if Path(output_name).name != output_name:
            raise ValueError("Composite output_model_name must be a filename without directories.")
        output_names = [
            output_name if name == selected_name else Path(component.model_path).name
            for name, component in components.items()
        ]
        if len(set(output_names)) != len(output_names):
            raise ValueError("Composite output ONNX filenames must be unique.")

        compiled = self._run_openvino(selected, config, str(output_dir))
        compiled.model_attributes = {**(selected.model_attributes or {}), **(compiled.model_attributes or {})}
        for name, component in components.items():
            if name == selected_name:
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
        return CompositeModelHandler(
            list(components.values()),
            list(components),
            model_path=output_dir,
            model_attributes=deepcopy(model.model_attributes),
        )

    def _run_openvino(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        """Generate an OpenVINO EPContext model through the ORT plugin EP API."""
        if isinstance(model, CompositeModelHandler):
            raise NotImplementedError("OpenVINO EPContext generation currently supports single ONNX models only.")
        if config.weight_sharing:
            raise ValueError("weight_sharing is only supported for QNN composite models.")

        output_model_name = config.output_model_name or f"{Path(model.model_path).stem}_ctx.onnx"
        return self._generate_plugin_ep_context_binary(
            ep_package_name="onnxruntime_ep_openvino",
            model_path=model.model_path,
            output_model_path=resolve_onnx_path(output_model_path, output_model_name),
            device=self.accelerator_spec.accelerator_type,
            compile_flow=config.compile_flow,
            provider_options=config.provider_options,
            session_options=config.session_options,
            model_compiler_options=config.model_compiler_options,
            ep_device_filters=config.ep_device_filters,
            ep_registration_name=config.ep_registration_name,
            embed_context=config.embed_context,
            disable_cpu_fallback=config.disable_cpu_fallback,
        )

    @staticmethod
    def _ep_device_matches(ep_device, filters: dict) -> bool:
        for key, value in filters.items():
            if key == "device_id":
                device_id = int(value, 16) if isinstance(value, str) else value
                if ep_device.device.device_id != device_id:
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
        device: Union[Device, str],
        compile_flow: str,
        provider_options: Optional[dict] = None,
        session_options: Optional[dict] = None,
        model_compiler_options: Optional[dict] = None,
        ep_device_filters: Optional[dict] = None,
        ep_registration_name: Optional[str] = None,
        embed_context: bool = False,
        disable_cpu_fallback: bool = False,
    ) -> ONNXModelHandler:
        """Compile an ONNX model with a plugin EP and retain its compiled context files."""
        import onnx
        import onnxruntime as ort

        if compile_flow not in ("aot", "jit"):
            raise ValueError(f"Unknown compile_flow {compile_flow!r}; expected 'aot' or 'jit'.")
        if compile_flow == "jit" and model_compiler_options:
            raise ValueError("model_compiler_options are only supported with compile_flow='aot'.")
        compiler_kwargs = {}
        if compile_flow == "aot":
            if not hasattr(ort, "ModelCompiler"):
                raise RuntimeError("OpenVINO AOT compilation requires an ONNX Runtime build with ModelCompiler.")
            compiler_kwargs = cls._build_model_compiler_kwargs(ort, model_compiler_options)
            embed_context = compiler_kwargs.setdefault("embed_compiled_data_into_model", embed_context)

        try:
            ep_package = importlib.import_module(ep_package_name)
        except ImportError as e:
            raise ImportError(
                f"Install the plugin EP package with 'pip install {ep_package_name.replace('_', '-')}'."
            ) from e
        ep_name = ep_package.get_ep_name()
        registration_name = ep_registration_name or ep_name

        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        if output_model_path.exists() or any(output_model_path.parent.glob(f"{output_model_path.stem}*.bin")):
            raise FileExistsError(f"EPContext output already exists at {output_model_path}; use a new output location.")

        session_options = deepcopy(session_options or {})
        context_options = {
            "ep.context_enable": "1",
            "ep.context_embed_mode": int(embed_context),
            "ep.context_file_path": str(output_model_path),
        }
        if compile_flow == "jit":
            session_options.update(context_options)
        else:
            # ModelCompiler sets these entries from its embedding option and compile_to_file path.
            for key in context_options:
                session_options.pop(key, None)
        session_options["session.disable_cpu_ep_fallback"] = int(disable_cpu_fallback)
        sess_options = ort.SessionOptions()
        for key, value in session_options.items():
            if key in ("log_severity_level", "log_verbosity_level"):
                setattr(sess_options, key, int(value))
            else:
                sess_options.add_session_config_entry(key, str(value))

        completed = False
        try:
            with registered_ep_library(ort, registration_name, ep_package.get_library_path()):
                device_type = get_ort_hardware_device_type(device)
                selected = [
                    ep_device
                    for ep_device in ort.get_ep_devices()
                    if ep_device.ep_name == ep_name
                    and ep_device.device.type == device_type
                    and cls._ep_device_matches(ep_device, ep_device_filters or {})
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected exactly one {ep_name} {device} device matching {ep_device_filters}, found"
                        f" {len(selected)}. Set ep_device_filters to select the target device."
                    )
                sess_options.add_provider_for_devices(selected, deepcopy(provider_options or {}))
                if compile_flow == "aot":
                    ort.ModelCompiler(sess_options, str(model_path), **compiler_kwargs).compile_to_file(
                        str(output_model_path)
                    )
                else:
                    ort.InferenceSession(str(model_path), sess_options=sess_options)

            if not output_model_path.is_file():
                raise FileNotFoundError(f"EPContext model was not generated at {output_model_path}.")
            graphs = [onnx.load(output_model_path, load_external_data=False).graph]
            has_context = False
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
                    if not (output_model_path.parent / context_file).is_file():
                        raise FileNotFoundError(
                            f"EPContext binary {context_file!r} was not generated for {output_model_path}."
                        )
            if not has_context:
                raise RuntimeError(f"No EPContext nodes were generated in {output_model_path}.")
            result = ONNXModelHandler(model_path=output_model_path.parent, onnx_file_name=output_model_path.name)
            completed = True
            return result
        finally:
            if not completed:
                output_model_path.unlink(missing_ok=True)
                for binary_path in output_model_path.parent.glob(f"{output_model_path.stem}*.bin"):
                    binary_path.unlink()

    @staticmethod
    def _build_model_compiler_kwargs(ort, model_compiler_options: Optional[dict]) -> dict:
        """Validate compiler keywords and convert JSON scalar and enum values to their ORT types."""
        if not model_compiler_options:
            return {}

        import inspect

        from pydantic import TypeAdapter

        parameters = inspect.signature(ort.ModelCompiler.__init__).parameters
        defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if name not in ("self", "sess_options", "input_model_path_or_bytes")
            and not name.endswith("_func")
            and parameter.default is not inspect.Parameter.empty
        }
        options = model_compiler_options or {}
        unknown = set(options) - set(defaults)
        if unknown:
            raise ValueError(f"Unsupported model_compiler_options: {sorted(unknown)}. Supported: {sorted(defaults)}")
        result = {}
        for key, option_value in options.items():
            value = option_value
            default = defaults[key]
            enum_type = type(default)
            if hasattr(enum_type, "__members__"):
                if isinstance(value, str):
                    members = value.split("|") if key == "flags" else [value]
                    converted = None
                    for name in members:
                        try:
                            member = enum_type.__members__[name.strip()]
                        except KeyError as e:
                            raise ValueError(
                                f"Invalid {key} member {name!r}. Expected one of {list(enum_type.__members__)}"
                            ) from e
                        converted = member if converted is None else converted | member
                    value = converted
                else:
                    value = enum_type(value)
            elif isinstance(default, (bool, int)):
                value = TypeAdapter(type(default)).validate_python(value)
            result[key] = value
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

    def _run_qnn(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        """Generate QNN context binaries, including composite models and shared contexts."""
        from onnxruntime import __version__ as OrtVersion
        from onnxruntime import get_available_providers

        if version.parse(OrtVersion).release <= version.parse("1.23.2").release:
            assert self.accelerator_spec.execution_provider in get_available_providers(), (
                f"Execution provider {self.accelerator_spec.execution_provider} is not available. Available providers:"
                f" {get_available_providers()}"
            )

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
        for key, value in session_options.items():
            # a few options are typed attributes on SessionOptions rather than string config entries.
            # setting them with add_session_config_entry would silently have no effect.
            if key in ("log_severity_level", "log_verbosity_level"):
                setattr(sess_options, key, int(value))
            else:
                sess_options.add_session_config_entry(key, str(value))

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
