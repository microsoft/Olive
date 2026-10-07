# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import logging
import platform
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Optional, Union

from packaging import version

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.hardware.constants import ExecutionProvider
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    get_context_bin_file_names,
    process_llm_pipeline,
    update_llm_pipeline_genai_config_gpu_ctxbin,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

_OPENVINO_EP_PLUGIN_PACKAGE = "onnxruntime_ep_openvino"
_EP_LIBRARY_REGISTRATION_LOCK = RLock()


@contextmanager
def registered_ep_library(ort, registration_name: str, library_path: str):
    """Register an EP plugin library, unregistering on exit only if this call actually registered it.

    EP registration is process-global and keyed by registration_name; if that name is already
    registered by another caller, we reuse it and leave it in place rather than tearing down state
    we don't own.
    """
    with _EP_LIBRARY_REGISTRATION_LOCK:
        owns_registration = True
        try:
            ort.register_execution_provider_library(registration_name, library_path)
        except Exception as e:
            if "already registered" not in str(e):
                raise
            owns_registration = False
            logger.debug("Execution provider %s already registered, reusing it.", registration_name)
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
                    "Whether to enable weight sharing between the component models. Only applicable to composite"
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
                    " log_severity_level and log_verbosity_level which are set as typed attributes. For the OpenVINO"
                    " plugin EP, use EPContext entries such as 'ep.context_enable', 'ep.context_embed_mode',"
                    " 'ep.enable_weightless', and 'ep.context_file_path'; provider options use the"
                    " 'ep.openvinoexecutionprovider.<key>' prefix."
                ),
            ),
            "disable_cpu_fallback": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to disable CPU fallback.",
            ),
            "model_compiler_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "OpenVINO plugin EP only. Options forwarded to onnxruntime.ModelCompiler's constructor:"
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
                    "OpenVINO plugin EP only. Filters to select a specific OrtEpDevice when the plugin exposes"
                    " multiple devices (e.g. virtual NPU platforms for AOT cross-compilation). Keys are matched"
                    " against each device's ep_metadata, e.g. {'npu_platform': '5010'} to target Panther Lake."
                ),
            ),
            "ep_registration_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "OpenVINO plugin EP only. Name used to register the EP library with ONNX Runtime. Defaults to the"
                    " plugin's EP name. Use a '.virtual' suffixed name (e.g. 'ovep_registration.virtual') to enumerate"
                    " virtual (meta) devices for ahead-of-time cross-platform compilation."
                ),
            ),
            "output_model_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "OpenVINO plugin EP only. File name for the generated EPContext model, written in the input"
                    " model's directory. Defaults to '<input_model_stem>_ctx.onnx'."
                ),
            ),
            "compile_flow": PassConfigParam(
                type_=str,
                default_value="aot",
                description=(
                    "OpenVINO plugin EP only. 'aot' compiles via ORT's ModelCompiler (Compile API, no session); 'jit'"
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
        if self.accelerator_spec.execution_provider == ExecutionProvider.OpenVINOExecutionProvider:
            return self._run_openvino(model, config, output_model_path)

        # session created using providers argument so will use the ort.get_available_providers()
        # TODO(jambayk): consider switching to the new EP API for Windows
        from onnxruntime import __version__ as OrtVersion
        from onnxruntime import get_available_providers

        # TODO(jambayk): validate and support other NPU EPs
        assert self.accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider, (
            "Only QNNExecutionProvider is supported for now."
        )

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

        return result

    def _run_openvino(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        """Generate an EPContext ONNX model for the OpenVINO plugin EP (aot: ModelCompiler, jit: InferenceSession)."""
        if isinstance(model, CompositeModelHandler):
            raise NotImplementedError("OpenVINO EPContext generation does not yet support composite models.")

        # write the epctx model into the pass output location (matches the QNN path)
        input_model_path = Path(model.model_path)
        output_model_name = config.output_model_name or f"{input_model_path.stem}_ctx.onnx"
        result = self._generate_plugin_ep_context_binary(
            ep_package_name=_OPENVINO_EP_PLUGIN_PACKAGE,
            model_path=model.model_path,
            output_model_path=resolve_onnx_path(output_model_path, output_model_name),
            compile_flow=config.compile_flow,
            provider_options=config.provider_options,
            session_options=config.session_options,
            model_compiler_options=config.model_compiler_options,
            ep_device_filters=config.ep_device_filters,
            ep_registration_name=config.ep_registration_name,
        )

        result.model_attributes = {**(model.model_attributes or {}), **(result.model_attributes or {})}
        result.model_attributes["ep"] = self.accelerator_spec.execution_provider
        result.model_attributes["device"] = str(self.accelerator_spec.accelerator_type).upper()
        if config.provider_options:
            result.model_attributes["provider_options"] = config.provider_options
        if config.ep_device_filters:
            result.model_attributes["ep_device_filters"] = config.ep_device_filters

        return result

    @staticmethod
    def _ep_device_matches(ep_device, filters: dict) -> bool:
        """Match an OrtEpDevice against filters, checking ep_metadata and the hardware device_id (hex)."""
        for key, value in filters.items():
            if key == "device_id":
                if f"{ep_device.device.device_id:X}".lower() != str(value).lower():
                    return False
            elif str(ep_device.ep_metadata.get(key)) != str(value):
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
    ) -> ONNXModelHandler:
        """Generate an EPContext model with a plugin EP. EP-agnostic: a new EP only needs its plugin package and options.

        compile_flow 'aot' compiles via ORT's ModelCompiler (no inference); 'jit' creates an InferenceSession that
        compiles and writes the EPContext model. Options are passed to ORT verbatim; nothing is set by Olive.
        """
        import onnxruntime as ort

        try:
            ep_package = importlib.import_module(ep_package_name)
        except ImportError as e:
            raise ImportError(
                f"The plugin EP package '{ep_package_name}' is required. Install it with"
                f" 'pip install {ep_package_name.replace('_', '-')}'."
            ) from e

        ep_name = ep_package.get_ep_name()
        registration_name = ep_registration_name or ep_name

        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)

        # refuse to overwrite an existing epctx model from a previous run
        existing = [output_model_path] if output_model_path.exists() else []
        existing += list(output_model_path.parent.glob(f"{output_model_path.stem}*.bin"))
        if existing:
            raise FileExistsError(
                f"EPContext file already exists: {output_model_path}. Remove it (and its .bin files) or choose a"
                " different output_model_name before re-running."
            )

        # set user session options verbatim as ORT session config entries
        sess_options = ort.SessionOptions()
        for key, value in (session_options or {}).items():
            sess_options.add_session_config_entry(key, str(value))

        with registered_ep_library(ort, registration_name, ep_package.get_library_path()):
            selected = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
            if ep_device_filters:
                selected = [d for d in selected if cls._ep_device_matches(d, ep_device_filters)]
            if len(selected) != 1:
                available = [(d.ep_name, dict(d.ep_metadata)) for d in ort.get_ep_devices() if d.ep_name == ep_name]
                raise ValueError(
                    f"Expected exactly one OrtEpDevice for {ep_name} matching filters {ep_device_filters}, found"
                    f" {len(selected)}. Add filters (e.g. 'npu_platform') to disambiguate. Available: {available}"
                )
            sess_options.add_provider_for_devices(selected, provider_options or {})

            if compile_flow == "aot":
                compiler_kwargs = cls._build_model_compiler_kwargs(ort, model_compiler_options)
                ort.ModelCompiler(sess_options, str(model_path), **compiler_kwargs).compile_to_file(
                    str(output_model_path)
                )
            elif compile_flow == "jit":
                if model_compiler_options:
                    raise ValueError("model_compiler_options are only used with compile_flow 'aot'.")
                sess_options.add_session_config_entry("ep.context_file_path", str(output_model_path))
                ort.InferenceSession(str(model_path), sess_options=sess_options)
            else:
                raise ValueError(f"Unknown compile_flow '{compile_flow}'. Expected 'aot' or 'jit'.")

        assert output_model_path.exists(), f"Context binary not found at {output_model_path}"

        # return the stub with dir + file name so only the epctx model and its .bin are saved
        return ONNXModelHandler(
            model_path=output_model_path.parent,
            onnx_file_name=output_model_path.name,
        )

    @staticmethod
    def _build_model_compiler_kwargs(ort, model_compiler_options: Optional[dict]) -> dict:
        """Coerce 'model_compiler_options' into onnxruntime.ModelCompiler constructor kwargs.

        Accepted keys and their coercion are derived from the ModelCompiler.__init__ signature, so the pass tracks the
        ORT API automatically: the value is coerced to the type of the parameter's default. Enum params (e.g.
        graph_optimization_level, flags) take a member name or a '|'-joined set of names; bool/int params take their
        respective type; everything else is passed verbatim.
        """
        import inspect

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

        def coerce(default, value):
            enum_type = type(default)
            if hasattr(enum_type, "__members__"):  # pybind enum (GraphOptimizationLevel, OrtCompileApiFlags)
                result = None
                for name in str(value).split("|"):
                    member = enum_type.__members__[name.strip()]
                    result = member if result is None else result | member
                return result
            if isinstance(default, bool):  # check before int: bool is an int subclass
                return bool(value)
            if isinstance(default, int):
                return int(value)
            return value  # str / path-like — passed verbatim

        return {key: coerce(settable[key], value) for key, value in opts.items()}

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
        provider_options = provider_options or {}
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
        session_options = session_options or {}
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
