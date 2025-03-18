# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import platform
from copy import deepcopy
from pathlib import Path
from typing import Dict, Type, Union

from packaging import version

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_context_bin_file_names, resave_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class EPContextBinaryGenerator(Pass):
    """Generate EP specific context binary for the model."""

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
            "disable_cpu_fallback": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to disable CPU fallback.",
            ),
        }

    def _run_for_config(
        self,
        model: Union[ONNXModelHandler, CompositeModelHandler],
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        from onnxruntime import __version__ as OrtVersion
        from onnxruntime import get_available_providers

        # TODO(jambayk): validate and support other NPU EPs
        assert (
            self.accelerator_spec.execution_provider == "QNNExecutionProvider"
        ), "Only QNNExecutionProvider is supported for now."
        assert self.accelerator_spec.execution_provider in get_available_providers(), (
            f"Execution provider {self.accelerator_spec.execution_provider} is not available. Available providers:"
            f" {get_available_providers()}"
        )

        generate_kwargs = {
            "execution_provider": self.accelerator_spec.execution_provider,
            "provider_options": config.provider_options,
            "embed_context": config.embed_context,
            "disable_cpu_fallback": config.disable_cpu_fallback,
        }

        if isinstance(model, ONNXModelHandler):
            return self._generate_context_binary(
                model_path=model.model_path,
                output_model_path=resolve_onnx_path(output_model_path, f"{Path(model.model_path).stem}_ctx.onnx"),
                **generate_kwargs,
            )

        if config.weight_sharing and (version.parse(OrtVersion).release < version.parse("1.22.0").release):
            raise ValueError("weight sharing is only supported in onnxruntime >= 1.22.0")

        output_model_path = Path(output_model_path).with_suffix("")

        # name: model
        component_map = dict(model.get_model_components())

        new_component_models = {}
        new_model_attributes = deepcopy(model.model_attributes) or {}
        if llm_pipeline := (model.model_attributes or {}).get("llm_pipeline"):
            new_llm_pipeline = {}

            # resave embeddings model
            embeddings_model_path = output_model_path / "embeddings.onnx"
            resave_model(component_map[llm_pipeline["embeddings"]].model_path, embeddings_model_path)
            new_component_models["embeddings"] = ONNXModelHandler(
                model_path=output_model_path, onnx_file_name=embeddings_model_path.name
            )
            new_llm_pipeline["embeddings"] = "embeddings"

            # iterate over the context/iterator models
            new_llm_pipeline["context"] = []
            new_llm_pipeline["iterator"] = []
            for ctx_model_name, iter_model_name in zip(llm_pipeline["context"], llm_pipeline["iterator"]):
                # potentially share context binary between corresponding context/iterator components
                new_ctx_model_name = f"{ctx_model_name}_ctx"
                new_iter_model_name = f"{iter_model_name}_ctx"
                new_component_models.update(
                    self._generate_composite_binaries(
                        model_paths_map={
                            new_ctx_model_name: component_map[ctx_model_name].model_path,
                            new_iter_model_name: component_map[iter_model_name].model_path,
                        },
                        output_model_dir=output_model_path,
                        generate_kwargs=generate_kwargs,
                        weight_sharing=config.weight_sharing,
                    )
                )
                new_llm_pipeline["context"].append(new_ctx_model_name)
                new_llm_pipeline["iterator"].append(new_iter_model_name)

            # resave the lm_head model
            lm_head_model_path = output_model_path / "lm_head.onnx"
            resave_model(component_map[llm_pipeline["lm_head"]].model_path, lm_head_model_path)
            new_component_models["lm_head"] = ONNXModelHandler(
                model_path=output_model_path, onnx_file_name=lm_head_model_path.name
            )
            new_llm_pipeline["lm_head"] = "lm_head"

            new_model_attributes["llm_pipeline"] = new_llm_pipeline
        else:
            new_component_models = self._generate_composite_binaries(
                model_paths_map={f"{name}_ctx": component.model_path for name, component in component_map.items()},
                output_model_dir=output_model_path,
                generate_kwargs=generate_kwargs,
                weight_sharing=config.weight_sharing,
            )
        return CompositeModelHandler(
            list(new_component_models.values()),
            list(new_component_models.keys()),
            model_attributes=new_model_attributes,
        )

    @classmethod
    def _generate_composite_binaries(
        cls,
        model_paths_map: Dict[str, str],
        output_model_dir: str,
        generate_kwargs: Dict[str, str],
        weight_sharing: bool = False,
    ) -> Dict[str, ONNXModelHandler]:
        """Generate context binary for each model in the composite model.

        :param model_paths_map: Map of model names to model paths.
        :param output_model_dir: Directory to save the output model files.
        :param generate_kwargs: Additional arguments for the context binary generation.
        :param weight_sharing: Whether to enable weight sharing between the models.
        :return: Map of model names to ONNXModelHandler for the generated context binaries.
        """
        new_models = {}
        for idx, (model_name, model_path) in enumerate(model_paths_map.items()):
            generate_kwargs = deepcopy(generate_kwargs)
            if weight_sharing:
                generate_kwargs["share_ep_contexts"] = True
                generate_kwargs["stop_share_ep_contexts"] = idx == len(model_paths_map) - 1
                generate_kwargs["embed_context"] = False
                generate_kwargs["ignore_missing_cb_bin"] = idx != len(model_paths_map) - 1

            new_models[model_name] = cls._generate_context_binary(
                model_path=model_path,
                output_model_path=Path(output_model_dir) / f"{model_name}.onnx",
                **generate_kwargs,
            )

        return new_models

    @staticmethod
    def _generate_context_binary(
        model_path: str,
        output_model_path: Union[str, Path],
        execution_provider: str,
        provider_options: dict,
        embed_context: bool = False,
        disable_cpu_fallback: bool = False,
        share_ep_contexts: bool = False,
        stop_share_ep_contexts: bool = False,
        ignore_missing_cb_bin: bool = False,
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
        :return: ONNXModelHandler for the generated context binary.
        """
        from onnxruntime import InferenceSession, SessionOptions

        # prepare provider options
        provider_options = provider_options or {}
        if execution_provider == "QNNExecutionProvider":
            provider_options["backend_path"] = "libQnnHtp.so" if platform.system() == "Linux" else "QnnHtp.dll"
            if share_ep_contexts:
                provider_options["enable_htp_weight_sharing"] = "1"

        # prepare session options
        session_options = SessionOptions()
        session_options.add_session_config_entry("ep.context_enable", "1")
        session_options.add_session_config_entry("ep.context_embed_mode", str(int(embed_context)))
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", str(int(disable_cpu_fallback)))
        session_options.add_session_config_entry("ep.share_ep_contexts", str(int(share_ep_contexts)))
        session_options.add_session_config_entry("ep.stop_share_ep_contexts", str(int(stop_share_ep_contexts)))

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
        session_options.add_session_config_entry("ep.context_file_path", str(output_model_path))

        # create the inference session
        logger.debug("Creating context binary for model %s", str(model_path))
        InferenceSession(
            model_path,
            sess_options=session_options,
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
