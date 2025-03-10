# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import platform
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type, Union

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_file, resave_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from onnxruntime import SessionOptions

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
            "compose": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to compose the component models into a single model. Only applicable to composite models."
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
        # validate and support other NPU EPs
        assert self.accelerator_spec.execution_provider == "QNNExecutionProvider", "Only QNN EP is supported for now."

        from onnxruntime import SessionOptions

        provider_options = config.provider_options or {}
        provider_options.update(
            {
                "backend_path": "libQnnHtp.so" if platform.system() == "Linux" else "QnnHtp.dll",
            }
        )

        session_options = SessionOptions()
        session_options.add_session_config_entry("ep.context_enable", "1")
        session_options.add_session_config_entry("ep.context_embed_mode", str(int(config.embed_context)))
        session_options.add_session_config_entry(
            "session.disable_cpu_ep_fallback", str(int(config.disable_cpu_fallback))
        )

        if isinstance(model, ONNXModelHandler):
            return self._generate_context_binary(
                model_path=model.model_path,
                output_model_path=resolve_onnx_path(output_model_path, f"{Path(model.model_path).stem}_ctx.onnx"),
                execution_provider=self.accelerator_spec.execution_provider,
                provider_options=provider_options,
                session_options=session_options,
            )

        output_model_path = Path(output_model_path).with_suffix("")

        # set options for weight sharing
        if config.weight_sharing:
            provider_options["enable_htp_weight_sharing"] = "1"
            session_options.add_session_config_entry("ep.share_ep_contexts", "1")

        # name: model
        component_map = dict(model.get_model_components())

        if llm_pipeline := (model.model_attributes or {}).get("llm_pipeline"):
            raise NotImplementedError(
                f"Generating context binary for {llm_pipeline} is not supported. Please implement this method."
            )

        # create context binary for each component
        new_component_names = []
        new_component_models = []
        for idx, (component_name, component_model) in enumerate(component_map.items()):
            if config.weight_sharing and idx == len(component_map) - 1:
                # stop context sharing at the last component
                session_options.add_session_config_entry("ep.stop_share_ep_contexts", "1")
            new_component_names.append(component_name)
            new_component_models.append(
                self._generate_context_binary(
                    model_path=component_model.model_path,
                    output_model_path=output_model_path / f"{component_name}_ctx.onnx",
                    execution_provider=self.accelerator_spec.execution_provider,
                    provider_options=provider_options,
                    session_options=session_options,
                    ignore_missing_cb_bin=config.weight_sharing and (idx != len(component_map) - 1),
                )
            )

        if config.compose:
            raise NotImplementedError("Composing context binary is not supported yet.")

        return CompositeModelHandler(new_component_models, new_component_names)

    @staticmethod
    def _generate_context_binary(
        model_path: str,
        output_model_path: Union[str, Path],
        execution_provider: str,
        provider_options: dict,
        session_options: "SessionOptions",
        ignore_missing_cb_bin: bool = False,
    ) -> ONNXModelHandler:
        """Generate context binary for the model.

        :param model_path: Path to the model file.
        :param output_model_path: Path to the output model file.
        :param execution_provider: Execution provider to use.
        :param provider_options: Provider options for the execution provider.
        :param session_options: Session options for the execution provider.
        :return: ONNXModelHandler with the context binary.
        """
        from onnxruntime import InferenceSession, get_available_providers

        assert (
            execution_provider in get_available_providers()
        ), f"Execution provider {execution_provider} is not available. Available providers: {get_available_providers()}"

        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)

        # hardlink/copy the original model into a tempdir under parent_dir
        # for easier clean up and file management
        # TODO(jambayk): avoid this if the cost is too high
        with tempfile.TemporaryDirectory(dir=output_model_path.parent, prefix="olive_tmp") as tmp_dir:
            # resave the model to a temp dir
            temp_model_path = Path(tmp_dir) / Path(model_path).name
            resave_model(model_path, temp_model_path)

            # path to create the context binary
            tmp_ctx_path = Path(tmp_dir) / output_model_path.name
            session_options.add_session_config_entry("ep.context_file_path", str(tmp_ctx_path))

            # create the inference session
            logger.debug("Creating context binary for model %s", str(model_path))
            InferenceSession(
                str(temp_model_path),
                sess_options=session_options,
                providers=[execution_provider],
                provider_options=[provider_options],
            )

            assert tmp_ctx_path.exists(), f"Context binary not found at {tmp_ctx_path}"

            # load and resave the _ctx.onnx model so that it doesn't refer to the original external data files
            model_proto = onnx.load(str(tmp_ctx_path))
            tmp_ctx_path.unlink()
            model_proto_to_file(model_proto, tmp_ctx_path)

            # move _ctx.onnx, .bin and external data files to the output model path
            has_external_data = resave_model(
                tmp_ctx_path, output_model_path, ignore_missing_cb_bin=ignore_missing_cb_bin
            )

        return ONNXModelHandler(
            model_path=output_model_path.parent if has_external_data else output_model_path,
            onnx_file_name=output_model_path.name if has_external_data else None,
        )
