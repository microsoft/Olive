# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Type, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, HfModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config

logger = logging.getLogger(__name__)


class OptimumConversion(Pass):
    """Convert a Hugging Face PyTorch model to ONNX model using the Optimum export function."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "target_opset": PassConfigParam(
                type_=int, default_value=14, description="The version of the default (ai.onnx) opset to target."
            ),
            "components": PassConfigParam(
                type_=List[str],
                default_value=None,
                description=(
                    "List of component models to export. E.g. ['decoder_model', 'decoder_with_past_model']. None means"
                    " export all components."
                ),
            ),
            "fp16": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to use fp16 precision to load torch model and then convert it to onnx.",
            ),
            "device": PassConfigParam(
                type_=str, default_value="cpu", description="The device to use to do the export. Defaults to 'cpu'."
            ),
            "extra_args": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Extra arguments to pass to the `optimum.exporters.onnx.main_export` function.",
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: Type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.fp16 and config.device != "cuda":
            logger.info("OptimumConversion: fp16 is set to True, but device is not set to cuda.")
            return False

        return True

    def _run_for_config(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        from optimum import version as optimum_version
        from optimum.exporters.onnx import main_export as export_optimum_model
        from packaging import version

        extra_args = deepcopy(config.extra_args) or {}
        extra_args.update(
            {
                "opset": config.target_opset,
                "fp16": config.fp16,
                "device": config.device,
            }
        )
        if model.load_kwargs and "trust_remote_code" not in extra_args:
            extra_args["trust_remote_code"] = model.load_kwargs.trust_remote_code

        if version.parse(optimum_version.__version__) < version.parse("1.14.0"):
            logger.warning(
                "The behavior of Optimum onnx exporter changed in version 1.14.0 with the introduction of `legacy`"
                " option. You are using an older version of optimum so it will use the legacy behavior and the output"
                " model/s may not be the same as the latest version. Please upgrade to the latest version of optimum if"
                " you do not want the legacy behavior!"
            )
            if "legacy" in extra_args:
                logger.warning(
                    "`legacy` option is set in the extra_args, but it is ignored because you are using optimum<1.14.0."
                )
                del extra_args["legacy"]

        # export directly to the output path
        # TODO(anyone): consider using a temporary directory to export the model and then save the relevant components
        export_optimum_model(model.model_name_or_path, output_model_path, **extra_args)

        # check the exported components
        exported_models = [name.stem for name in Path(output_model_path).iterdir() if name.suffix == ".onnx"]
        if config.components:
            assert all(
                component in exported_models for component in config.components
            ), f"Components {config['components']} are not exported. Only {exported_models} are exported."
        components = config.components or exported_models
        logger.debug("Exported models are: %s. Returning components: %s.", exported_models, components)

        # if there is only one component, return it directly
        if len(components) == 1:
            # will always return an onnx model handler with folder as the model path
            return ONNXModelHandler(model_path=output_model_path, onnx_file_name=f"{components[0]}.onnx")

        # if there are multiple components, return a composite model
        model_components = []
        model_component_names = []
        for component_name in components:
            # Note: since conversion is done directly to the output path, all components are in the same folder
            # this is not the same as for other composite models where each component is in a separate subfolder
            model_components.append(
                ONNXModelHandler(
                    model_path=output_model_path,
                    onnx_file_name=f"{component_name}.onnx",
                    model_attributes=model.model_attributes,
                )
            )
            model_component_names.append(component_name)

        return CompositeModelHandler(model_components, model_component_names)
