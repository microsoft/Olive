# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler, PyTorchModelHandler
from olive.model.config.hf_config import HfConfig
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class OptimumConversion(Pass):
    """Convert a Hugging Face PyTorch model to ONNX model using the Optimum export function."""

    _requires_user_script = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
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
        config.update(get_external_data_config())
        return config

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})

        if search_point.get("fp16") and search_point.get("device") != "cuda":
            logger.info("OptimumConversion: fp16 is set to True, but device is not set to cuda.")
            return False

        return True

    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        from optimum import version as optimum_version
        from optimum.exporters.onnx import main_export as export_optimum_model
        from packaging import version

        extra_args = deepcopy(config["extra_args"]) or {}
        extra_args.update(
            {
                "opset": config["target_opset"],
                "fp16": config["fp16"],
                "device": config["device"],
            }
        )
        hf_config = model.hf_config or HfConfig()
        if hf_config.from_pretrained_args and "trust_remote_code" not in extra_args:
            extra_args["trust_remote_code"] = hf_config.from_pretrained_args.trust_remote_code

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

        with tempfile.TemporaryDirectory() as temp_dir:
            export_optimum_model(
                model.model_path or hf_config.model_name,
                temp_dir,
                **extra_args,
            )

            exported_models = [name.stem for name in Path(temp_dir).iterdir() if name.suffix == ".onnx"]
            logger.debug(f"Exported models: {exported_models}")
            if config["components"]:
                assert all(
                    component in exported_models for component in config["components"]
                ), f"Components {config['components']} are not exported. Only {exported_models} are exported."
            components = config["components"] or exported_models
            logger.debug(f"Components to export: {components}")

            # if there is only one component, return it directly
            if len(components) == 1:
                output_model_path = resolve_onnx_path(output_model_path)
                model_proto = onnx.load(Path(temp_dir) / f"{components[0]}.onnx")
                return model_proto_to_olive_model(model_proto, output_model_path, config)

            # if there are multiple components, return a composite model
            model_components = []
            model_component_names = []
            for component_name in components:
                # save the component model to a sub-directory
                component_output_path = resolve_onnx_path(Path(output_model_path).with_suffix("") / component_name)
                component_proto = onnx.load(Path(temp_dir) / f"{component_name}.onnx")
                component_handler = model_proto_to_olive_model(component_proto, component_output_path, config)
                component_handler.model_attributes = model.model_attributes
                # add the component model to the composite model
                model_components.append(component_handler)
                model_component_names.append(component_name)

            return CompositeModelHandler(model_components, model_component_names)
