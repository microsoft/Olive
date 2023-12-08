# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler, OptimumModelHandler
from olive.model.config.hf_config import HfConfig
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class OptimumConversion(Pass):
    """Convert a Optimum model to ONNX model using the Optimum export function."""

    _requires_user_script = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_opset": PassConfigParam(
                type_=int, default_value=14, description="The version of the default (ai.onnx) opset to target."
            ),
            "fp16": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to use fp16 precision to load torch model and then convert it to onnx.",
            ),
            "device": PassConfigParam(
                type_=str, default_value="cpu", description="The device to use to do the export. Defaults to 'cpu'."
            ),
            "no_post_process": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Whether to skip post-processing the exported model.",
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
        self, model: OptimumModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        assert len(model.model_components) > 0

        from optimum import version as optimum_version
        from optimum.exporters.onnx import main_export as export_optimum_model
        from packaging import version

        # TODO(jambayk): export into temp dir and then move to sub-dirs of output_model_path
        # so that we only keep the final model files in the output_model_path
        # and track external data if present
        config["extra_args"] = config["extra_args"] or {}
        config["extra_args"].update(
            {
                "opset": config["target_opset"],
                "fp16": config["fp16"],
                "no_post_process": config["no_post_process"],
                "device": config["device"],
            }
        )
        hf_config = deepcopy(model.hf_config) or HfConfig()
        if version.parse(optimum_version.__version__) >= version.parse("1.14.0"):
            # Optimum 1.14.0 needs to be run in legacy mode to support older versions of transformers
            # TODO(trajep): deprecated legacy after fully test with the model using optimum merging
            config["extra_args"]["legacy"] = True

        export_optimum_model(
            model.model_path or hf_config.model_name,
            output_model_path,
            **config["extra_args"],
        )

        onnx_model_components = [
            ONNXModelHandler(str(Path(output_model_path) / model_component), model_attributes=model.model_attributes)
            for model_component in model.model_components
        ]
        onnx_model_component_names = [Path(model_component).stem for model_component in model.model_components]

        if len(onnx_model_components) == 1:
            return ONNXModelHandler(Path(output_model_path) / model.model_components[0])

        return CompositeModelHandler(onnx_model_components, onnx_model_component_names)
