#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_file, resave_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class EstimateNPULatency(Pass):
    """Returns latency estimates for the model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "target_device": PassConfigParam(
                type_=str, required=False, description="Target device type", default_value="stx"
            )
        }

    @classmethod
    def validate_config(cls, config: type[BasePassConfig], accelerator_spec: AcceleratorSpec) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.target_device and config.target_device not in ["stx"]:
            logger.warning("Unsupported target device type: %s", config.target_device)
            return False

        return True

    def _run_for_config(
        self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        perf_installed = True
        try:
            from estimator.config import EstimatorSettings
            from estimator.run import run_perf_estimate
        except ImportError:
            perf_installed = False
            logger.warning("Estimator module not found. Skipping EstimateNPULatency pass.")

        if not isinstance(model, ONNXModelHandler):
            raise ValueError("Model must be an instance of ONNXModelHandler")

        input_model_path = model.model_path

        # Bypass if perf estimator package not installed
        if perf_installed:
            EstimatorSettings.model_path = f"{input_model_path}"

            # Override default parameters if specified
            if config.target_device:
                EstimatorSettings.target_device = config.target_device
            EstimatorSettings.initialized = True

            logger.info(
                "Running perf estimator for model path: %s and target device: %s",
                input_model_path,
                EstimatorSettings.target_device,
            )

            run_perf_estimate(EstimatorSettings)

            logger.info("Finish running perf estimator pass")

        # return the original model
        output_model_path = Path(resolve_onnx_path(output_model_path, Path(model.model_path).name))
        has_external_data = resave_model(model.model_path, output_model_path)
        onnx_model = model.load_model()
        model_proto_to_file(onnx_model, output_model_path)

        return ONNXModelHandler(
            model_path=output_model_path.parent if has_external_data else output_model_path,
            onnx_file_name=output_model_path.name if has_external_data else None
        )
