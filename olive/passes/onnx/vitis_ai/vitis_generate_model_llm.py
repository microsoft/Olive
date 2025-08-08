#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class VitisGenerateModelLLM(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec):
        return {
            "packed_const": PassConfigParam(
                type_=bool, default_value=False, description="Enable packed constants optimization in NPU export."
            ),
            "cpu_only": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Run only model builder OGA CPU only model, skip NPU-related steps.",
            ),
        }

    def _run_for_config(
        self, model: HfModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        logger.info("[DEBUG] Running VitisGenerateModelLLM with config: %s", config)
        from model_generate import generate_npu_model

        input_model_path = model.model_path
        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[VitisGenerateModelLLM] Generating Vitis NPU model from: %s", input_model_path)
        logger.info("[VitisGenerateModelLLM] Output directory: %s", output_dir)
        logger.info("[VitisGenerateModelLLM] Packed constants: %s", config.packed_const)

        # Generate the NPU model
        generate_npu_model(
            input_model=str(input_model_path),
            output_dir=str(output_dir),
            packed_const=config.packed_const,
            cpu_only=config.cpu_only,
        )
        return ONNXModelHandler(
            model_path=output_dir,
            onnx_file_name="model.onnx",
        )
