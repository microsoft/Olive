#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path
from typing import Union

from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class VitisGenerateModelLLM(Pass):
    """Olive pass for generating NPU models using AMD Vitis toolchain.

    Supports multiple flows:
    - full_fusion: NPU full fusion model (default)
    - gpt_oss: GPT-OSS specific flow (via model_type)

    Input can be:
    - HfModelHandler: For models coming from quantization pass
    - ONNXModelHandler: For pre-quantized models like GPT-OSS

    All flows output model.onnx for consistency.
    """

    @classmethod
    def _default_config(cls, accelerator_spec):
        return {
            "optimize": PassConfigParam(
                type_=str,
                default_value="full_fusion",
                description="Optimization mode: 'full_fusion', 'decode'.",
            ),
            "model_type": PassConfigParam(
                type_=str,
                default_value=None,
                description="Model type for special handling: 'gpt_oss' or None.",
            ),
            "script_option": PassConfigParam(
                type_=str,
                default_value="jit_npu",
                description="JIT mode: 'jit_npu' (default) or 'non_jit'.",
            ),
            "use_ep": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Use RyzenAI Execution Provider flow.",
            ),
            "no_prune_logits": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Disable pruning of logits (lm_head) during model partitioning.",
            ),
            "max_seq_len": PassConfigParam(
                type_=int,
                default_value=4096,
                description="Maximum sequence length for optimization.",
            ),
            "packed_const": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Enable packed constants in NPU export (legacy flow only).",
            ),
            "cpu_only": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Run only model builder OGA CPU only model, skip NPU-related steps.",
            ),
            "basic": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use basic NPU flow.",
            ),
            "npu_op_version": PassConfigParam(
                type_=str,
                default_value="v2",
                description="NPU LLM op version: 'v2'.",
            ),
        }

    def _run_for_config(
        self, model: Union[HfModelHandler, ONNXModelHandler], config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        from model_generate import generate_npu_model

        # Get input model path - handle both HfModelHandler and ONNXModelHandler
        if isinstance(model, ONNXModelHandler):
            # For ONNX models, get the directory containing the model
            input_model_path = Path(model.model_path)
            if input_model_path.is_file():
                input_model_path = input_model_path.parent
        else:
            # For HF models, use model_path directly
            input_model_path = Path(model.model_path)

        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating Vitis NPU model from: %s", input_model_path)
        logger.info("Output directory: %s", output_dir)
        logger.info(
            "Configuration: optimize=%s, model_type=%s, script_option=%s, use_ep=%s, no_prune_logits=%s",
            config.optimize,
            config.model_type,
            config.script_option,
            config.use_ep,
            config.no_prune_logits,
        )

        generate_npu_model(
            input_model=str(input_model_path),
            output_dir=str(output_dir),
            packed_const=config.packed_const,
            script_option=config.script_option,
            cpu_only=config.cpu_only,
            optimize=config.optimize,
            max_seq_len=config.max_seq_len,
            npu_op_version=config.npu_op_version,
            basic=config.basic,
            use_ep=config.use_ep,
            no_prune_logits=config.no_prune_logits,
            model_type=config.model_type,
        )

        onnx_file_name = "model.onnx"

        logger.info("NPU model generated: %s", output_dir / onnx_file_name)

        return ONNXModelHandler(
            model_path=output_dir,
            onnx_file_name=onnx_file_name,
        )
