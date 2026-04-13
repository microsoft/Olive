#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path
from typing import Optional, Union

from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class VitisGenerateModelLLM(Pass):
    """Olive pass for generating optimized NPU/Hybrid models using AMD Vitis toolchain.

    Uses model_generate v2 API which wraps onnx_utils optimize.

    Supported modes:
    - npu: NPU-only model with configurable recipe (default)
    - hybrid: Hybrid model (NPU prefill + GPU token)

    Supported recipes (NPU mode):
    - full_fusion: Maximum performance (default)
    - token_fusion: Better TPS with NPU token fusion
    - eager: Eager execution (simplest, most compatible)
    - basic: MatMulNBits only (safest for new/unsupported models)

    Input can be:
    - HfModelHandler: For models coming from quantization pass
    - ONNXModelHandler: For pre-quantized or pre-exported OGA models
    """

    @classmethod
    def _default_config(cls, accelerator_spec):
        return {
            "mode": PassConfigParam(
                type_=str,
                default_value="npu",
                description="Generation mode: 'npu' or 'hybrid'.",
            ),
            "recipe": PassConfigParam(
                type_=str,
                default_value="full_fusion",
                description=(
                    "NPU recipe: 'full_fusion' (max performance), 'token_fusion' (better TPS), "
                    "'eager' (simplest), 'basic' (safest for new models). Ignored for hybrid mode."
                ),
            ),
            "mem_optimize": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Optimize for 16GB laptops (adds --priority memory).",
            ),
            "model_name": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description="Input ONNX file name to optimize. Default: 'model.onnx'.",
            ),
            "output_model_name": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description="Output ONNX file name. Default: same as model_name.",
            ),
            "extra_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Additional onnx_utils options passed as key-value pairs. "
                    "Examples: {'model_type': 'gpt-oss'}, {'max_seq_len': '4096'}, "
                    "{'no_prune_logits': 'true'}."
                ),
            ),
        }

    def _run_for_config(
        self, model: Union[HfModelHandler, ONNXModelHandler], config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        from model_generate import generate_model

        # Resolve input model path
        if isinstance(model, ONNXModelHandler):
            input_model_path = Path(model.model_path)
            if input_model_path.is_file():
                input_model_path = input_model_path.parent
        else:
            input_model_path = Path(model.model_path)

        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating Vitis model from: %s", input_model_path)
        logger.info("Mode: %s, Recipe: %s", config.mode, config.recipe)

        generate_model(
            mode=config.mode,
            output_dir=str(output_dir),
            input_model=str(input_model_path),
            recipe=config.recipe,
            mem_optimize=config.mem_optimize,
            model_name=config.model_name,
            output_model_name=config.output_model_name,
            extra_options=config.extra_options,
            verbose=1,
        )

        # Determine output ONNX filename
        output_model_name = config.output_model_name or config.model_name or "model.onnx"

        logger.info("Model generated: %s", output_dir / output_model_name)

        return ONNXModelHandler(
            model_path=output_dir,
            onnx_file_name=output_model_name,
        )
