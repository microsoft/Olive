# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, Type

import onnx
import onnxscript
from onnxscript.rewriter import pattern

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class MatMulToConvTransform(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        model_proto = model.load_model()
        
        def matmul_pattern(op, input_a, input_b):
            return op.MatMul(input_a, input_b)

        def matmul_to_conv_replacement(op, input_a, input_b):
            input_a_rank = len(input_a.shape)
            input_b_rank = len(input_b.shape)
            
            # Only handle 2D×2D and 3D×2D cases
            if not ((input_a_rank == 2 and input_b_rank == 2) or (input_a_rank == 3 and input_b_rank == 2)):
                logger.info(f"Skipping MatMul with shapes {input_a.shape} x {input_b.shape}")
                return op.MatMul(input_a, input_b)
                
            # Step 1: Convert inputs to 4D tensors through unsqueeze as needed
            if input_a_rank == 2:  # 2D×2D case: [M, K] × [K, N]
                # Add batch dimension and spatial dimension: [M, K] to [1, M, K, 1]
                padded_a = op.Unsqueeze(
                    op.Unsqueeze(input_a, op.Constant(value_ints=[0])), 
                    op.Constant(value_ints=[3])
                )
            elif input_a_rank == 3:  # 3Dx2D case: [B, M, K] x [K, N]
                # Add spatial dimension: [B, M, K] to [B, M, K, 1]
                padded_a = op.Unsqueeze(input_a, op.Constant(value_ints=[3]))
                
            # Convert weights: [K, N] to [1, K, N, 1]
            padded_b = op.Unsqueeze(
                op.Unsqueeze(input_b, op.Constant(value_ints=[0])),
                op.Constant(value_ints=[3])
            )
                
            # Step 2: Transpose for Conv format
            if input_a_rank == 2:
                # [1, M, K, 1] to [1, K, M, 1]
                transposed_a = op.Transpose(padded_a, perm=[0, 2, 1, 3])
            elif input_a_rank == 3:
                # [B, M, K, 1] to [B, K, M, 1]
                transposed_a = op.Transpose(padded_a, perm=[0, 2, 1, 3])
                
            # [1, K, N, 1] to [N, K, 1, 1]
            transposed_b = op.Transpose(padded_b, perm=[2, 1, 0, 3])
            
            # Step 3: Apply Conv operation
            conv_output = op.Conv(transposed_a, transposed_b)
            
            # Step 4: Transpose Conv output back
            # [1, N, M, 1] or [B, N, M, 1] to [1, M, N, 1] or [B, M, N, 1]
            transposed_out = op.Transpose(conv_output, perm=[0, 2, 1, 3])
            
            # Step 5: Squeeze back to original dimensions
            if input_a_rank == 2:
                # [1, M, N, 1] to [M, N]
                return op.Squeeze(
                    op.Squeeze(transposed_out, op.Constant(value_ints=[3])),
                    op.Constant(value_ints=[0])
                )
            elif input_a_rank == 3:
                # [B, M, N, 1] to [B, M, N]
                return op.Squeeze(transposed_out, op.Constant(value_ints=[3]))

        def has_non4d_inputs(context, input_a, input_b) -> bool:
            is_applicable = len(input_a.shape) != 4 or len(input_b.shape) != 4
            if is_applicable:
                logger.debug(f"Found MatMul with non-4D inputs: {input_a.name} (shape={input_a.shape}), {input_b.name} (shape={input_b.shape})")
            return is_applicable

        matmul_to_conv_rule = pattern.RewriteRule(
            matmul_pattern,
            matmul_to_conv_replacement,
            has_non4d_inputs,
            verbose=1
        )

        transformed_model_proto = onnxscript.rewriter.rewrite(
            model_proto,
            pattern_rewrite_rules=[matmul_to_conv_rule],
        )
        logger.info(f"MatMul to Conv transformation applied.")
        
        return model_proto_to_olive_model(transformed_model_proto, output_model_path, config)
