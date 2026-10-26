# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

from olive.constants import MSFT_DOMAIN
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, ir_model_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ONNXModelHandler

logger = logging.getLogger(__name__)


class OnnxMoEQuantization(Pass):
    """Quantize constant MoE expert banks to raw, symmetric CUDA QMoE INT4/INT8.

    Routing, activation precision, biases and expert row order are unchanged.
    Weights use [experts, output, input/pack] layout with group scales along
    the input dimension. CUDA prepacking requires a compatible ONNX Runtime
    build supporting weights_prepacked=0. Repeated nodes share initializers,
    but each CUDA kernel may maintain its own prepacked copy. Separate FC3
    nodes are left unchanged because CUDA integer QMoE requires fused banks.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(type_=int, default_value=4, description="Expert weight bits: 4 or 8."),
            "block_size": PassConfigParam(
                type_=int, default_value=128, description="Input-dimension group size: 64 or 128."
            ),
            "nodes_to_exclude": PassConfigParam(type_=list, default_value=None),
            "nodes_to_include": PassConfigParam(type_=list, default_value=None),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if config.bits not in (4, 8) or config.block_size not in (64, 128):
            raise ValueError("MoE quantization requires bits=4 or 8 and block_size=64 or 128.")
        ir_model = model.load_ir_model()
        packed_weights = {}
        count = 0
        for node in list(ir_model.graph.all_nodes()):
            if node.domain != MSFT_DOMAIN or node.op_type != "MoE":
                continue
            if node.name in (config.nodes_to_exclude or []):
                continue
            if config.nodes_to_include and node.name not in config.nodes_to_include:
                continue
            if len(node.inputs) > 6 and node.inputs[6] is not None:
                logger.warning("Skipping %s: CUDA integer QMoE does not support separate FC3 banks.", node.name)
                continue
            banks = [node.inputs[i] if i < len(node.inputs) else None for i in (2, 4, 6)]
            if any(value is not None and value.const_value is None for value in banks):
                logger.warning(
                    "Skipping %s: expert banks must be constant; fold weight-only transforms first.", node.name
                )
                continue
            if banks[0] is None or banks[1] is None:
                raise ValueError(f"MoE node {node.name} is missing an expert bank.")

            inputs = list(node.inputs[:2])
            for index, bank in zip((2, 4, 6), banks):
                bias = node.inputs[index + 1] if index + 1 < len(node.inputs) else None
                if bank is None:
                    inputs.extend((None, None, bias))
                    continue
                if bank.const_value.dtype not in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
                    raise ValueError("CUDA QMoE expert banks require float16 or bfloat16.")
                key = (node.graph, bank)
                if key not in packed_weights:
                    packed, scales = self.quantize_experts(bank.const_value.numpy(), config.bits, config.block_size)
                    quantized = ir.Value(name=f"{bank.name}_Q{config.bits}", const_value=ir.tensor(packed))
                    scale_value = ir.Value(name=f"{bank.name}_scales", const_value=ir.tensor(scales))
                    node.graph.register_initializer(quantized)
                    node.graph.register_initializer(scale_value)
                    packed_weights[key] = (quantized, scale_value)
                inputs.extend((*packed_weights[key], bias))

            attributes = dict(node.attributes)
            attributes.update(
                expert_weight_bits=config.bits,
                block_size=config.block_size,
                quant_type="int",
                weights_prepacked=0,
            )
            replacement = ir.node("QMoE", domain=MSFT_DOMAIN, inputs=inputs, attributes=attributes, name=node.name)
            for old, new in zip(node.outputs, replacement.outputs):
                new.name, new.type, new.shape = old.name, old.type, old.shape
            ir.convenience.replace_nodes_and_values(
                node.graph, node, [node], [replacement], node.outputs, replacement.outputs
            )
            count += 1
        ir_model.graph.opset_imports[MSFT_DOMAIN] = 1
        ir.passes.common.RemoveUnusedNodesPass()(ir_model)
        logger.info("Quantized %d MoE nodes with %d shared expert banks.", count, len(packed_weights))
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        return ir_model_to_olive_model(ir_model, output_model_path, config)

    @staticmethod
    def quantize_experts(weights: np.ndarray, bits: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Pack [E, N, K] weights using the effective stored scale precision."""
        if bits not in (4, 8) or block_size not in (64, 128):
            raise ValueError("MoE quantization requires bits=4 or 8 and block_size=64 or 128.")
        if weights.ndim != 3 or weights.shape[-1] % block_size:
            raise ValueError("Expert weights must be [E, N, K] with K divisible by block_size.")
        if str(weights.dtype) not in ("float32", "float16", "bfloat16"):
            raise ValueError("Expert weights must be float32, float16 or bfloat16.")
        blocks = weights.astype(np.float32).reshape(*weights.shape[:2], -1, block_size)
        if not np.isfinite(blocks).all():
            raise ValueError("Expert weights must be finite.")
        maximum = np.max(np.abs(blocks), axis=-1)
        midpoint = 1 << (bits - 1)
        scales = np.where(maximum == 0, 1, maximum / (midpoint - 1)).astype(weights.dtype)
        effective_scales = scales.astype(np.float32)
        if not np.isfinite(effective_scales).all() or np.any(effective_scales <= 0):
            raise ValueError("Expert scales must remain finite and positive in the activation dtype.")
        # BF16/F16 narrowing can change code selection: divide by the stored scale.
        codes = np.clip(np.rint(blocks / effective_scales[..., None]), -midpoint, midpoint - 1).reshape(weights.shape)
        codes = (codes + midpoint).astype(np.uint8)
        pack = 8 // bits
        packed = np.zeros((*weights.shape[:2], weights.shape[-1] // pack), dtype=np.uint8)
        for index in range(pack):
            packed |= codes[..., index::pack] << (index * bits)
        return packed, scales
