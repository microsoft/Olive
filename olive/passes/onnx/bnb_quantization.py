# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from pathlib import Path
from typing import Optional

import onnx
from packaging import version

from olive.constants import Precision
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class OnnxBnb4Quantization(Pass):
    """Quantize MatMul nodes in ONNX model using 4bit FP4/NF4 quantization."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "precision": PassConfigParam(
                type_=Optional[Precision],
                default_value=None,
                description="The quantization type. Only 'fp4' and 'nf4' are supported.",
            ),
            "quantized_modules": PassConfigParam(
                type_=list[str],
                description=(
                    "The list of modules to quantize. Node names will be matched as '.*[./]{module}[./]MatMul$'. If not"
                    " specified, all MatMul nodes will be quantized except those specified in nodes_to_exclude."
                ),
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of node names to exclude from quantization.",
            ),
        }
        config.update(get_external_data_config())
        return config

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        # Only fp4 and nf4 are supported
        return not config.precision or config.precision in {Precision.FP4, Precision.NF4}

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime import __version__ as OrtVersion

        assert version.parse(OrtVersion) >= version.parse("1.16.2"), (
            "MatMulBnb4Quantizer is only supported in onnxruntime >= 1.16.2"
        )

        from onnxruntime.quantization.matmul_bnb4_quantizer import MatMulBnb4Quantizer

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        precision = config.precision.value if config.precision else None
        quantized_modules = config.quantized_modules
        if model.model_attributes:
            quantized_modules = quantized_modules or model.model_attributes.get("quantized_modules")

            # extract precision from model_attributes if not specified in config
            if not precision:
                quant_config = model.model_attributes.get("quantization_config") or {}

                if quant_config.get("load_in_8bit"):
                    raise ValueError("load_in_8bit is not supported. Only 4-bit quantization is supported.")

                precision = quant_config.get("bnb_4bit_quant_type")
                if not precision:
                    raise ValueError("precision is required.")

                if quant_config.get("bnb_4bit_use_double_quant"):
                    logger.info(
                        "bnb_4bit_use_double_quant is set to True but double quantization is not supported. Ignoring."
                    )
        assert precision in {"fp4", "nf4"}, f"quant_type must be one of 'fp4' or 'nf4'. Got {precision}."
        quant_type_enum = getattr(MatMulBnb4Quantizer, precision.upper())

        # load the model
        onnx_model = model.load_model()

        # get nodes to exclude from quantization
        nodes_to_exclude = config.nodes_to_exclude or []

        # find all MatMul nodes in the graph
        matmul_nodes = self._find_matmul_nodes(onnx_model.graph)
        # filter based on quantized_modules
        quantized_modules = set(quantized_modules or [])
        nodes_to_exclude = nodes_to_exclude + [
            node
            for node in matmul_nodes
            if quantized_modules and not any(re.match(f".*[./]{key}[./]MatMul$", node) for key in quantized_modules)
        ]

        # quantize the model
        quantizer = MatMulBnb4Quantizer(
            onnx_model, quant_type=quant_type_enum, block_size=64, nodes_to_exclude=nodes_to_exclude
        )
        quantizer.process()
        # topologically sort the graph at the end since previous optimizations may have broken it
        quantizer.model.topological_sort()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    @classmethod
    def _find_matmul_nodes(cls, graph: onnx.GraphProto) -> list[str]:
        """Find all MatMul nodes in the graph and return their names."""
        matmul_nodes = []
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    # recursive call to take care of sub-graph
                    matmul_nodes += cls._find_matmul_nodes(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        # recursive call to take care of sub-graph
                        matmul_nodes += cls._find_matmul_nodes(subgraph)
            if node.op_type == "MatMul":
                matmul_nodes.append(node.name)

        return matmul_nodes
