# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path

import numpy as np
import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


# TODO(anyone): Move from onnxruntime.transformers.onnx_model.OnnxModel to OnnxDAG
# or reimplement logic using onnx-rewriter
# no need to create a new instance of OnnxModel for each optimization
class ModelOptimizer:
    def __init__(self, source_model_path):
        self.source_model_path = str(source_model_path)
        self.model = onnx.load(self.source_model_path)

    def fuse_reshape_operations(self):
        # Remove unnecessary Reshape operator. Consecutive Reshape operators with latter's input being "[-1]"
        # i.e. flatten the input, the former Reshape operator is useless."""
        from onnxruntime.transformers.onnx_model import OnnxModel as TransformersOnnxModel

        o_model = TransformersOnnxModel(self.model)
        o_producers = o_model.output_name_to_node()
        o_consumers = o_model.input_name_to_nodes()

        expected_constant_value = np.array([-1])

        num_changed = 0
        for o_reshape_node in o_model.get_nodes_by_op_type("Reshape"):
            input_node_0 = o_producers.get(o_reshape_node.input[0])

            previous_is_reshape = input_node_0 and (input_node_0.op_type == "Reshape")
            current_value = o_model.get_constant_value(o_reshape_node.input[1])
            current_flattens = np.array_equal(current_value, expected_constant_value)
            only_consumer = len(o_consumers[o_reshape_node.input[0]]) == 1

            if previous_is_reshape and current_flattens and only_consumer:
                o_reshape_node.input[0] = input_node_0.input[0]
                input_node_0.input.remove(input_node_0.input[0])
                num_changed += 1

        if num_changed > 0:
            logger.debug("Fused %d redundant Reshape operators", num_changed)
            o_model.prune_graph()

    def onnxscript_optimize(self):
        try:
            import onnxscript
        except ImportError:
            logger.warning("Please install `onnxscript` to apply more optimization.")
            return
        try:
            logger.debug("Running onnxscript optimizer")
            self.model = onnxscript.optimizer.optimize(self.model)
        except Exception as e:
            if "TypeInferenceError" in str(e):
                logger.info(
                    "onnxscript optimizer failed with %s. Rerunning with shape inference disabled.",
                    str(e),
                )
                self.model = onnxscript.optimizer.optimize(self.model, onnx_shape_inference=False)
            else:
                raise

    def onnxoptimizer_optimize(self):
        try:
            from onnxoptimizer import optimize
        except ImportError:
            logger.warning("Please install `onnxoptimizer` to apply more optimization.")
            return

        self.model = optimize(self.model)


class OnnxPeepholeOptimizer(Pass):
    """Optimize ONNX model by fusing nodes."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # optimize model
        peephole_optimizer = ModelOptimizer(model.model_path)
        peephole_optimizer.onnxscript_optimize()
        peephole_optimizer.onnxoptimizer_optimize()
        peephole_optimizer.fuse_reshape_operations()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(peephole_optimizer.model, output_model_path, config)
