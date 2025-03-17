
from pathlib import Path
from typing import Dict, List, Type, Union

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.olive_pass import Pass

from olive.passes.pass_config import BasePassConfig,PassConfigParam, ParamCategory
from olive.model import ONNXModelHandler
from typing import Dict, Any, Union
from onnx import helper, numpy_helper
import onnx
from pathlib import Path
import onnx
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.model import ONNXModelHandler
from olive.passes.vaip.npu_model_gen.preprocess import process_model
import os

class ONNXRuntimeShapeInferencePass(Pass):
    """Perform shape inference on ONNX models using ONNX Runtime's implementation."""

    @classmethod
    def _default_config(cls,accelerator_spec: AcceleratorSpec):
        return {
            "check_model": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Validate model after shape inference",
            ),
            "strict_mode": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Enable strict shape inference mode",
            )
        }

    def _run_for_config(self, model: ONNXModelHandler, config: dict, output_model_path: str):
        # Get input model path and load
        model_path = model.model_path
        output_model_path = Path(output_model_path) / "model_shape_inferred.onnx"

        # Perform shape inference using ONNX Runtime's implementation
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        inferred_model = SymbolicShapeInference.infer_shapes(
            onnx.load(model_path),
            auto_merge=True
        )

        # Save the inferred model
        onnx.save(inferred_model, output_model_path)

        # Optional model validation
        if config.check_model:
            onnx.checker.check_model(output_model_path)

        return ONNXModelHandler(output_model_path)

class AddCastNodes(Pass):
    """
    This pass adds cast nodes above and below a given operator type (optype).
    """
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "op_types": PassConfigParam(
                type_= List,
                required=True,
                description="The operator type around which cast nodes will be added."
            ),
            "cast_to": PassConfigParam(
                type_=str,
                required=True,
                description="The data type to which the input and output should be cast. E.g., 'float32', 'float16'."
            )
        }
    def _run_for_config(self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str) -> ONNXModelHandler:
            optypes = config.op_types
            cast_to = config.cast_to

            onnx_model = model.load_model()

            # Collect indices of nodes to process
            nodes_to_process = []
            for i, node in enumerate(onnx_model.graph.node):
                if node.op_type in optypes:
                    nodes_to_process.append(i)

            # Process nodes in reverse order to handle insertions correctly
            for i in sorted(nodes_to_process, reverse=True):
                node = onnx_model.graph.node[i]

                input_name = node.input[0]
                output_name = node.output[0]

                cast_input_name = input_name + "_cast_input"
                cast_output_name = output_name + "_cast_output"

                # Create input Cast node (casts to desired type)
                cast_input_node = helper.make_node(
                    'Cast',
                    inputs=[input_name],
                    outputs=[cast_input_name],
                    to=getattr(onnx.TensorProto, cast_to.upper())
                )

                # Create output Cast node (casts back to original type)
                cast_output_node = helper.make_node(
                    'Cast',
                    inputs=[cast_output_name],  # Now takes the modified output
                    outputs=[output_name],      # Restores original output name
                    to=onnx.TensorProto.FLOAT   # Adjust to original type if needed
                )

                # Update the original node's input/output
                node.input[0] = cast_input_name
                node.output[0] = cast_output_name  # Node now outputs to cast_output_name

                # Insert cast_input_node before the original node
                onnx_model.graph.node.insert(i, cast_input_node)
                # Insert cast_output_node after the original node (now at i+1)
                onnx_model.graph.node.insert(i + 2, cast_output_node)

            onnx.save(onnx_model, output_model_path + 'model.onnx')
            return ONNXModelHandler(output_model_path + 'model.onnx')


class RyzenLLMPreprocess(Pass):
    """
    This pass 
    - adds cast nodes
    - Changes domain to com.amd 
    - Fuse SSMLP and GQO oprator
    """
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "custom_ops": PassConfigParam(
                type_= bool,
                required=False,
                default_value=True,
                description="Pass this if AMD custom_ops required."
            ),
            "fuse": PassConfigParam(
                type_=bool,
                required=False,
                default_value=False,
                description="Pass this if fused SSMLP+GQO model is required"
            ),
            "fuse_SSMLP": PassConfigParam(
                type_=bool,
                required=False,
                default_value=True,
                description="Pass this if fused SSMLP model is required"
            ),
            "fuse_GQO": PassConfigParam(
                type_=bool,
                required=False,
                default_value=False,
                description="Pass this if fused GQO model is required"
            ),
            "pack_const": PassConfigParam(
                type_=bool,
                required=False,
                default_value=False,
                description="Pack constants for MatMulNbits"
            )
        }
    def _run_for_config(self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str) -> ONNXModelHandler:
            custom_op = config.custom_ops
            fuse = config.fuse
            fuse_SSMLP=config.fuse_SSMLP
            fuse_GQO=config.fuse_GQO
            pack_const=config.pack_const
            onnx_path = model.model_path
            args={
                 "input_model":onnx_path,
                 "output_model":os.path.join(output_model_path,"model.onnx"),
                 "custom_ops":custom_op,
                 "fuse":fuse,
                 "fuse_SSMLP":fuse_SSMLP,
                 "fuse_GQO":fuse_GQO,
                 "packed_const":pack_const
            }
            try:
                process_model(args)
            except Exception as e:
                 raise
                 
            return ONNXModelHandler(os.path.join(output_model_path,"model.onnx"))
