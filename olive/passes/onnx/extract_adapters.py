# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type

import numpy as np
import onnx

from olive.common.utils import WeightsFileFormat, save_weights
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import LORA_NAME_PATTERNS, get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExtractAdapters(Pass):
    """Extract adapter weights from ONNX model and save them as external weights file.

    If make_inputs is False, model proto is invalid after this pass as the adapter weights point to non-existent
    external files. Inference session must be created by first loading the adapter weights using
    SessionOptions.add_external_initializers.

    If make_inputs is True, the adapter weights are inputs to the model and must be provided during inference.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "make_inputs": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Convert adapter weights to inputs. If false, the adapter weights will be set as initializers with"
                    " external data."
                ),
            ),
            "dynamic_lora_r": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether the model uses dynamic shape for lora_r. Only used if make_inputs is True. Valid only for"
                    " float modules."
                ),
            ),
            "optional_inputs": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Create default initializers (empty tensor with lora_r dimension set to 0) for the adapter weights,"
                    " if inputs not provided during inference. Only used if make_inputs is True. Valid only for float"
                    " modules."
                ),
            ),
            "save_format": PassConfigParam(
                type_=WeightsFileFormat,
                default_value=WeightsFileFormat.ONNX_ADAPTER,
                description="Format to save the weights in.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # create a dag from the model
        dag = OnnxDAG.from_model_path(model.model_path)
        # remove unnecessary identity nodes
        dag.remove_identity_nodes()

        # dictionary to store adapter weights
        weights = {}
        # keep track of float and quantized modules
        float_modules = set()
        quant_modules = set()

        # nodes to remove at the end
        nodes_to_remove = set()
        for node_name in dag.get_node_names():
            op_type = dag.get_node_op_type(node_name)
            if op_type not in {"MatMul", "MatMulNBits"} or not any(
                re.match(pattern, node_name) for pattern in LORA_NAME_PATTERNS
            ):
                # not a lora module
                continue

            # new name for the float weight
            new_weight_name = self._create_new_weight_name(node_name)
            # new names for quantized weight and parameters
            # zero point is optional if symmetric
            quantized_suffices = [".quant.weight", ".quant.scale", ".quant.zero_point"]
            new_quantized_names = [new_weight_name.replace(".weight", suffix) for suffix in quantized_suffices]

            if op_type == "MatMul":
                # float or QDQ quantized
                # original weight name
                old_weight_name = dag.get_node_inputs(node_name)[1]

                if dag.is_input(old_weight_name):
                    # nothing to do here
                    continue
                elif dag.is_initializer(old_weight_name):
                    # weight is an float initializer
                    # create initializer with new weight name
                    self._externalize_initializer(dag, weights, old_weight_name, new_weight_name)

                    # change input to the new name
                    dag.replace_node_input(node_name, old_weight_name, new_weight_name)

                    # add the module to the float modules
                    float_modules.add(new_weight_name.replace(".weight", ""))
                elif dag.get_node_op_type(dag.get_producer(old_weight_name)) == "DequantizeLinear":
                    # weight is QDQ quantized
                    # get the dequantize node
                    old_dequantize_name = dag.get_producer(old_weight_name)
                    old_dequantize_node = dag.get_node(old_dequantize_name)

                    # zero point is optional so we keep track of used inputs
                    used_inputs = []
                    # create new initializers for the dequantize node
                    for old_input, new_input in zip(old_dequantize_node.inputs, new_quantized_names):
                        self._externalize_initializer(dag, weights, old_input, new_input)
                        used_inputs.append(new_input)

                    # create a new dequantize node
                    # NOTE: We could directly modify the original dequantize node but this assumes that the dequantize
                    # node is not used elsewhere
                    # this cannot be guaranteed (for instance, if the float model has lora modules with same weights,
                    # they might all share the same dequantize node)
                    new_dequantize_proto = onnx.NodeProto()
                    new_dequantize_proto.CopyFrom(old_dequantize_node.proto)
                    # change node name
                    new_dequantize_proto.name = new_weight_name.replace("weight", "dequantize")
                    # change input names
                    for i, new_input in enumerate(used_inputs):
                        new_dequantize_proto.input[i] = new_input
                    # change output name
                    new_dequantize_proto.output[0] = new_weight_name

                    # add new dequantize node
                    dag.add_node(new_dequantize_proto, old_dequantize_node.graph_idx)

                    # replace input to the new name
                    dag.replace_node_input(node_name, old_weight_name, new_weight_name)

                    # add old dequantize node to remove
                    nodes_to_remove.add(old_dequantize_name)

                    # add the module to the quant modules
                    quant_modules.add(new_weight_name.replace(".weight", ".quant"))
            elif op_type == "MatMulNBits":
                # weight is Nbits quantized
                # create empty initializers and change node inputs
                for old_input, new_input in zip(dag.get_node_inputs(node_name)[1:], new_quantized_names):
                    self._externalize_initializer(dag, weights, old_input, new_input)
                    dag.replace_node_input(node_name, old_input, new_input)

                # add the module to the quant modules
                quant_modules.add(new_weight_name.replace(".weight", ".quant"))

        if not weights:
            logger.info("No lora modules found in the model. Returning the original model.")
            return model

        # remove old dequantize nodes
        for node_name in nodes_to_remove:
            dag.remove_node(node_name)

        if config.make_inputs:
            if quant_modules and config.dynamic_lora_r:
                # MatMulNBits has static K,N dimensions which are set as attributes
                # No use case for DequantizeLinear with dynamic lora_r
                logger.info("Quantized modules do not support dynamic_lora_r. Ignoring.")

            # create inputs for the weights
            for weight_name in weights:
                dag.convert_initializer_to_input(weight_name)
                self._make_dynamic_optional(dag, weights, weight_name, config)

        # update the model with the changes
        dag.update()

        # save the weights
        weights_path = save_weights(weights, Path(output_model_path).parent / "adapter_weights", config.save_format)

        # save the model
        output_model = model_proto_to_olive_model(
            dag.model,
            output_model_path,
            config,
            external_initializers_file_name=weights_path.name if not config.make_inputs else None,
            constant_inputs_file_name=weights_path.name if config.make_inputs else None,
        )
        output_model.model_attributes = deepcopy(model.model_attributes) or {}
        # add adapter weights to the model attributes
        output_model.model_attributes["additional_files"] = additional_files = output_model.model_attributes.get(
            "additional_files", []
        )
        additional_files.append(str(weights_path))
        # save information about the weights in the model attributes
        weights_info = {name: [list(value.shape), str(value.dtype)] for name, value in weights.items()}
        if not config.make_inputs:
            output_model.model_attributes["external_initializers"] = weights_info
        else:
            output_model.model_attributes["constant_inputs"] = weights_info
        return output_model

    @staticmethod
    def _create_new_weight_name(old_name: str) -> str:
        """Create new weight name based on old name.

        The new weight name is of the form model.layers.0.self_attn.q_proj.lora_A.quant.weight
        """
        weight_name = old_name[1:] if old_name.startswith("/") else old_name
        matmul_name = weight_name.split("/")[-1]
        return (
            weight_name.replace("/", ".")
            .replace("default.", "lora_A.")
            .replace("default_1.", "lora_B.")
            .replace("default_0.", "lora_A.")
            .replace("default_0_1.", "lora_B.")
            .replace(matmul_name, "weight")
        )

    @staticmethod
    def _copy_initializer(old_initializer: onnx.TensorProto, new_name: str) -> onnx.TensorProto:
        """Copy initializer with a new name and dummy external data location."""
        from onnx.external_data_helper import set_external_data

        # create a new initializer
        new_initializer = onnx.TensorProto()
        # copy the old initializer
        new_initializer.CopyFrom(old_initializer)
        # set the new name
        new_initializer.name = new_name
        # raw_data is required for set_external_data
        if not new_initializer.HasField("raw_data"):
            new_initializer.raw_data = b""
        set_external_data(new_initializer, location="dummy-location.bin")
        # clear the data fields
        new_initializer.ClearField("raw_data")
        new_initializer.ClearField("float_data")
        return new_initializer

    @classmethod
    def _externalize_initializer(cls, dag: OnnxDAG, weights: Dict[str, "NDArray"], old_name: str, new_name: str):
        """Create a new initializer with the same shape and type as the old initializer.

        The initializer points to a dummy external location.
        Add the new initializer to the graph and store the weight in a dictionary.

        :param dag: OnnxDAG object
        :param weights: dictionary to store the weights
        :param old_name: name of the initializer to copy
        :param new_name: new initializer name
        """
        assert dag.is_initializer(old_name), f"{old_name} is not an initializer"

        old_proto = dag.get_io(old_name).proto[-1]

        # store the weight in a dictionary
        weights[new_name] = onnx.numpy_helper.to_array(old_proto)

        # copy initializer
        new_initializer = cls._copy_initializer(old_proto, new_name)
        # add the new initializer to the graph
        dag.add_initializer(new_initializer, dag.get_io(old_name).graph_idx)

    @classmethod
    def _make_dynamic_optional(
        cls, dag: OnnxDAG, weights: Dict[str, "NDArray"], name: str, config: Type[BasePassConfig]
    ):
        """Make the input dynamic and optional."""
        if "quant" in name:
            # dynamic shape not supported for quantized modules
            # cannot have empty tensor as default values, so create default initializers of the same shape
            # scales must be zero to make the dequantized weights zero
            # quant weight and zeros points also made zero to be clean and consistent
            if config.optional_inputs:
                initializer_proto = onnx.numpy_helper.from_array(np.zeros_like(weights[name]), name)
                dag.add_initializer(initializer_proto, 0, keep_input=True)

            return

        # lora r dimension index
        dim_idx = 1 if "lora_A" in name else 0

        # make the input dynamic
        if config.dynamic_lora_r:
            dag.make_input_dim_dynamic(name, dim_idx, "lora_r")

        # create default initializer with the lora_r dimension set to 0
        if config.optional_inputs:
            shape = list(weights[name].shape)
            shape[dim_idx] = 0
            initializer_proto = onnx.numpy_helper.from_array(np.zeros(shape, dtype=weights[name].dtype), name)
            dag.add_initializer(initializer_proto, 0, keep_input=True)
