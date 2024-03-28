# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.onnx.utils import OnnxDAG
from olive.passes.pass_config import PassConfigParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExtractAdapters(Pass):
    """Extract adapter weights from model and save them as external weights file.

    Model proto is invalid after this pass as the adapter weights point to non-existing external files.
    Inference session must be created by first loading the adapter weights using
    SessionOptions.add_external_initializers.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {}

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        if "lora_modules" not in model.model_attributes:
            raise ValueError("Model does not contain lora_modules attribute")

        # create a dag from the model
        dag = OnnxDAG.from_model_path(model.model_path)
        # remove unnecessary identity nodes
        self.remove_identity_nodes(dag)

        # get lora modules
        lora_modules = model.model_attributes["lora_modules"]
        lora_name_patterns = self._get_lora_name_patterns(lora_modules)

        # dictionary to store adapter weights
        weights = {}

        # nodes to remove at the end
        nodes_to_remove = set()

        for node_name in list(dag.nodes.keys()):
            if dag.get_op_type(node_name) != "MatMul" or not any(
                re.match(pattern, node_name) for pattern in lora_name_patterns
            ):
                # not a lora module
                continue

            # new name for the weight
            new_weight_name = self._create_new_weight_name(node_name)

            # original weight name
            old_weight_name = dag.nodes[node_name].inputs[1]

            if dag.is_input(old_weight_name):
                # nothing we can do here
                continue
            elif dag.is_initializer(old_weight_name):
                # weight is an initializer (not quantized)
                # create initializer with new weight name
                self._create_empty_initializer(dag, weights, old_weight_name, new_weight_name)

                # change input to the new name
                dag.replace_node_input(node_name, old_weight_name, new_weight_name)
            elif dag.get_op_type(dag.get_producer(old_weight_name)) == "DequantizeLinear":
                # weight is quantized
                # get the dequantize node
                old_dequantize_name = dag.get_producer(old_weight_name)
                old_dequantize_node = dag.nodes[old_dequantize_name]

                # new names for the dequantize node inputs
                suffixes = ["quant.weight", "quant.scale", "quant.zero_point"]
                new_input_names = [new_weight_name.replace("weight", suffix) for suffix in suffixes]

                # zero point is optional so we keep track of used inputs
                used_inputs = []
                # create new initializers for the dequantize node
                for old_input, new_input in zip(old_dequantize_node.inputs, new_input_names):
                    self._create_empty_initializer(dag, weights, old_input, new_input)
                    used_inputs.append(new_input)

                # create a new dequantize node
                # NOTE: We could directly modify the original dequantize node but this assumes that the dequantize node
                # is not used elsewhere
                # this cannot be guaranteed (for instance, if the float model has lora modules with same weights, they
                # might all share the same dequantize node)
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
                dag.add_node(new_dequantize_proto, dag.nodes[old_dequantize_name].graph_idx)

                # replace input to the new name
                dag.replace_node_input(node_name, old_weight_name, new_weight_name)

                # add old dequantize node to remove
                nodes_to_remove.add(old_dequantize_name)

        # remove old dequantize nodes
        for node_name in nodes_to_remove:
            dag.remove_node(node_name, check_no_consumers=True)

        # update the model with the changes
        dag.update()

        # save the weights
        # TODO(jambayk): Consider other methods for saving the weights
        # safetensors is an option but it is not available on ARM64 Windows
        weights_path = Path(output_model_path).parent / "adapter_weights.npz"
        np.savez(weights_path, **weights)

        # save the model
        output_model = model_proto_to_olive_model(
            dag.model,
            output_model_path,
            external_data_config={"save_as_external_data": True, "all_tensors_to_one_file": True},
            external_initializers_name=weights_path.name,
        )
        output_model.model_attributes = deepcopy(model.model_attributes)
        output_model.model_attributes["external_initializers"] = list(weights.keys())
        return output_model

    @staticmethod
    def remove_identity_nodes(dag: OnnxDAG):
        """Remove unnecessary identity nodes from the graph."""
        nodes_to_remove = set()
        for node_name in list(dag.nodes.keys()):
            if dag.get_op_type(node_name) != "Identity" or dag.is_output_producer(node_name):
                continue

            # change the input of consumers to the input of the identity node
            for consumer in dag.get_consumers(node_name):
                dag.replace_node_input(consumer, dag.nodes[node_name].outputs[0], dag.nodes[node_name].inputs[0])

            # remove the identity node
            nodes_to_remove.add(node_name)

        for node_name in nodes_to_remove:
            dag.remove_node(node_name, check_no_consumers=True)
        logger.debug("Removed %d Identity nodes", len(nodes_to_remove))

    @staticmethod
    def _get_lora_name_patterns(lora_modules: List[str]) -> List[str]:
        """Get the node name patterns for lora modules."""
        return [f".*[./]{key}[./]{name}[./]MatMul$" for key in lora_modules for name in ["default", "default_1"]]

    @staticmethod
    def _create_new_weight_name(old_name: str) -> str:
        """Create new weight name based on old name.

        The new weight name is of the form model.layers.0.self_attn.q_proj.lora_A.quant.weight
        """
        weight_name = old_name[1:] if old_name.startswith("/") else old_name
        return (
            weight_name.replace("/", ".")
            .replace("default.", "lora_A.")
            .replace("default_1.", "lora_B.")
            .replace("MatMul", "weight")
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
    def _create_empty_initializer(cls, dag: OnnxDAG, weights: Dict[str, "NDArray"], old_name: str, new_name: str):
        """Create an empty initializer with the same shape and type as the old initializer.

        Add the new initializer to the graph and store the weight in a dictionary.

        :param dag: OnnxDAG object
        :param weights: dictionary to store the weights
        :param old_name: name of the initializer to copy
        :param new_name: new initializer name
        """
        assert dag.is_initializer(old_name), f"{old_name} is not an initializer"

        old_proto = dag.ios[old_name].proto

        # store the weight in a dictionary
        weights[new_name] = onnx.numpy_helper.to_array(old_proto)

        # copy initializer
        new_initializer = cls._copy_initializer(old_proto, new_name)
        # add the new initializer to the graph
        dag.add_initializer(new_initializer, dag.ios[old_name].graph_idx)
