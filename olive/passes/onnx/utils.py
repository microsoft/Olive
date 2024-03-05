# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import tempfile
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Union

import onnx
from onnx import AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto, numpy_helper
from onnx.shape_inference import infer_shapes_path

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Field
from olive.common.utils import onnx_dtype_to_np_dtype

if TYPE_CHECKING:
    from onnx import ModelProto

logger = logging.getLogger(__name__)


class SpecialInput(str, Enum):
    """Special inputs for ONNX nodes."""

    INPUT = "__input__"  # user input
    INITIALIZER = "__initializer__"  # constant initializer


class SpecialOutput(str, Enum):
    """Special outputs for ONNX nodes."""

    OUTPUT = "__output__"  # model output


class OnnxNode(ConfigBase):
    """ONNX node."""

    op_type: str
    inputs: List[str]
    outputs: List[str]
    graph_idx: int
    proto: NodeProto  # reference to the node in the model graph

    def __str__(self):
        return f"{self.op_type}({', '.join(self.inputs)} -> {', '.join(self.outputs)})"


class OnnxIO(ConfigBase):
    """ONNX input/output.

    Behaves similar to labeled edges in a graph but can connect to multiple nodes.
    """

    dtype: str = None
    shape: List = None
    source: str = None
    destination: List[str] = Field(default_factory=list)
    graph_idx: int
    proto: Union[ValueInfoProto, TensorProto]
    scalar_value: Union[int, float] = None

    def __str__(self):
        return f"{self.proto.name}({self.dtype}, {self.shape})"


class OnnxDAG:
    """ONNX model as a directed acyclic graph (DAG)."""

    def __init__(self, model: "ModelProto"):
        self.model = model
        self.graphs = self.get_all_graphs(model)
        self.nodes: Dict[str, OnnxNode] = {}
        self.ios: Dict[str, OnnxIO] = {}
        self.connections = defaultdict(list)

        # traverse the graphs and populate nodes, ios, and connections
        for idx, graph in enumerate(self.graphs):
            self.process_io(graph, self.ios, idx)
            for node in graph.node:
                self.process_node(node, self.nodes, self.ios, self.connections, idx)

    @staticmethod
    def get_all_graphs(model: "ModelProto") -> List[GraphProto]:
        """Get all graphs in the model."""
        all_graphs = []
        graph_queue = [model.graph]
        while graph_queue:
            graph = graph_queue.pop(0)
            all_graphs.append(graph)
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == AttributeProto.AttributeType.GRAPH:
                        assert isinstance(attr.g, GraphProto)
                        graph_queue.append(attr.g)
                    if attr.type == AttributeProto.AttributeType.GRAPHS:
                        for g in attr.graphs:
                            assert isinstance(g, GraphProto)
                            graph_queue.append(g)
        return all_graphs

    @staticmethod
    def _get_io_type_shape(io: ValueInfoProto) -> Dict:
        """Get the type and shape of an input/output."""
        tensor_type = io.type.tensor_type
        if tensor_type.elem_type == 0:
            # sequence type
            # TODO(jambayk): add support for different types
            # refer to https://github.com/lutzroeder/netron/blob/main/source/onnx.js#L1424
            tensor_type = io.type.sequence_type.elem_type.tensor_type
        data_type = onnx_dtype_to_np_dtype(tensor_type.elem_type)
        shape = [dim.dim_param if dim.dim_param else dim.dim_value for dim in tensor_type.shape.dim]
        return {
            "dtype": data_type,
            "shape": shape,
        }

    @staticmethod
    def _get_scalar_value(proto, shape: List[int]) -> Union[int, float]:
        """Get scalar value from an initializer or constant node."""
        scalar_value = None
        if shape in ([], [1]):
            value = numpy_helper.to_array(proto).item()
            if isinstance(value, (int, float)):
                scalar_value = value
        return scalar_value

    @classmethod
    def process_io(cls, graph: GraphProto, ios: Dict[str, OnnxIO], graph_idx: int):
        """Process inputs, outputs, initializers, and value_info.

        This will populate ios. Should be called before adding nodes.
        """
        for i in graph.input:
            ios[i.name] = OnnxIO(
                proto=i,
                source=SpecialInput.INPUT,
                graph_idx=graph_idx,
                **cls._get_io_type_shape(i),
            )
        for o in graph.output:
            ios[o.name] = OnnxIO(
                proto=o,
                destination=[SpecialOutput.OUTPUT],
                graph_idx=graph_idx,
                **cls._get_io_type_shape(o),
            )
        for initializer in graph.initializer:
            shape = list(initializer.dims)
            # extract scalar value
            scalar_value = cls._get_scalar_value(initializer, shape)
            ios[initializer.name] = OnnxIO(
                proto=initializer,
                source=SpecialInput.INITIALIZER,
                graph_idx=graph_idx,
                dtype=onnx_dtype_to_np_dtype(initializer.data_type),
                shape=shape,
                scalar_value=scalar_value,
            )
        for vi in graph.value_info:
            ios[vi.name] = OnnxIO(
                proto=vi,
                graph_idx=graph_idx,
                **cls._get_io_type_shape(vi),
            )
        return ios

    @classmethod
    def process_node(
        cls,
        node_proto: NodeProto,
        nodes: Dict[str, OnnxNode],
        ios: Dict[str, OnnxIO],
        connections: Dict[str, List[str]],
        graph_idx: int,
    ):
        """Process a node and populate the nodes and connections attributes."""
        name = node_proto.name
        onnx_node = OnnxNode(
            proto=node_proto,
            op_type=node_proto.op_type,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            graph_idx=graph_idx,
        )
        nodes[name] = onnx_node

        for i in node_proto.input:
            ios[i].destination.append(name)
            parent = ios[i].source
            if parent not in [SpecialInput.INPUT, SpecialInput.INITIALIZER]:
                connections[parent].append(name)

        for o in node_proto.output:
            if ios[o].source is not None:
                raise ValueError(f"Output {o} is already connected to another node.")
            ios[o].source = name
            # get scalar value if node is a constant
            if node_proto.op_type == "Constant":
                for attr in node_proto.attribute:
                    if attr.name == "value":
                        ios[o].scalar_value = cls._get_scalar_value(attr.t, ios[o].shape)
                        break
            for destination in ios[o].destination:
                if destination != SpecialOutput.OUTPUT and destination not in connections[name]:
                    connections[name].append(destination)

    def add_node(self, node_proto: NodeProto, graph_idx: int):
        """Add a node to the graph.

        This adds the node to the `nodes` attribute and connects them using the `ios` attribute.
        """
        self.process_node(node_proto, self.nodes, self.ios, self.connections, graph_idx)

    def remove_node(self, node_name: str, check_no_consumers: bool = False):
        """Remove a node from the graph."""
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist in the graph.")

        if check_no_consumers and self.connections[node_name]:
            raise ValueError(f"Node {node_name} has consumers.")

        node = self.nodes.pop(node_name)
        for i in node.inputs:
            self.ios[i].destination.remove(node_name)
            parent = self.ios[i].source
            if parent not in [SpecialInput.INPUT, SpecialInput.INITIALIZER]:
                self.connections[parent].remove(node_name)

        for o in node.outputs:
            self.ios[o].source = None

        del self.connections[node_name]

    def replace_nodes(self, old_node_names: List[str], new_node_proto: NodeProto):
        """Replace a chain of nodes in the graph with a new node."""
        inputs = []
        outputs = []
        graph_idx = self.nodes[old_node_names[0]].graph_idx
        for idx, node in enumerate(old_node_names):
            # get the inputs and outputs of the chain
            inputs.extend(
                self.nodes[node].inputs
                if idx == 0
                else [i for i in self.nodes[node].inputs if self.ios[i].source != old_node_names[idx - 1]]
            )

            if idx == len(old_node_names) - 1:
                outputs.extend(self.nodes[node].outputs)
                continue

            # check that the node is connected to the next node in the chain
            if len(self.connections[node]) != 1:
                raise ValueError(f"Node {node} is not connected to exactly one node.")
            if self.connections[node][0] != old_node_names[idx + 1]:
                raise ValueError(f"Node {node} is not connected to the next node in the chain.")
            if self.nodes[node].graph_idx != graph_idx:
                raise ValueError(f"Node {node} is not in the same graph as the other nodes.")

        # constant scalar inputs are absorbed by the new node
        constant_scalar = [i for i in inputs if self.ios[i].scalar_value is not None]
        # check that the inputs and outputs match
        if (set(inputs) - set(constant_scalar)) != set(new_node_proto.input):
            raise ValueError("Inputs do not match.")
        if set(outputs) != set(new_node_proto.output):
            raise ValueError("Outputs do not match.")

        # remove the old nodes
        for node in old_node_names[::-1]:
            self.remove_node(node)
        # remove constant nodes whose inputs are not needed anymore
        for i in constant_scalar:
            node = self.ios[i].source
            if not self.connections[node]:
                self.remove_node(node)

        # add the new node
        self.add_node(new_node_proto, graph_idx)

    def replace_node_input(self, node_name: str, old_input: str, new_input: str):
        """Replace an input of a node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist in the graph.")
        if old_input not in self.nodes[node_name].inputs:
            raise ValueError(f"Input {old_input} does not exist in node {node_name}.")
        if new_input not in self.ios:
            raise ValueError(f"Input {new_input} does not exist in the graph.")

        node = self.nodes[node_name]
        # update the node object and proto
        node.inputs[node.inputs.index(old_input)] = new_input
        proto_updated = False
        for i in range(len(node.inputs)):
            if node.proto.input[i] == old_input:
                node.proto.input[i] = new_input
                proto_updated = True
        if not proto_updated:
            raise ValueError(f"Input {old_input} does not exist in node {node_name} proto.")

        # update the ios
        self.ios[old_input].destination.remove(node_name)
        self.ios[new_input].destination.append(node_name)

        # update the connections
        old_parent = self.ios[old_input].source
        if old_parent not in [SpecialInput.INPUT, SpecialInput.INITIALIZER]:
            self.connections[old_parent].remove(node_name)

        new_parent = self.ios[new_input].source
        if new_parent not in [SpecialInput.INPUT, SpecialInput.INITIALIZER]:
            self.connections[new_parent].append(node_name)

    def get_shape(self, io_name: str) -> List:
        """Get the shape of an input/output."""
        return list(self.ios[io_name].shape)

    def get_op_type(self, node_name: str) -> str:
        """Get the op type of a node."""
        return self.nodes[node_name].op_type

    def get_consumers(self, node_name: str) -> List[str]:
        """Get the consumers of a node."""
        return list(self.connections[node_name])

    def get_input_names(self, node_name: str) -> List[str]:
        """Get the input names of a node."""
        return list(self.nodes[node_name].inputs)

    def get_input_names_or_scalar(self, node_name: str) -> List[Union[str, int, float]]:
        """Get the input names of a node. If the input is a scalar, return the scalar value."""
        inputs = self.get_input_names(node_name)
        for idx, input_name in enumerate(inputs):
            if self.ios[input_name].scalar_value is not None:
                inputs[idx] = self.ios[input_name].scalar_value
        return inputs

    def get_output_names(self, node_name: str) -> List[str]:
        """Get the output names of a node."""
        return list(self.nodes[node_name].outputs)

    def get_input_shapes(self, node_name: str) -> List:
        """Get the input shapes of a node."""
        return [self.ios[i].shape for i in self.nodes[node_name].inputs]

    def get_output_shapes(self, node_name: str) -> List:
        """Get the output shapes of a node."""
        return [self.ios[o].shape for o in self.nodes[node_name].outputs]

    def get_input_dtypes(self, node_name: str) -> List:
        """Get the input dtypes of a node."""
        return [self.ios[i].dtype for i in self.nodes[node_name].inputs]

    def get_output_dtypes(self, node_name: str) -> List:
        """Get the output dtypes of a node."""
        return [self.ios[o].dtype for o in self.nodes[node_name].outputs]

    def get_node_protos(self, node_names: List[str]) -> List[NodeProto]:
        """Get the node protos from the graph."""
        for name in node_names:
            if name not in self.nodes:
                return None

        return [self.nodes[name].proto for name in node_names]

    def get_model_inputs(self) -> List[str]:
        """Get the model inputs."""
        return [i for i, io in self.ios.items() if io.source == SpecialInput.INPUT]

    def get_model_outputs(self) -> List[str]:
        """Get the model outputs."""
        return [o for o, io in self.ios.items() if SpecialOutput.OUTPUT in io.destination]

    def _topological_sort_util(self, v: str, visited: Set[str], order: List[str]):
        # keep track of the nodes to visit
        stack = [v]

        while stack:
            v = stack.pop()
            visited.add(v)

            for neighbor in self.connections[v]:
                if neighbor not in visited:
                    # remember to come back to this node
                    stack.append(v)
                    # visit the neighbor
                    stack.append(neighbor)
                    break
            else:
                order.insert(0, v)

    def topological_sort(self):
        visited = set()
        order = []

        for v in self.nodes:
            if v not in visited:
                self._topological_sort_util(v, visited, order)

        return order

    def update(self):
        """Update the graph proto with the latest nodes and connections."""
        node_order = self.topological_sort()

        for idx, graph in enumerate(self.graphs):
            # update the nodes in the graph proto
            nodes = [self.nodes[name].proto for name in node_order if self.nodes[name].graph_idx == idx]

            # assume inputs, outputs and initializers have not changed
            value_info = []
            for io in self.ios.values():
                if io.graph_idx != idx:
                    continue
                if io.source in [None, SpecialInput.INPUT, SpecialInput.INITIALIZER]:
                    # skip inputs, initializers
                    # skip if parent node is removed
                    continue
                if not io.destination or SpecialOutput.OUTPUT in io.destination:
                    # skip output
                    # skip if destination nodes are removed
                    continue
                value_info.append(io.proto)

            # update the graph proto
            graph.ClearField("node")
            graph.node.extend(nodes)
            graph.ClearField("value_info")
            graph.value_info.extend(value_info)

    def remove_redundant_cast_nodes(self):
        """Remove redundant cast nodes from the graph."""
        model_outputs = self.get_model_outputs()
        removed_count = 0
        for node_name in list(self.nodes):
            node = self.nodes[node_name]
            if node.op_type != "Cast":
                continue

            if self.get_input_dtypes(node_name) != self.get_output_dtypes(node_name):
                continue

            if node.outputs[0] in model_outputs:
                # we don't want to remove the cast node if it's an output
                # TODO(jambayk): handle this
                continue

            for destination in list(self.ios[node.outputs[0]].destination):
                self.replace_node_input(destination, node.outputs[0], node.inputs[0])

            self.remove_node(node_name, check_no_consumers=True)
            removed_count += 1
        logger.info("Removed %d redundant cast nodes.", removed_count)

    @classmethod
    def from_model_path(cls, model_path: Union[str, Path]) -> "OnnxDAG":
        """Load an ONNX model with shape inference and create an DAG."""
        with tempfile.NamedTemporaryFile(dir=Path(model_path).parent) as tmpfile:
            shape_infer_model_path = tmpfile.name
            # infer_shapes_path can be used for model >2GB, and infer_shapes cannot.
            infer_shapes_path(model_path, shape_infer_model_path)
            model = onnx.load(shape_infer_model_path)

        return cls(model)

    @classmethod
    def from_model(cls, model: "ModelProto", do_shape_infer: bool = True) -> "OnnxDAG":
        """Create a DAG for the model."""
        if do_shape_infer:
            # shape infer needs file path for model >2GB
            # so always save the model to a temporary file to avoid memory issues
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model.onnx"
                onnx.save(model, model_path, save_as_external_data=True, location="model.onnx.data")

                return cls.from_model_path(model_path)

        return cls(model)
