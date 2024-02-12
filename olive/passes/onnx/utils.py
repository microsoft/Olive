# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Union

import onnx
from onnx import AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto
from onnx.shape_inference import infer_shapes_path

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Field
from olive.common.utils import onnx_dtype_to_np_dtype

if TYPE_CHECKING:
    from onnx import ModelProto


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
    proto: NodeProto  # reference to the node in the model graph


class OnnxIO(ConfigBase):
    """ONNX input/output.

    Behaves similar to labeled edges in a graph but can connect to multiple nodes.
    """

    dtype: str = None
    shape: List = None
    source: str = None
    destination: List[str] = Field(default_factory=list)
    proto: Union[ValueInfoProto, TensorProto]


class OnnxDAG:
    """ONNX graph as a directed acyclic graph (DAG)."""

    def __init__(self, graph: GraphProto):
        self.proto = graph
        self.nodes = {}
        self.ios = {}
        self.connections = defaultdict(list)

        # traverse the graph and populate nodes, ios, and connections
        self.process_io(graph, self.ios)
        for node in graph.node:
            self.process_node(node, self.nodes, self.ios, self.connections)

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

    @classmethod
    def process_io(cls, graph: GraphProto, ios: Dict[str, OnnxIO]):
        """Process inputs, outputs, initializers, and value_info.

        This will populate ios. Should be called before adding nodes.
        """
        for i in graph.input:
            ios[i.name] = OnnxIO(
                proto=i,
                source=SpecialInput.INPUT,
                **cls._get_io_type_shape(i),
            )
        for o in graph.output:
            ios[o.name] = OnnxIO(
                proto=o,
                destination=[SpecialOutput.OUTPUT],
                **cls._get_io_type_shape(o),
            )
        for initializer in graph.initializer:
            ios[initializer.name] = OnnxIO(
                proto=initializer,
                source=SpecialInput.INITIALIZER,
                dtype=onnx_dtype_to_np_dtype(initializer.data_type),
                shape=list(initializer.dims),
            )
        for vi in graph.value_info:
            ios[vi.name] = OnnxIO(
                proto=vi,
                **cls._get_io_type_shape(vi),
            )
        return ios

    @staticmethod
    def process_node(
        node_proto: NodeProto, nodes: Dict[str, OnnxNode], ios: Dict[str, OnnxIO], connections: Dict[str, List[str]]
    ):
        """Process a node and populate the nodes and connections attributes."""
        name = node_proto.name
        onnx_node = OnnxNode(
            proto=node_proto, op_type=node_proto.op_type, inputs=list(node_proto.input), outputs=list(node_proto.output)
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
            for destination in ios[o].destination:
                if destination != SpecialOutput.OUTPUT and destination not in connections[name]:
                    connections[name].append(destination)

    def add_node(self, node_proto: NodeProto):
        """Add a node to the graph.

        This adds the node to the `nodes` attribute and connects them using the `ios` attribute.
        """
        self.process_node(node_proto, self.nodes, self.ios, self.connections)

    def remove_node(self, node_name: str):
        """Remove a node from the graph."""
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist in the graph.")

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

        # check that the inputs and outputs match
        if set(inputs) != set(new_node_proto.input):
            raise ValueError("Inputs do not match.")
        if set(outputs) != set(new_node_proto.output):
            raise ValueError("Outputs do not match.")

        # remove the old nodes
        for node in old_node_names[::-1]:
            self.remove_node(node)

        # add the new node
        self.add_node(new_node_proto)

    def get_op_type(self, node_name: str) -> str:
        """Get the op type of a node."""
        return self.nodes[node_name].op_type

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

    @classmethod
    def from_model_path(cls, model_path: Union[str, Path]) -> Tuple["ModelProto", List["OnnxDAG"]]:
        """Load an ONNX model with shape inference and create a DAG for each graph."""
        with tempfile.NamedTemporaryFile(dir=Path(model_path).parent) as tmpfile:
            shape_infer_model_path = tmpfile.name
            # infer_shapes_path can be used for model >2GB, and infer_shapes cannot.
            infer_shapes_path(model_path, shape_infer_model_path)
            model = onnx.load(shape_infer_model_path)

        dags = []
        graph_queue = [model.graph]
        while graph_queue:
            graph = graph_queue.pop(0)
            dags.append(cls(graph))
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == AttributeProto.AttributeType.GRAPH:
                        assert isinstance(attr.g, GraphProto)
                        graph_queue.append(attr.g)
                    if attr.type == AttributeProto.AttributeType.GRAPHS:
                        for g in attr.graphs:
                            assert isinstance(g, GraphProto)
                            graph_queue.append(g)
        return model, dags

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

        nodes = [self.nodes[name].proto for name in node_order]
        # assume inputs, outputs and initializers have not changed
        value_info = []
        for io in self.ios.values():
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
        self.proto.ClearField("node")
        self.proto.node.extend(nodes)
        self.proto.ClearField("value_info")
        self.proto.value_info.extend(value_info)
