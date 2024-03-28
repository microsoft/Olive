# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Union

import onnx
from onnx import AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Field

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


class OnnxIO(ConfigBase):
    """ONNX input/output.

    Behaves similar to labeled edges in a graph but can connect to multiple nodes.
    """

    source: str = None
    destination: List[str] = Field(default_factory=list)
    graph_idx: int
    proto: Union[ValueInfoProto, TensorProto] = None


class OnnxDAG:
    """ONNX model as a directed acyclic graph (DAG)."""

    def __init__(self, model: "ModelProto"):
        self.model = model
        self.graphs = self.get_all_graphs(self.model)
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

    @classmethod
    def process_io(cls, graph: GraphProto, ios: Dict[str, OnnxIO], graph_idx: int):
        """Process inputs, outputs, initializers, and value_info.

        This will populate ios. Should be called before adding nodes.
        """
        for i in graph.input:
            ios[i.name] = OnnxIO(proto=i, source=SpecialInput.INPUT, graph_idx=graph_idx)
        for o in graph.output:
            ios[o.name] = OnnxIO(proto=o, destination=[SpecialOutput.OUTPUT], graph_idx=graph_idx)
        for initializer in graph.initializer:
            ios[initializer.name] = OnnxIO(proto=initializer, source=SpecialInput.INITIALIZER, graph_idx=graph_idx)
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
            if i not in ios:
                raise ValueError(
                    f"Input {i} does not exist in the graph. Please process the nodes in topological order."
                )
            ios[i].destination.append(name)
            parent = ios[i].source
            if parent not in [SpecialInput.INPUT, SpecialInput.INITIALIZER]:
                connections[parent].append(name)

        for o in node_proto.output:
            if o == "":
                # some nodes have unnamed, unused outputs
                continue
            if o not in ios:
                ios[o] = OnnxIO(graph_idx=graph_idx)
            elif ios[o].source is not None:
                raise ValueError(f"Output {o} is already connected to another node.")
            ios[o].source = name
            for destination in ios[o].destination:
                if destination != SpecialOutput.OUTPUT:
                    connections[name].append(destination)

    def add_initializer(self, initializer: TensorProto, graph_idx: int):
        """Add an initializer to the graph."""
        if initializer.name in self.ios:
            raise ValueError(f"Initializer {initializer.name} already exists in the graph.")

        self.ios[initializer.name] = OnnxIO(proto=initializer, source=SpecialInput.INITIALIZER, graph_idx=graph_idx)

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
        num_updated = 0
        for i in range(len(node.inputs)):
            if node.proto.input[i] == old_input:
                node.proto.input[i] = new_input

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

                num_updated += 1

        if num_updated < 1:
            raise ValueError(f"Input {old_input} does not exist in node {node_name} proto.")

    def get_op_type(self, node_name: str) -> str:
        """Get the operator type of a node."""
        return self.nodes[node_name].op_type

    def is_input(self, name: str) -> bool:
        """Check if an input is a user input."""
        return self.ios[name].source == SpecialInput.INPUT

    def is_initializer(self, name: str) -> bool:
        """Check if an input is an initializer."""
        return self.ios[name].source == SpecialInput.INITIALIZER

    def get_producer(self, io_name: str) -> str:
        """Get the producer of an input/output."""
        return self.ios[io_name].source

    def get_consumers(self, node_name: str) -> List[str]:
        """Get the consumers of a node.

        :param node_name: name of the node. It can also be an input or initializer.
        """
        if node_name in self.ios and self.ios[node_name].source in [SpecialInput.INPUT, SpecialInput.INITIALIZER]:
            return list(self.ios[node_name].destination)

        return list(self.connections[node_name])

    def is_output_producer(self, node_name: str) -> bool:
        """Check if a node is an output producer."""
        return any(SpecialOutput.OUTPUT in self.ios[o].destination for o in self.nodes[node_name].outputs)

    def _topological_sort_util(self, v: str, visited: Set[str], order: List[str]):
        # keep track of the nodes to visit
        stack = [v]

        while stack:
            v = stack.pop()
            visited.add(v)

            for neighbor in self.get_consumers(v):
                if neighbor not in visited:
                    # remember to come back to this node
                    stack.append(v)
                    # visit the neighbor
                    stack.append(neighbor)
                    break
            else:
                order.insert(0, v)

    def topological_sort(self):
        """Sort the nodes in topological order.

        :param include_inputs: include model inputs in the order.
        :param include_initializers: include model initializers in the order.
        """
        visited = set()
        order = []

        for v in list(self.nodes.keys()):
            if v not in visited:
                self._topological_sort_util(v, visited, order)

        return order

    def update(self):
        """Update the graph proto with the latest nodes and connections."""
        node_order = self.topological_sort()

        for idx, graph in enumerate(self.graphs):
            # update the nodes in the graph proto
            nodes = [
                self.nodes[name].proto
                for name in node_order
                if self.nodes[name].graph_idx == idx
                and (len(self.get_consumers(name)) > 0 or self.is_output_producer(name))
            ]

            # update the initializers in the graph proto
            initializers = [
                io.proto
                for i, io in self.ios.items()
                if io.source == SpecialInput.INITIALIZER and io.graph_idx == idx and len(self.get_consumers(i)) > 0
            ]

            # update the graph proto
            graph.ClearField("node")
            graph.node.extend(nodes)
            graph.ClearField("initializer")
            graph.initializer.extend(initializers)

    @classmethod
    def from_model_path(cls, model_path: Union[str, Path]) -> "OnnxDAG":
        """Load an ONNX model and create an DAG."""
        return cls(onnx.load(model_path))
