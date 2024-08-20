# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Union

import onnx
from onnx import AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Field
from olive.common.utils import StrEnumBase

if TYPE_CHECKING:
    from onnx import ModelProto

logger = logging.getLogger(__name__)


class SpecialInput(StrEnumBase):
    """Special inputs for ONNX nodes."""

    INPUT = "__input__"  # user input
    INITIALIZER = "__initializer__"  # constant initializer
    INPUT_INITIALIZER = "__input_initializer__"  # input + initializer

    @classmethod
    def is_special_input(cls, value: str) -> bool:
        try:
            cls(value)
            return True
        except ValueError:
            # check if value is a valid input name
            return False

    @classmethod
    def is_input(cls, value: str) -> bool:
        return value in {cls.INPUT, cls.INPUT_INITIALIZER}

    @classmethod
    def is_initializer(cls, value: str) -> bool:
        return value in {cls.INITIALIZER, cls.INPUT_INITIALIZER}


class SpecialOutput(StrEnumBase):
    """Special outputs for ONNX nodes."""

    OUTPUT = "__output__"  # model output


class OnnxNode(ConfigBase):
    """ONNX node."""

    op_type: str
    inputs: List[str]
    outputs: List[str]
    graph_idx: int
    # reference to the node in the model graph
    # can't be serialized to JSON, but we don't need it
    proto: NodeProto


class OnnxIO(ConfigBase):
    """ONNX input/output.

    Behaves similar to labeled edges in a graph but can connect to multiple nodes.
    """

    source: str = None
    destination: List[str] = Field(default_factory=list)
    graph_idx: int
    # reference to the protobuf object
    # can't be serialized to JSON, but we don't need it
    proto: List[Union[ValueInfoProto, TensorProto]] = Field(default_factory=list)


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
            self._process_io(graph, self.ios, idx)
            for node in graph.node:
                self._process_node(node, self.nodes, self.ios, self.connections, idx)

    @staticmethod
    def get_all_graphs(model: "ModelProto") -> List[GraphProto]:
        """Get all graphs in the model.

        :param model: ONNX model.
        :return: list of graphs in the model.
        """
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
    def _process_io(graph: GraphProto, ios: Dict[str, OnnxIO], graph_idx: int):
        """Process inputs, outputs, and initializers in the graph.

        This will populate ios. Should be called before adding nodes.

        :param graph: ONNX graph.
        :param ios: dictionary to store the inputs, outputs, and initializers.
        :param graph_idx: index of the graph in the model.
        """
        for i in graph.input:
            ios[i.name] = OnnxIO(proto=[i], source=SpecialInput.INPUT, graph_idx=graph_idx)
        for o in graph.output:
            ios[o.name] = OnnxIO(proto=[o], destination=[SpecialOutput.OUTPUT], graph_idx=graph_idx)
        for initializer in graph.initializer:
            if initializer.name in ios:
                # it can be both an input and an initializer
                io = ios[initializer.name]
                io.proto.append(initializer)
                io.source = SpecialInput.INPUT_INITIALIZER
            else:
                ios[initializer.name] = OnnxIO(
                    proto=[initializer],
                    source=SpecialInput.INPUT_INITIALIZER,
                    graph_idx=graph_idx,
                )
        for vi in graph.value_info:
            if vi.name not in ios:
                ios[vi.name] = OnnxIO(proto=[vi], graph_idx=graph_idx)
        return ios

    @staticmethod
    def _process_node(
        node_proto: NodeProto,
        nodes: Dict[str, OnnxNode],
        ios: Dict[str, OnnxIO],
        connections: Dict[str, List[str]],
        graph_idx: int,
        overwrite_input_initializers: bool = False,
    ):
        """Process a node and populate the nodes and connections attributes.

        :param node_proto: ONNX node.
        :param nodes: dictionary to store the nodes.
        :param ios: dictionary to store the inputs, outputs, and initializers.
        :param connections: dictionary to store the connections between nodes.
        :param graph_idx: index of the graph in the model.
        :param overwrite_input_initializers: whether to overwrite the inputs and/or initializers if a node
            output is already present as an one. If False, it will raise an error.
        """
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
            if i == "":
                # some nodes have unnamed, unused inputs
                continue
            if i not in ios:
                raise ValueError(
                    f"Input {i} does not exist in the graph. Please process the nodes in topological order."
                )
            ios[i].destination.append(name)
            parent = ios[i].source
            if not SpecialInput.is_special_input(parent):
                connections[parent].append(name)

        for o in node_proto.output:
            if o == "":
                # some nodes have unnamed, unused outputs
                continue
            if o not in ios:
                ios[o] = OnnxIO(graph_idx=graph_idx)
            elif ios[o].source is not None and not (
                overwrite_input_initializers and SpecialInput.is_special_input(ios[o].source)
            ):
                # if the output's original source is an input/initializer, we can overwrite it
                raise ValueError(f"Output {o} is already connected to another node.")
            ios[o] = OnnxIO(source=name, destination=ios[o].destination, graph_idx=graph_idx)
            for destination in ios[o].destination:
                if destination != SpecialOutput.OUTPUT:
                    connections[name].append(destination)

    def add_input(self, input_proto: ValueInfoProto, graph_idx: int, keep_initializer: bool = False):
        """Add an input to the graph.

        :param input_proto: ValueInfoProto of the input.
        :param graph_idx: index of the graph in the model.
        :param keep_initializer: whether to keep the initializer if it exists with the same name.
        """
        self._add_special_input(input_proto, graph_idx, SpecialInput.INPUT, keep_initializer)

    def add_initializer(self, initializer: TensorProto, graph_idx: int, keep_input: bool = False):
        """Add an initializer to the graph.

        :param initializer: TensorProto of the initializer.
        :param graph_idx: index of the graph in the model.
        :param keep_input: whether to keep the input if it exists with the same name.
        """
        self._add_special_input(initializer, graph_idx, SpecialInput.INITIALIZER, keep_input)

    def _add_special_input(
        self,
        proto: Union[ValueInfoProto, TensorProto],
        graph_idx: int,
        i_type: SpecialInput,
        keep_existing: bool = False,
    ):
        """Add a special input to the graph.

        :param proto: ValueInfoProto or TensorProto of the input.
        :param graph_idx: index of the graph in the model.
        :param i_type: type of the special input.
        :param keep_existing: whether to keep the existing input/initializer if it exists with the same name.
        """
        name = proto.name
        proto_list = [proto]
        other_type = SpecialInput.INITIALIZER
        insert_idx = 0
        if i_type == SpecialInput.INITIALIZER:
            other_type = SpecialInput.INPUT
            insert_idx = 1
        if name in self.ios and not (keep_existing and self.ios[name].source == other_type):
            raise ValueError(f"{i_type} {name} already exists in the graph.")
        elif name in self.ios:
            # keep the other type
            proto_list.insert(insert_idx, self.ios[name].proto[0])

        self.ios[name] = OnnxIO(proto=proto_list, source=i_type, graph_idx=graph_idx)

    def convert_initializer_to_input(self, initializer_name: str):
        """Convert an initializer to an input.

        :param initializer_name: name of the initializer to convert to an input.
        """
        io = self.ios[initializer_name]
        if io.source != SpecialInput.INITIALIZER:
            raise ValueError(f"{initializer_name} is not an initializer.")

        # update the ios
        self.ios[initializer_name] = OnnxIO(
            proto=[onnx.helper.make_tensor_value_info(initializer_name, io.proto[0].data_type, io.proto[0].dims)],
            source=SpecialInput.INPUT,
            destination=io.destination,
            graph_idx=io.graph_idx,
        )

    def make_input_dim_dynamic(self, input_name: str, dim_idx: int, dim_param: str):
        """Make a dimension of an input dynamic.

        This assumes changing the dimension of the input doesn't break the graph, especially if it has gone through
        shape inference.

        :param input_name: name of the input.
        :param dim_idx: index of the dimension to make dynamic.
        :param dim_value: symbolic value of the dimension.
        """
        if not self.is_input(input_name):
            raise ValueError(f"{input_name} is not an input.")

        io_proto = self.ios[input_name].proto[0]

        # graph inputs are required to have a shape to provide the rank
        shape = io_proto.type.tensor_type.shape
        if dim_idx >= len(shape.dim):
            raise ValueError(f"Input {input_name} has rank {len(shape.dim)} but trying to access dim {dim_idx}.")

        for idx, dim in enumerate(shape.dim):
            if idx != dim_idx:
                continue

            if dim.HasField("dim_param"):
                raise ValueError(f"Can't replace existing dynamic dim {dim.dim_param} with {dim_param}")

            dim.Clear()
            dim.dim_param = dim_param
            break

    def add_node(self, node_proto: NodeProto, graph_idx: int, overwrite_input_initializers: bool = False):
        """Add a node to the graph.

        This adds the node to the `nodes` attribute and connects them using the `ios` attribute.

        :param node_proto: ONNX node.
        :param graph_idx: index of the graph in the model.
        :param overwrite_input_initializers: whether to overwrite the inputs and/or initializers if a node
            output is already present as an one. If False, it will raise an error.
        """
        self._process_node(node_proto, self.nodes, self.ios, self.connections, graph_idx, overwrite_input_initializers)

    def remove_node(self, node_name: str, check_no_consumers: bool = False):
        """Remove a node from the graph.

        :param node_name: name of the node to remove.
        :param check_no_consumers: whether to check if the node has consumers.
            If True, it will raise an error if the node has consumers.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist in the graph.")

        if check_no_consumers and self.connections[node_name]:
            raise ValueError(f"Node {node_name} has consumers.")

        node = self.nodes.pop(node_name)

        # remove node from the connections
        for i in node.inputs:
            self.ios[i].destination.remove(node_name)
            parent = self.ios[i].source
            if parent not in [SpecialInput.INPUT, SpecialInput.INITIALIZER]:
                self.connections[parent].remove(node_name)

        for o in node.outputs:
            self.ios[o].source = None

        del self.connections[node_name]

    def replace_node_input(self, node_name: str, old_input: str, new_input: str):
        """Replace an input of a node.

        :param node_name: name of the node.
        :param old_input: name of the input to replace.
        :param new_input: name of the new input.
        """
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
                if SpecialInput.is_special_input(old_parent):
                    self.connections[old_parent].remove(node_name)

                new_parent = self.ios[new_input].source
                if not SpecialInput.is_special_input(new_parent):
                    self.connections[new_parent].append(node_name)

                num_updated += 1

        if num_updated < 1:
            raise ValueError(f"Input {old_input} does not exist in node {node_name} proto.")

    def get_node_names(self) -> List[str]:
        """Get the names of all nodes in the graph.

        :return: list of node names.
        """
        return list(self.nodes.keys())

    def get_node(self, node_name: str) -> OnnxNode:
        """Get the node object.

        :param node_name: name of the node.
        :return: OnnxNode object.
        """
        return self.nodes[node_name]

    def get_node_op_type(self, node_name: str) -> str:
        """Get the operator type of a node.

        :param node_name: name of the node.
        :return: operator type of the node.
        """
        return self.nodes[node_name].op_type

    def get_node_inputs(self, node_name: str) -> List[str]:
        """Get the input names of a node.

        :param node_name: name of the node.
        :return: list of input names.
        """
        return list(self.nodes[node_name].inputs)

    def get_node_outputs(self, node_name: str) -> List[str]:
        """Get the output names of a node.

        :param node_name: name of the node.
        :return: list of output names.
        """
        return list(self.nodes[node_name].outputs)

    def get_io(self, io_name: str) -> OnnxIO:
        """Get the input/output object.

        :param io_name: name of the input/output.
        :return: OnnxIO object.
        """
        return self.ios[io_name]

    def is_input(self, io_name: str) -> bool:
        """Check if an input/output is a user input.

        :param io_name: name of the input/output.
        :return: True if the input/output is a user input.
        """
        return SpecialInput.is_input(self.ios[io_name].source)

    def is_initializer(self, io_name: str) -> bool:
        """Check if an input/output is an initializer.

        :param io_name: name of the input/output.
        :return: True if the input/output is an initializer.
        """
        return SpecialInput.is_initializer(self.ios[io_name].source)

    def is_output(self, io_name: str) -> bool:
        """Check if an input/output is an output.

        :param io_name: name of the input/output.
        :return: True if the input/output is an output.
        """
        return SpecialOutput.OUTPUT in self.ios[io_name].destination

    def get_producer(self, io_name: str) -> str:
        """Get the producer of an input/output.

        :param io_name: name of the input/output.
        :return: name of node that produces the input/output.
        """
        return self.ios[io_name].source

    def get_consumers(self, node_name: str) -> List[str]:
        """Get the consumers of a node.

        :param node_name: name of the node. It can also be an input or initializer.
        :return: list of names of nodes that consume one/more outputs of the node.
        """
        if node_name in self.ios and SpecialInput.is_special_input(self.ios[node_name].source):
            return list(self.ios[node_name].destination)

        return list(self.connections[node_name])

    def is_output_producer(self, node_name: str) -> bool:
        """Check if a node is an output producer.

        :param node_name: name of the node.
        :return: True if the node produces one/more outputs that are also model outputs.
        """
        return any(SpecialOutput.OUTPUT in self.ios[o].destination for o in self.nodes[node_name].outputs)

    def _topological_sort_util(self, v: str, visited: Set[str], order: List[str]):
        """Do depth-first search starting from node v.

        Iterative instead of recursive to avoid stack overflow for large graphs.
        """
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

    def topological_sort(self) -> List[str]:
        """Sort the nodes in topological order.

        :return: list of node names in topological order.
        """
        visited = set()
        order = []

        for v in list(self.nodes.keys()):
            if v not in visited:
                self._topological_sort_util(v, visited, order)

        return order

    def update(self):
        """Update the graph proto with the latest inputs, outputs, initializers, and nodes."""
        node_order = self.topological_sort()

        for idx, graph in enumerate(self.graphs):
            # update the nodes in the graph proto
            nodes = [
                self.nodes[name].proto
                for name in node_order
                if self.nodes[name].graph_idx == idx
                and (len(self.get_consumers(name)) > 0 or self.is_output_producer(name))
            ]

            # update the inputs, outputs, and initializers in the graph proto
            inputs = []
            outputs = []
            initializers = []
            for i, io in self.ios.items():
                if io.graph_idx != idx:
                    # not in the current graph
                    continue
                if self.is_output(i):
                    # outputs are handled separately
                    outputs.append(io.proto[0])
                    continue

                # inputs, initializers or intermediate connections
                if len(self.get_consumers(i)) == 0:
                    # no consumers, so don't add it to the graph proto
                    continue
                if self.is_input(i):
                    inputs.append(io.proto[0])
                elif self.is_initializer(i):
                    initializers.append(io.proto[-1])

            # update the graph proto
            graph.ClearField("node")
            graph.node.extend(nodes)
            graph.ClearField("input")
            graph.input.extend(inputs)
            graph.ClearField("initializer")
            graph.initializer.extend(initializers)
            graph.ClearField("output")
            graph.output.extend(outputs)

    def remove_identity_nodes(self):
        """Remove identity nodes from the graph."""
        nodes_to_remove = set()
        for node_name in self.get_node_names():
            if self.get_node_op_type(node_name) != "Identity" or self.is_output_producer(node_name):
                continue

            # change the input of consumers to the input of the identity node
            for consumer in self.get_consumers(node_name):
                self.replace_node_input(
                    consumer, self.get_node_outputs(node_name)[0], self.get_node_inputs(node_name)[0]
                )

            # remove the identity node
            nodes_to_remove.add(node_name)

        for node_name in nodes_to_remove:
            self.remove_node(node_name, check_no_consumers=True)
        logger.debug("Removed %d Identity nodes", len(nodes_to_remove))

    @classmethod
    def from_model_path(cls, model_path: Union[str, Path]) -> "OnnxDAG":
        """Load an ONNX model and create an self.

        :param model_path: path to the ONNX model.
        :return: OnnxDAG object.
        """
        return cls(onnx.load(model_path))
