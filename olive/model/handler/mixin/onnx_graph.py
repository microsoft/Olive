# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx
from onnx import AttributeProto, GraphProto


class OnnxGraphMixin:
    """Provide the following model graph functionalites.

    * get graph nodes
    * get graph io config
    * get graph initializer
    * get graph output name to node mapping
    """

    def nodes(self):
        for graph in self.get_all_graphs():
            yield from graph.node

    def get_graph(self):
        if self.graph is not None:
            return self.graph
        self.graph = self.load_model().graph
        return self.graph

    def get_all_graphs(self):
        if self.all_graphs is not None:
            return self.all_graphs
        self.all_graphs = []
        graph_queue = [self.get_graph()]
        while graph_queue:
            graph = graph_queue.pop(0)
            self.all_graphs.append(graph)
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == AttributeProto.AttributeType.GRAPH:
                        assert isinstance(attr.g, GraphProto)
                        graph_queue.append(attr.g)
                    if attr.type == AttributeProto.AttributeType.GRAPHS:
                        for g in attr.graphs:
                            assert isinstance(g, GraphProto)
                            graph_queue.append(g)
        return self.all_graphs

    def output_name_to_node(self):
        output_name_to_node = {}
        for node in self.nodes():
            for output_name in node.output:
                if output_name:  # could be empty when it is optional
                    output_name_to_node[output_name] = node
        return output_name_to_node

    def get_initializer(self, name):
        for graph in self.get_all_graphs():
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor
        return None

    def get_graph_io_config(self):
        try:
            from onnx.helper import tensor_dtype_to_np_dtype
        except ImportError:
            from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

            def tensor_dtype_to_np_dtype(tensor_type):
                return TENSOR_TYPE_TO_NP_TYPE[tensor_type]

        # external data is not needed for io config parsing
        # the .onnx model already contains all of the graph information
        # this method works whether the external data is in the same directory or not
        model = onnx.load(self.model_path, load_external_data=False)
        io_config = {
            "input_names": [],
            "input_shapes": [],
            "input_types": [],
            "output_names": [],
            "output_shapes": [],
            "output_types": [],
        }
        for prefix, ios in [("input", model.graph.input), ("output", model.graph.output)]:
            for io in ios:
                # get name, type, shape
                name = io.name
                tensor_type = io.type.tensor_type
                if tensor_type.elem_type == 0:
                    # sequence type
                    # TODO(jambayk): add support for different types
                    # refer to https://github.com/lutzroeder/netron/blob/main/source/onnx.js#L1424
                    tensor_type = io.type.sequence_type.elem_type.tensor_type
                data_type = str(tensor_dtype_to_np_dtype(tensor_type.elem_type))
                shape = [dim.dim_param if dim.dim_param else dim.dim_value for dim in tensor_type.shape.dim]

                # append to io_config
                io_config[f"{prefix}_names"].append(name)
                io_config[f"{prefix}_types"].append(data_type)
                io_config[f"{prefix}_shapes"].append(shape)

        return io_config
