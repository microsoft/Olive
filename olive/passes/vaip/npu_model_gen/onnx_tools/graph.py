##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import copy
import math
import warnings

import numpy
import onnx

from .node import create_node
from .tensor import STATIC_TENSOR, DYNAMIC_TENSOR
from .tensor import get_attribute_data, Tensor, volume
from .utils import VERSION, tuple2str, ModelConfig


def __shape_of_initializer__(initial):
    shape = []
    # for nb in tensor.shape.dim
    for nb in initial.dims:
        shape.append(nb)
    return shape


_SHAPE_TENSORS = {
    "Reshape": ("1of2",),
    "Resize": ("2of3", "3of4", "1of2"),
    "Upsample": ("2of3", "3of4", "1of2"),
    "Expand": ("1of2",),
    "Slice": ("1,2of3", "1,2,3of4", "1,2,3,4of5"),
    "ConstantOfShape": ("0of1",),
    "Tile": ("1of2",),
    "Range": ("0,1,2of3",),
    "OneHot": ("1of3",),
    "TopK": ("1of2",),
    "Pad": (
        "1of2",
        "1of3",
    ),
    "NonMaxSuppression": ("2of5", "2of4"),
    "Split": ("1of2",),
    "Unsqueeze": ("1of2",),
    "Squeeze": ("1of2",),
    "ReduceSum": ("1of2",),
    "ReduceMean": ("1of2",),
}


def _contains_shape_tensor(n):
    nodeset = _SHAPE_TENSORS.keys()
    shape_tensors = []
    if n.op_type in nodeset:
        tensor_descs = _SHAPE_TENSORS[n.op_type]
        for desc in tensor_descs:
            strs = desc.split("of")
            indice = strs[0]
            count = int(strs[1])
            if len(n.input) == count:
                indistr = indice.split(",")
                for istr in indistr:
                    shape_tensors.append(n.input[int(istr)])
    return shape_tensors


class ValueExpr:
    def __init__(self, srcrange: [], dstrange: []):
        self.alpha = 1
        self.beta = 0
        self.factor = 0
        self.truncmode = 0
        self.build_expr(srcrange, dstrange)

    def error(self, srcrange, dstrange):
        err = 0
        for sval, dval in zip(srcrange, dstrange):
            err += abs(self.__call__(sval) - dval)
        return err

    def build_expr(self, srcrange: [], dstrange: []):
        lastidx = len(srcrange) - 1
        srclen = srcrange[lastidx] - srcrange[0]
        dstlen = dstrange[lastidx] - dstrange[0]
        if srclen <= dstlen:
            self.factor = 0
            self.alpha = round(dstlen / srclen)
            self.beta = round(dstrange[lastidx] - srcrange[lastidx] * self.alpha)
        else:
            self.factor = 1
            self.alpha = round(srclen / dstlen)
            self.beta = round(dstrange[lastidx] - srcrange[lastidx] / self.alpha)

        test = [self.__call__(x) for x in srcrange]
        diff = 0
        for x0, x1 in zip(test, dstrange):
            diff += x0 - x1
        if diff != 0:
            self.truncmode = 1

    def __call__(self, x):
        if self.factor == 0:
            y = self.alpha * x + self.beta
        else:
            y = x / self.alpha + self.beta
        return self.truncate(y)

    def truncate(self, x):
        if self.truncmode == 0:
            return math.ceil(x)
        else:
            return math.floor(x)


class ShapeEngine:
    def __init__(self, input_desc):
        self.input_desc = input_desc
        self.variables = {}
        self.tensor_desc = {}
        self.tensor_epxr = {}
        for key in input_desc.keys():
            self.add_tensor_desc(key, input_desc[key])

    def update_variable(self, key, val):
        self.variables[key] = val

    def __get_shape_from_desc__(self, desc):
        shape = []
        for val in desc:
            if isinstance(val, int):
                shape.append(val)
            else:
                shape.append(self.variables[val])
        return shape

    def add_tensor_desc(self, tensor, desc):
        self.tensor_desc[tensor] = desc

    def update_tensor_desc(self, tensor, axis, vkey):
        desc = self.tensor_desc[tensor]
        assert not isinstance(desc[axis], int)
        desc[axis] = vkey

    def get_tensor_desc(self, tensor):
        if tensor not in self.tensor_desc.keys():
            return None
        return self.tensor_desc[tensor]

    def add_expr(self, var_name, src_name, expr):
        self.tensor_epxr[var_name] = [src_name, expr]
        self.variables[var_name] = 0

    def update_variables(self):
        for key in self.tensor_epxr.keys():
            expr = self.tensor_epxr[key]
            self.variables[key] = expr[1](self.variables[expr[0]])

    def get_tensorshape(self, tname):
        desc = self.tensor_desc[tname]
        shape = self.__get_shape_from_desc__(desc)
        return shape

    def generate_input(self):
        tmp_input = {}
        for key in self.input_desc:
            shape = self.get_tensorshape(key)
            tmp_input[key] = numpy.zeros(shape)
        return tmp_input


class Graph:
    def __init__(self, g: onnx.GraphProto, mcfg: ModelConfig):
        self.cfg = mcfg
        self.nodemap = {}
        self.tensormap = {}
        self.producedby = {}
        self.consumedby = {}
        self.rawgraph = g
        self.initials = []
        self.dynamics = []
        self.input = []
        self.output = []
        self.valid_shape = False
        self.valid_profile = False

        if g is not None:
            self.__init_graph_from_onnxproto__(g, self.cfg.node_rename)
            if self.cfg.if_fixed_branch is not None:
                self.__remove_if__()
            if self.cfg.fixed_topk > 0:
                self.__set_fixed_topk(self.cfg.fixed_topk)
            self.__constant_search__(self.cfg.constant_folding)
            self.__update_nodes_tensors__(self.cfg.constant_folding)
            self.__find_shape_tensors__()

    def log(self, str):
        if self.cfg.verbose:
            print(str)

    def __update_nodes_tensors__(self, constant_folding):
        from .utils import timer

        tm = timer()
        if constant_folding:
            rmlist = []
            for key in self.nodemap:
                if self.nodemap[key].constant:
                    rmlist.append(key)
            self.log(f"GraphProto Nodes Count:{len(self.rawgraph.node)}")
            self.log(f"Contant folding {rmlist[:3]}... {len(rmlist)} Nodes")
            for key in rmlist:
                self.nodemap.pop(key)

        if constant_folding:
            for name in self.nodemap.keys():
                for tname in self.nodemap[name].input:
                    if self.tensormap[tname].type == STATIC_TENSOR:
                        if tname not in self.initials:
                            self.initials.append(tname)
                    if self.tensormap[tname].type == DYNAMIC_TENSOR:
                        if tname not in self.dynamics:
                            self.dynamics.append(tname)
                for tname in self.nodemap[name].output:
                    if tname not in self.dynamics:
                        self.dynamics.append(tname)
        else:
            for initial in self.rawgraph.initializer:
                self.initials.append(initial.name)
            self.dynamics.extend(self.input)
            for name in self.nodemap.keys():
                for tname in self.nodemap[name].input:
                    if tname not in self.initials and tname not in self.dynamics:
                        self.dynamics.append(tname)
                for tname in self.nodemap[name].output:
                    if tname not in self.dynamics:
                        self.dynamics.append(tname)

        valid_tensors = []
        valid_tensors.extend(self.initials)
        valid_tensors.extend(self.dynamics)
        rm_list = []
        for tname in self.tensormap.keys():
            if tname not in valid_tensors:
                rm_list.append(tname)
        for tname in rm_list:
            self.tensormap.pop(tname)
        # graph_input=self.input
        # graph_output=self.output

        self.input = []
        self.output = []

        #### Fix for swapped inputs

        for tensor in self.rawgraph.input:
            if tensor.name not in self.producedby and tensor.name in self.dynamics:
                if tensor.name not in self.input:
                    self.input.append(tensor.name)
        for tensor in self.rawgraph.output:
            if tensor.name not in self.consumedby and tensor.name in self.dynamics:
                if tensor.name not in self.output:
                    self.output.append(tensor.name)

        #### Default
        # for name in self.nodemap.keys():
        #     node = self.nodemap[name]
        #     for tensor in node.input:
        #         if tensor not in self.producedby and tensor in self.dynamics:
        #             if tensor not in self.input:
        #                 self.input.append(tensor)
        #     for tensor in node.output:
        #         if tensor not in self.consumedby or len(self.consumedby[tensor]) == 0:
        #             if tensor not in self.output:
        #                 self.output.append(tensor)

        self.__update_consumer_producer__()
        self.log(f"Update Nodes Tensors  Time Elapsed {tm.stop()}")

    def __is_node_constant__(self, node):
        constant_node = True
        if node.op_type in ["DequantizeLinear"]:
            return False
        for tname in node.input:
            if self.tensormap[tname].type == DYNAMIC_TENSOR:
                constant_node = False
                break
        return constant_node

    def __constant_search__(self, constant_folding):
        from .utils import timer

        tm = timer()
        for name in self.nodemap.keys():
            node = self.nodemap[name]
            if hasattr(node, "constant"):
                continue
            constant_node = self.__is_node_constant__(node)
            node.constant = constant_node
            if constant_node:
                search_nodes = [name]
                while len(search_nodes) > 0:
                    this_node = self.nodemap[search_nodes[0]]
                    search_nodes.pop(0)
                    if constant_folding:
                        itensors = []
                        for input in this_node.input:
                            itensors.append(self.tensormap[input])
                            if self.tensormap[input].numpy is None and input != "":
                                warnings.warn(
                                    f"Tensor {input} has shape only, {name} may has wrong value infer result"
                                )
                        otensors = []
                        for output in this_node.output:
                            otensors.append(self.tensormap[output])
                        this_node.value_infer(itensors, otensors)
                        if len(otensors) > 0:
                            for i, output in enumerate(this_node.output):
                                self.tensormap[output].type = STATIC_TENSOR
                    else:
                        for i, output in enumerate(this_node.output):
                            self.tensormap[output].type = STATIC_TENSOR
                    for output in this_node.output:
                        if output in self.consumedby:
                            for consumer in self.consumedby[output]:
                                cnode = self.nodemap[consumer]
                                if self.__is_node_constant__(cnode):
                                    cnode.constant = True
                                    search_nodes.append(consumer)
        self.log(f"Constant Search Time Elapsed {tm.stop()}")

    def __update_consumer_producer__(self):
        self.producedby = {}
        self.consumedby = {}
        for name in self.nodemap:
            node = self.nodemap[name]
            for tensor in node.input:
                if tensor not in self.consumedby:
                    self.consumedby[tensor] = []
                self.consumedby[tensor].append(node.name)
            for tensor in node.output:
                if tensor not in self.producedby:
                    self.producedby[tensor] = []
                self.producedby[tensor].append(node.name)

    def update_tensor_relations(self):
        self.__update_consumer_producer__()
        self.dynamics = list(self.producedby.keys())

    def __remove_if__(self):
        prekeys = list(self.nodemap.keys())
        remove_list = []
        selected_branch = (
            "else_branch" if self.cfg.if_fixed_branch == "else" else "then_branch"
        )
        for n in prekeys:
            node = self.nodemap[n]
            if node.op_type == "If":
                # remove condition node chain
                sub_input = []
                for node_proto in node.attr[selected_branch].node:
                    for ipn in node_proto.input:
                        sub_input.append(ipn)
                node_list = [node.name]
                while len(node_list):
                    remove = True
                    sname = node_list.pop(0)
                    snode = self.nodemap[sname]
                    for ipn in snode.input:
                        if (
                            len(self.consumedby[ipn]) == 1 and ipn not in sub_input
                        ):  # consumed by this node
                            node_list.append(self.producedby[ipn][0])
                    if remove:
                        remove_list.append(sname)
                # add branch node
                names = []
                for node_proto in node.attr[selected_branch].node:
                    self.__add_node_from_proto__(node_proto)
                    names.append(node_proto.name)

                # revise output name
                preoutput = node.attr[selected_branch].output[0].name
                newoutput = node.output[0]
                for newname in names:
                    newnode = self.nodemap[newname]
                    for tensor in newnode.input:
                        if tensor not in self.tensormap.keys():
                            self.tensormap[tensor] = Tensor(tensor)
                    for tensor in newnode.output:
                        if tensor not in self.tensormap.keys():
                            self.tensormap[tensor] = Tensor(tensor)
                        if tensor in self.consumedby:
                            for consumer in self.consumedby[tensor]:
                                self.nodemap[newnode.name].nextnodes.append(
                                    self.nodemap[consumer]
                                )
                    for i, v in enumerate(newnode.output):
                        if v == preoutput:
                            newnode.output[i] = newoutput
                            self.producedby[newoutput].append(newname)
        for n in remove_list:
            self.remove_node(n)

    def __set_fixed_topk(self, fixed_k):
        prekeys = list(self.nodemap.keys())
        fixed_k_tensor_name = "constant_fixed_topk_k"
        remove_list = []
        topk_node = 0
        for n in prekeys:
            node = self.nodemap[n]
            if node.op_type == "TopK":
                # remove condition node chain
                node_list = []
                ktensor = node.input[1]
                if len(self.consumedby[ktensor]) == 1:
                    node_list = [self.producedby[ktensor][0]]
                while len(node_list):
                    remove = True
                    sname = node_list.pop(0)
                    snode = self.nodemap[sname]
                    for ipn in snode.input:
                        if len(self.consumedby[ipn]) == 1:  # consumed by this node
                            node_list.append(self.producedby[ipn][0])
                    if remove:
                        remove_list.append(sname)
                topk_node += 1
                node.input[1] = fixed_k_tensor_name
        if topk_node > 0:
            self.add_initial(fixed_k_tensor_name, numpy.array([fixed_k]))
        for n in remove_list:
            self.remove_node(n)

    def __add_node_from_proto__(self, node_proto):
        newnode = create_node(node_proto)
        if len(newnode.name) == 0:
            newnode.name = newnode.op_type + "_" + str(self.node_count)
        self.node_count += 1
        for tensor in node_proto.input:
            if tensor in self.producedby:
                for producer in self.producedby[tensor]:
                    newnode.prevnodes.append(self.nodemap[producer])
            if tensor not in self.consumedby:
                self.consumedby[tensor] = []
            self.consumedby[tensor].append(newnode.name)
            newnode.input.append(tensor)
        for tensor in node_proto.output:
            if tensor not in self.producedby:
                self.producedby[tensor] = []
            self.producedby[tensor].append(newnode.name)
            newnode.output.append(tensor)
        self.nodemap[newnode.name] = newnode

    def __init_graph_from_onnxproto__(self, g, noderename):
        if g is None:
            return
        self.node_count = 0
        from .utils import timer

        tm = timer()
        tm.start()
        for node in g.node:
            newnode = create_node(node)
            if noderename or len(newnode.name) == 0:
                newnode.name = newnode.op_type + "_" + str(self.node_count)
            self.node_count += 1
            for tensor in node.input:
                if tensor in self.producedby:
                    for producer in self.producedby[tensor]:
                        newnode.prevnodes.append(self.nodemap[producer])
                if tensor not in self.consumedby:
                    self.consumedby[tensor] = []
                self.consumedby[tensor].append(newnode.name)
                newnode.input.append(tensor)
            for tensor in node.output:
                if tensor not in self.producedby:
                    self.producedby[tensor] = []
                self.producedby[tensor].append(newnode.name)
                newnode.output.append(tensor)

            self.nodemap[newnode.name] = newnode

        self.log(f"Node Init Time Elapsed {tm.stop()}")

        tm.start()
        # init initials first
        for initial in g.initializer:
            tensor = Tensor(initial)
            self.tensormap[initial.name] = tensor

        for input in g.input:
            if input.name not in self.tensormap.keys():
                self.tensormap[input.name] = Tensor(input)
            # self.input.append(input.name)

        for output in g.output:
            if output.name not in self.tensormap.keys():
                self.tensormap[output.name] = Tensor(output)
            # self.output.append(output.name)

        # init dynamic tensor info
        for valinfo in g.value_info:
            if valinfo.name not in self.tensormap.keys():
                tensor = Tensor(valinfo)
                self.tensormap[valinfo.name] = tensor
        self.log(f"Tensor Init Time Elapsed {tm.stop()}")

        tm.start()
        # dynamic tensor info
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            for tensor in node.input:
                if tensor not in self.tensormap.keys():
                    self.tensormap[tensor] = Tensor(tensor)
            for tensor in node.output:
                if tensor not in self.tensormap.keys():
                    self.tensormap[tensor] = Tensor(tensor)
                if tensor in self.consumedby:
                    for consumer in self.consumedby[tensor]:
                        self.nodemap[node.name].nextnodes.append(self.nodemap[consumer])
        self.log(f"IO Tensor Init Time Elapsed {tm.stop()}")

        self.sparse_model = False
        for key in self.tensormap.keys():
            tensor = self.tensormap[key]
            if tensor.sparsity is not None and tensor.sparsity["ratio"] > 0.4:
                self.sparse_model = True
                break

    def __find_shape_tensors__(self):
        self.shape_tensors = []
        for n in self.nodemap.keys():
            shape_tensors = _contains_shape_tensor(self.nodemap[n])
            for st in shape_tensors:
                self.shape_tensors.append(st)
        self.shape_tensors = set(self.shape_tensors)
        # print(self.shape_tensors)
        for tensor in self.shape_tensors:
            if tensor not in self.initials and tensor in self.producedby.keys():
                searchnodes = self.producedby[tensor]
                while len(searchnodes) > 0:
                    nextnodes = []
                    for nname in searchnodes:
                        node = self.nodemap[nname]
                        node.shape_calc = True
                        if node.op_type == "Shape":
                            continue
                        for input in node.input:
                            if (
                                input not in self.initials
                                and input in self.producedby.keys()
                            ):
                                producers = self.producedby[input]
                                nextnodes.extend(
                                    [
                                        p
                                        for p in producers
                                        if self.nodemap[p].shape_calc == False
                                    ]
                                )
                    searchnodes = nextnodes

    def __get_subnodes_byio__(self, inputs: [], outputs: []):
        graph_level0 = []
        graph_level1 = []
        graph_level2 = []
        searchlist = outputs
        while len(searchlist):
            newlist = []
            for name in searchlist:
                if name in self.consumedby:
                    # print("---", self.consumedby[name])
                    for consumer in self.consumedby[name]:
                        if consumer in graph_level2:
                            continue
                        graph_level2.append(consumer)
                        for tensor in self.nodemap[consumer].output:
                            newlist.append(tensor)
            searchlist = newlist

        searchlist = inputs
        while len(searchlist):
            newlist = []
            for name in searchlist:
                if name in self.consumedby:
                    for consumer in self.consumedby[name]:
                        # if consumer == '/model/layers.0/attn/MultiHeadAttention':
                        #     print("we found mha")
                        if consumer in graph_level2 or consumer in graph_level1:
                            continue
                        graph_level1.append(consumer)
                        for tensor in self.nodemap[consumer].output:
                            newlist.append(tensor)
            searchlist = newlist

        for node in self.nodemap.keys():
            if node not in graph_level1 and node not in graph_level2:
                graph_level0.append(node)

        # if '/model/layers.0/attn/MultiHeadAttention' in graph_level2:
        #     print("mha in lvl2")
        return graph_level0, graph_level1, graph_level2

    def add_initial(self, name, data):
        from .tensor import create_initial_Tensor

        self.initials.append(name)
        if isinstance(data, numpy.ndarray):
            self.tensormap[name] = create_initial_Tensor(name, data)
        else:
            raise NotImplementedError("unsupported data type")

    def add_dynamic(self, name, data):
        from .tensor import create_dynamic_Tensor

        self.dynamics.append(name)
        if isinstance(data, numpy.ndarray):
            self.tensormap[name] = create_dynamic_Tensor(name, data)
        else:
            raise NotImplementedError("unsupported data type")

    def get_subgraph(self, inputs: [], outputs: []):
        graph_level0, graph_level1, graph_level2 = self.__get_subnodes_byio__(
            inputs, outputs
        )

        graph_level0 = Graph(self.get_onnxgraph_by_nodenames(graph_level0), self.cfg)
        graph_level1 = Graph(self.get_onnxgraph_by_nodenames(graph_level1), self.cfg)
        graph_level2 = Graph(self.get_onnxgraph_by_nodenames(graph_level2), self.cfg)

        group_outputs = [graph_level0.output, graph_level1.output, graph_level2.output]
        group_inputs = [graph_level0.input, graph_level1.input, graph_level2.input]

        extern_outputs = []
        extern_inputs = []
        for ele in group_outputs:
            extern_outputs.extend(ele)
        extern_outputs = set(extern_outputs)

        for ele in group_inputs:
            extern_inputs.extend(ele)
        extern_inputs = set(extern_inputs)

        for inputs in group_inputs:
            extern_outputs = extern_outputs - set(inputs)

        for outputs in group_outputs:
            extern_inputs = extern_inputs - set(outputs)

        if len(extern_inputs) != len(self.input):
            warnings.warn(
                "subgraph input and output tensors can not reverse to raw graph."
            )

        # if len(extern_outputs) != len(self.output):
        #     warnings.warn(
        #         "subgraph input and output tensors can not reverse to raw graph."
        #     )

        return graph_level0, graph_level1, graph_level2

    def get_initials_from_nodenames(self, nodenames):
        initializer = []
        enqueued = []
        for name in nodenames:
            for input in self.nodemap[name].input:
                if input in self.initials and input not in enqueued:
                    proto = self.tensormap[input].make_tensor_proto()
                    if proto is not None:
                        initializer.append(proto)
                    enqueued.append(input)
        return initializer

    def remove_node(self, nodename):
        node = self.nodemap[nodename]
        # update producer
        for o in node.output:
            if o in self.producedby.keys():
                if nodename in self.producedby[o]:
                    self.producedby[o].remove(nodename)
                if len(self.producedby[o]) == 0:
                    self.producedby.pop(o)

                # if o in self.output:
                #     self.output.remove(o)

        # update consumer
        for i in node.input:
            if i in self.consumedby.keys():
                self.consumedby[i].remove(nodename)
        self.nodemap.pop(nodename)

    def remove_subtree(self, nodename, nodeset=None):
        if nodeset is not None:
            if nodename in self.nodemap:  # may be already removed?
                node = self.nodemap[nodename]

                for n in node.nextnodes:
                    nodeset.add(n.name)
                    nodeset.union(self.remove_subtree(n.name, nodeset))
                return nodeset
            return nodeset

        nodeset = set()
        nodeset.add(nodename)
        nodeset.union(self.remove_subtree(nodename, nodeset))

        for node in nodeset:
            if node not in self.nodemap:
                continue
            self.remove_node(node)

    def remove_dangling_nodes(self):
        try_to_remove = True
        while try_to_remove:
            try_to_remove = False
            for nodename in list(self.nodemap.keys()):
                node = self.nodemap[nodename]
                node_used = False

                for o in node.output:
                    if o in self.output:
                        node_used = True

                for nnode in node.nextnodes:
                    if nnode.name in self.nodemap:
                        node_used = True
                        break

                if node_used:
                    continue

                self.remove_node(nodename)
                try_to_remove = True

    def skip_node(self, nodename):
        node = self.nodemap[nodename]
        count = 0
        for input in node.input:
            if input in self.dynamics:
                count += 1
                indtensor = input
        if count == 1 and len(node.output) == 1:
            self.consumedby[indtensor].remove(nodename)
            if node.output[0] in self.consumedby:
                for con in self.consumedby[node.output[0]]:
                    con_node = self.nodemap[con]
                    for i in range(len(con_node.input)):
                        if con_node.input[i] == node.output[0]:
                            con_node.input[i] = indtensor
                            self.consumedby[indtensor].append(con)
                            break
            else:
                for pro in self.producedby[indtensor]:
                    pro_node = self.nodemap[pro]
                    assert len(pro_node.output) == 1
                    pro_node.output[0] = node.output[0]
        self.nodemap.pop(nodename)

    def fuse_subgraph_node_names(
        self, nodes: [str], nodeop: str, nodename: str, keep_attr=True
    ):
        _inputs, _outputs = self.get_iotensors(nodes, remove_initials=False)
        newnode = onnx.helper.make_node(nodeop, _inputs, _outputs, name=nodename)
        count = 0
        if keep_attr:
            for node in nodes:
                for attribute in self.nodemap[node].proto.attribute:

                    matmuls = ["gate", "up", "down", "o_proj"]
                    for muls in matmuls:
                        if (
                            muls in self.nodemap[node].name
                            and "Group" not in self.nodemap[node].name
                        ):
                            attribute.name = muls + "_" + attribute.name

                    if (
                        "Cast" not in attribute.name
                        and "Cast" not in self.nodemap[node].op_type
                    ):
                        attr = onnx.helper.make_attribute(
                            # self.nodemap[node].proto.op_type
                            # + str(count)
                            # + "_"
                            attribute.name,
                            get_attribute_data(attribute),
                        )
                        newnode.attribute.append(attr)
                # count += 1
        from .node import Node

        # changes for PSU0/1
        if any("MatMul_nhwc_to_nchw" in node for node in nodes):
            attrs = ["accuracy_level", "bits", "block_size", "K", "N"]
            for names in attrs:
                if "acc" in names:
                    val = 0
                elif names == "bits":
                    val = 4
                elif "block" in names:
                    val = 128
                else:
                    val = 0
                attr = onnx.helper.make_attribute(names, val)
                newnode.attribute.append(attr)
            # temp = copy.deepcopy(_inputs[0])
            # _inputs[0] = copy.deepcopy(_inputs[4])
            # _inputs[4] = temp
            _inputs = _inputs[0:4]

        for name in nodes:
            self.remove_node(name)
        newnode = Node(newnode)
        newnode.input = _inputs
        newnode.output = _outputs
        newnode.domain = "com.amd"

        if any("GroupQuery" in node for node in nodes):
            out = newnode.output.pop()
            newnode.output.insert(0, out)
        # if any("SkipLayer" in node for node in nodes):
        #     newnode.output.pop(0)
        # print("Fused Node outputs", newnode.output)
        for i in _inputs:
            self.consumedby[i].append(nodename)
            if i in self.producedby.keys():
                for node in self.producedby[i]:
                    newnode.prevnodes.append(self.nodemap[node])
        for o in _outputs:
            self.producedby[o] = [nodename]
            if o in self.consumedby.keys():
                for node in self.consumedby[o]:
                    newnode.nextnodes.append(self.nodemap[node])
        self.nodemap[nodename] = newnode

    def fuse_postop_node_names(self, nodes: [str], append_attr: bool, **extrakeys):
        _inputs, _outputs = self.get_iotensors(nodes, remove_initials=False)
        mainnode = self.nodemap[nodes[0]]
        newnode = onnx.helper.make_node(
            mainnode.op_type, _inputs, _outputs, name=mainnode.name
        )
        count = 0
        for attr in mainnode.proto.attribute:
            if attr.name == "postop_count":
                count = onnx.helper.get_attribute_value(attr)
            else:
                newnode.attribute.append(attr)

        for key in extrakeys:
            attr = onnx.helper.make_attribute(key, extrakeys[key])
            newnode.attribute.append(attr)

        if append_attr:
            for nname in nodes:
                if nname == mainnode.name:
                    continue
                pnode = self.nodemap[nname]
                attr = onnx.helper.make_attribute("postop_" + str(count), pnode.op_type)
                newnode.attribute.append(attr)
                for attribute in pnode.proto.attribute:
                    attr = onnx.helper.make_attribute(
                        "postop_" + str(count) + "_" + attribute.name,
                        get_attribute_data(attribute),
                    )
                    newnode.attribute.append(attr)
                count += 1

        attr = onnx.helper.make_attribute("postop_count", count)
        newnode.attribute.append(attr)

        for name in nodes:
            self.remove_node(name)
        newnode = create_node(newnode)
        newnode.input = _inputs
        newnode.output = _outputs
        for i in _inputs:
            if i in self.consumedby.keys():
                self.consumedby[i].append(mainnode.name)
            if i in self.producedby.keys():
                newnode.prevnodes.append(self.producedby[i])
        for o in _outputs:
            self.producedby[o] = [mainnode.name]
            if o in self.consumedby.keys():
                newnode.nextnodes.append(self.consumedby[o])
        self.nodemap[mainnode.name] = newnode

    def fuse_subgraph_iotensors(
        self,
        inputs: [],
        outputs: [],
        nodeop: str,
        name_prefix: str = None,
        keep_attr=True,
    ):
        _, nodes, _ = self.__get_subnodes_byio__(inputs, outputs)
        from .fusion import create_descs_from_nodenames, FusionPattern

        descs = create_descs_from_nodenames(self, nodes)
        pattern = FusionPattern(descs)
        nodesls = pattern.search_pattern(self)
        for i, nodes in enumerate(nodesls):
            name = name_prefix + "_" + str(i) if name_prefix is not None else nodes[0]
            self.fuse_subgraph_node_names(nodes, nodeop, name, keep_attr)

    def get_onnxgraph_by_nodenames(self, nodenames):
        if len(nodenames):
            _inputs0, _outputs0 = self.get_iotensors(nodenames, remove_initials=False)
            # _inputs0 = self.input
            # _outputs0 = self.output
            graph_level0 = self.topsort_nodes(nodenames, _inputs0)
            # graph_level0 = self.nodemap
            subgraph = self.make_graph_onnx(
                graph_level0, "subgraph", _inputs0, _outputs0
            )
            return subgraph
        return None

    def save_model(
        self,
        f: str,
        shape_only: bool = False,
        no_shape: bool = False,
        rawmodel: onnx.ModelProto = None,
    ):
        if len(self.nodemap.keys()) == 0:
            warnings.warn(f"Empty graph {f} to save")
            return
        graph = self.make_graph_onnx(
            self.nodemap.keys(),
            "graph",
            self.input,
            self.output,
            with_initializer=True,
            with_shape_info=True,
        )
        if graph is not None and f is not None:
            attr = {}
            attr["producer_name"] = "dd_helper"
            attr["producer_version"] = "v" + VERSION
            model = onnx.helper.make_model(graph, **attr)
            if rawmodel is not None:
                model.ir_version = rawmodel.ir_version
                model.opset_import.pop()
                for opset in rawmodel.opset_import:
                    model.opset_import.append(opset)
            model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
            onnx.save_model(model, f)

    def save_model_function(
        self,
        f: str,
        shape_only: bool = False,
        no_shape: bool = False,
        rawmodel: onnx.ModelProto = None,
        functions=None,
    ):
        if len(self.nodemap.keys()) == 0:
            warnings.warn(f"Empty graph {f} to save")
            return
        graph = self.make_graph_onnx(
            self.nodemap.keys(),
            "graph",
            self.input,
            self.output,
            with_initializer=True,
            with_shape_info=True,
        )
        if graph is not None and f is not None:
            attr = {}
            attr["producer_name"] = "dd_helper"
            attr["producer_version"] = "v" + VERSION
            if not functions:
                model = onnx.helper.make_model(graph, **attr)
            else:
                model = onnx.helper.make_model(graph, functions=functions, **attr)
            if rawmodel is not None:
                model.ir_version = rawmodel.ir_version
                model.opset_import.pop()
                for opset in rawmodel.opset_import:
                    model.opset_import.append(opset)
            # onnx.save_model(model, f)
            onnx.save(
                model,
                f,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="fused_model.data",
                size_threshold=1024,
                convert_attribute=False,
            )

    def make_graph_onnx(
        self,
        nodenames,
        gname,
        inputnames,
        outputnames,
        with_initializer=True,
        with_shape_info=True,
    ):
        nodes = []
        for name in nodenames:
            nodes.append(self.nodemap[name].make_nodeproto())

        initializer = None
        if with_initializer:
            initializer = self.get_initials_from_nodenames(nodenames)

        inputs = []
        outputs = []
        for name in inputnames:
            if name in self.tensormap:
                proto = self.tensormap[name].make_value_proto(make_dummy=True)
                if proto is not None:
                    inputs.append(proto)
            else:
                inputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        for name in outputnames:
            if name in self.tensormap:
                proto = self.tensormap[name].make_value_proto(make_dummy=True)
                if proto is not None:
                    outputs.append(proto)
        value_infos = []
        if with_shape_info:
            for key in self.dynamics:
                if key in self.input or key in self.output:
                    continue
                tensor = self.tensormap[key]
                vinfo = tensor.make_value_proto()
                if vinfo is None:
                    continue
                value_infos.append(vinfo)
            if not with_initializer:
                for key in self.initials:
                    tensor = self.tensormap[key]
                    vinfo = tensor.make_value_proto()
                    if vinfo is None:
                        continue
                    value_infos.append(vinfo)
        graph = onnx.helper.make_graph(
            nodes=nodes,
            name=gname,
            inputs=inputs,
            outputs=outputs,
            initializer=initializer,
            value_info=value_infos,
        )
        return graph

    def topsort_nodes(self, node_names, input_names):
        # update
        produced_by = {}
        # print("nodenames len in topsort fn = ", len(node_names))
        for name in node_names:
            node = self.nodemap[name]
            for tname in node.output:
                produced_by[tname] = name

        consumed_by = {}
        dependencies = {}
        for name in node_names:
            node = self.nodemap[name]
            count = 0
            for tname in node.input:
                if tname in produced_by.keys() and tname != "":
                    # if name =="/model/layers.0/attn/MultiHeadAttention":
                    #     print(tname)
                    count += 1
                    if tname in consumed_by.keys():
                        consumed_by[tname].append(name)
                    else:
                        consumed_by[tname] = [name]

            # if name =="/model/layers.0/attn/MultiHeadAttention":
            #     print("Dependencies of ",name)
            #     print(count)
            dependencies[name] = count
        # print("len of dependencies DS = ", len(dependencies))

        ordered_nodes = []
        queue = []
        while True:
            # print("nodenames len in topsort fn = ", len(node_names))
            for name in node_names:
                if dependencies[name] == 0:
                    queue.append(name)
                    ordered_nodes.append(name)
                    dependencies[name] -= 1
                    # print(dependencies[name])
            if len(queue) == 0:
                break
            # print("len of q in topsort fn = ", len(queue))
            for name in queue:
                node = self.nodemap[name]
                # print("Node =", name)
                for o in node.output:
                    if o in consumed_by.keys():  # or o in self.output:
                        for con in consumed_by[o]:
                            # if con == "/model/layers.0/attn/MultiHeadAttention":
                            #     print("child is MHA")
                            dependencies[con] -= 1
                            # print(con, dependencies[con])
            queue.clear()
        return ordered_nodes

    def graph_reorder_nodes(self):
        # print("In the reorder function before topsort len =",len(self.nodemap))
        ordered_nodes = self.topsort_nodes(self.nodemap.keys(), self.input)
        # print("In the reorder function len=",len(ordered_nodes))
        new_map = {}
        for nname in ordered_nodes:
            new_map[nname] = self.nodemap[nname]
        self.nodemap = new_map
        self.update_tensor_relations()

    def get_iotensors(self, nodenames, remove_initials=True):
        intensors = []
        outtensors = []
        # print(nodenames)
        for name in nodenames:
            for input in self.nodemap[name].input:
                # if len(input) == 0:
                #     continue
                if remove_initials and input in self.initials:
                    continue
                if input in self.producedby and len(input) != 0:
                    producers = self.producedby[input]
                    inner = True
                    for producer in producers:
                        if producer not in nodenames:
                            inner = False
                            break
                    if inner:
                        continue
                # if input not in intensors:
                intensors.append(input)

            for output in self.nodemap[name].output:
                if remove_initials and output in self.initials:
                    continue
                if output in self.consumedby:
                    consumers = self.consumedby[output]
                    inner = True
                    for consumer in consumers:
                        if consumer not in nodenames:
                            inner = False
                            break
                    if inner:
                        continue
                if output not in outtensors:
                    outtensors.append(output)

        return intensors, outtensors

    def update_input_by_map(self, inputs: {}):
        for key in inputs.keys():
            if key in self.tensormap.keys():
                self.tensormap[key].update_tensor(inputs[key])

    def get_dynamic_tensors(self):
        dtensors = {}
        import copy

        for key in self.dynamics:
            dtensors[key] = copy.deepcopy(self.tensormap[key])
        return dtensors

    def check_inputs(self):
        for name in self.input:
            shape = self.tensormap[name].shape
            for val in shape:
                if isinstance(val, str):
                    return False, name
                if val < 0:
                    return False, name
        return True, None

    def shape_infer(self, inputs: {} = None):
        self.valid_shape = False
        if inputs is not None:
            self.update_input_by_map(inputs)
        in_valid, tname = self.check_inputs()
        if not in_valid:
            raise ValueError(
                f"The input tensor {tname}'s shape {self.tensormap[tname].shape2str()} is not valid, Please set it to a valid shape."
            )
        self.shapeinfer_optime_map = {}
        from .utils import timer

        tm = timer()
        for key in self.nodemap.keys():
            tm.start()
            node = self.nodemap[key]
            itensors = []
            for input in node.input:
                itensors.append(self.tensormap[input])
            otensors = []
            for output in node.output:
                otensors.append(self.tensormap[output])

            if node.shape_calc:
                node.value_infer(itensors, otensors)
            else:
                node.shape_infer(itensors, otensors)

            if node.op_type in self.shapeinfer_optime_map.keys():
                self.shapeinfer_optime_map[node.op_type] += tm.stop()
            else:
                self.shapeinfer_optime_map[node.op_type] = tm.stop()
        self.log(self.shapeinfer_optime_map)
        self.valid_shape = True

    def value_infer(self, inputs: {}):
        self.update_input_by_map(inputs)
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            itensors = []
            for input in node.input:
                itensors.append(self.tensormap[input])
            otensors = []
            for output in node.output:
                otensors.append(self.tensormap[output])
            node.value_infer(itensors, otensors)
        outputs = []
        for output in self.output:
            outputs.append(self.tensormap[output].numpy)
        return outputs

    def add_dump_tensors(self, dump_tensor_names: []):
        for name in dump_tensor_names:
            if name in self.tensormap.keys():
                if name not in self.output:
                    self.output.append(name)

    def shape_regress(self, input_desc: {}, input_range: {}):
        shapeengine = ShapeEngine(input_desc)
        for key in input_range.keys():
            shapeengine.update_variable(key, input_range[key][1])
        tmp_input = shapeengine.generate_input()
        self.shape_infer(tmp_input)
        maxtensormap = self.get_dynamic_tensors()

        for key in input_range.keys():
            shapeengine.update_variable(key, input_range[key][0])
        tmp_input = shapeengine.generate_input()
        self.shape_infer(tmp_input)
        mintensormap = self.get_dynamic_tensors()

        for key in mintensormap.keys():
            if key in input_desc.keys():
                continue
            shape_desc = []
            minshape = mintensormap[key].shape
            maxshape = maxtensormap[key].shape
            for i, a in zip(minshape, maxshape):
                if i == a:
                    shape_desc.append(i)
                else:
                    shape_desc.append("")
            shapeengine.add_tensor_desc(key, shape_desc)

        vcount = 0
        for key in input_range.keys():
            minv = input_range[key][0]
            maxv = input_range[key][1]
            vranges = [(minv + maxv) // 2, maxv]
            shapes_range = []

            for val in vranges:
                shapeengine.update_variable(key, val)
                tmpinputs = shapeengine.generate_input()
                self.shape_infer(tmpinputs)
                shapes_range.append(self.get_dynamic_tensors())

            shapeengine.update_variable(key, input_range[key][0])
            srcrange = [
                minv,
            ] + vranges
            dstranges = [srcrange]
            tensor_range_map = {}

            def range_in_rangelist(range, rangelist):
                for i, r in enumerate(rangelist):
                    flag = True
                    for sv, dv in zip(range, r):
                        if sv != dv:
                            flag = False
                            break
                    if flag:
                        return i
                return -1

            for vkey in mintensormap.keys():
                if vkey in input_desc.keys():
                    continue
                shapes = [
                    mintensormap[vkey].shape,
                    shapes_range[0][vkey].shape,
                    shapes_range[1][vkey].shape,
                ]
                for i in range(len(shapes[0])):
                    if shapes[0][i] != shapes[1][i] or shapes[0][i] != shapes[2][i]:
                        newrange = [val[i] for val in shapes]
                        idx = range_in_rangelist(newrange, dstranges)
                        if idx == -1:
                            nodename = self.producedby[vkey][0]
                            srcvalkey = []
                            for iname in self.nodemap[nodename].input:
                                desc = shapeengine.get_tensor_desc(iname)
                                if desc is not None:
                                    for ele in desc:
                                        if not isinstance(ele, int) and ele.isnumeric():
                                            if int(ele) >= vcount:
                                                srcvalkey.append(ele)
                            dstranges.append(newrange)
                            vidx = vcount + len(dstranges) - 1
                            if len(srcvalkey) == 0:
                                expr = ValueExpr(dstranges[0], newrange)
                                err = expr.error(dstranges[0], newrange)
                                srckey = key
                            else:
                                # TODO multiple src variables comparison
                                srckey = srcvalkey[0]
                                expr = ValueExpr(tensor_range_map[srckey], newrange)
                                err = expr.error(tensor_range_map[srckey], newrange)
                            if err > 0:
                                warnings.warn(
                                    f"src key {srckey} dst key {vidx} error:{err}"
                                )

                            shapeengine.add_expr(str(vidx), srckey, expr)
                            tensor_range_map[str(vidx)] = newrange
                            valkey = str(vidx)
                        else:
                            if idx == 0:
                                valkey = key
                            else:
                                vidx = vcount + idx
                                valkey = str(vidx)
                        shapeengine.update_tensor_desc(vkey, i, valkey)
            vcount += len(dstranges)
        return shapeengine

    def compress_memory(self, size_padding=64):
        tensor_in_mem = copy.deepcopy(self.input)
        tensor_mem_per_node = []
        tensor_consumed = {}
        for node in self.nodemap.keys():
            node_ = self.nodemap[node]
            for name in node_.output:
                tensor_in_mem.append(name)
            new_list = copy.deepcopy(tensor_in_mem)
            tensor_mem_per_node.append(new_list)
            for name in node_.input:
                if name in self.consumedby:
                    if name in tensor_consumed:
                        tensor_consumed[name].append(node)
                    else:
                        tensor_consumed[name] = [node]
                    consumers = self.consumedby[name]
                    if len(tensor_consumed[name]) == len(consumers):
                        if name in tensor_in_mem:
                            tensor_in_mem.remove(name)
        for output in self.output:
            if output not in tensor_in_mem:
                warnings.warn(f"tensor list is wrong, {output} is missing!")

        compress_mem = {}
        mem_tags = []
        mem_block = []
        for nodetensors in tensor_mem_per_node:
            # clear mem tags
            for i, tag in enumerate(mem_tags):
                if tag == "":
                    continue
                if tag not in nodetensors:
                    premerge = False
                    if i >= 1 and mem_tags[i - 1] == "":
                        premerge = True
                    nextmerge = False
                    if i < len(mem_tags) - 1 and mem_tags[i + 1] == "":
                        nextmerge = True
                    if premerge and nextmerge:
                        newblock = [
                            mem_block[i - 1][0],
                            mem_block[i + 1][0]
                            + mem_block[i + 1][1]
                            - mem_block[i - 1][0],
                        ]
                        mem_tags.pop(i)
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_block.pop(i)
                        mem_tags[i - 1] = ""
                        mem_block[i - 1] = newblock
                    elif premerge:
                        newblock = [
                            mem_block[i - 1][0],
                            mem_block[i][0] + mem_block[i][1] - mem_block[i - 1][0],
                        ]
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_tags[i - 1] = ""
                        mem_block[i - 1] = newblock
                    elif nextmerge:
                        newblock = [
                            mem_block[i][0],
                            mem_block[i + 1][0] + mem_block[i + 1][1] - mem_block[i][0],
                        ]
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_tags[i] = ""
                        mem_block[i] = newblock
                    else:
                        mem_tags[i] = ""

            # push tensor mem to block
            for tname in nodetensors:
                t_ = self.tensormap[tname]
                size_ = t_.get_memsize()
                size_ = int((size_ + size_padding - 1) // size_padding * size_padding)
                if tname in mem_tags:
                    continue
                block_found = False
                for i, tag in enumerate(mem_tags):
                    if tag == "":
                        if mem_block[i][1] >= size_:
                            remain_size = mem_block[i][1] - size_
                            if remain_size > 1024 * 1024:
                                remain_block = [mem_block[i][0] + size_, remain_size]
                                mem_tags[i] = tname
                                mem_block[i][1] = size_
                                mem_tags.insert(i + 1, "")
                                mem_block.insert(i + 1, remain_block)
                            else:
                                mem_tags[i] = tname
                            block_found = True
                            break
                if not block_found:
                    lastidx = len(mem_tags) - 1
                    if lastidx >= 0 and mem_tags[lastidx] == "":
                        mem_block[lastidx][1] = size_
                        mem_tags[lastidx] = tname
                    else:
                        mem_tags.append(tname)
                        if lastidx >= 0:
                            mem_block.append(
                                [mem_block[lastidx][0] + mem_block[lastidx][1], size_]
                            )
                        else:
                            mem_block.append([0, size_])

                idx = mem_tags.index(tname)
                compress_mem[tname] = mem_block[idx]

        lastblock = mem_block[-1]
        compress_size = lastblock[0] + lastblock[1]
        # print(compress_mem)
        # validate memory
        for nodetensors in tensor_mem_per_node:
            maxmem = 0
            overlap = False
            for tname in nodetensors:
                block = compress_mem[tname]
                if block[0] + block[1] > maxmem:
                    maxmem = block[0] + block[1]
            for tname in nodetensors:
                block0 = compress_mem[tname]
                if block0[1] == 0:
                    continue
                for tname1 in nodetensors:
                    if tname1 == tname:
                        continue
                    block1 = compress_mem[tname1]
                    if block1[1] == 0:
                        continue
                    if block0[0] >= block1[0] and block0[0] < block1[0] + block1[1]:
                        overlap = True
                        laptensor = [tname, tname1]
                        break
                    if block1[0] >= block0[0] and block1[0] < block0[0] + block0[1]:
                        overlap = True
                        laptensor = [tname, tname1]
                        break
            if maxmem > compress_size:
                warnings.warn(
                    f"Wrong compress total memory size:{compress_size}. larger size detected:{maxmem}"
                )
            if overlap:
                if (
                    laptensor[0] != "" and laptensor[1] != ""
                ):  # some chaos models use this empty name as an empty tensor name
                    warnings.warn(
                        f"Memory overlap detected!{laptensor[0]} v.s. {laptensor[1]}"
                    )

        raw_memsize = 0
        for tname in self.dynamics:
            if tname in self.input or tname in self.output:
                continue
            raw_memsize += self.tensormap[tname].get_memsize()

        self.log(f"Raw memory size (without parameter memory): {raw_memsize:,} bytes")
        self.log(f"Compressed memory size: {compress_size:,} bytes")
        self.log(f"Comression ratio: {compress_size / raw_memsize * 100:.3f}%")
        return compress_mem, compress_size

    def get_compute_graph(self):
        cg = copy.copy(self)
        nodes = []
        rmnodes = []
        for key in cg.nodemap.keys():
            node = cg.nodemap[key]
            if node.shape_calc:
                rmnodes.append(key)
                continue
            dummy_node = True
            for output in node.output:
                dummy = True
                if output in cg.consumedby.keys():
                    for consumer in cg.consumedby[output]:
                        if not cg.nodemap[consumer].shape_calc:
                            dummy = False
                            break
                else:
                    dummy = False
                dummy_node = dummy_node and dummy
            if dummy_node:
                rmnodes.append(key)
                searchnodes = [key]
                while len(searchnodes) > 0:
                    this_node = self.nodemap[searchnodes[0]]
                    searchnodes.pop(0)
                    for input in this_node.input:
                        if input in self.producedby:
                            if len(self.consumedby[input]) == 1:
                                pnodename = self.producedby[input][0]
                                if pnodename not in rmnodes:
                                    rmnodes.append(pnodename)
                                searchnodes.append(pnodename)
                continue
            nodes.append(node.name)
        self.log(f"Total Nodes:{len(cg.nodemap)} Removed Shape Nodes:{len(rmnodes)}")
        for key in rmnodes:
            cg.remove_node(key)
        self.log(f"Compute Graph Nodes:{len(cg.nodemap)}")

        cg.dynamics = []
        cg.dynamics.extend(cg.input)
        cg.dynamics.extend(cg.output)
        for name in cg.nodemap.keys():
            for output in cg.nodemap[name].output:
                if name not in cg.dynamics:
                    cg.dynamics.append(output)

        return cg

    def profile(self):
        self.valid_profile = False
        if not self.valid_shape:
            warnings.warn("Please perform a valid shape_infer() before profile().")
            return
        params_flag_map = {}
        for key in self.initials:
            params_flag_map[key] = 0

        self.macs = [0.0, 0.0]
        self.params = 0
        self.memory = 0
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            itensors = []
            _params = 0
            _memory = 0
            max_sparsity = 0
            block_sparsity = {"blocksize": (1, 1), "blockratio": 0, "ratio": 0}
            for input in node.input:
                tensor = self.tensormap[input]
                itensors.append(tensor)
                if input in self.initials:
                    if params_flag_map[input] == 0:
                        elesize = volume(self.tensormap[input].get_shape())
                        _params += elesize
                        _memory += elesize * self.tensormap[input].get_elementsize()
                    params_flag_map[input] += 1
                if (
                    tensor.sparsity is not None
                    and tensor.sparsity["ratio"] > max_sparsity
                ):
                    max_sparsity = tensor.sparsity["ratio"]
                    block_sparsity = tensor.sparsity
            otensors = []
            for output in node.output:
                otensors.append(self.tensormap[output])
                if node.op_type == "Constant":
                    # Constant's output tensors are already counted as weight tensors
                    continue
                _memory += self.tensormap[output].get_memsize()
            macs = node.profile(itensors, otensors)
            outshape = (0,)
            if len(node.output) > 0:
                outshape = self.tensormap[node.output[0]].get_shape()
                outshape = (0,) if len(outshape) == 0 else outshape
            inshape = (0,)
            if len(node.input) > 0:
                inshape = self.tensormap[node.input[0]].get_shape()
                inshape = (0,) if len(inshape) == 0 else inshape

            node.macs = macs
            node.inshape = inshape
            node.outshape = outshape
            node.params = _params
            node.memory = _memory
            node.sparsity = block_sparsity
            self.macs[0] += macs[0]
            self.macs[1] += macs[1]
            self.params += _params
            self.memory += _memory

        self.valid_profile = True

    def print_node_map(self, f: str = None, metric="MACs", exclude_ops=None):
        if not self.valid_profile:
            warnings.warn("Please perform a valid profile() before print_node_map().")
            return
        from tabulate import tabulate

        assert metric in ["MACs", "FLOPs"]
        print_sparse_table = self.sparse_model
        saveformat = "txt"
        splitch = "x"

        if f is not None and ".csv" in f:
            saveformat = "csv"
            csvformat = True
        else:
            csvformat = False

        ptable = []

        forward_macs = int(round(self.macs[0]))
        backward_macs = int(round(self.macs[1]))
        backward_valid = backward_macs > 0
        params = int(self.params)
        memory = int(self.memory)

        shared_size = 0
        for key in self.tensormap.keys():
            if key in self.initials:
                if key in self.consumedby.keys() and len(self.consumedby[key]) > 1:
                    tensor = self.tensormap[key]
                    shared_size += volume(tensor.get_shape())

        if shared_size > 1024:
            print()
            print("*" * 64)
            print(f"Please note that Weight Tensors Sharing is detected:")
            for key in self.tensormap.keys():
                if key in self.initials:
                    if key in self.consumedby.keys() and len(self.consumedby[key]) > 1:
                        print(f"Tensor:{key} ")
                        print("Shared by: ")
                        for node in self.consumedby[key]:
                            print("           ", node)
                        print()
            print("*" * 64)

        factor = 1
        if metric == "FLOPs":
            factor = 2

        def num2str(num, csv=False):
            if csv:
                return "{}".format(num)
            else:
                return "{:,}".format(num)

        params += 1e-18
        forward_macs += 1e-18
        backward_macs += 1e-18
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            if exclude_ops is not None and node.op_type in exclude_ops:
                continue
            row = [key, self.nodemap[key].op_type]
            if print_sparse_table:
                sparsity = node.sparsity
                row.append(tuple2str(sparsity["blocksize"], splitch))
                row.append("{:.2%}".format(sparsity["blockratio"]))
                row.append("{:.2%}".format(sparsity["ratio"]))
            row.append(num2str(int(node.macs[0]) * factor, csvformat))
            row.append("{:.2%}".format(node.macs[0] / forward_macs))
            if backward_valid:
                row.append(num2str(int(node.macs[1]) * factor, csvformat))
                row.append("{:.2%}".format(node.macs[1] / backward_macs))
            row.append(num2str(int(node.memory), csvformat))
            row.append("{:.2%}".format(node.memory / memory))
            row.append(num2str(int(node.params), csvformat))
            row.append("{:.2%}".format(node.params / params))
            row.append(tuple2str(node.inshape, splitch))
            row.append(tuple2str(node.outshape, splitch))

            ptable.append(row)
        row = ["Total", "_"]
        if print_sparse_table:
            row.append("_")
            row.append("_")
            row.append("_")
        row.append(num2str(int(forward_macs * factor), csvformat))
        row.append("100%")
        if backward_valid:
            row.append(num2str(int(backward_macs * factor), csvformat))
            row.append("100%")
        row.append(num2str(int(memory), csvformat))
        row.append("100%")
        row.append(num2str(int(params), csvformat))
        row.append("100%")
        row.append("_")
        row.append("_")

        ptable.append(row)
        header = ["Name", "Type"]
        if print_sparse_table:
            header.append("Sparse Pattern")
            header.append("Sparse Block Ratio")
            header.append("Sparse Ratio")
        header.extend(["Forward_" + metric, "FPercent"])
        if backward_valid:
            header.extend(["Backward_" + metric, "BPercent"])
        header.extend(
            ["Memory", "MPercent", "Params", "PPercent", "InShape", "OutShape"]
        )

        if f is None:
            print(tabulate(ptable, headers=header))
        else:
            fp = open(f, "w")
            if saveformat == "csv":
                headerstr = ""
                for i, item in enumerate(header):
                    headerstr += item
                    if i < len(header) - 1:
                        headerstr += ","
                headerstr += "\n"
                fp.write(headerstr)
                for row in ptable:
                    str = ""
                    for i, ele in enumerate(row):
                        str += ele
                        if i != len(row) - 1:
                            str += ","
                    str += "\n"
                    fp.write(str)
            else:
                fp.write(tabulate(ptable, headers=header))
            fp.close()
