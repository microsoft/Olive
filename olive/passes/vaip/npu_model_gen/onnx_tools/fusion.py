##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy
import onnx.helper

from .graph import Graph
from .node import Node

ShapeOps = ["Flatten", "Reshape", "Unsqueeze", "Squeeze"]


def removeShapeOps(g: Graph):
    rmlist = []
    for n in g.nodemap.keys():
        node = g.nodemap[n]
        if node.op_type in ShapeOps:
            rmlist.append(n)
    for n in rmlist:
        g.skip_node(n)
    g.update_tensor_relations()
    return g


def createSerialOpChain(oplist: list[str]):
    chain = []
    for i, op in enumerate(oplist):
        inport = [] if i == 0 else [[-1, str(i - 1), -1]]
        outport = [] if i == len(oplist) - 1 else [[-1, str(i + 1), -1]]
        nodedesc = {
            "name": str(i),
            "op": op,
            "attrs": [],
            "inport": inport,
            "outport": outport,
        }
        chain.append(nodedesc)
    return chain


def createSerialPattern(oplist: list[str]):
    chain = createSerialOpChain(oplist)
    return FusionPattern(chain)


class AttrExpr:
    def __init__(self, raw: []):
        self.attrname = raw[0]
        self.expr = raw[1]
        if len(raw) == 4:
            self.idx = raw[2]
            self.num = raw[3]
        else:
            self.num = raw[2]
            self.idx = -1

    def __call__(self, x):
        if hasattr(x, self.attrname):
            if self.idx == -1:
                return self.logical(x.__getattribute__(self.attrname))
            else:
                return self.logical(x.__getattribute__(self.attrname)[self.idx])
        return False

    def logical(self, attr):
        if self.expr == "<=":
            return attr <= self.num
        if self.expr == "<":
            return attr < self.num
        if self.expr == ">=":
            return attr >= self.num
        if self.expr == ">":
            return attr > self.num
        if self.expr == "==":
            return attr == self.num


class NodeCondition:
    def __init__(self, attrmap: {}):
        self.op = attrmap["op"]
        self.name = attrmap["name"]
        self.attexprs = []
        for att in attrmap["attrs"]:
            self.attexprs.append(AttrExpr(att))
        self.outport = attrmap["outport"]
        self.inport = attrmap["inport"]

    def is_node(self, node: Node):
        flag = True

        if isinstance(self.op, list):
            flag &= node.op_type in self.op
        else:
            if self.op != "Any":
                flag &= node.op_type == self.op
        for attrexpr in self.attexprs:
            flag &= attrexpr(node)
        return flag


def create_descs_from_nodenames(graph: Graph, nodenames: [str]):
    nodedesc = []
    consumed_by = {}
    produced_by = {}
    for name in nodenames:
        node = graph.nodemap[name]
        desc = {
            "name": node.name,
            "op": node.op_type,
            "attrs": [],
            "inport": [],
            "outport": [],
        }
        nodedesc.append(desc)
        for t in node.output:
            if t != "":
                consumed_by[t] = []
                produced_by[t] = name

    for i, name in enumerate(nodenames):
        node = graph.nodemap[name]
        for j, t in enumerate(node.input):
            if t in consumed_by.keys() and t != "":
                pnode = produced_by[t]
                pidx = nodenames.index(pnode)
                prodnode = graph.nodemap[pnode]
                nodedesc[pidx]["outport"].append([prodnode.output.index(t), name, j])
                nodedesc[i]["inport"].append([j, pnode, prodnode.output.index(t)])
    return nodedesc


class FusionPattern:
    def __init__(self, nodedescs: {}, inplace_fusion=False):
        self.nodedesc = {}
        self.first_key = nodedescs[0]["name"]
        self.inplace_fusion = inplace_fusion
        self.append_fusion = inplace_fusion
        for desc in nodedescs:
            self.nodedesc[desc["name"]] = NodeCondition(desc)

    def search_node(self, nodepair, graph, searched):
        curdescname = nodepair[0]
        curnodename = nodepair[1]

        desc = self.nodedesc[curdescname]
        node = graph.nodemap[curnodename]
        searched.append(curnodename)

        # expand in nodes
        invalid = False if len(desc.inport) > 0 else True
        uppaths = []
        for inset in desc.inport:
            inidx = inset[0]
            indesckey = inset[1]
            prev_outidx = inset[2]
            nextdesc = self.nodedesc[indesckey]
            if inidx == -1:
                for tname in node.input:
                    if tname not in graph.producedby:
                        continue
                    producer_node = graph.producedby[tname]
                    for nodename in producer_node:
                        nodeobject = graph.nodemap[nodename]
                        if (
                            prev_outidx == -1
                            or nodeobject.output.index(tname) == prev_outidx
                        ):
                            if nextdesc.is_node(nodeobject):
                                if nodename not in searched:
                                    invalid, uppath = self.search_node(
                                        (indesckey, nodename), graph, searched
                                    )
                                    if invalid:
                                        uppaths.append(uppath)
                                        break
                                else:
                                    invalid = True
            elif inidx < len(node.input):
                tname = node.input[inidx]
                if tname in graph.producedby:
                    producer_node = graph.producedby[tname]
                    for nodename in producer_node:
                        nodeobject = graph.nodemap[nodename]
                        if (
                            prev_outidx == -1
                            or nodeobject.output.index(tname) == prev_outidx
                        ):
                            if nextdesc.is_node(nodeobject):
                                if nodename not in searched:
                                    invalid, uppath = self.search_node(
                                        (indesckey, nodename), graph, searched
                                    )
                                    if invalid:
                                        uppaths.append(uppath)
                                        break
                                else:
                                    invalid = True

        if not invalid:
            searched.remove(curnodename)
            return False, None
        outpath = []
        for uppath in uppaths:
            if uppath is not None:
                for v in uppath:
                    outpath.append(v)
        outpath.append(nodepair)

        outvalid = False if len(desc.outport) > 0 else True
        downpaths = []
        for outset in desc.outport:
            outidx = outset[0]
            outdescky = outset[1]
            next_inidx = outset[2]
            nextdesc = self.nodedesc[outdescky]
            if outidx == -1:
                for output in node.output:
                    if outvalid:
                        break
                    if output in graph.consumedby:
                        consumed_nodes = graph.consumedby[output]
                        if self.append_fusion and len(consumed_nodes) > 1:
                            # inpalce_fusion the consumer op will be appended to this op as postop
                            # it requires that the output of this op is consumed by next op only
                            continue
                        for nodename in consumed_nodes:
                            nodeobject = graph.nodemap[nodename]
                            if (
                                next_inidx == -1
                                or nodeobject.input.index(output) == next_inidx
                            ):
                                if nextdesc.is_node(nodeobject):
                                    if nodename not in searched:
                                        outvalid, downpath = self.search_node(
                                            (outdescky, nodename), graph, searched
                                        )
                                        if outvalid:
                                            downpaths.append(downpath)
                                            break
                                    else:
                                        outvalid = True

            elif outidx < len(node.output):
                tname = node.output[outidx]
                if tname in graph.consumedby:
                    consumed_nodes = graph.consumedby[tname]
                    if self.append_fusion and len(consumed_nodes) > 1:
                        # inpalce_fusion the consumer op will be appended to this op as postop
                        # it requires that the output of this op is consumed by next op only
                        continue
                    for nodename in consumed_nodes:
                        nodeobject = graph.nodemap[nodename]
                        if (
                            next_inidx == -1
                            or nodeobject.input.index(tname) == next_inidx
                        ):
                            if nextdesc.is_node(nodeobject):
                                if nodename not in searched:
                                    outvalid, downpath = self.search_node(
                                        (outdescky, nodename), graph, searched
                                    )
                                    if outvalid:
                                        downpaths.append(downpath)
                                        break
                                else:
                                    outvalid = True
                                    break

        if outvalid:
            for downpath in downpaths:
                if downpath is not None:
                    for v in downpath:
                        outpath.append(v)
            return True, outpath
        else:
            searched.remove(curnodename)
            return False, None

    def search_pattern(self, graph: Graph):
        ls_nodes = []
        first_desc = self.nodedesc[self.first_key]
        # print("is node? ", first_desc.is_node)
        self.found_node_names = []
        self.tmp_names = []
        for name in graph.nodemap.keys():
            if name in self.found_node_names:
                continue
            node = graph.nodemap[name]
            # print("first_desc is_node(node)", name, first_desc.is_node(node))
            if first_desc.is_node(node):
                # print("printing node name getting thru is_node", name)
                searched = []
                valid, path = self.search_node((self.first_key, name), graph, searched)
                if valid:
                    desckeys = list(self.nodedesc.keys())
                    nodes = ["a"] * len(desckeys)
                    for val in path:
                        idx = desckeys.index(val[0])
                        nodes[idx] = val[1]
                    ls_nodes.append(nodes)
                    self.found_node_names.extend(nodes)
        return ls_nodes
