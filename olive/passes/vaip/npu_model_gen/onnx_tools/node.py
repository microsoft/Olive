##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy
import onnx
import math
import warnings
from typing import List
from .tensor import (
    get_attribute_data,
    volume,
    is_valid_ndarray,
    create_ndarray_f32,
    onnxdtype2npdtype,
    Tensor,
)
from .utils import NODE_REGISTRY

"""
Real MACs: the number of x86 instructions to finish numeric compute.
From a low-level view, float-point multiple and add is much expensive than
bit movement. But current hardware has added more and more float-point
units to accelerate this. So, we should treat every instruction equally to
measure the complexity of the model.


"""

MUL_MACS = 1
ADD_MACS = 1
CMP_MACS = 1
DIV_MACS = 4  # refers to vrcp14ps
# following refers to https://github.com/reyoung/avx_mathfun
EXP_MACS = 32
POW_MACS = EXP_MACS  # similar with exp
LOG_MACS = 43
SIN_MACS = 39
COS_MACS = 39

TANH_MACS = EXP_MACS + 2 * ADD_MACS + DIV_MACS  # (e^2x-1)/(e^2x+1)
ATANH_MACS = LOG_MACS + 2 * ADD_MACS + DIV_MACS + MUL_MACS  # 1/2*ln((1+x)/(1-x))

SQRT_MACS = 24  # refers to vsqrtps
ATAN_MACS = (
    SQRT_MACS + 3 * MUL_MACS + MUL_MACS + DIV_MACS
)  # refers to "Approximations to inverse tangent function"

RESIZE_LINEAR_MACS = 4
RESIZE_CUBIC_MACS = 8

_SHAPE_TENSORS = {
    "Reshape": ("1of2",),
    "Resize": ("2of3", "3of4"),
    "Slice": ("1,2of3", "1,2,3of4", "1,2,3,4of5"),
    "Expand": ("1of2",),
}


def _conv_output_shape(xin, pad, ksize, stride, dilation):
    return int((xin + pad - dilation * (ksize - 1) - 1) / stride + 1)


def _convtranspose_output_shape(xin, output_padding, pad, ksize, stride, dilation):
    return stride * (xin - 1) + output_padding + ((ksize - 1) * dilation + 1) - pad


def _auto_pad_valid_shape_calc(x, ksize, stride):
    return math.ceil((x - ksize + 1) / stride)


def _auto_pad_same_shape_calc(x, stride):
    return math.ceil((x) / stride)


def _pooling_shape_calc(inshape, pad, kshape, dilation, stride, ceilmode):
    outshape = (inshape + pad - ((kshape - 1) * dilation + 1)) / stride + 1
    if ceilmode:
        return math.ceil(outshape)
    return math.floor(outshape)


def _axes_neg2pos(len, axes):
    newaxes = []
    for axis in axes:
        if axis < 0:
            newaxes.append(len + axis)
        else:
            newaxes.append(axis)
    return newaxes


def _max_shape(shapes: [numpy.ndarray]):
    maxshape = shapes[0]
    maxvol = volume(maxshape)
    for shape in shapes:
        vol = volume(shape)
        if vol > maxvol:
            maxshape = shape
            maxvol = vol
        elif vol == maxvol:
            if len(shape) > len(maxshape):
                maxshape = shape
    return maxshape


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


def _get_shape(item):
    if isinstance(item, numpy.ndarray):
        return list(item.shape)
    elif isinstance(item, (list, tuple)):
        return list(item)
    return [1]


def _get_tensor(item):
    if isinstance(item, numpy.ndarray):
        return item
    elif isinstance(item, (list, tuple)):
        return create_ndarray_f32(item)


class Node:
    def __init__(self, n: onnx.NodeProto):
        self.name = n.name
        self.op_type = n.op_type
        self.nextnodes = []
        self.prevnodes = []
        self.output = []
        self.input = []
        self.proto = n
        self.domain = n.domain
        self.shape_calc = False
        self.attr = {}
        for att in n.attribute:
            self.attr[att.name] = onnx.helper.get_attribute_value(att)
            self.__setattr__(att.name, get_attribute_data(att))
            if att.name == "axes":
                if isinstance(self.axes, list):
                    self.axes = tuple(self.axes)

    def set_attr(self, key, val):
        self.attr[key] = val
        self.__setattr__(key, val)

    def add_default_value(self, attname, defaultvalue):
        if not hasattr(self, attname):
            setattr(self, attname, defaultvalue)

    def make_nodeproto(self):
        return onnx.helper.make_node(
            self.op_type,
            self.input,
            self.output,
            self.name,
            domain=self.domain,
            **self.attr,
        )

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        self.value_infer(intensors, outtensors)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        raise NotImplementedError(
            f"this Node {self.op_type}-{self.name} has no value_infer"
        )

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return [0, 0]


class FusedBase(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_tensor(intensors[0].get_numpy())

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return [0, 0]


class PWNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ADD_MACS
        self.ratio = max(1, len(self.input) - 1)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        inshapes = []
        for tensor in intensors:
            inshapes.append(tensor.get_shape())
        outtensors[0].update_shape(_max_shape(inshapes))
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outshape = outtensors[0].get_shape()
        macs = volume(outshape) * self.ratio * self.op_mac
        return [macs, 0]


class NpMathBase(Node):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ADD_MACS
        self.ratio = max(1, len(self.input) - 1)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        maxlen = 0
        for tensor in intensors:
            shape = tensor.get_shape()
            if len(shape) > maxlen:
                maxlen = len(shape)
        inshapes = []
        for tensor in intensors:
            shape = tensor.get_shape()
            for i in range(0, maxlen - len(shape)):
                shape = [
                    1,
                ] + shape
            inshapes.append(shape)
        outshape = []
        for i in range(maxlen):
            maxdim = 0
            for shape in inshapes:
                if shape[i] > maxdim:
                    maxdim = shape[i]
            outshape.append(maxdim)
        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outshape = outtensors[0].get_shape()
        macs = volume(outshape) * self.ratio * self.op_mac
        return [macs, 0]


@NODE_REGISTRY.register()
class SubNode(NpMathBase):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = intensors[0].get_numpy() - intensors[1].get_numpy()
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class AddNode(NpMathBase):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = intensors[0].get_numpy() + intensors[1].get_numpy()
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class MinNode(NpMathBase):
    def __init__(self, node):
        super().__init__(node)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = intensors[0].get_numpy()
        for i in range(1, len(intensors)):
            result = not numpy.minimum(result, intensors[i].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class MaxNode(NpMathBase):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = intensors[0].get_numpy()
        for i in range(1, len(intensors)):
            result = numpy.maximum(result, intensors[i].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class NegNode(NpMathBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_tensor(-intensors[0].get_numpy())


@NODE_REGISTRY.register()
class DivNode(NpMathBase):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = DIV_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if intensors[0].dtype == intensors[1].dtype:
            if intensors[0].dtype in [numpy.int64]:
                result = intensors[0].get_numpy() // intensors[1].get_numpy()
            else:
                result = intensors[0].get_numpy() / intensors[1].get_numpy()
        else:
            result = intensors[0].get_numpy() / intensors[1].get_numpy()
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class ModNode(NpMathBase):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ADD_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.mod(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class MulNode(NpMathBase):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = MUL_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = intensors[0].get_numpy() * intensors[1].get_numpy()
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class AbsNode(NpMathBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.abs(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class CeilNode(NpMathBase):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.ceil(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class ExpNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS
        self.ratio = 1

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.exp(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class SoftmaxNode(ExpNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS + DIV_MACS
        self.ratio = 1

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xexp = numpy.exp(intensors[0].get_numpy())
        result = xexp / numpy.sum(xexp, axis=self.axis, keepdims=True)
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class LogNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = LOG_MACS
        self.ratio = 1

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.log(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class ImageScalerNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = ADD_MACS + MUL_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class InstanceNormalizationNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = ADD_MACS + MUL_MACS + ADD_MACS + DIV_MACS


@NODE_REGISTRY.register()
class SqrtNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = SQRT_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.sqrt(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class PowNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = POW_MACS

    def value_infer(sel, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.power(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class SinNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = SIN_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.sin(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class CosNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = COS_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.cos(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class RangeNode(Node):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = 1

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        start = intensors[0].get_numpy()
        limit = intensors[1].get_numpy()
        delta = intensors[2].get_numpy()
        result = numpy.arange(start, limit, delta, dtype=intensors[0].dtype)
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class SigmoidNode(ExpNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = 1 / (1 + numpy.exp(-intensors[0].get_numpy()))
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class TanhNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = TANH_MACS
        self.ratio = 2

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.tanh(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class AtanNode(TanhNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ATAN_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.arctan(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class SignNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.sign(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class HardSigmoidNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + ADD_MACS + CMP_MACS * 2
        self.add_default_value("alpha", 0.2)
        self.add_default_value("beta", 0.5)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = max(0, min(1, self.alpha * intensors[0].get_numpy() + self.beta))
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class HardSwishNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS * 2 + ADD_MACS + CMP_MACS * 2
        self.add_default_value("alpha", 1 / 6)
        self.add_default_value("beta", 0.5)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = intensors[0].get_numpy() * max(
            0, min(1, self.alpha * intensors[0].get_numpy() + self.beta)
        )
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class ReluNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.clip(intensors[0].get_numpy(), 0, None)
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class PReluNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS + MUL_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class LeakyReluNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = self.op_mac = MUL_MACS + CMP_MACS
        self.add_default_value("alpha", 0.01)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        x = intensors[0].get_numpy()
        outtensor = numpy.zeros_like(x)
        for i in numpy.ndindex(x.shape):
            x = x[i]
            outtensor[i] = x if x >= 0 else x * self.alpha
        outtensors[0].update_tensor(outtensor)


@NODE_REGISTRY.register()
class SumNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS
        self.ratio = len(nodeproto.input) - 1

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        y = intensors[0].get_numpy()
        for i in range(1, len(intensors)):
            y = y + intensors[i].get_numpy()
        outtensors[0].update_tensor(y)


@NODE_REGISTRY.register()
class NonMaxSuppressionNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if len(intensors) >= 3:
            box_shape = intensors[0].get_shape()
            score_shape = intensors[1].get_shape()
            max_output_boxes_per_class = int(intensors[2].get_scalar())
            box_count = volume(box_shape[:-1])
            assert box_shape[1] == score_shape[2]
            assert box_shape[0] == score_shape[0]
            max_output_boxes_per_class = min(max_output_boxes_per_class, box_count)
            outtensors[0].update_shape([max_output_boxes_per_class, 3])
            outtensors[0].update_dtype(numpy.int64)
            return
        raise NotImplementedError()


@NODE_REGISTRY.register()
class LRNNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        macs = 0
        outvol = volume(outtensors[0].get_shape())
        outvol *= DIV_MACS + EXP_MACS + ADD_MACS + self.size * MUL_MACS
        macs += outvol
        return [macs, 0]


@NODE_REGISTRY.register()
class LessNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(numpy.bool_)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.less(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return [volume(outtensors[0].get_shape()) * CMP_MACS, 0]


@NODE_REGISTRY.register()
class LessOrEqualNode(LessNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.less_equal(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class NotNode(LessNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.logical_not(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class AndNode(LessNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.logical_and(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class OrNode(LessNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.logical_or(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class XorNode(LessNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.logical_xor(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class WhereNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        cond_shape = intensors[0].get_shape()
        x_shape = intensors[1].get_shape()
        y_shape = intensors[2].get_shape()
        outtensors[0].update_shape(_max_shape((cond_shape, x_shape, y_shape)))
        outtensors[0].update_dtype(intensors[1].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.where(
            intensors[0].get_numpy(), intensors[1].get_numpy(), intensors[2].get_numpy()
        )
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class TransposeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("perm", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        yshape = []
        if self.perm is None:
            yshape = xshape[::-1]
        else:
            for axis in self.perm:
                yshape.append(xshape[axis])
        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.transpose(intensors[0].get_numpy(), self.perm)
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class GemmNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("transA", 0)
        self.add_default_value("transB", 0)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        if self.__class__ == GemmNode:
            if self.transA > 0:
                xshape = xshape[::-1]
            else:
                xshape = xshape
            if self.transB > 0:
                yshape = xshape[:-1] + [
                    wshape[-2],
                ]
            else:
                yshape = xshape[:-1] + [
                    wshape[-1],
                ]
        else:
            # broadcast support
            batchshape = (
                xshape[:-2]
                if volume(xshape[:-2]) >= volume(wshape[:-2])
                else wshape[:-2]
            )
            yshape = batchshape + [xshape[-2], wshape[-1]]
        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if self.__class__ == MatMulNode:
            ashape = intensors[0].get_shape()
            bshape = intensors[1].get_shape()
            assert ashape[-1] == bshape[-2]
            result = numpy.matmul(intensors[0].get_numpy(), intensors[1].get_numpy())
            outtensors[0].update_tensor(result)
        if self.transA > 0:
            A = numpy.transpose(intensors[0].get_numpy())
        else:
            A = intensors[0].get_numpy()
        if self.transB > 0:
            B = numpy.transpose(intensors[1].get_numpy())
        else:
            B = intensors[1].get_numpy()
        C = numpy.matmul(A, B)
        if len(intensors) > 2:
            C = numpy.add(C, intensors[2].get_numpy())
        outtensors[0].update_tensor(C)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        yshape = outtensors[0].get_shape()
        if len(intensors) >= 2:
            weight_shape = intensors[1].get_shape()
            macs = volume(yshape)
            if self.__class__ == GemmNode:
                if self.transB > 0:
                    macs *= weight_shape[-1]
                else:
                    macs *= weight_shape[-2]
            else:
                if len(weight_shape) > 1:
                    macs *= weight_shape[-2]
            if len(intensors) == 3:
                macs += volume(yshape) * ADD_MACS
        else:
            raise NotImplementedError()
        return [macs, 0]


@NODE_REGISTRY.register()
class MatMulNode(GemmNode):
    pass


@NODE_REGISTRY.register()
class TileNode(Node):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        input = intensors[0].get_numpy()
        repeats = intensors[1].get_numpy()
        output = numpy.tile(input, repeats)
        outtensors[0].update_tensor(output)


@NODE_REGISTRY.register()
class GatherNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("axis", 0)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        idxshape = intensors[1].get_shape()
        axis = _axes_neg2pos(len(xshape), [self.axis])[0]
        yshape = []
        for i in range(len(xshape)):
            if i == axis:
                yshape.extend(idxshape)
            else:
                yshape.append(xshape[i])
        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        out = numpy.take(
            intensors[0].get_numpy(), intensors[1].get_numpy(), axis=self.axis
        )
        outtensors[0].update_tensor(out)


@NODE_REGISTRY.register()
class ClipNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS * 2
        self.ratio = 1

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if intensors[1].name == "":
            minval = None
        else:
            minval = intensors[1].get_numpy()
        if intensors[2].name == "":
            maxval = None
        else:
            maxval = intensors[2].get_numpy()
        y = numpy.clip(intensors[0].get_numpy(), minval, maxval)
        outtensors[0].update_tensor(y)


@NODE_REGISTRY.register()
class ReciprocalNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = DIV_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.reciprocal(intensors[0].get_numpy())
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class Relu6Node(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = CMP_MACS * 2
        self.ratio = 1

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.clip(intensors[0].get_numpy(), 0, 6)
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class ConstantNode(Node):
    def __init__(self, n):
        super().__init__(n)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(self.value.shape)
        outtensors[0].update_dtype(self.value.dtype.type)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_tensor(self.value)


@NODE_REGISTRY.register()
class ConcatNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outshape = intensors[0].get_shape()
        for i in range(len(intensors) - 1):
            shape = intensors[i + 1].get_shape()
            outshape[self.axis] += shape[self.axis]
        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        ins = []
        for tensor in intensors:
            ins.append(tensor.get_numpy())
        outtensor = numpy.concatenate(ins, self.axis)
        outtensors[0].update_tensor(outtensor)


@NODE_REGISTRY.register()
class GridSampleNode(Node):
    def __init__(self, n):
        super().__init__(n)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        r = intensors[1].shape[-1]
        out_shape = intensors[0].shape[:2] + intensors[1].shape[1 : 1 + r]
        outtensors[0].update_shape(out_shape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class DepthToSpaceNode(Node):
    def __init__(self, n):
        super().__init__(n)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        n, c, h, w = intensors[0].shape
        b = self.attr["blocksize"]
        outtensors[0].update_shape([n, c // (b * b), h * b, w * b])
        outtensors[0].update_dtype(intensors[0].dtype)


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/onehot.py
def one_hot(indices, depth, axis=-1, dtype=numpy.float32):  # type: ignore
    """Compute one hot from indices at a specific axis"""
    values = numpy.asarray(indices)
    rank = len(values.shape)
    depth_range = numpy.arange(depth)
    if axis < 0:
        axis += rank + 1
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = numpy.reshape(
        depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs)
    )
    values = numpy.reshape(numpy.mod(values, depth), ls + (1,) + rs)
    return numpy.asarray(targets == values, dtype=dtype)


@NODE_REGISTRY.register()
class OneHotNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("axis", -1)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        indices = intensors[0].get_numpy()
        depth = intensors[1].get_numpy()
        values = intensors[2].get_numpy()
        y = one_hot(indices, depth, self.axis)
        outtensors[0].update_tensor(y)


@NODE_REGISTRY.register()
class TriluNode(FusedBase):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value("upper", 1)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if len(intensors) == 2:
            k = intensors[1].get_numpy()
        else:
            k = numpy.array(0).astype(numpy.int64)
        if self.upper == 0:
            result = numpy.tril(intensors[0].get_numpy(), k)
        else:
            result = numpy.triu(intensors[0].get_numpy(), k)
        outtensors[0].update_tensor(result)


@NODE_REGISTRY.register()
class EinsumNode(Node):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        strs = self.equation.split(b",")
        self.ashape = strs[0].replace(b" ", b"")
        strs = strs[1].split(b"->")
        self.bshape = strs[0].replace(b" ", b"")
        self.cshape = strs[1].replace(b" ", b"")

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        shape = []
        map = {}
        shape0 = intensors[0].get_shape()
        shape1 = intensors[1].get_shape()
        for i, v in enumerate(shape0):
            map[self.ashape[i]] = v
        for i, v in enumerate(shape1):
            map[self.bshape[i]] = v
        for k in self.cshape:
            shape.append(map[k])
        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        macs = 1
        map = {}
        shape0 = intensors[0].get_shape()
        shape1 = intensors[1].get_shape()
        for i, v in enumerate(shape0):
            map[self.ashape[i]] = v
        for i, v in enumerate(shape1):
            map[self.bshape[i]] = v
        for key in map.keys():
            macs *= map[key]
        return [macs, 0]


@NODE_REGISTRY.register()
class UnsqueezeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("axes", [0])

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        inshape = intensors[0].get_shape()
        if len(intensors) == 2:
            axes = intensors[1].get_numpy()
        else:
            axes = self.axes
        newaxis_len = len(inshape) + len(axes)
        axes = _axes_neg2pos(newaxis_len, axes)
        newshape = []
        idx = 0
        for i in range(newaxis_len):
            if i in axes:
                newshape.append(1)
            else:
                newshape.append(inshape[idx])
                idx += 1
        outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensor = intensors[0].get_numpy()
        if len(intensors) == 2:
            axes = intensors[1].get_numpy()
        else:
            axes = self.axes
        for axis in axes:
            outtensor = numpy.expand_dims(outtensor, axis=axis)
        outtensors[0].update_tensor(outtensor)


@NODE_REGISTRY.register()
class SqueezeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("axes", [0])

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        inshape = intensors[0].get_shape()
        outshape = []
        if len(intensors) == 2:
            self.axes = intensors[1].get_numpy()
        axes = _axes_neg2pos(len(inshape), self.axes)
        for i in range(len(inshape)):
            if i in axes:
                continue
            else:
                outshape.append(inshape[i])
        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensor = intensors[0].get_numpy().copy()
        idx = 0
        if len(intensors) == 2:
            self.axes = intensors[1].get_numpy()
        for axis in self.axes:
            outtensor = numpy.squeeze(outtensor, axis=axis - idx)
            idx += 1
        outtensors[0].update_tensor(outtensor)


@NODE_REGISTRY.register()
class ShapeNode(Node):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        ret = numpy.array(intensors[0].get_shape(), dtype=numpy.int64)
        outtensors[0].update_tensor(ret)


@NODE_REGISTRY.register()
class GeluNode(Node):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].shape)


@NODE_REGISTRY.register()
class GroupQueryAttention(Node):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].shape)


@NODE_REGISTRY.register()
class LpNormalizationNode(Node):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].shape)


@NODE_REGISTRY.register()
class ResizeNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        roi = []
        sizes = []
        if len(intensors) == 2:
            scales = intensors[1].get_numpy()
        elif len(intensors) >= 3:
            roi = intensors[1].get_numpy()
            scales = intensors[2].get_numpy()
            if len(intensors) >= 4:
                sizes = intensors[3].get_numpy()

        newshape = []
        if is_valid_ndarray(sizes):
            if len(sizes) == len(xshape):
                newshape = sizes
            if len(sizes) == 2:
                newshape = xshape[:2] + sizes
        else:
            if is_valid_ndarray(scales):
                newshape = []
                for src, scale in zip(xshape, scales):
                    newshape.append(math.floor(src * scale))

        if is_valid_ndarray(newshape):
            if newshape.dtype != numpy.int64:
                newshape = newshape.astype(dtype=numpy.int64)
        outtensors[0].update_shape(list(newshape))
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        macs = 0
        outvol = volume(outtensors[0].get_shape())
        if self.mode == b"nearest":
            outvol *= 0
        elif self.mode == b"linear":
            outvol *= RESIZE_LINEAR_MACS
        elif self.mode == b"cubic":
            outvol *= RESIZE_CUBIC_MACS
        macs += outvol
        return [macs, 0]


@NODE_REGISTRY.register()
class UpsampleNode(ResizeNode):
    pass


@NODE_REGISTRY.register()
class PoolBase(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS
        self.add_default_value("kernel_shape", (3, 3))
        self.add_default_value("ceil_mode", 0)
        self.add_default_value("pads", (0, 0, 0, 0))
        self.add_default_value("strides", (1, 1))
        self.add_default_value("dilations", (1, 1))
        self.add_default_value("auto_pad", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        inshape = intensors[0].get_shape()
        if self.auto_pad is not None and self.auto_pad != b"NOTSET":
            outshape = inshape[:2]
            if self.auto_pad in [b"SAME_LOWER", b"SAME_UPPER"]:
                outshape += (_auto_pad_same_shape_calc(inshape[2], self.strides[0]),)
                if len(self.strides) == 2:
                    outshape += [
                        _auto_pad_same_shape_calc(inshape[3], self.strides[1]),
                    ]
            elif self.auto_pad == b"VALID":
                outshape += [
                    _auto_pad_valid_shape_calc(
                        inshape[2], self.kernel_shape[0], self.strides[0]
                    ),
                ]
                if len(self.strides) == 2:
                    outshape += [
                        _auto_pad_valid_shape_calc(
                            inshape[3], self.kernel_shape[1], self.strides[1]
                        ),
                    ]
        else:
            if len(self.kernel_shape) == 1:
                outshape = inshape[:2] + [
                    _pooling_shape_calc(
                        inshape[2],
                        self.pads[0] + self.pads[1],
                        self.kernel_shape[0],
                        self.dilations[0],
                        self.strides[0],
                        self.ceil_mode,
                    ),
                ]
            if len(self.kernel_shape) == 2:
                outshape = inshape[:2] + [
                    _pooling_shape_calc(
                        inshape[2],
                        self.pads[0] + self.pads[2],
                        self.kernel_shape[0],
                        self.dilations[0],
                        self.strides[0],
                        self.ceil_mode,
                    ),
                    _pooling_shape_calc(
                        inshape[3],
                        self.pads[1] + self.pads[3],
                        self.kernel_shape[1],
                        self.dilations[1],
                        self.strides[1],
                        self.ceil_mode,
                    ),
                ]
            if len(self.kernel_shape) == 3:
                outshape = inshape[:2] + [
                    _pooling_shape_calc(
                        inshape[2],
                        self.pads[0] + self.pads[0],
                        self.kernel_shape[0],
                        self.dilations[0],
                        self.strides[0],
                        self.ceil_mode,
                    ),
                    _pooling_shape_calc(
                        inshape[3],
                        self.pads[1] + self.pads[1],
                        self.kernel_shape[1],
                        self.dilations[1],
                        self.strides[1],
                        self.ceil_mode,
                    ),
                    _pooling_shape_calc(
                        inshape[4],
                        self.pads[2] + self.pads[2],
                        self.kernel_shape[2],
                        self.dilations[1],
                        self.strides[2],
                        self.ceil_mode,
                    ),
                ]
        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outshape = outtensors[0].get_shape()
        outvol = volume(outshape)
        macs = outvol * CMP_MACS * self.kernel_shape[0]
        if len(self.kernel_shape) == 2:
            macs *= self.kernel_shape[1]
        return [macs, 0]


@NODE_REGISTRY.register()
class AveragePoolNode(PoolBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS


@NODE_REGISTRY.register()
class MaxPoolNode(PoolBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        self.shape_infer(intensors, outtensors)
        ot = outtensors[0].get_numpy()
        for i in numpy.ndindex(ot.shape):
            batch = i[0]
            ocn = i[1]
            oh = i[2]
            ow = i[3]
            t = ot[i]
            ks = tuple(self.kernel_shape)
            for j in numpy.ndindex(ks):
                kh = j[0]
                kw = j[1]
                srch = oh * self.strides[0] + kh - self.pads[0]
                srcw = ow * self.strides[1] + kw - self.pads[1]
                if srch < 0 or srch >= xshape[2] or srcw < 0 or srcw >= xshape[3]:
                    continue
                else:
                    srcv = intensors[0].get_numpy()[batch, ocn, srch, srcw]
                t = max(srcv, t)
            ot[i] = t


@NODE_REGISTRY.register()
class DropoutNode(FusedBase):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)
        if len(outtensors) == 2:
            outtensors[1].update_shape(intensors[0].get_shape())
            outtensors[1].update_dtype(numpy.bool_)


@NODE_REGISTRY.register()
class GlobalAveragePoolNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        inshape = intensors[0].get_shape()
        shape = inshape[0:2]
        for i in range(2, len(inshape)):
            shape += (1,)
        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        x = intensors[0].get_numpy()
        h = x.shape[2]
        w = x.shape[3]
        y = numpy.zeros(x.shape[:2], dtype=numpy.float32)
        for i in numpy.ndindex(y.shape):
            t = 0
            for j in numpy.ndindex((h, w)):
                xi = i + j
                t += x[xi]
            t /= h * w
            y[i] = t
        outtensors[0].update_tensor(y)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        inshape = intensors[0].get_shape()
        outshape = outtensors[0].get_shape()
        macs = volume(inshape) * ADD_MACS + volume(outshape) * DIV_MACS
        return [macs, 0]


@NODE_REGISTRY.register()
class ExpandNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        expandshape = intensors[1].get_numpy().tolist()
        if not isinstance(expandshape, list):
            expandshape = [
                expandshape,
            ]
        yshape = []
        if len(xshape) < len(expandshape):
            for i in range(len(xshape), len(expandshape)):
                xshape = [
                    1,
                ] + xshape
        for x, e in zip(xshape, expandshape):
            yshape.append(max(x, e))
        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        output = intensors[0].get_numpy() * numpy.ones(
            intensors[1].get_numpy(), dtype=intensors[0].dtype
        )
        outtensors[0].update_tensor(output)


@NODE_REGISTRY.register()
class PadNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("pads", None)
        self.add_default_value("value", 0)
        self.add_default_value("mode", "constant")

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        inshape = intensors[0].get_shape()
        newshape = []
        if self.pads is None:
            if len(intensors) > 1:
                pads = intensors[1].get_numpy()
                for i, v in enumerate(inshape):
                    newshape.append(v + pads[i] + pads[i + len(inshape)])
        else:
            for i, v in enumerate(inshape):
                newshape.append(v + self.pads[i] + self.pads[i + len(inshape)])
        newshape = [int(val) for val in newshape]
        outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if self.pads is not None:
            pads = self.pads
        elif len(intensors) > 1:
            pads = intensors[1].get_numpy()
        nrank = len(pads) // 2
        start = pads[:nrank].reshape(nrank, 1)
        end = pads[nrank:].reshape(nrank, 1)
        pad_width = numpy.concatenate([start, end], axis=-1)
        value = self.value
        if len(intensors) > 2:
            value = intensors[2].get_numpy()
        outtensor = numpy.pad(
            intensors[0].get_numpy(),
            pad_width=pad_width,
            mode=self.mode.decode("utf-8"),
            constant_values=value,
        )
        outtensors[0].update_tensor(outtensor)


@NODE_REGISTRY.register()
class IdentityNode(FusedBase):
    pass


@NODE_REGISTRY.register()
class ErfNode(FusedBase):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensor = numpy.zeros_like(intensors[0].get_numpy())
        for i in numpy.ndindex(intensors[0].shape):
            outtensor[i] = math.erf(intensors[0].get_numpy()[i])
        outtensors[0].update_tensor(outtensor)


@NODE_REGISTRY.register()
class BatchNormalizationNode(FusedBase):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value("epsilon", 1e-05)
        self.add_default_value("momentum", 0.9)
        self.add_default_value("training_mode", int(0))

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        x = intensors[0].get_numpy()
        scale = intensors[1].get_numpy()
        b = intensors[2].get_numpy()
        mean = intensors[3].get_numpy()
        var = intensors[4].get_numpy()
        y = numpy.zeros_like(x)
        for i in numpy.ndindex(y.shape):
            cn = i[1]
            sqrt_var = math.sqrt(var[cn] + self.epsilon)
            sm = scale[cn] / sqrt_var
            sv = b[cn]
            m = mean[cn]
            y[i] = (x[i] - m) * sm + sv
        outtensors[0].update_tensor(y)

    # Fusion of batchnorm is determined by inference engine, here just gives the MACs.
    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        base = volume(outtensors[0].get_shape())
        base *= ADD_MACS + SQRT_MACS + DIV_MACS + ADD_MACS + MUL_MACS
        return [base, 0]


@NODE_REGISTRY.register()
class FlattenNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.add_default_value("axis", None)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        x = intensors[0].get_numpy()
        if self.axis is None:
            y = x.reshape((x.shape[0], -1))
        else:
            vol = 1
            for i in range(self.axis):
                vol *= x.shape[i]
            y = x.reshape((vol, -1))
        outtensors[0].update_tensor(y)


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/argmax.py
def argmax_use_numpy(
    data: numpy.ndarray, axis: int = 0, keepdims: int = 1
) -> numpy.ndarray:
    result = numpy.argmax(data, axis=axis)
    if keepdims == 1:
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


@NODE_REGISTRY.register()
class ArgMaxNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value("axis", 0)
        self.add_default_value("keepdims", 1)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        data = intensors[0].get_numpy()
        out = argmax_use_numpy(data, self.axis, self.keepdims)
        outtensors[0].update_tensor(out)


@NODE_REGISTRY.register()
class ArrayFeatureExtractorNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[1].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class ZipMapNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(
            [
                intensors[0].get_shape()[0],
            ]
        )
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class SliceNode(Node):
    def __init__(self, n):
        super(SliceNode, self).__init__(n)
        self.add_default_value("steps", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        inshape = intensors[0].get_shape()
        if len(intensors) == 1:
            starts = self.starts
            ends = self.ends
            axes = self.axes
            if self.steps is None:
                steps = [1] * len(starts)
            else:
                steps = self.steps
        else:
            elesize = len(intensors[1].get_numpy())
            starts = intensors[1].get_numpy()
            ends = intensors[2].get_numpy()
            if len(intensors) == 3:
                # undef beheviour of bidaf-9.onnx
                axes = [0]
                steps = [1]
            else:
                axes = [0] * elesize
                steps = [1] * elesize
            if len(intensors) >= 4:
                axes = intensors[3].get_numpy()
            if len(intensors) >= 5:
                steps = intensors[4].get_numpy()

        axes = _axes_neg2pos(len(inshape), axes)
        newshape = inshape.copy()
        for a in axes:
            newshape[a] = 0
        for s, e, a, st in zip(starts, ends, axes, steps):
            if s < 0:
                s = max(0, inshape[a] + s)
            else:
                s = max(s, 0)
            if e < 0:
                e = max(0, inshape[a] + e)
            else:
                e = min(e, inshape[a])
            tmp = abs(e - s)
            newshape[a] += abs(math.ceil(tmp / st))
        outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        data = intensors[0].get_numpy()
        datashape = intensors[0].get_shape()
        if len(intensors) == 3:
            starts = intensors[1].get_numpy()
            ends = intensors[2].get_numpy()
            x = data[starts[0] : ends[0]]
        if len(intensors) == 4:
            starts = intensors[1].get_numpy()
            ends = intensors[2].get_numpy()
            axes = intensors[3].get_numpy()
            index = 0
            x = data.copy()
            for i in axes:
                if i == 0:
                    x = x[starts[index] : ends[index], ...]
                if i == 1:
                    x = x[:, starts[index] : ends[index], ...]
                if i == 2:
                    x = x[:, :, starts[index] : ends[index], ...]
                if i == 3:
                    x = x[:, :, :, starts[index] : ends[index], ...]
                if i == 4:
                    x = x[:, :, :, :, starts[index] : ends[index], ...]
                index += 1
        if len(intensors) == 5:
            starts = intensors[1].get_numpy()
            ends = intensors[2].get_numpy()
            axes = intensors[3].get_numpy()
            steps = intensors[4].get_numpy()
            index = 0
            x = data.copy()
            for i in axes:
                if i == 0:
                    x = x[starts[index] : ends[index] : steps[index], ...]
                if i == 1:
                    x = x[:, starts[index] : ends[index] : steps[index], ...]
                if i == 2:
                    x = x[:, :, starts[index] : ends[index] : steps[index], ...]
                if i == 3:
                    x = x[:, :, :, starts[index] : ends[index] : steps[index], ...]
                if i == 4:
                    x = x[:, :, :, :, starts[index] : ends[index] : steps[index], ...]
                index += 1
        if len(intensors) == 1:
            index = 0
            x = data.copy()
            for i in self.axes:
                if i == 0:
                    x = x[self.starts[index] : self.ends[index], ...]
                if i == 1:
                    x = x[:, self.starts[index] : self.ends[index], ...]
                if i == 2:
                    x = x[:, :, self.starts[index] : self.ends[index], ...]
                if i == 3:
                    x = x[:, :, :, self.starts[index] : self.ends[index], ...]
                if i == 4:
                    x = x[:, :, :, :, self.starts[index] : self.ends[index], ...]
                index += 1
        outtensors[0].update_tensor(x)


@NODE_REGISTRY.register()
class ReduceMeanNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("axes", None)
        self.add_default_value("keepdims", 1)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        yshape = []
        if len(intensors) == 2:
            axes = intensors[1].get_numpy()
        else:
            axes = self.axes
        if axes is None:
            return [[1]]
        else:
            axes = _axes_neg2pos(len(xshape), axes)

        for i in range(len(xshape)):
            if i in axes:
                if self.keepdims:
                    yshape.append(1)
            else:
                yshape.append(xshape[i])
        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if len(intensors) == 2:
            axes = intensors[1].get_numpy()
        else:
            axes = self.axes
        reduced = numpy.mean(
            intensors[0].get_numpy(), axis=axes, keepdims=self.keepdims == 1
        )
        outtensors[0].update_tensor(reduced)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        vol = volume(intensors[0].get_shape())
        return [vol * ADD_MACS, 0]


@NODE_REGISTRY.register()
class ReduceProdNode(ReduceMeanNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if len(intensors) == 2:
            axes = intensors[1].get_numpy()
        else:
            axes = self.axes
        reduced = numpy.prod(
            intensors[0].get_numpy(), axis=axes, keepdims=self.keepdims == 1
        )
        outtensors[0].update_tensor(reduced)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        datashape = intensors[0].get_shape()
        vol = volume(datashape)
        return [vol * MUL_MACS, 0]


@NODE_REGISTRY.register()
class ReduceSumNode(ReduceMeanNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if len(intensors) == 2:
            axes = tuple(intensors[1].get_numpy().tolist())
        else:
            axes = self.axes
        reduced = numpy.sum(
            intensors[0].get_numpy(), axis=axes, keepdims=self.keepdims == 1
        )
        outtensors[0].update_tensor(reduced)


@NODE_REGISTRY.register()
class ReduceMinNode(ReduceMeanNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        data = intensors[0].get_numpy()
        if len(intensors) == 2:
            axes = intensors[1].get_numpy()
        else:
            axes = self.axes
        reduced = numpy.minimum.reduce(data, axis=axes, keepdims=self.keepdims == 1)
        outtensors[0].update_tensor(reduced)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        datashape = intensors[0].get_shape()
        vol = volume(datashape)
        return [vol * CMP_MACS, 0]


@NODE_REGISTRY.register()
class ReduceMaxNode(ReduceMinNode):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        data = intensors[0].get_numpy()
        if len(intensors) == 2:
            axes = intensors[1].get_numpy()
        else:
            axes = self.axes
        reduced = numpy.maximum.reduce(data, axis=axes, keepdims=self.keepdims == 1)
        outtensors[0].update_tensor(reduced)


@NODE_REGISTRY.register()
class TopKNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.add_default_value("axis", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        k = intensors[1].get_numpy()[0]
        # when the input tensor only contain 1 dimension, the axis attribute (default: 0) may not appear in the node
        if len(xshape) == 1 and self.axis is None:
            self.axis = 0
        axis = _axes_neg2pos(len(xshape), [self.axis])[0]
        newshape = []
        for i in range(len(xshape)):
            if i == axis:
                newshape.append(k)
            else:
                newshape.append(xshape[i])
        outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)
        if len(outtensors) == 2:
            outtensors[1].update_shape(newshape)
            outtensors[1].update_dtype(numpy.int64)


@NODE_REGISTRY.register()
class ScanNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("num_scan_inputs", None)
        self.add_default_value("scan_input_directions", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if len(self.output) == 2:
            outtensors[0].update_shape(intensors[3].get_shape())
            outtensors[0].update_dtype(intensors[3].dtype)
            outtensors[1].update_shape(intensors[3].get_shape())
            outtensors[1].update_dtype(intensors[3].dtype)
        # TODO
        outtensors[0].update_shape([1, 1])
        outtensors[0].update_dtype(intensors[3].dtype)
        outtensors[1].update_shape([1, 1])
        outtensors[1].update_dtype(intensors[3].dtype)
        outtensors[2].update_shape(
            [
                1,
            ]
        )
        outtensors[2].update_dtype(intensors[3].dtype)
        outtensors[3].update_shape(intensors[3].get_shape())
        outtensors[3].update_dtype(intensors[3].dtype)
        outtensors[4].update_shape(intensors[3].get_shape())
        outtensors[4].update_dtype(intensors[3].dtype)


@NODE_REGISTRY.register()
class CompressNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.add_default_value("axis", None)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        ret = numpy.compress(
            intensors[1].get_numpy(), intensors[0].get_numpy(), self.axis
        )
        outtensors[0].update_tensor(ret)


@NODE_REGISTRY.register()
class HardmaxNode(PWNode):
    pass


@NODE_REGISTRY.register()
class CategoryMapperNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = 0


@NODE_REGISTRY.register()
class LSTMNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("direction", None)
        self.add_default_value("hidden_size", None)
        # TODO(Shinji) add activation types support in attributes

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        seq_len = xshape[0]
        batch = xshape[1]
        num_dir = wshape[0]
        h_len = wshape[1] // 4
        if self.hidden_size is not None:
            assert h_len == self.hidden_size
        outtensors[0].update_shape([seq_len, num_dir, batch, h_len])
        outtensors[0].update_dtype(intensors[0].dtype)
        if len(outtensors) > 1:
            outtensors[1].update_shape([num_dir, batch, h_len])
            outtensors[1].update_dtype(intensors[0].dtype)
            if len(outtensors) > 2:
                outtensors[2].update_shape([num_dir, batch, h_len])
                outtensors[2].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        rshape = intensors[2].get_shape()
        bshape = intensors[3].get_shape()
        seq = xshape[0]
        batch = xshape[1]
        num_dir = wshape[0]
        h_len = wshape[1] // 4
        if self.hidden_size is not None:
            assert h_len == self.hidden_size
        ht_size = volume([batch, seq, h_len])
        # ft = sig(W*X+U*Ht-1+B)
        # it = sig(W*X+U*Ht-1+B)
        # ot = sig(W*X+U*Ht-1+B)
        # ct' = tanh(W*X+U*Ht-1+B)
        # ct = ft*Ct-1+it*ct'
        # ht = ot*tanh(ct)
        gemm_macs = (volume(wshape) + volume(rshape)) * batch * seq
        gemm_bias_macs = volume(bshape) * ADD_MACS * batch * seq
        gemm_add_macs = ht_size * ADD_MACS * 4
        sig_macs = ht_size * EXP_MACS * 3
        tanh_macs = ht_size * TANH_MACS * 2
        blend_macs = ht_size * (ADD_MACS + MUL_MACS + MUL_MACS + MUL_MACS)
        macs = (
            gemm_macs
            + gemm_bias_macs
            + sig_macs
            + tanh_macs
            + blend_macs
            + gemm_add_macs
        )
        return [macs, 0]


@NODE_REGISTRY.register()
class ConvNode(Node):
    def __init__(self, n):
        super(ConvNode, self).__init__(n)
        self.add_default_value("auto_pad", None)
        self.add_default_value("pads", (0, 0, 0, 0, 0, 0))
        self.add_default_value("strides", (1, 1, 1))
        self.add_default_value("dilations", (1, 1, 1))
        self.add_default_value("group", 1)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        shape = []
        if self.auto_pad is not None and self.auto_pad != b"NOTSET":
            if self.auto_pad in [b"SAME_LOWER", b"SAME_UPPER"]:
                shape = (xshape[0], wshape[0], math.ceil(xshape[2] / self.strides[0]))
                if len(xshape) == 4:
                    shape += (math.ceil(xshape[3] / self.strides[1]),)
            elif self.auto_pad == b"VALID":
                oh = math.ceil((xshape[2] - wshape[2] + 1) / self.strides[0])
                shape = (xshape[0], wshape[0], oh)
                if len(xshape) == 4:
                    ow = math.ceil((xshape[3] - wshape[3] + 1) / self.strides[1])
                    shape += (ow,)
            else:
                assert 0

        else:
            if len(xshape) == 5:
                od = _conv_output_shape(
                    xshape[2],
                    self.pads[0] + self.pads[3],
                    wshape[2],
                    self.strides[0],
                    self.dilations[0],
                )
                oh = _conv_output_shape(
                    xshape[3],
                    self.pads[1] + self.pads[4],
                    wshape[3],
                    self.strides[1],
                    self.dilations[1],
                )
                ow = _conv_output_shape(
                    xshape[4],
                    self.pads[2] + self.pads[5],
                    wshape[4],
                    self.strides[2],
                    self.dilations[2],
                )
                shape = (xshape[0], wshape[0], od, oh, ow)
            if len(xshape) == 4:
                oh = _conv_output_shape(
                    xshape[2],
                    self.pads[0] + self.pads[2],
                    wshape[2],
                    self.strides[0],
                    self.dilations[0],
                )
                ow = _conv_output_shape(
                    xshape[3],
                    self.pads[1] + self.pads[3],
                    wshape[3],
                    self.strides[1],
                    self.dilations[1],
                )
                shape = (xshape[0], wshape[0], oh, ow)
            elif len(xshape) == 3:
                oh = _conv_output_shape(
                    xshape[2],
                    self.pads[0] + self.pads[1],
                    wshape[2],
                    self.strides[0],
                    self.dilations[0],
                )
                shape = (xshape[0], wshape[0], oh)
        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if self.group != 1:
            raise NotImplementedError()
        self.shape_infer(intensors, outtensors)
        outshape = outtensors[0].get_shape()
        outtensor = outtensors[0].get_numpy()
        has_bias = len(intensors) > 2
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        if len(wshape) != 4:
            raise NotImplementedError()

        reduce_shape = tuple(wshape[1:])
        for i in numpy.ndindex(tuple(outshape)):
            batch = i[0]
            ocn = i[1]
            oh = i[2]
            ow = i[3]
            t = outtensor[i]
            if has_bias:
                t = intensors[2].get_numpy()[ocn]
            for j in numpy.ndindex(reduce_shape):
                icn = j[0]
                kh = j[1]
                kw = j[2]
                srch = oh * self.strides[0] + kh * self.dilations[0] - self.pads[0]
                srcw = ow * self.strides[1] + kw * self.dilations[1] - self.pads[1]
                if srch < 0 or srch >= xshape[2] or srcw < 0 or srcw >= xshape[3]:
                    srcv = 0
                else:
                    srcv = intensors[0].get_numpy()[batch, icn, srch, srcw]
                wv = intensors[1].get_numpy()[(ocn,) + j]
                t += srcv * wv
            outtensor[i] = t
        outtensors[0].update_tensor(outtensor)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        macs = 0
        if len(outtensors) == 1:
            if len(intensors) == 3 or len(intensors) == 2:
                kernel_shape = intensors[1].get_shape()
                outvol = volume(outtensors[0].get_shape())
                reduce_shape = kernel_shape[1:]
                reduce_vol = volume(reduce_shape)
                macs += outvol * reduce_vol * MUL_MACS
                if len(intensors) > 2:
                    macs += outvol * ADD_MACS
        return [macs, 0]


@NODE_REGISTRY.register()
class ReduceL2Node(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("axes", None)
        self.add_default_value("keepdims", 1)
        self.axes = tuple(self.axes) if self.axes is not None else None

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        reduced = numpy.sqrt(
            numpy.sum(
                intensors[0].get_numpy() * intensors[0].get_numpy(),
                axis=self.axes,
                keepdims=self.keepdims == 1,
            )
        )
        outtensors[0].update_tensor(reduced)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        vol = volume(intensors[0].get_shape())
        return [vol * (ADD_MACS + SQRT_MACS), 0]


@NODE_REGISTRY.register()
class CumSumNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = ADD_MACS
        self.ratio = 1
        self.add_default_value("exclusive", 0)
        self.add_default_value("reverse", 0)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        if self.exclusive == 0 and self.reverse == 0:
            y = numpy.cumsum(intensors[0].get_numpy(), intensors[1].get_numpy())
            outtensors[0].update_tensor(y)
            return
        raise NotImplementedError(
            f"CumSum doesnt support {self.exclusive} {self.reverse}"
        )


@NODE_REGISTRY.register()
class NonZeroNode(Node):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        condi = intensors[0].get_numpy()
        result = numpy.array(numpy.nonzero(condi), dtype=numpy.int64)
        if volume(result.shape) == 0:
            condi = numpy.ones_like(intensors[0].get_numpy())
            result = numpy.array(numpy.nonzero(condi), dtype=numpy.int64)
        outtensors[0].update_tensor(result)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return [volume(outtensors[0].get_shape()) * CMP_MACS, 0]


@NODE_REGISTRY.register()
class EqualNode(Node):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.equal(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return [volume(outtensors[0].get_shape()) * CMP_MACS, 0]


@NODE_REGISTRY.register()
class FloorNode(FusedBase):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        ret = numpy.floor(intensors[0].get_numpy())
        outtensors[0].update_tensor(ret)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return [volume(outtensors[0].get_shape()) * CMP_MACS, 0]


@NODE_REGISTRY.register()
class RoiAlignNode(Node):
    def __init__(self, node):
        super().__init__(node)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        rois_shape = intensors[1].get_shape()
        num_rois = rois_shape[0]
        assert rois_shape[1] == 4
        batch_shape = intensors[2].get_shape()
        assert batch_shape[0] == num_rois
        if (
            len(xshape) == 4
            and self.output_height is not None
            and self.output_width is not None
        ):
            newshape = [num_rois, xshape[1], self.output_height, self.output_width]
            outtensors[0].update_shape(newshape)
            outtensors[0].update_dtype(intensors[0].dtype)
        else:
            raise NotImplementedError()


@NODE_REGISTRY.register()
class ScatterElementsNode(Node):
    def __init__(self, node):
        super().__init__(node)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/scatternd.py
def scatter_nd_impl(data, indices, updates, reduction="none"):  # type: ignore
    # Check tensor shapes
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1] :]

    # Compute output
    output = numpy.copy(data)
    for i in numpy.ndindex(indices.shape[:-1]):
        rawoutput = output[tuple(indices[i])]
        # NOTE: The order of iteration in this loop is not specified.
        if reduction == "add":
            rawoutput += updates[i]
        elif reduction == "mul":
            rawoutput *= updates[i]
        elif reduction == "max":
            rawoutput = numpy.maximum(rawoutput, updates[i])
        elif reduction == "min":
            rawoutput = numpy.minimum(rawoutput, updates[i])
        else:
            rawoutput = updates[i]
        output[tuple(indices[i])] = rawoutput
    return output


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gathernd.py
def gather_nd_impl(
    data: numpy.ndarray, indices: numpy.ndarray, batch_dims: int
) -> numpy.ndarray:
    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    # The list of data/indice shape of batch_dims
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array
    batch_dims_size = 1

    # Check the shape of indice and data are identicial for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below

    # Compute shape of output array
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
    reshaped_data = data.reshape((batch_dims_size,) + data.shape[batch_dims:])

    # gather each scalar value from 'data'
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim, *gather_index)])
    return numpy.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)


@NODE_REGISTRY.register()
class ScatterNDNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        data = intensors[0].get_numpy()
        indices = intensors[1].get_numpy()
        updates = intensors[2].get_numpy()
        outtensors[0].update_tensor(scatter_nd_impl(data, indices, updates))


@NODE_REGISTRY.register()
class GatherNDNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        data_shape = intensors[0].get_shape()
        indice_shape = intensors[1].get_shape()
        batch_dims = 0
        # Note the data rank - will be reused multiple times later
        data_rank = len(data_shape)

        # Check input tensors' shape/rank condition
        assert indice_shape[-1] <= data_rank

        # The list of data/indice shape of batch_dims
        batch_dims_shape = []

        # The number of elements in the batch_dims for data/indice array
        batch_dims_size = 1

        # Check the shape of indice and data are identicial for batch dims.
        for i in range(batch_dims):
            batch_dims_shape.append(indice_shape[i])
            batch_dims_size *= indice_shape[i]

        # Compute output of the op as below

        # Compute shape of output array
        output_shape = (
            batch_dims_shape + list(indice_shape)[batch_dims:-1]
            if (indice_shape[-1] == data_rank - batch_dims)
            else batch_dims_shape
            + list(indice_shape)[batch_dims:-1]
            + list(data_shape)[batch_dims + indice_shape[-1] :]
        )
        outtensors[0].update_shape(output_shape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        data = intensors[0].get_numpy()
        indices = intensors[1].get_numpy()
        ret = gather_nd_impl(data, indices, 0)
        outtensors[0].update_tensor(ret)


@NODE_REGISTRY.register()
class RandomUniformLikeNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value("dtype", None)
        self.add_default_value("high", 1.0)
        self.add_default_value("low", 0.0)
        self.add_default_value("seed", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        if self.dtype is None:
            outtensors[0].update_dtype(intensors[0].dtype)
        else:
            outtensors[0].update_dtype(onnxdtype2npdtype(self.dtype))


@NODE_REGISTRY.register()
class RandomNormalLikeNode(RandomUniformLikeNode):
    pass


@NODE_REGISTRY.register()
class GreaterNode(Node):
    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        result = numpy.greater(intensors[0].get_numpy(), intensors[1].get_numpy())
        outtensors[0].update_tensor(result)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outshape = outtensors[0].get_shape()
        return [volume(outshape) * CMP_MACS, 0]


@NODE_REGISTRY.register()
class DequantizeLinearNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + ADD_MACS
        self.ratio = 1

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[1].dtype)


@NODE_REGISTRY.register()
class LayerNormalizationNode(Node):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.add_default_value("axis", -1)
        self.add_default_value("epsilon ", 1e-05)
        self.add_default_value("stash_type", 1)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        tshape = intensors[0].get_shape()
        axis = _axes_neg2pos(len(tshape), [self.axis])[0]
        vol = volume(tshape)
        tshape[axis] = 1
        vol2 = volume(tshape)
        return [
            vol * (MUL_MACS * 3 + +ADD_MACS * 4)
            + vol2 * (ADD_MACS + SQRT_MACS + DIV_MACS),
            0,
        ]


@NODE_REGISTRY.register()
class QuantizeLinearNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + ADD_MACS
        self.ratio = 1

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[2].dtype)


@NODE_REGISTRY.register()
class MatMulIntegerNode(GemmNode):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        super().shape_infer(intensors, outtensors)
        outtensors[0].update_dtype(numpy.int32)


@NODE_REGISTRY.register()
class QLinearMatMulNode(GemmNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("transA", None)
        self.add_default_value("transB", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[3].get_shape()

        if self.__class__ == GemmNode:
            if self.transA is not None and self.transA > 0:
                xshape = xshape[::-1]
            else:
                xshape = xshape
            if self.transB is not None and self.transB > 0:
                yshape = xshape[:-1] + [
                    wshape[-2],
                ]
            else:
                yshape = xshape[:-1] + [
                    wshape[-1],
                ]
        else:
            yshape = xshape[:-1] + [
                wshape[-1],
            ]
        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[-1].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        weight_shape = intensors[3].get_shape()
        macs = volume(xshape)
        if self.__class__ == GemmNode:
            macs *= weight_shape[0]
        else:
            macs *= weight_shape[-1]
        return [macs, 0]


@NODE_REGISTRY.register()
class QLinearConvNode(ConvNode):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[3].get_shape()
        shape = []
        if self.auto_pad is not None and self.auto_pad != b"NOTSET":
            if self.auto_pad in [b"SAME_LOWER", b"SAME_UPPER"]:
                shape = [xshape[0], wshape[0], math.ceil(xshape[2] / self.strides[0])]
                if len(xshape) == 4:
                    shape += [
                        math.ceil(xshape[3] / self.strides[1]),
                    ]
        else:
            if len(xshape) == 4:
                oh = _conv_output_shape(
                    xshape[2],
                    self.pads[0] + self.pads[2],
                    wshape[2],
                    self.strides[0],
                    self.dilations[0],
                )
                ow = _conv_output_shape(
                    xshape[3],
                    self.pads[1] + self.pads[3],
                    wshape[3],
                    self.strides[1],
                    self.dilations[1],
                )
                shape = [xshape[0], wshape[0], oh, ow]
            elif len(xshape) == 3:
                oh = _conv_output_shape(
                    xshape[2],
                    self.pads[0] + self.pads[1],
                    wshape[2],
                    self.strides[0],
                    self.dilations[0],
                )
                shape = [xshape[0], wshape[0], oh]
        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(intensors[-2].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        macs = 0
        outshape = outtensors[0].get_shape()
        if len(outtensors) == 1:
            kernel_shape = intensors[3].get_shape()
            if len(kernel_shape) > 3:
                outvol = volume(outshape)
                macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
            elif len(kernel_shape) == 3:
                outvol = volume(outshape)
                macs += outvol * kernel_shape[1] * kernel_shape[2]
            else:
                outvol = 0
                raise NotImplementedError()
            if len(intensors) == 9:
                macs += outvol * ADD_MACS
        return [macs, 0]


@NODE_REGISTRY.register()
class ConvIntegerNode(ConvNode):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        shape = []
        if self.auto_pad is not None and self.auto_pad != b"NOTSET":
            if self.auto_pad in [b"SAME_LOWER", b"SAME_UPPER"]:
                shape = [xshape[0], wshape[0], math.ceil(xshape[2] / self.strides[0])]
                if len(xshape) == 4:
                    shape += [
                        math.ceil(xshape[3] / self.strides[1]),
                    ]
        else:
            if len(xshape) == 4:
                oh = _conv_output_shape(
                    xshape[2],
                    self.pads[0] + self.pads[2],
                    wshape[2],
                    self.strides[0],
                    self.dilations[0],
                )
                ow = _conv_output_shape(
                    xshape[3],
                    self.pads[1] + self.pads[3],
                    wshape[3],
                    self.strides[1],
                    self.dilations[1],
                )
                shape = [xshape[0], wshape[0], oh, ow]
            elif len(xshape) == 3:
                oh = _conv_output_shape(
                    xshape[2],
                    self.pads[0] + self.pads[1],
                    wshape[2],
                    self.strides[0],
                    self.dilations[0],
                )
                shape = [xshape[0], wshape[0], oh]
        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(numpy.int32)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        macs = 0
        outshape = outtensors[0].get_shape()
        if len(outtensors) == 1:
            kernel_shape = intensors[1].get_shape()
            if len(kernel_shape) > 3:
                outvol = volume(outshape)
                macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
            elif len(kernel_shape) == 3:
                outvol = volume(outshape)
                macs += outvol * kernel_shape[1] * kernel_shape[2]
            else:
                outvol = 0
                raise NotImplementedError()
        return [macs, 0]


@NODE_REGISTRY.register()
class ConvTransposeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("pads", (0, 0, 0, 0))
        self.add_default_value("output_padding", (0, 0, 0, 0))
        self.add_default_value("strides", (1, 1))
        self.add_default_value("dilations", (1, 1))
        self.add_default_value("output_shape", (0, 0))
        self.add_default_value("group", 1)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        shape = []
        outc = self.group * wshape[1]
        if len(xshape) == 5:
            od = _convtranspose_output_shape(
                xshape[2],
                self.output_padding[0],
                self.pads[0] + self.pads[3],
                wshape[2],
                self.strides[0],
                self.dilations[0],
            )
            ow = _convtranspose_output_shape(
                xshape[3],
                self.output_padding[1],
                self.pads[1] + self.pads[4],
                wshape[3],
                self.strides[1],
                self.dilations[1],
            )
            oh = _convtranspose_output_shape(
                xshape[4],
                self.output_padding[2],
                self.pads[2] + self.pads[5],
                wshape[4],
                self.strides[2],
                self.dilations[2],
            )
            shape = [xshape[0], outc, od, ow, oh]
            if volume(self.output_shape) != 0:
                shape[2:] = self.output_shape
        if len(xshape) == 4:
            ow = _convtranspose_output_shape(
                xshape[2],
                self.output_padding[0],
                self.pads[0] + self.pads[2],
                wshape[2],
                self.strides[0],
                self.dilations[0],
            )
            oh = _convtranspose_output_shape(
                xshape[3],
                self.output_padding[1],
                self.pads[1] + self.pads[3],
                wshape[3],
                self.strides[1],
                self.dilations[1],
            )
            shape = [xshape[0], outc, ow, oh]
            if volume(self.output_shape) != 0:
                shape[2:] = self.output_shape
        elif len(xshape) == 3:
            ow = _convtranspose_output_shape(
                xshape[2],
                self.output_padding[0],
                self.pads[0] + self.pads[1],
                wshape[2],
                self.strides[0],
                self.dilations[0],
            )
            shape = [xshape[0], outc, ow]
            if volume(self.output_shape) != 0:
                shape[2] = self.output_shape[0]
        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        macs = 0
        if len(outtensors) == 1:
            if len(intensors) == 3 or len(intensors) == 2:
                kernel_shape = intensors[1].get_shape()
                outvol = volume(outtensors[0].get_shape())
                reduce_shape = kernel_shape[1:]
                reduce_vol = volume(reduce_shape)
                macs += outvol * reduce_vol * MUL_MACS
                if len(intensors) > 2:
                    macs += outvol * ADD_MACS
        return [macs, 0]


@NODE_REGISTRY.register()
class ReshapeNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        srcshape = intensors[0].get_shape()
        if not is_valid_ndarray(intensors[1].get_numpy()):
            outtensors[0].update_shape(
                [
                    1,
                ]
            )
            outtensors[0].update_dtype(intensors[0].dtype)
            return
        shape = intensors[1].get_numpy()
        newshape = []
        for i in range(len(shape)):
            if shape[i] == 0:
                newshape.append(int(srcshape[i]))
            else:
                newshape.append(int(shape[i]))
        sum = volume(newshape)
        raw = volume(srcshape)
        if sum < 0:
            remain = raw // -sum
            for i, val in enumerate(newshape):
                if val == -1:
                    newshape[i] = remain
                    break
        assert raw == volume(newshape)
        outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        shape = []
        xtensor = intensors[0].get_numpy()
        stensor = intensors[1].get_numpy()
        for i, v in enumerate(stensor):
            if v == 0:
                shape.append(xtensor.shape[i])
            else:
                shape.append(v)
        ret = xtensor.reshape(shape)
        outtensors[0].update_tensor(ret)


@NODE_REGISTRY.register()
class GatherElementsNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.add_default_value("axis", 0)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[1].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        x = intensors[0].get_numpy()
        indice = intensors[1].get_numpy()
        outtensor = numpy.zeros_like(indice)
        for i in numpy.ndindex(outtensor.shape):
            idx = list(i)
            idx[self.axis] = indice[i]
            outtensor[i] = x[tuple(idx)]
        outtensors[0].update_tensor(outtensor)


@NODE_REGISTRY.register()
class GRUNode(Node):
    # TODO(Shinji) add activation types support in attributes
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        seq_len = xshape[0]
        batch = xshape[1]
        num_dir = wshape[0]
        h_len = wshape[1] // 3
        outtensors[0].update_shape([seq_len, num_dir, batch, h_len])
        outtensors[0].update_dtype(intensors[0].dtype)
        outtensors[1].update_shape([num_dir, batch, h_len])
        outtensors[1].update_dtype(intensors[0].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        rshape = intensors[2].get_shape()
        bshape = intensors[3].get_shape()
        batch = xshape[1]
        seq = xshape[0]
        h_len = wshape[1] // 3
        ht_size = volume([batch, seq, h_len])
        # r = sigmoid(Wr*X+Wr*Ht-1+br)
        # z = sigmoid(Wz*X+Wz*Ht-1+bz)
        # h' = tanh(W*X+W*(r*Ht-1)+bh)
        # Ht = (1-z)*Ht-1 + z*h'
        gemm_macs = (volume(wshape) + volume(rshape)) * batch * seq
        gemm_bias_macs = volume(bshape) * ADD_MACS * batch * seq
        gemm_add_macs = ht_size * ADD_MACS * 3
        sig_macs = ht_size * EXP_MACS * 2
        tanh_macs = ht_size * TANH_MACS
        blend_macs = ht_size * (ADD_MACS + MUL_MACS + MUL_MACS + ADD_MACS + MUL_MACS)
        macs = (
            gemm_macs
            + gemm_bias_macs
            + gemm_add_macs
            + sig_macs
            + tanh_macs
            + blend_macs
        )
        return [macs, 0]


@NODE_REGISTRY.register()
class ConstantOfShapeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("value", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(list(intensors[0].get_numpy()))
        if self.value is None:
            outtensors[0].update_dtype(numpy.float32)
        else:
            outtensors[0].update_dtype(self.value.dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        arr = numpy.zeros(intensors[0].get_numpy(), dtype=self.value.dtype)
        if self.value is not None and len(self.value) == 1:
            arr.fill(self.value[0])
        outtensors[0].update_tensor(arr)


@NODE_REGISTRY.register()
class CastNode(Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(onnxdtype2npdtype(self.to))

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_tensor(
            intensors[0].get_numpy().astype(onnxdtype2npdtype(self.to))
        )


@NODE_REGISTRY.register()
class SplitNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value("axis", None)
        self.add_default_value("split", None)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        end = 0
        inshape = intensors[0].get_shape()
        if self.split is None:
            if len(intensors) == 2:
                split = intensors[1].get_numpy()
            else:
                split = [inshape[self.axis] // 2] * 2
        else:
            split = self.split
        axis = _axes_neg2pos(len(inshape), [self.axis])[0]
        shapes = []
        for v in split:
            shape = []
            for i in range(len(inshape)):
                if i == axis:
                    if end + v < inshape[axis]:
                        shape.append(v)
                    else:
                        shape.append(inshape[axis] - end)
                else:
                    shape.append(inshape[i])
            end += v
            shapes.append(shape)
        for i in range(len(shapes)):
            outtensors[i].update_shape(shapes[i])
            outtensors[i].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        splitpos = []
        end = 0
        inshape = intensors[0].get_shape()
        if self.split is None:
            if len(intensors) == 2:
                split = intensors[1].get_numpy()
            else:
                split = [inshape[self.axis] // 2]
        else:
            split = self.split

        axis = _axes_neg2pos(len(inshape), [self.axis])[0]
        for v in split:
            if end + v >= inshape[axis]:
                break
            splitpos.append(end + v)
            end += v
        ret = numpy.split(intensors[0].get_numpy(), splitpos, axis)
        for i, t in enumerate(ret):
            outtensors[i].update_tensor(t)


def create_node(n: onnx.NodeProto):
    node_class = NODE_REGISTRY.get(n.op_type + "Node")
    if node_class != None:
        instance = node_class(n)
        return instance
    # warnings.warn(
    #     f"node {n.op_type} is not registed for profiling, return 0 Macs and 0 params as default. "
    #     f"Use NODEPROFILER_REGISTRY to register your profiler for this node."
    # )
    return Node(n)
