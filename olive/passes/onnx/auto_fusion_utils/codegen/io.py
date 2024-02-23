# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    sympy = None
    create_expand_pow_optimization = None
else:
    import sympy
    from sympy.codegen.rewriting import create_expand_pow_optimization


SHAPE_TYPE = List[Union[str, int]]


class KernelIO:
    def __init__(self, output_shape: SHAPE_TYPE):
        # output shape of the kernel
        # since there is no reduction, the output shape has the maximum possible dimensions
        self.rank = len(output_shape) or 1
        self.output_shape = self.make_sympy_shape(output_shape)

        # strides for the output shape
        self.strides = [sympy.Integer(1)]
        for i in range(self.rank - 2, -1, -1):
            self.strides.insert(0, self.strides[0] * self.output_shape[i + 1])

        # mapping from input name to systematic input name
        self.input_shapes = {}
        self.input_strides = {}

        # which input do we see the dim first from
        self.dim_sources = {}
        self.used_dims = set()

    @staticmethod
    def make_sympy_shape(shape: SHAPE_TYPE):
        return [sympy.Integer(dim) if isinstance(dim, int) else sympy.Symbol(dim) for dim in shape or [1]]

    def add_input(self, name: str, shape: SHAPE_TYPE):
        if name in self.input_shapes:
            # assume shape is the same
            return

        self.input_shapes[name] = self.make_sympy_shape(shape)

        for idx, dim in enumerate(shape):
            if isinstance(dim, int):
                continue
            if dim not in self.dim_sources:
                self.dim_sources[dim] = (name, idx)

        strides = []
        full_shape = [sympy.Integer(1)] * (self.rank - len(shape)) + self.input_shapes[name]
        running_stride = sympy.Integer(1)
        for i in range(self.rank - 1, -1, -1):
            if self.output_shape[i] == full_shape[i]:
                strides.insert(0, running_stride)
                running_stride *= full_shape[i]
            else:
                strides.insert(0, sympy.Integer(0))
        self.input_strides[name] = strides

        if any(val == sympy.Integer(0) for val in strides):
            for idx, val in enumerate(strides):
                if val != sympy.Integer(0):
                    self.used_dims.add(idx)

    def get_symbolic_dims(self):
        return [dim.name for dim in self.output_shape if isinstance(dim, sympy.Symbol)]

    def get_dim_indices(self):
        dim_indices = []
        for idx in range(self.rank):
            if idx not in self.used_dims:
                continue
            div_str = f" // {maybe_parenthesize(str(self.strides[idx]))}" if idx != self.rank - 1 else ""
            mod_str = f" % {maybe_parenthesize(str(self.output_shape[idx]))}" if self.strides[idx] != 0 else ""
            dim_indices.append(f"y_{idx}_idx = y_idx{div_str}{mod_str}")
        return dim_indices

    def get_input_idx(self, input_name: str):
        if self.input_shapes[input_name] == self.output_shape:
            return "y_idx"
        dim_indices = self.make_sympy_shape([f"y_{idx}_idx" for idx in range(self.rank)])
        expand_opt = create_expand_pow_optimization(6)
        return str(expand_opt(sympy_dot(dim_indices, self.input_strides[input_name])))

    def get_dim_source(self, dim_name: str) -> Tuple[str, int]:
        return self.dim_sources[dim_name]

    @classmethod
    def from_kernel_info(cls, kernel_info) -> "KernelIO":
        kernel_io = cls(kernel_info["output_shape"])
        for input_name, input_shape in kernel_info["shapes"].items():
            kernel_io.add_input(input_name, input_shape)
        return kernel_io


def maybe_parenthesize(name: str) -> str:
    if not re.match("^[A-Za-z0-9_.]*$", name):
        return f"({name})"
    return name


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))
