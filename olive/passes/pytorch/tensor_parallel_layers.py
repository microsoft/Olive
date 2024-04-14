# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Tensor Parallel layers.
# This layers could replace corresponding PyTorch layers.
# --------------------------------------------------------------------------

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

# pylint: disable=unsubscriptable-object


class AllReduce(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Value) -> torch.Value:  # pylint: disable=arguments-differ
        if torch.onnx.is_in_onnx_export():
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def symbolic(g: torch.Graph, x: torch.Value) -> torch.Value:
        return g.op("com.microsoft::AllReduce", x).setType(x.type())


class TensorParallelColumnLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        world_size=1,
        process_group: torch.distributed.ProcessGroup = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.use_bias = bias
        self.world_size = world_size
        self.in_features = in_features
        self.out_features = out_features // self.world_size
        # We change from traditional `nn.Linear` and remove unnecessary `torch.Tensor.transpose` operation
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def load_rank_weights(self, rank, world_size):
        weight = self.weight.chunk(world_size)[rank]
        self.weight = nn.Parameter(weight.contiguous())
        if self.use_bias:
            bias = self.bias.chunk(world_size)[rank]
            self.bias = nn.Parameter(bias.contiguous())

    def reset_parameters(self) -> None:
        # From `torch.nn.Linear`
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # pylint: disable=protected-access
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        # From `torch.nn.Linear`
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward(self, ip: torch.Tensor) -> torch.Tensor:
        return F.linear(ip, weight=self.weight, bias=self.bias)  # pylint: disable=not-callable


class TensorParallelRowLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        world_size=1,
        process_group: torch.distributed.ProcessGroup = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.use_bias = bias
        self.world_size = world_size
        self.in_features = in_features // self.world_size
        self.out_features = out_features
        # We change from traditional `nn.Linear` and remove unnecessary `torch.Tensor.transpose` operation
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def load_rank_weights(self, rank, world_size):
        weight = self.weight.chunk(world_size, dim=1)[rank]
        self.weight = nn.Parameter(weight.contiguous())

    def reset_parameters(self) -> None:
        # From `torch.nn.Linear`
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # pylint: disable=protected-access
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, ip: torch.Tensor) -> torch.Tensor:
        out = F.linear(ip, weight=self.weight, bias=self.bias)  # pylint: disable=not-callable
        return AllReduce.apply(out)

    def extra_repr(self) -> str:
        # From `torch.nn.Linear`
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
