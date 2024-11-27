# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
try:
    import torch_tensorrt  # noqa: F401 # pylint: disable=unused-import
except ImportError:
    raise ImportError("Please install torch_tensorrt with: pip install torch-tensorrt") from None

import io
import logging
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import tensorrt as trt
from torch_tensorrt.fx import compile  # noqa: A004 # pylint: disable=redefined-builtin
from torch_tensorrt.fx import InputTensorSpec, TRTInterpreter, TRTModule
from torch_tensorrt.fx.tracer.acc_tracer import acc_tracer

if TYPE_CHECKING:
    import torch


class TRTLinearLayer(TRTModule):
    def forward(self, inputs):
        """Forward pass of the module. Casts input to fp16 and casts output back to original data type."""
        import torch

        return super().forward(inputs.to(torch.float16)).to(inputs.dtype)


def compile_trt_model(torch_module: "torch.nn.Module", hidden_states: "torch.Tensor", batch_size: int, seqlen: int):
    """Compile a torch module to a TensorRT module.

    :param torch_module: The torch module to compile. Only torch.nn.Linear modules are supported currently.
    :param hidden_states: The input tensor to the torch module.
    :param batch_size: The batch size of the input tensor.
    :param seqlen: The maximum sequence length of the input tensor. seqlen dimension is treated as dynamic.
    """
    import torch

    # disable logging from torch_tensorrt
    # torch_tensorrt logs are very verbose and produces multiple lines of log per module
    # this makes the log file very large and hard to read
    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)

    if not isinstance(torch_module, torch.nn.Linear):
        raise TypeError("torch_module must be a torch.nn.Linear module. Other modules are not supported.")

    # whether the batch and seqlen dimensions are flattened
    flattened = len(hidden_states.shape) == 2

    # redirect stdout to avoid printing
    f = io.StringIO()
    with redirect_stdout(f):
        # compile torch module to TensorRT module
        low_model = compile(torch_module.to(torch.float16), [hidden_states.to(torch.float16)]).eval()
    f.close()
    acc_model = acc_tracer.trace(low_model, [hidden_states.to(torch.float16)])
    # get input specs
    hidden_size = hidden_states.shape[-1]
    if not flattened:
        shape = [batch_size, -1, hidden_size]
        shape_ranges = [
            ((batch_size, 1, hidden_size), (batch_size, 100, hidden_size), (batch_size, seqlen, hidden_size))
        ]
    else:
        shape = [-1, hidden_size]
        shape_ranges = [
            ((batch_size, hidden_size), (batch_size * 100, hidden_size), (batch_size * seqlen, hidden_size))
        ]
    input_specs = [InputTensorSpec(shape=shape, dtype=torch.float16, shape_ranges=shape_ranges)]
    # create TensorRT module
    interpreter = TRTInterpreter(acc_model, input_specs, explicit_batch_dimension=True, logger_level=trt.Logger.ERROR)
    result = interpreter.run(sparse_weights=True)
    return TRTLinearLayer(result.engine, result.input_names, result.output_names)
