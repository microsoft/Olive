# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.passes.pytorch.gptq import GptqQuantizer
from olive.passes.pytorch.lora import LoftQ, LoRA, QLoRA
from olive.passes.pytorch.quantization_aware_training import QuantizationAwareTraining
from olive.passes.pytorch.sparsegpt import SparseGPT
from olive.passes.pytorch.tensor_parallel import PyTorchTensorParallel
from olive.passes.pytorch.torch_trt_conversion import TorchTRTConversion

__all__ = [
    "GptqQuantizer",
    "LoftQ",
    "LoRA",
    "PyTorchTensorParallel",
    "QLoRA",
    "QuantizationAwareTraining",
    "SparseGPT",
    "TorchTRTConversion",
]
