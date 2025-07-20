#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from typing import Optional
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.torch.quantization.config.config import QuantizationSpec

class CustomPercentileObserver(PerTensorMinMaxObserver):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)
        self.forward_count = 0
        self.tensor_range = None

    def filter_function(self, idx):
        return idx % 20 < 8

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if self.filter_function(self.forward_count):
            max_val = torch.max(x_orig)
            min_val = torch.min(x_orig)
            tensor_range = torch.maximum(torch.abs(max_val), torch.abs(min_val))
            if self.tensor_range is None:
                self.tensor_range = tensor_range
            self.tensor_range = torch.minimum(self.tensor_range, tensor_range).to(x_orig.dtype)
            self.min_val = self.tensor_range
            self.max_val = self.tensor_range
        self.forward_count += 1
        return x_orig
