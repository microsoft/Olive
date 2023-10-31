# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch


class ArgmaxSampling(torch.nn.Module):
    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True)
