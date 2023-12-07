# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch


class ArgmaxSampling(torch.nn.Module):
    def forward(self, logits, seq_lens):
        next_tokens = torch.argmax(logits, dim=-1, keepdim=False)
        indices = seq_lens - 1
        return next_tokens.gather(1, indices.view(-1, 1))
