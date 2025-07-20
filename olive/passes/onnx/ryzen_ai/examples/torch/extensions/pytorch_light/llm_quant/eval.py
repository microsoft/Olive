#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from torch import nn
from tqdm import tqdm

@torch.no_grad()
def eval_ppl(model, dataloader, device, args):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    seqlen = args.seqlen
    nsamples = len(dataloader)
    nlls = []

    iterator = iter(dataloader)
    for i in tqdm(range(nsamples)):
        batch = next(iterator)[0].to(device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].to(device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    model.config.use_cache = use_cache

    return ppl
