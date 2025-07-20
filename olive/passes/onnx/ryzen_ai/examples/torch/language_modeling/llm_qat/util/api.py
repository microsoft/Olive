#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import time

import torch
from torch import nn
from datasets import load_dataset

from .utils import AverageMeter
from llm_eval.evaluation import ppl_eval

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver


def weight_only_quantize(model, loader, quant_scheme, group_size):
    if quant_scheme in ["w_uint4_asym", "w_int4_sym"]:
        dtype = Dtype.uint4 if 'unint4' in quant_scheme else Dtype.int4
        symmetric = False if 'asym' in quant_scheme else True
        WEIGHT_SPEC = QuantizationSpec(dtype=dtype,
                                       observer_cls=PerGroupMinMaxObserver,
                                       symmetric=symmetric,
                                       scale_type=ScaleType.float,
                                       round_method=RoundType.half_even,
                                       qscheme=QSchemeType.per_group,
                                       ch_axis=1,
                                       is_dynamic=False,
                                       group_size=group_size)
    else:
        raise Exception(f"Not implement for other quant scheme {quant_scheme}")

    QUANT_SPEC = QuantizationConfig(weight=WEIGHT_SPEC)
    quant_config = Config(global_quant_config=QUANT_SPEC)

    quantizer = ModelQuantizer(quant_config)

    model = quantizer.quantize_model(model, loader)
    model = quantizer.freeze(model)

    return model, quant_config

def full_finetune(model, tokenizer, finetune_loader, optimizer, num_epoch, output_dir):
    main_device = next(model.parameters()).device
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    best_ppl = ppl_eval(model, testenc, main_device)
    print(f"\n[QUARK-INFO]: Perplexity Test of Wikitext2 before Fine-Tuning: {best_ppl}")

    epoch_iters = len(finetune_loader)
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()

    tic = time.time()

    for epoch in range(num_epoch):
        model.train()
        print(f"\n[QUARK-INFO]: Start Fine-Tuning - Epoch {epoch + 1}:")
        for i_iter, sample in enumerate(finetune_loader):
            input = sample[0].to(main_device)
            output = model(input).logits.to(main_device)

            loss_fct = nn.CrossEntropyLoss()
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_time.update(time.time() - tic)
            tic = time.time()
            ave_loss.update(loss.item())
            ave_acc.update(torch.exp(loss).item())

            if i_iter == 0 or (i_iter + 1) % 50 == 0:
                msg = 'Epoch:[{}/{}] Iter:[{}/{}], Time: {:.2f}, lr: {:.6f}, Loss: {:.2f}, Acc: {:.2f}' .format(
                    epoch, num_epoch, i_iter + 1, epoch_iters,
                    batch_time.average(), optimizer.param_groups[0]['lr'], ave_loss.average(), ave_acc.average())
                print(msg)

        ppl = ppl_eval(model, testenc, main_device)
        print(f"\n[QUARK-INFO]: Perplexity Test of Wikitext2 after Fine-Tuning - Epoch {epoch + 1}: {ppl}")

        torch.save({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "ppl": ppl,
        }, os.path.join(output_dir, 'last.pth'))
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "ppl": ppl,
            }, os.path.join(output_dir, 'best.pth'))
    return
