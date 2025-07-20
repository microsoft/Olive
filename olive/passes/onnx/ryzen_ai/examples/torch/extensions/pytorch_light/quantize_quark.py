#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
import argparse
import torch
from transformers import AutoModelForCausalLM
from llm_quant.data import get_dataloader
from llm_quant.eval import eval_ppl
from pytorchlight_config import Dtype, Config, BRECQ, ExporterConfig, PytorchlightModelQuantizer, PytorchlightModelExporter
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig
from quark.torch.quantization.config.type import QSchemeType


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="llama-7b", help="Opt: opt_1.3b, gpt2_small, bloom_7b1,falcon_7b,llama-7b")
parser.add_argument('--model_path', type=str, default="", help= "Model path, if not defined, it will be automatically determined based on args.model")
parser.add_argument('--model_dtype', type=str, default='float32', help= "Opt:float32,float16 etc.accoding to your device memory and torch datatype support")
parser.add_argument('--dataset', type=str, default='wikitext2-raw', help='Default: wikitext2.')
parser.add_argument('--dataset_path', type=str, default="/group/dphi_algo_scratch_08/zijunx/data/", help='')
parser.add_argument('--seqlen', type=int, default=1024, help='Sequence length. Default: 1024.')

parser.add_argument('--q_opt_samples', type=int, default=-1, help='Default: -1 SAME as valdata')
parser.add_argument('--eval', action='store_true', help='Eval')
parser.add_argument('--brecq', action='store_true', help='Eval')
parser.add_argument('--export', action='store_true', help='Export')
parser.add_argument('--seed', type=int, default=0, help='Default: 0')
parser.add_argument('--example', choices=['int_k', 'bfp16', 'brecq'], default='int_k', help='Options: int_k, bfp16, brecq')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def set_seed(seed):
    torch.manual_seed(seed)

def main():
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    # Load data
    logging.info(f"Data {args.dataset} loading...")
    q_opt_data, val_data = get_dataloader(args.model_path, args)


    if args.example == 'int_k':
        GLOBAL_SPEC = QuantizationSpec(dtype=Dtype.int8, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False)
        GLOBAL_CONFIG = QuantizationConfig(weight=GLOBAL_SPEC, input_tensors=GLOBAL_SPEC)
        quant_config = Config(global_quant_config=GLOBAL_CONFIG)
        quantizer = PytorchlightModelQuantizer(quant_config)
        qmodel = quantizer.quantize_model(model, q_opt_data)


    if args.example == 'bfp16':
        GLOBAL_SPEC = QuantizationSpec(dtype=Dtype.bfp16, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False)
        GLOBAL_CONFIG = QuantizationConfig(weight=GLOBAL_SPEC, input_tensors=GLOBAL_SPEC)
        quant_config = Config(global_quant_config=GLOBAL_CONFIG)
        quantizer = PytorchlightModelQuantizer(quant_config)
        qmodel = quantizer.quantize_model(model, None)

    # Export onnx
    if args.export:
        freeze_model = quantizer.freeze(qmodel)
        export_config = ExporterConfig(pytorch_light_export_config={}, json_export_config={})
        exporter = PytorchlightModelExporter(config=export_config, export_dir="./")
        input_args = next(iter(q_opt_data))[0].to(device=device)
        exporter.export_onnx_model(freeze_model, input_args)

    # Run brecq
    if args.example == 'brecq':
        GLOBAL_SPEC = QuantizationSpec(dtype=Dtype.int8, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False)
        GLOBAL_CONFIG = QuantizationConfig(weight=GLOBAL_SPEC, input_tensors=GLOBAL_SPEC)
        quant_config = Config(global_quant_config=GLOBAL_CONFIG)
        quant_config.algo_config = BRECQ()
        quantizer = PytorchlightModelQuantizer(quant_config)
        qmodel = quantizer.quantize_model(model, q_opt_data)

    # Evaluation quant model
    if args.eval:
        ppl_quantized = eval_ppl(qmodel, val_data, device, args)
        logging.info(f"Quant eval perplexity: {ppl_quantized}")

if __name__ == '__main__':
    main()
