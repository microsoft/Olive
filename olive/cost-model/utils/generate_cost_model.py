# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
from models import get_llama2_7b, get_llama2_13b, get_phi3_mini_4k

def print_number(n, names = ['',' Thousand',' Million',' Billion',' Trillion']):
    import math

    n = float(n)
    nameidx = max(0,min(len(names)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.2f}{}'.format(n / 10**(3 * nameidx), names[nameidx])

def print_memory_size(n):
    return print_number(n, ['',' KB',' MB',' GB',' TB', 'PB'])

def print_flops(n):
    return print_number(n, ['',' KFlops',' mFLops',' GFlops',' TFlops', 'PFlops'])

def main():
    parser = argparse.ArgumentParser(description="Meru")
    parser.add_argument("--model_name", type=str, help="Model Name")
    parser.add_argument("--batch_size", type=int, default = 1, help="Model Name")
    parser.add_argument("--prompt_size", type=int, default = 1000, help="Prompt Size")
    parser.add_argument("--output_size", type=int, default = 1000, help="Size of generated text")
    parser.add_argument("--weight_precision", type=str, default = "fp16", help="Model weight precision")
    parser.add_argument("--kv_precision", type=str, default = "fp16", help="KV Cache precision")
    args = parser.parse_args()

    print(f"Model name : {args.model_name}")

    m = None
    if args.model_name == "llama2-7b":
        m = get_llama2_7b()
    elif args.model_name == "llama2-13b":
        m = get_llama2_13b()
    elif args.model_name == "phi3-mini-4k":
        m = get_phi3_mini_4k()
    else:
        raise NotImplementedError(f"I do not suport this model {args.model_name} yet")

    m.set_batch_size(args.batch_size)
    m._kv_precision = args.kv_precision
    m._weight_precision = args.weight_precision

    pc = print_number(m.param_count())
    mm = print_memory_size(m.param_count()*2)
    print(f"Parameter count : {pc}")
    print(f"Memory required to load the model {mm}")

    pfc = print_flops(m.flops(args.prompt_size))
    pm = print_memory_size(m.memory_for_token_processing(args.prompt_size))
    print(f"FLOPs required to process the prompt ({args.prompt_size} tokens) : {pfc}")
    print(f"Memory required to process the prompt ({args.prompt_size} tokens) : {pm}")

    ofc = print_flops(m.flops(args.prompt_size + args.output_size))
    om = print_memory_size(m.memory_for_token_processing(args.prompt_size + args.output_size))
    print(f"FLOPs required to generate the ouptput ({args.output_size} tokens) : {ofc}")
    print(f"Memory required to generate the ouptput ({args.output_size} tokens) : {om}")

    print(f"Tokens \t Total Memory \t kv size \t Total Flops ")
    for t in range(0, args.prompt_size+1, int((args.prompt_size)/10)):
        nt = t if t != 0 else 1
        kv = print_memory_size(m.kv_size(nt))
        tm = print_memory_size(m.param_count()*2 + m.kv_size(nt))
        f = print_flops(m.flops(nt))
        print(f"{nt} \t {tm} \t {kv} \t {f}")

    print("Finished prompt processing")

    for t in range(args.prompt_size, args.prompt_size+args.output_size+1, int((args.output_size)/10)):
        nt = t if t != 0 else 1
        kv = print_memory_size(m.kv_size(nt))
        tm = print_memory_size(m.param_count()*2 + m.kv_size(nt))
        f = print_flops(m.flops(nt))
        print(f"{nt} \t {tm} \t {kv} \t {f}")

    m.print_cost_model(args.prompt_size, args.output_size)

if __name__ == "__main__":
    main()