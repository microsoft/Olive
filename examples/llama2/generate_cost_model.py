# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
from pathlib import Path

from olive.common.constants import DEFAULT_HF_TASK
from olive.common.hf.mappings import MODELS_TO_EMBEDDINGS_MAPPING
from olive.model import HfModelHandler

# ruff: noqa: T201

precision_to_bytes = {
    "fp32": 4,
    "fp16": 2,
    "fp8": 1,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "int4": 0.5,
    "uint4": 0.5,
    "nf4": 0.5,
    "fp4": 0.5,
}


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument("-m", "--model_name_or_path", type=str, required=True, help="Name of the model")
    parser.add_argument(
        "-t", "--task", type=str, default=DEFAULT_HF_TASK, help="Task for which the huggingface model is used."
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument(
        "-p", "--weight_precision", type=str, default="fp16", choices=precision_to_bytes.keys(), help="Weight precision"
    )
    parser.add_argument("--exclude_embeds", action="store_true", help="Exclude embeddings from the cost model")
    parser.add_argument("--exclude_lm_head", action="store_true", help="Exclude lm head from the cost model")

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Output csv path for the cost model. If not provided, the path with the <model_name_or_path.name>.csv",
    )

    args = parser.parse_args(raw_args)

    load_kwargs = {}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    model_handler = HfModelHandler(args.model_name_or_path, task=args.task, **load_kwargs)

    # exclude modules
    modules_to_exclude = set()
    if args.exclude_embeds:
        modules_to_exclude.update(
            MODELS_TO_EMBEDDINGS_MAPPING.get(model_handler.get_hf_model_type(), ["model.embed_tokens"])
        )
    if args.exclude_lm_head:
        modules_to_exclude.add("lm_head")

    # model costs
    costs = {}
    for name, module in model_handler.load_model().named_modules():
        if module._modules or name in modules_to_exclude:
            # has children or excluded
            continue

        num_params = sum(p.numel() for p in module.parameters())
        num_bytes = num_params * precision_to_bytes[args.weight_precision]
        costs[name] = (num_params, num_bytes)

    # write to csv
    output_path = Path(args.output_path or Path(args.model_name_or_path).name).with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("module,num_params,num_bytes\n")
        for module, (num_params, num_bytes) in costs.items():
            f.write(f"{module},{num_params},{num_bytes}\n")

    print(f"Cost model written to {output_path}")
    print(sum(num_bytes for _, num_bytes in costs.values()))


if __name__ == "__main__":
    main()
