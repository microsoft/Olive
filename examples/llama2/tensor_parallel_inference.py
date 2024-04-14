# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os
import pprint
import sys
from pathlib import Path

import onnxruntime
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

# ruff: noqa: T201, T203


def _run_pytorch(model, inputs):
    return model(**inputs)


def _mpipool_worker(args):
    filepath, local_rank, world_size, inputs = args

    os.environ["OMPI_COMM_WORLD_RANK"] = str(local_rank)
    os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)

    from mpi4py.MPI import COMM_WORLD

    local_rank = COMM_WORLD.Get_rank()
    COMM_WORLD.barrier()

    print(f"rank: {local_rank}, filepath: {filepath}")

    session = onnxruntime.InferenceSession(
        filepath,
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": str(local_rank)}],
    )
    return session.run(None, inputs)[0]


def _run_onnx(filepath, world_size, inputs):
    from mpi4py.futures import MPIPoolExecutor
    from mpi4py.MPI import COMM_WORLD

    args = [(filepath.format(rank), rank, world_size, inputs) for rank in range(world_size)]
    with MPIPoolExecutor(max_workers=world_size) as executor:
        outputs = executor.map(_mpipool_worker, args)
        executor.shutdown()

    COMM_WORLD.barrier()

    return list(outputs)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        dest="model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model Id to load",
    )
    parser.add_argument(
        "--filename-pattern",
        dest="filename_pattern",
        type=str,
        help="Onnx model file name pattern to use for distributed run",
    )
    parser.add_argument("--world-size", dest="world_size", type=int, help="World size for distributed run")
    parser.add_argument(
        "--compare",
        dest="compare",
        action="store_true",
        default=False,
        help="Compare results from distributed session to non-distributed session",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
    args = parser.parse_args()

    prompt = "Is it normal to have a dark ring around the iris of my eye?"
    device = "cuda"

    tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
    tokens = tokenizer(prompt, return_tensors="pt")
    tokenizer = None

    config = LlamaConfig.from_pretrained(args.model_id)
    num_heads = config.num_key_value_heads
    head_size = config.hidden_size // config.num_attention_heads
    batch_size, past_seq_len = 2, 0

    model = LlamaForCausalLM.from_pretrained(args.model_id, torch_dtype=config.torch_dtype, config=config)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    position_ids = tokens.attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(tokens.attention_mask == 0, 1)

    pytorch_inputs = {
        "input_ids": tokens.input_ids.to(device),
        "attention_mask": tokens.attention_mask.to(device),
        "position_ids": position_ids.to(device),
    }
    pytorch_outputs = _run_pytorch(model, pytorch_inputs)

    if args.debug:
        with (Path.cwd() / "pytorch_output.txt").open("wt") as strm:
            pprint.pprint(pytorch_outputs, stream=strm)
            strm.flush()

    onnx_inputs = {
        "input_ids": tokens.input_ids.numpy(),
        "attention_mask": tokens.attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
    }
    for i in range(config.num_hidden_layers):
        onnx_inputs[f"past_key_values.{i}.key"] = torch.rand(
            batch_size, num_heads // args.world_size, past_seq_len, head_size, dtype=torch.float16
        ).numpy()
        onnx_inputs[f"past_key_values.{i}.value"] = torch.rand(
            batch_size, num_heads // args.world_size, past_seq_len, head_size, dtype=torch.float16
        ).numpy()

    onnx_outputs = _run_onnx(args.filename_pattern, args.world_size, onnx_inputs)

    if args.debug:
        with (Path.cwd() / "onnx_output.txt").open("wt") as strm:
            pprint.pprint(onnx_outputs, stream=strm)
            strm.flush()

    if args.compare and (pytorch_outputs is not None) and (onnx_outputs is not None):
        import numpy as np

        pytorch_outputs = pytorch_outputs["logits"].cpu().numpy()

        results = {}
        for i in range(args.world_size):
            results[f"pytorch vs. onnx_{i:02d}"] = np.fabs(np.median(pytorch_outputs - onnx_outputs[i]))

            if i > 0:
                results[f"onnx_00 vs. onnx_{i:02d}"] = np.fabs(np.median(onnx_outputs[0] - onnx_outputs[i]))

        if args.debug:
            pprint.pprint(results)

        atol = 3e-3
        if not np.all(np.array(list(results.values())) < atol):
            raise RuntimeError("Inference test failed!")

    print("Inference test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(_main())


# python3 tensor_parallel_inference.py      \
#   --filename-pattern model_{:02d}.onnx    \
#   --world-size 2                          \
#   [--model-id meta-llama/Llama-2-7b-hf]   \
#   [--debug]
#
# python3 tensor_parallel_inference.py      \
#   --filename-pattern model_{:02d}.onnx    \
#   --world-size 2                          \
#   [--model-id meta-llama/Llama-2-7b-hf]   \
#   [--compare]                             \
#   [--debug]
