# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import torch
import torch.distributed as dist
from mpi4py.MPI import COMM_WORLD
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

# NOTE(shaahji): Optimum model logic load expects a directory path as input (instead of the usual file path).
# Also, it scans the folder for the first available .onnx model. Olive exports all distributed/ranked models
# in the same folder which makes it difficult to run this script as is. The models in the Olive output folder
# need to be reorganized so there's only a single model in the folder. So, instead of,
#   * output_folder/model_00.onnx
#   * output_folder/model_01.onnx
#   * output_folder/model_02.onnx
#   * output_folder/model_03.onnx
#
# reorganize the folder to the following -
#   * output_folder/model_00/model_00.onnx
#   * output_folder/model_01/model_01.onnx
#   * output_folder/model_02/model_02.onnx
#   * output_folder/model_03/model_03.onnx
#
# Ideally, Olive logic should be updated to generate each ranked model in its own folder.

# ruff: noqa: T201


rank = COMM_WORLD.Get_rank()
world_size = COMM_WORLD.Get_size()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
device = torch.device(f"cuda:{rank}")
dist.init_process_group("nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(device)
torch.cuda.empty_cache()

model_id = "meta-llama/Llama-2-7b-hf"
model_path = "models/tensor_parallel/tensor_parallel-conversion-transformers_optimization_fp16/gpu-cuda_model/model_{:02d}".format(  # noqa: E501
    rank
)
prompt = "What is an apple?"
# prompt = "Is it normal to have a dark ring around the iris of my eye?"

model_path = Path(model_path)

if not (model_path / "config.json").exists():
    config = AutoConfig.from_pretrained(model_id)
    config.save_pretrained(model_path)
else:
    config = AutoConfig.from_pretrained(model_path)

if not (model_path / "generation_config.json").exists():
    gen_config = GenerationConfig.from_pretrained(model_id)
    gen_config.save_pretrained(model_path)
else:
    gen_config = GenerationConfig.from_pretrained(model_path)

# config.num_attention_heads = config.num_attention_heads // world_size
config.num_key_value_heads = config.num_key_value_heads // world_size

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

model = ORTModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    generation_config=gen_config,
    use_io_binding=True,
    provider="CUDAExecutionProvider",
    provider_options={"device_id": str(rank)},
)
outputs = model.generate(**inputs, max_new_tokens=256)
print(rank, tokenizer.decode(outputs[0], skip_special_tokens=True))


# mpirun -n 4 -x NCCL_DEBUG=INFO python3 tensor_parallel_generate.py
