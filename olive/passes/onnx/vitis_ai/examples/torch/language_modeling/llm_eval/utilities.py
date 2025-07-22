#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ruff: noqa: T201

import json
import random

import numpy as np
import onnxruntime_genai as og
import torch
from lm_eval.evaluator import (
    eval_logger,
)
from tqdm import tqdm


def get_dtype(dtype_arg):
    if dtype_arg == "fp32":
        dtype = torch.float32
    if dtype_arg == "fp16":
        dtype = torch.float16
    if dtype_arg == "bf16":
        dtype = torch.bfloat16
    return dtype


def _adjust_config(task_dict, predict_only=False, num_fewshot=None, fewshot_random_seed=1234, gen_kwargs=None):
    adjusted_task_dict = {}
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            adjusted_task_dict = {
                **adjusted_task_dict,
                task_name: _adjust_config(task_obj),
            }

        else:
            if task_obj.get_config("output_type") == "generate_until":
                if gen_kwargs is not None:
                    task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

            if predict_only:
                eval_logger.info(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
                # we have to change the class properties post-hoc. This is pretty hacky.
                task_obj.override_metric(metric_name="bypass")

            # override tasks' fewshot values to the provided num_fewshot arg value
            # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
            if num_fewshot is not None:
                if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                    eval_logger.info(
                        f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                    )
                else:
                    eval_logger.warning(
                        f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                    )
                    task_obj.set_config(key="num_fewshot", value=num_fewshot)
            else:
                # if num_fewshot not provided, and the task does not define a default one, default to 0
                if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                    task_obj.set_config(key="num_fewshot", value=0)
            # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
            task_obj.set_fewshot_seed(seed=fewshot_random_seed)

            adjusted_task_dict[task_name] = task_obj

    return adjusted_task_dict


def set_seeds(args):
    random.seed(args.random_seed)
    np.random.seed(args.numpy_random_seed)
    torch.manual_seed(args.torch_random_seed)


def oga_generation(args, inputs, model_dir, filename, seq_len=512, max_seq_len=1024):
    # defined PSU prompt
    PSU_PROMPT = "Please solve following problem and explain it to me. Then give me final answer at the end with a single number preceded by string '#### '. "
    # load the eos_token_id
    with open(str(args.import_model_dir + "genai_config.json")) as f:
        config = json.load(f)
    eos_token_id = config["model"]["eos_token_id"]

    set_seeds(args)

    def model_load(model_dir):
        model = og.Model(model_dir)
        return model

    def get_tokenizer(model):
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        return tokenizer, tokenizer_stream

    model = model_load(model_dir)
    tokenizer, tokenizer_stream = get_tokenizer(model)
    outputs = []
    with open(filename, "w") as file:
        for i in tqdm(range(len(inputs))):
            if args.case == "default":
                prompt = inputs[i]
            elif (args.case == "psu_prompt" or args.case == "psu_prompt_eos_stop") and args.tasks == "tinyGSM8k":
                # preprending PSU Prompt
                prompt = PSU_PROMPT + inputs[i]

            input_tokens = tokenizer.encode(prompt)[:seq_len]

            search_options = {}
            params = og.GeneratorParams(model)
            params.input_ids = input_tokens

            search_options["max_length"] = max_seq_len
            params.set_search_options(**search_options)
            generator = og.Generator(model, params)

            num_output_tokens = 0
            tokens = []
            response = ""
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]

                # early stopping w/eos
                if args.case == "psu_prompt_eos_stop" and new_token == eos_token_id:
                    print(f"****eos triggered, {new_token}****")
                    break
                tokens.append(new_token)
                response += tokenizer_stream.decode(new_token)
                num_output_tokens += 1
            del generator

            # saving OGA generations
            file.write(response + f"\n{args.eor}\n")
            file.flush()
            outputs.append(response)
    return outputs
