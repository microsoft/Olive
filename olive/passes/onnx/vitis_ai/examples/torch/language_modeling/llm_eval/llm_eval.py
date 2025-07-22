#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ruff: noqa: T201
import argparse
import json
import os
import sys
from typing import Optional, Type

import custom_lm_eval_harness
import datasets
import lm_eval
import onnxruntime_genai as og
from datasets import load_dataset
from evaluation import (
    meteor_eval,
    mlperf_rouge_eval,
    ppl_eval,
    ppl_eval_for_kv_cache,
    rouge_eval,
    rouge_meteor_generations,
)
from lm_eval import utils
from lm_eval.__main__ import cli_evaluate, parse_eval_args, setup_parser
from lm_eval.api.model import T
from lm_eval.api.task import Task
from lm_eval.evaluator_utils import get_task_list
from lm_eval.tasks import TaskManager, get_task_dict
from onnxruntime import InferenceSession
from optimum.onnxruntime import ORTModelForCausalLM
from quark.torch import ModelImporter
from torch import nn as nn
from transformers import AutoConfig, AutoTokenizer, tokenization_utils_base
from utilities import _adjust_config, oga_generation

# TODO: Using sys.path.append is bad practice.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_utils.model_preparation import get_model, get_model_type, get_tokenizer


# Get model for PPL
def prepare_model(
    model_dir: str,
    model_reload: bool,
    import_file_format: str,
    import_model_dir: str,
    seq_len: int,
    device: str,
    multi_gpu: bool,
    ppl: bool,
) -> tuple[nn.Module, AutoTokenizer, tokenization_utils_base.BatchEncoding]:
    model, model_dtype = get_model(model_dir, device=device, multi_gpu=multi_gpu)
    model_type = get_model_type(model)

    if import_file_format == "onnx_format":
        if ppl:
            # ppl onnx models loaded via OGA
            model_obj = og.Model(import_model_dir)
        else:
            # for rouge, meteor, lm_eval_harness, onnx models loaded via optimum ORT
            session = InferenceSession(import_model_dir, providers=["CPUExecutionProvider"])
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            model_obj = ORTModelForCausalLM(session, config, use_cache=True, use_io_binding=False)
    else:
        if model_reload:
            print(f"\nRestore quantized model from {import_file_format} file ...")
            importer = ModelImporter(model_info_dir=import_model_dir, saved_format=import_file_format)

            model = importer.import_model_info(model)
        model_obj = model

    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = get_tokenizer(model_dir, max_seq_len=seq_len, model_type=model_type)

    if args.num_eval_data != -1:
        testenc = tokenizer("\n\n".join(testdata["text"][: args.num_eval_data]), return_tensors="pt")
    else:
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return model_obj, tokenizer, testenc


# Save evaluation results to JSON file
def save_evaluation_results(args: argparse.Namespace) -> None:
    import shutil
    from pathlib import Path

    if args.tasks is not None:
        output_path = Path(args.output_path)
        json_files = list(output_path.glob("*.json"))
        with open(json_files[0]) as f:
            harness_results = json.load(f)
            results["harness_metrics"] = harness_results["results"]
        shutil.rmtree(output_path)

    # for ppl, rouge, meteor
    results["quark_metrics"] = {}
    results["quark_metrics"].update(quark_metrics)
    metric_output_dir = os.path.join(args.metrics_output_dir, current_time)
    metric_output_dir = Path(metric_output_dir)
    metric_output_dir.mkdir(parents=True, exist_ok=True)
    metric_output_json_path = metric_output_dir / "evaluation_results.json"
    with open(metric_output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[INFO]: Evaluation results are saved in {metric_output_json_path}.")


# Create model for tasks
def create_from_arg_obj(cls: Type[T], arg_dict: dict, additional_config: Optional[dict] = None) -> T:
    """Overrides the HFLM.create_from_arg_obj"""
    model_dir = arg_dict.pop("pretrained", None)
    model_reload = arg_dict.pop("model_reload", None)
    import_model_dir = arg_dict.pop("import_model_dir", None)
    device = arg_dict.pop("device", None)
    multi_gpu = arg_dict.pop("multi_gpu", None)

    model, model_dtype = get_model(model_dir, device=device, multi_gpu=multi_gpu)
    model_obj = cls(model)

    if model_reload:
        import_file_format = arg_dict.pop("import_file_format", None)

        importer = ModelImporter(model_info_dir=import_model_dir, saved_format=import_file_format)
        importer.import_model_info(model_obj.model)

    return model_obj


# Retrieve dataset in offline mode
def get_dataset(
    task_name,
    num_fewshot,
    limit=None,
    cache_requests=False,
    rewrite_requests_cache=False,
    system_instruction=None,
    apply_chat_template=False,
    fewshot_as_multiturn=False,
):
    def save_data(data, filename):
        with open(filename + ".txt", "w") as file:
            for sample in data:
                file.write(sample + "\n<EOR>\n")

        with open(filename + ".json", "w") as f:
            json.dump(data, f, indent=4)

    if limit is not None:
        limit = int(limit)

    task_manager = TaskManager("INFO")
    task_dict = get_task_dict(task_name, task_manager)
    task_dict = _adjust_config(task_dict, num_fewshot=args.num_fewshot)

    eval_tasks = get_task_list(task_dict)
    for task_output in eval_tasks:
        task: Task = task_output.task
        task.build_all_requests(
            limit=limit,
            rank=0,
            world_size=1,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=None,
            tokenizer_name=getattr(lm, "tokenizer_name", "") if apply_chat_template else "",
        )

        inputs = []
        references = []
        doc_iterator = task.doc_iterator(rank=0, limit=limit, world_size=1)
        for doc_id, doc in doc_iterator:
            if task._config.num_fewshot > 0 or num_fewshot is not None:
                # get the input w/the fewshot examples
                input = task.instances[doc_id].arguments[0]
                inputs.append(input)
            elif num_fewshot == 0 or num_fewshot is None:
                inputs.append(task.doc_to_text(doc))

            references.append(task.doc_to_target(doc))

        # SAVING BOTH JSON AND TXT FILES
        save_data(inputs, f"{task_name}_inputs_limit-{str(limit)}")
        save_data(references, f"{task_name}_references_limit-{str(limit)}")

    print("Task Inputs saved -- coming from lm-evaluation-harness")
    print("Task References saved -- coming from lm-evaluation-harness")


def setup_parser_with_modelopt_args():
    """Args: The following are the key args in `lm-evaluation-harness`, please refer to https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py#L65 for full args,
    --model (`str`): Name of model. Default `hf`.
    --tasks ('str'): List of task names or task groupings to evaluate on. Can be in comma-separated format `task1,task2`, and defalut to be None.
    --model_args (`str`): Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`.
    --num_fewshot (`int`): Number of examples in few-shot context. Default to be None.
    --batch_size ('str'): Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.
    --max_batch_size (`int`): Maximal batch size to try with --batch_size auto. Defalut None.
    --device (`str`): Device to use (e.g. cuda, cuda:0, cpu). Default None.
    --output_path (`str`): The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used. Default None.
    --limit (`float`): Limit the number of examples per task. Default None.
    --use_cache (`str`): A path to a sqlite db file for caching model responses. `None` if not caching. Default None.
    --cache_requests: Speed up evaluation by caching the building of dataset requests. `None` if not caching. Choices ["true", "refresh", "delete"]. Default None.
    --apply_chat_template (`str`): If True, apply chat template to the prompt. Default False.
    """
    parser = setup_parser()

    # additional args in Quark
    parser.add_argument("--seq_len", type=int, help="Sequence length of data", default=512)
    parser.add_argument("--max_new_toks", type=int, help="Maximum tokens generated by model", default=512)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--model_reload", help="safetensors or pth model reload", action="store_true")
    parser.add_argument("--import_model_dir", type=str, help="directory of hf or quark model", default=None)
    parser.add_argument(
        "--import_file_format",
        type=str,
        help="file_format for importing. If you export hf_format, you should use 'hf_format' for reloading.",
        default="quark_format",
        choices=["quark_format", "hf_format", "onnx_format"],
    )
    parser.add_argument("--ppl", action="store_true")
    parser.add_argument("--use_ppl_eval_for_kv_cache", action="store_true")
    parser.add_argument(
        "--use_ppl_eval_for_kv_cache_context_size",
        type=int,
        help="Context size used in PPL evaluation for KV cache.",
        default=1024,
    )
    parser.add_argument(
        "--use_ppl_eval_for_kv_cache_sample_size",
        type=int,
        help="Sample size used in PPL evaluation for KV cache.",
        default=512,
    )
    parser.add_argument(
        "--use_ppl_eval_for_kv_cache_patch_size",
        type=int,
        help="Patch size used in PPL evaluation for KV cache.",
        default=None,
    )
    parser.add_argument(
        "--num_eval_data",
        help="Number of samples for evaluation. The default value is -1, which means the entire dataset is used for evaluation.",
        type=int,
        default=-1,
    )
    parser.add_argument("--metrics_output_dir", default=None, type=str, help="Output path of json with metrics.")
    parser.add_argument("--rouge", action="store_true")
    parser.add_argument("--meteor", action="store_true")
    parser.add_argument(
        "--datasets",
        help="comma seperated dataset selection for rouge or meteor evaluation",
        type=str,
        metavar="datset1,dataset2",
    )
    parser.add_argument(
        "--mode",
        help="standard (end-to-end generation & evals), offline (decoupled generation and evals)",
        default="standard",
        type=str,
    )
    parser.add_argument("--mlperf_rouge", action="store_true")
    parser.add_argument("--eval_data_dir", help="Dataset for evaluation", type=str, default=None)

    # setting all the random seeds -- aligned with the default vals in lm_eval_harness
    parser.add_argument("--random_seed", type=int, required=False, default=0, help="random seed")
    parser.add_argument("--numpy_random_seed", type=int, required=False, default=1234, help="np rand seed")
    parser.add_argument("--torch_random_seed", type=int, required=False, default=1234, help="torch rand seed")

    # arguments for offline mode
    parser.add_argument(
        "--eor",
        type=str,
        required=False,
        default="<EOR>",
        help="token differentiating between responses--needed for parsing",
    )
    parser.add_argument(
        "--outputs_path", type=str, required=False, default=None, help="directory of predictions.txt or references.txt"
    )
    parser.add_argument(
        "--retrieve_dataset", help="retrieve inputs and references for specified dataset", action="store_true"
    )
    parser.add_argument(
        "--eval_mode", help="run evaluation on provided predictions.txt for specified task", action="store_true"
    )
    parser.add_argument("--oga_references", help="get OGA model references for pretrained model", action="store_true")
    parser.add_argument("--inputs_path", type=str, help="directory of inputs.txt", default=None)
    parser.add_argument("--model_name", type=str, required=False, default="modelName", help="offline_mode model name")
    parser.add_argument(
        "--case",
        type=str,
        help="Offline mode case to run",
        default="default",
        choices=["default", "psu_prompt", "psu_prompt_eos_stop"],
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser_with_modelopt_args()
    args = parse_eval_args(parser)
    model_args = utils.simple_parse_args_string(args.model_args)

    if args.trust_remote_code:
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        model_args["trust_remote_code"] = True
        args.trust_remote_code = None

    args.model_args = model_args

    if args.metrics_output_dir is not None:
        from datetime import datetime

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.tasks is not None:
            vars(args)["output_path"] = os.path.join(args.metrics_output_dir, current_time)

    # functionality for end-to-end model predictions and evaluation
    if args.mode == "standard":
        quark_metrics = {}
        results = {}

        # PPL
        if args.ppl or args.use_ppl_eval_for_kv_cache or args.rouge or args.meteor or args.mlperf_rouge:
            # load the model
            model_obj, tokenizer, testenc = prepare_model(
                model_args["pretrained"],
                args.model_reload,
                args.import_file_format,
                args.import_model_dir,
                args.seq_len,
                args.device,
                args.multi_gpu,
                args.ppl,
            )

            if args.ppl or args.use_ppl_eval_for_kv_cache:
                # eval model ppl
                if args.ppl:
                    ppl = ppl_eval(model_obj, testenc, args.device, args.import_file_format)
                    print(f"\n[INFO] Perplexity: {ppl.item()}")
                    quark_metrics["Perplexity"] = ppl.item()
                elif args.use_ppl_eval_for_kv_cache:
                    ppl_eval_for_kv_cache(
                        model_obj,
                        testenc,
                        args.use_ppl_eval_for_kv_cache_context_size,
                        args.use_ppl_eval_for_kv_cache_sample_size,
                        args.use_ppl_eval_for_kv_cache_patch_size,
                        args.device,
                    )

            # ROUGE and METEOR
            if args.rouge or args.meteor:
                requested_datasets = args.datasets.split(",")
                for dataset in requested_datasets:
                    generations = rouge_meteor_generations(args, dataset, model_obj, tokenizer)
                    if args.rouge:
                        rouge_scores = rouge_eval(dataset, generations)
                        print(f"\n[INFO] {dataset} ROUGE: {rouge_scores}")
                        quark_metrics[f"{dataset} ROUGE"] = rouge_scores
                    if args.meteor:
                        meteor_scores = meteor_eval(dataset, generations)
                        print(f"\n[INFO] {dataset} METEOR: {meteor_scores}")
                        quark_metrics[f"{dataset} METEOR"] = meteor_scores

            # eval model mlperf_rouge
            if args.mlperf_rouge:
                mlperf_rouge_eval(args, model_obj, model_args["pretrained"], args.device, args.batch_size)

        # LM EVAL HARNESS TASKS
        if args.tasks is not None:
            if args.model == "hf":
                model_args.update(
                    {
                        "model_reload": args.model_reload,
                        "import_file_format": args.import_file_format,
                        "import_model_dir": args.import_model_dir,
                        "device": args.device,
                        "parallelize": args.multi_gpu,
                    }
                )
                model_obj = custom_lm_eval_harness.LMEvalModelWrapper(**model_args)
                args.model = model_obj
                cli_evaluate(args)
            else:
                lm_eval.api.registry.get_model(args.model).create_from_arg_obj = classmethod(create_from_arg_obj)
                model_args.update(
                    {
                        "model_reload": args.model_reload,
                        "import_file_format": args.import_file_format,
                        "import_model_dir": args.import_model_dir,
                        "device": args.device,
                        "parallelize": args.multi_gpu,
                    }
                )
                cli_evaluate(args)

    # decoupled predictions and evaluations
    elif args.mode == "offline":
        if "," in args.tasks:
            raise ValueError("Please provide only 1 task")

        task_name = args.tasks
        if args.retrieve_dataset:
            print("Retrieving dataset...")
            get_dataset(task_name, num_fewshot=args.num_fewshot, limit=args.limit)

        if args.eval_mode:
            print("In eval mode...")
            if args.outputs_path is None:
                raise TypeError("Please specify a path to references.txt or predictions.txt ")

            model_args.update({"outputs_path": args.outputs_path, "eor": args.eor})
            eval_LM = custom_lm_eval_harness.LMEvalModelGenWrapper(**model_args)
            args.model = eval_LM
            cli_evaluate(args)

        if args.oga_references:
            print("Saving OGA references...")
            # parse the inputs.json file
            with open(str(args.inputs_path)) as f:
                inputs = json.load(f)
            filename = f"{args.model_name}_{args.tasks}_limit-{args.limit}_{args.case}.txt"
            if args.import_file_format == "onnx_format":
                references = oga_generation(args, inputs, args.import_model_dir, filename)

    # save evaluation results
    if args.metrics_output_dir is not None:
        save_evaluation_results(args)
