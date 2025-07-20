#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import argparse
import json
from quark.torch.pruning.config import Config, OSSCARConfig, BlockwiseTuningConfig
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_utils.data_preparation import get_calib_dataloader
from llm_eval.evaluation import eval_model
from llm_utils.model_preparation import get_model, get_model_type, get_tokenizer, set_seed, save_model


def get_config(args: argparse.Namespace, model_type: str) -> Config:

    if args.pruning_algo == 'osscar':
        algo_config_file = 'models/' + model_type + '/osscar_config.json'
        with open(algo_config_file, 'r') as file:
            algo_config_info = json.load(file)
        pruning_algo_config = OSSCARConfig.from_dict(algo_config_info)

        if args.blockwise_tuning:
            blockwise_tuning_config_file = 'models/' + model_type + '/pruning_blockwise_tuning_config.json'
            with open(blockwise_tuning_config_file, 'r') as file:
                blockwise_tuning_config_info = json.load(file)
            blockwise_tuning_config = BlockwiseTuningConfig.from_dict(blockwise_tuning_config_info)
        else:
            blockwise_tuning_config = None
    else:
        pruning_algo_config = None

    pruning_config = Config(
        algo_config = pruning_algo_config,
        blockwise_tuning_config = blockwise_tuning_config
    )

    if args.pruning_algo is not None and model_type is None:
        raise ValueError(f"{args.pruning_algo} is not tested for current model")

    return pruning_config


def main(args: argparse.Namespace) -> None:
    # 1. Define original model
    print("\n[INFO]: Loading model ...")
    set_seed(args.seed)
    model, model_dtype = get_model(args.model_dir, args.data_type, args.device, args.multi_gpu)
    model_type = get_model_type(model)
    tokenizer = get_tokenizer(args.model_dir, max_seq_len=args.seq_len, model_type=model_type)

    from quark.shares.utils.log import ScreenLogger
    logger = ScreenLogger(__name__)

    # 3. Define calibration dataloader.
    print("\n[INFO]: Loading dataset ...")
    # When the model is small, accelerate will place it on the last device
    main_device = model.device if args.multi_gpu else args.device
    calib_dataloader = get_calib_dataloader(dataset_name=args.dataset,
                                            tokenizer=tokenizer,
                                            batch_size=args.batch_size,
                                            num_calib_data=args.num_calib_data,
                                            seqlen=args.seq_len,
                                            device=main_device)
    # 3. Pruning
    if not args.skip_pruning:
        # 3-1. Set pruning configuration
        pruning_config = get_config(args, model_type)

        # 3-2. In-place replacement of model modules with pruning versions.
        from quark.torch import ModelPruner
        model_pruner = ModelPruner(pruning_config)
        model = model_pruner.pruning_model(model, calib_dataloader)

        # 4. Export
        if args.save_pruned_model:
            print("\n[INFO]: Save pruned model ...")
            save_model(model, None, args.save_dir)

    # 5. (Optional) Model Evaluation
    if not args.skip_evaluation:
        print("\n[INFO]: Evaluating ...")
        eval_model(args, model, main_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        help="Specify where the HuggingFace model is. This example support Llama, OPT models",
                        required=True)
    parser.add_argument("--dataset",
                        help="Dataset for calibration",
                        default="pileval",
                        choices=[
                            "pileval", "wikitext", "pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark",
                            "HuggingFaceH4/ultrachat_200k"
                        ])
    parser.add_argument("--device", help="Device for running the pruner", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument("--data_type",
                        help="Datatype of the model",
                        default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--seq_len", type=int, help="Sequence length of data", default=512)
    parser.add_argument("--skip_pruning", action='store_true')
    parser.add_argument("--skip_evaluation", action='store_true')

    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument(
        "--eval_batch_size",
        type=str,
        default=8,
        metavar="auto|auto:N|N",
        help="Batch size for evaluation. Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1."
    )
    parser.add_argument("--max_eval_batch_size",
                        type=int,
                        default=None,
                        metavar="N",
                        help="Maximal batch size to try with --batch_size auto.")
    parser.add_argument("--num_calib_data", help="Number of samples for calibration.", type=int, default=512)
    parser.add_argument(
        "--num_eval_data",
        help=
        "Number of samples for evaluation. The default value is -1, which means the entire dataset is used for evaluation.",
        type=int,
        default=-1)
    parser.add_argument("--num_fewshot",
                        type=int,
                        default=None,
                        metavar="N",
                        help="Number of examples in few-shot context")
    parser.add_argument("--tasks",
                        default=None,
                        type=str,
                        metavar="task1,task2",
                        help="Comma-separated list of task names or task groupings to evaluate on.")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt."
    )

    parser.add_argument("--pruning_algo", help="Pruning Algorithms.", default="osscar", choices=["osscar", None])

    parser.add_argument("--blockwise_tuning", help="Providing `--blockwise_tuning` will apply blockwise_tuning after pruning.", action="store_true")

    parser.add_argument("--torch_compile", help="Model torch compile", action="store_true")

    parser.add_argument("--save_pruned_model", help="pruned model save", action='store_true')
    parser.add_argument(
        "--save_dir",
        help="Directory to save model parameters as safetensors or pth, in the case when --save_pruned_model is used.",
        default="model_params")

    parser.add_argument("--use_ppl_eval_for_kv_cache", action="store_true")
    parser.add_argument("--ppl_eval_for_kv_cache_context_size",
                        type=int,
                        help="Context size used in PPL evaluation for KV cache.",
                        default=1024)
    parser.add_argument("--ppl_eval_for_kv_cache_sample_size",
                        type=int,
                        help="Sample size used in PPL evaluation for KV cache.",
                        default=512)
    parser.add_argument("--ppl_eval_for_kv_cache_patch_size",
                        type=int,
                        help="Patch size used in PPL evaluation for KV cache.",
                        default=None)
    parser.add_argument("--seed",
                        type=int,
                        help="random seed.",
                        default=42)

    args = parser.parse_args()

    main(args)
