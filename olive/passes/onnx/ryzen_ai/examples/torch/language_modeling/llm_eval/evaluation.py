#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import torch
import json
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Any, Union, List, Optional
import datetime
import math
import argparse
import evaluate
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import numpy as np
import nltk
import time


def eval_model(args: argparse.Namespace, model: nn.Module, main_device, save_metrics_to_csv: bool = False, output_dir: Union[Path, str] = 'metrics_output_dir', multimodal: bool = False):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True,)

    if args.num_eval_data != -1 and not args.use_mlperf_rouge:
        testenc = tokenizer("\n\n".join(testdata['text'][:args.num_eval_data]), return_tensors='pt')
    else:
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    metrics = []
    main_device = model.device

    # eval kv_cache ppl
    if args.use_ppl_eval_for_kv_cache:
        ppl_eval_for_kv_cache(model, testenc, args.ppl_eval_for_kv_cache_context_size,
                              args.ppl_eval_for_kv_cache_sample_size, args.ppl_eval_for_kv_cache_patch_size,
                              main_device)
    # eval model ppl
    else:
        ppl = ppl_eval(model, testenc, main_device)
        print("\n[INFO] Perplexity: {}".format(ppl.item()))
        metrics.append(['Perplexity', ppl.cpu().numpy()])
    # eval model mlperf_rouge
    if args.use_mlperf_rouge:
        mlperf_rouge_eval(args, model, args.model_dir, main_device, args.eval_batch_size)

    # eval tasks
    if args.tasks is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        task_eval(model, tokenizer, args.eval_batch_size, args.max_eval_batch_size, args.tasks, args.num_fewshot, args.apply_chat_template, device=main_device, multimodal=multimodal)  # batch_size: N|auto|auto:1

    # save result into csv
    if save_metrics_to_csv:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        evaluation_metrics_path = Path(output_dir, 'evaluation_metrics.csv').as_posix()
        with open(evaluation_metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for metric in metrics:
                writer.writerow(metric)
        print(f"[INFO] Saved evaluation_metrics to {evaluation_metrics_path}.")


@torch.no_grad()
def ppl_eval(model: nn.Module, testenc: AutoTokenizer, dev: str, file_format: str = "hf_format") -> None:
    if file_format != "onnx_format":
        model.eval()
    # Set sequence length as 2048 for wikitext dataset evaluation
    seqlen_for_eval = 2048
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen_for_eval

    testenc = testenc.to(dev)
    nlls = []

    if file_format == "onnx_format":
        import onnxruntime_genai as og
        params = og.GeneratorParams(model)
        params.try_graph_capture_with_max_batch_size(1)
        search_options = {}
        search_options["max_length"] = seqlen_for_eval + 1
        params.set_search_options(**search_options)

    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen_for_eval):((i + 1) * seqlen_for_eval)].to(dev)
        if file_format == "onnx_format":
            # onnx model logits using oga
            params.input_ids = batch.cpu().numpy()
            generator = og.Generator(model, params)
            generator.compute_logits()
            shift_logits = torch.tensor(generator.get_output("logits")[0][:-1]).to(dev)
        else:
            lm_logits = model(batch)['logits']
            shift_logits = lm_logits[:, :-1, :].contiguous()

        shift_labels = testenc[:, (i * seqlen_for_eval):((i + 1) * seqlen_for_eval)][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen_for_eval
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen_for_eval))

    return ppl


@torch.no_grad()
def task_eval(model: nn.Module, tokenizer: AutoTokenizer, batch_size: int = 1, max_batch_size: Optional[int] = None, tasks: Optional[List[str]] = None, num_fewshot: Optional[int] = None, apply_chat_template: bool = False, device: Optional[str] = 'cuda', multimodal: bool = False, output_path: Optional[str] = None) -> None:
    import sys
    from typing import Optional, Type
    from lm_eval.__main__ import cli_evaluate, setup_parser
    from lm_eval.api.model import T
    from lm_eval.models.huggingface import HFLM

    def create_from_arg_obj(cls: Type[T], arg_dict: dict, additional_config: Optional[dict] = None) -> T:
        model_obj = arg_dict.pop('pretrained')
        tokenizer = arg_dict.pop('tokenizer')
        model_obj = cls(pretrained=model_obj, tokenizer=tokenizer)
        return model_obj

    HFLM.create_from_arg_obj = classmethod(create_from_arg_obj)
    parser = setup_parser()
    parser.set_defaults(
                        model='hf-multimodal' if multimodal else 'hf',
                        model_args={'pretrained': model, 'tokenizer': tokenizer},
                        batch_size=batch_size,
                        max_batch_size=max_batch_size,
                        tasks=tasks,
                        num_fewshot=num_fewshot,
                        apply_chat_template=apply_chat_template,
                        output_path=output_path
                       )
    temp_args = sys.argv
    sys.argv = [sys.argv[0]]
    args = parser.parse_args()
    sys.argv = temp_args

    cli_evaluate(args)


# use_ppl_eval_for_kv_cache
def update_model_kwargs_for_generation(
    outputs,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
) -> Dict[str, Any]:
    """
    Update model arguments for next generation.
    """
    # update past_key_values
    for key in ["past_key_values", "mems", "past_buckets_states"]:
        if key in outputs:
            model_kwargs["past_key_values"] = outputs[key]

    if "state" in outputs:
        model_kwargs["state"] = outputs.state

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

    return model_kwargs


@torch.no_grad()
def get_cumulative_logprob(model: AutoModelForCausalLM, input_tokens: torch.Tensor, future_context: list,
                           dev: str) -> float:
    """
    Calculate the cumulative logprob of the given future context.
    Parameters:
        model : AutoModelForCausalLM
        input_tokens : torch.Tensor
        future_context : list
            A list of token IDs, typically adjacent to the input token. Used for calculating cumulative log probabilities.
        dev : str
            The compute device.
    Returns:
        float
            The cumulative log probabilities of input_tokens and future_context.
    """
    # init
    input_tokens = input_tokens.to(dev)
    model_kwargs = {
        'use_cache': True,
        'attention_mask': torch.ones(input_tokens.shape, dtype=int).to(dev),
    }

    cumulative_logpro = 0
    for next_token in future_context:
        model_inputs = model.prepare_inputs_for_generation(input_tokens, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

        # get the output of the last decoder layer
        hidden_states = outputs.hidden_states[-1].clone().detach()

        # get logits
        selected_token_indices = torch.Tensor([(model_inputs['input_ids'].numel() - 1)]).to(int).to(dev)
        logits = model.lm_head(hidden_states[0, :, :].index_select(0, selected_token_indices))
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # get the specific token's logprob and accumulate
        cumulative_logpro += logprobs[0, next_token]

        # update for next step
        input_tokens = torch.cat((input_tokens, torch.tensor([
            [
                next_token,
            ],
        ]).to(dev)), dim=-1)
        model_kwargs = update_model_kwargs_for_generation(outputs,
                                                          model_kwargs,
                                                          is_encoder_decoder=model.config.is_encoder_decoder)
    return cumulative_logpro.item()


@torch.no_grad()
def ppl_eval_for_kv_cache(model: nn.Module, testenc: torch.Tensor, context_size: int, sample_size: int, patch_size: int,
                          dev: str) -> None:
    """
    A perplexity-computing test for the KV cache system.
    Parameters:
        model : nn.Module
        testenc : torch.Tensor
            The input token IDs.
        context_size : int
            The size of the context used for generation.
        sample_size : int
            The number of the output generated by the given context size. This his variable also \
            defines the size of the individual patch.
        patch_size : int
            The size of patches, if specified, will determine the number of patches. If not, \
            the number of patches will be determined by sample_size and the number of input tokens.
        dev : str
    Returns:
        None
    """
    print(f"Initializing @ {datetime.datetime.now()}")

    # Prepare parameters
    my_enc = testenc.input_ids
    n_samples = sample_size
    n_patches = math.ceil((my_enc.numel() - context_size - 1) / n_samples)
    if patch_size is not None:
        n_patches = patch_size

    ppl = 0.0
    num_tokens_generated = 0
    starting_time = datetime.datetime.now()
    print(f'Starting generation @ {starting_time} '
          f'will try to process {n_patches} patch(es), '
          f'generating {n_samples} tokens in each patch '
          f'from the initial context of {context_size} tokens.')
    for idx in range(n_patches):
        context = my_enc[:, idx * n_samples:idx * n_samples + context_size]
        upper_boundary = min((idx + 1) * n_samples + context_size, my_enc.numel())
        future_context = my_enc[0, idx * n_samples + context_size:upper_boundary].tolist()

        logprobs = get_cumulative_logprob(model=model, input_tokens=context, future_context=future_context, dev=dev)

        ppl -= logprobs
        num_tokens_generated += len(future_context)

        print(f'Iteration {idx + 1} of {n_patches} Intermediate '
              'Estimates:\n'
              f'\tCross-entropy_intermediate={ppl / num_tokens_generated}\n'
              f'\tPerplexity_intermediate={math.exp(ppl / num_tokens_generated)}')

    ending_time = datetime.datetime.now()
    print(f'Done @ {ending_time} after processing for '
          f'{ending_time - starting_time} generated {num_tokens_generated} tokens.')
    print(f'Integral Cross-Entropy={ppl} Average Cross-Entropy='
          f'{ppl / num_tokens_generated} PPL={math.exp(ppl / num_tokens_generated)}')


def rouge_meteor_generations(args: argparse.Namespace, dataset: str, model: nn.Module, tokenizer: AutoTokenizer) -> Dict[str, str]:

    def get_generation(args, sample, input_field):
        inputs = tokenizer(
            sample[input_field],
            return_tensors="pt",
            max_length=args.seq_len,
            truncation=True,
        )

        if args.import_file_format == "onnx_format" and "chatglm3-6b" in args.model_args['pretrained'].lower():
            with open(args.import_model_dir + "/genai_config.json") as f:
                genai_config = json.load(f)
            config = AutoConfig.from_pretrained(args.model_args["pretrained"], trust_remote_code=True)
            config.num_key_value_heads = genai_config["model"]["decoder"]["num_key_value_heads"]
            past_key_values = [
                (
                    torch.zeros((args.batch_size, config.num_key_value_heads, inputs["input_ids"].shape[1], config.hidden_size // config.num_attention_heads)),
                    torch.zeros((args.batch_size, config.num_key_value_heads, inputs["input_ids"].shape[1], config.hidden_size // config.num_attention_heads))
                )
                for i in range(config.num_layers)
            ]
            model.past_key_values = past_key_values

            output_ids = model.generate(
                inputs["input_ids"].to(args.device),
                max_new_tokens=args.max_new_toks,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=past_key_values
            )
        else:
            output_ids = model.generate(
                inputs["input_ids"].to(args.device),
                max_new_tokens=args.max_new_toks,
                pad_token_id=tokenizer.eos_token_id,
            )

        sample["predicted"] = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        return sample

    def get_dataset_sample(dataset):
        if args.num_eval_data != -1:
            return Dataset.from_dict(dataset[:args.num_eval_data])
        else:
            return dataset

    # 1. load dataset, 2. retrieve a subsample of the dataset, if provided (for quicker evals) 3. generate preds
    if(dataset == "xsum"):
        dataset = load_dataset("xsum", split="test", trust_remote_code=True)
        dataset = get_dataset_sample(dataset)
        print("Generating predictions for XSUM Dataset")
        dataset_generations = dataset.map(lambda x: get_generation(args, x, input_field="document"))
    elif(dataset == "cnn_dm"):
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
        dataset = get_dataset_sample(dataset)
        print("Generating predictions for CNN_DM Dataset")
        dataset_generations = dataset.map(lambda x: get_generation(args, x, input_field="article"))
    elif(dataset == "samsum"):
        dataset = load_dataset("samsum", split="test", trust_remote_code=True)
        dataset = get_dataset_sample(dataset)
        print("Generating predictions for SAMSUM Dataset")
        dataset_generations = dataset.map(lambda x: get_generation(args, x, input_field="dialogue"))

    return dataset_generations


def rouge_eval(dataset, generations) -> Dict[str, float]:

    rouge = evaluate.load("rouge")
    if(dataset == "xsum"):
        print("Calculating Rouge on XSUM Dataset")
        rouge_preds = rouge.compute(
            predictions=generations["predicted"], references=generations["summary"]
        )
    if(dataset == "cnn_dm"):
        print("Calculating Rouge on CNN DM Dataset")
        rouge_preds = rouge.compute(
            predictions=generations["predicted"], references=generations["highlights"]
        )
    if(dataset == "samsum"):
        print("Calculating Rouge on SAMSUM Dataset")
        rouge_preds = rouge.compute(
            predictions=generations["predicted"], references=generations["summary"]
        )

    for k, val in rouge_preds.items():
        rouge_preds[k] = float(val)
    return rouge_preds


def meteor_eval(dataset, generations) -> Dict[str, float]:

    meteor = evaluate.load("meteor")
    if(dataset == "xsum"):
        print("Calculating Rouge on XSUM Dataset")
        meteor_preds = meteor.compute(
            predictions=generations["predicted"], references=generations["summary"]
        )
    if(dataset == "cnn_dm"):
        print("Calculating Rouge on CNN DM Dataset")
        meteor_preds = meteor.compute(
            predictions=generations["predicted"], references=generations["highlights"]
        )
    if(dataset == "samsum"):
        print("Calculating Rouge on SAMSUM Dataset")
        meteor_preds = meteor.compute(
            predictions=generations["predicted"], references=generations["summary"]
        )

    for k, val in meteor_preds.items():
        meteor_preds[k] = float(val)

    return meteor_preds


def calculate_rouge_score(model_outputs, ref_outputs) -> Dict[str, float]:
    metric = evaluate.load("rouge")

    m_preds = [pred.strip() for pred in model_outputs]
    m_targets = [target.strip() for target in ref_outputs]

    # rougeLSum expects newline after each sentence
    m_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in m_preds]
    m_targets = ["\n".join(nltk.sent_tokenize(target)) for target in m_targets]
    m_result = metric.compute(
        predictions=m_preds, references=m_targets, use_stemmer=True, use_aggregator=False
    )
    m_rouge_result = {k: round(np.mean(v) * 100, 4) for k, v in m_result.items()}

    return m_rouge_result


def evaluate_openorca(df: pd.DataFrame, result_keys: dict) -> str:
    print("Evaluating OpenOrca score...")
    gen_output = df[f"{result_keys['result']}"].tolist()
    gt_output = df.output.tolist()
    score = calculate_rouge_score(gen_output, gt_output)
    gen_token_len = df[result_keys['length']].tolist()
    gen_token_per_sample = sum(gen_token_len) / len(gen_token_len)
    print(f"OpenOrca score: {score}, gen_token_per_sample: {gen_token_per_sample}")
    return score


@torch.no_grad()
def mlperf_rouge_infer(args: argparse.Namespace, model: nn.Module, model_dir: str, main_device: str, batch_size: str) -> pd.DataFrame:
    G_MAX_OUTPUT_SEQLEN = 1024

    df = pd.read_pickle(args.eval_data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left", trust_remote_code=True,)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # gen parameter. We stop at 1024
    gen_kwargs = {
        "max_new_tokens": G_MAX_OUTPUT_SEQLEN,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
    }

    # Start inference
    BS = int(batch_size)
    bidx = 0
    model.eval()

    input_tokens = []
    input_tokens_lens = []
    output_tokens = []
    output_tokens_lens = []
    output_texts = []

    tic = time.time()
    n_samples = min(len(df), args.num_eval_data)
    for idx in range(0, n_samples, BS):
        tac = time.time()
        print(f"Processing {idx}/{n_samples}, time: {tac - tic}s")
        sidx = idx
        eidx = min(sidx + BS, n_samples)

        # We use batch_encode_plus for batch inference.
        batch_texts = df['input'][sidx:eidx].tolist()
        batch_ids = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", padding=True)
        tok_input_length = batch_ids['attention_mask'].sum(
            axis=1).to(torch.int32).tolist()
        input_tokens_lens += tok_input_length
        tok_input_id = batch_ids['input_ids'].to(torch.int32).tolist()
        # Remove eos from the input id
        tok_input_id = [[element for element in sublist if element !=
                        tokenizer.eos_token_id] for sublist in tok_input_id]
        input_tokens += tok_input_id

        batch_ids = batch_ids.to(main_device)
        _, length = batch_ids.input_ids.shape
        outputs = model.generate(**batch_ids, num_return_sequences=1, **gen_kwargs)

        output_ids = outputs[:, length:].cpu().tolist()
        output_tokens += output_ids

        # Filter out EOS
        id_filtered = [[num for num in sublist if num !=
                        tokenizer.eos_token_id] for sublist in output_ids]
        output_id_len = [len(out) for out in id_filtered]
        output_tokens_lens += output_id_len

        # Detokenizer
        output_msgs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        output_texts += output_msgs
        bidx += 1

    # Assemble the output
    output_df = df[:len(output_tokens)].copy()
    output_df["ref_output"] = output_texts
    output_df["tok_ref_output"] = output_tokens
    output_df["tok_ref_output_length"] = output_tokens_lens

    return output_df


def mlperf_rouge_eval(args: argparse.Namespace, model: nn.Module, model_dir: str, main_device: str, batch_size: str) -> None:
    nltk.download('punkt_tab')
    result_keys = {
        "result": "ref_output",
        "length": "tok_ref_output_length"
    }
    df = mlperf_rouge_infer(args, model, model_dir, main_device, batch_size)
    evaluate_openorca(df, result_keys)
