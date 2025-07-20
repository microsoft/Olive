#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
import logging
import os
import random
from datasets import load_dataset
import sys
sys.path.append(os.path.realpath('../../'))
import onnxruntime_genai as oga

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

WEIGHTS_NAME = 'pytorch_model.bin'
from transformers import (
    AutoTokenizer
)

logger = logging.getLogger(__name__)

# MODEL_CLASSES = {
#     "llama2": (LlamaConfig, LlamaForCausalLM, LlamaTokenizer),
# }


class TextDataset(Dataset):

    def __init__(self, tokenizer, args, block_size=512):

        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = ''
        for i in testdata:
            text += i['text']
        self.examples = []
        tokenized_text = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(text))
        for i in range(0,
                       len(tokenized_text) - block_size + 1,
                       block_size):  # Truncate in block of block_size
            self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i:i + block_size]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(args, tokenizer, evaluate=True):
    dataset = TextDataset(
        tokenizer,
        args,
        block_size=args.block_size,
    )
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def evaluate_onnx(args, model, tokenizer, prefix=""):
    from torch.nn import CrossEntropyLoss
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_data = ''
    for i in testdata:
        test_data += i['text']

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.per_gpu_eval_batch_size)
    sampler = eval_dataloader.sampler


    search_options = {
        'min_length': 1,
        'max_length': args.block_size + 1,
    }
    params = oga.GeneratorParams(model)
    params.set_search_options(**search_options)


    logger.info("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0

    count = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        with torch.no_grad():
            params.input_ids = inputs
            generator = oga.Generator(model, params)
            generator.compute_logits()

            # Shift so that tokens < n predict n
            lm_logits = generator.get_output("logits")[0]
            shift_logits = torch.from_numpy(lm_logits[..., :-1, :]).contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))

            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
        count += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument(
        "--block_size",
        default=1024,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    parser.add_argument("--per_gpu_eval_batch_size",
                        default=1,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--do_onnx_eval",
                        action="store_true",
                        help="evaluate onnx model")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=False,
        cache_dir=None,
    )
    tokenizer.add_bos_token = False

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, "Pad token cannot be set!"

    if args.block_size <= 0:
        args.block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model

    # Evaluation
    results = {}

    if args.do_onnx_eval:
        logger.info("Evaluate the following onnx model: %s",
                    args.model_name_or_path)
        global_step = ""
        # Load model
        prefix = 'onnx'
        model = oga.Model(args.model_name_or_path)

        result = evaluate_onnx(args, model, tokenizer, prefix=prefix)
        result = dict(
            (k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)


if __name__ == "__main__":
    main()
