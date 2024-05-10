# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import copy
import json
import os
import sys
import time

import onnxruntime_genai as og
from packaging import version

# ruff: noqa


def _main():
    parser = argparse.ArgumentParser(description="End-to-end token generation loop example for gen-ai model")
    parser.add_argument(
        "model", type=str, help="Onnx model folder path (must contain genai_config.json and model.onnx)"
    )
    parser.add_argument("-pr", "--prompts", nargs="*", required=False, help="Input prompts to generate tokens from")
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        help="This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. Note that diversity_penalty is only effective if group beam search is enabled.",
    )
    parser.add_argument(
        "--do_sample", type=bool, help="Whether or not to use sampling ; use greedy decoding otherwise."
    )
    parser.add_argument(
        "--early_stopping",
        type=bool,
        help="Whether to stop the beam search when at least num_beams sentences are finished per batch or not.",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        help="Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.",
    )
    parser.add_argument("--max_length", type=int, help="Max number of tokens to generate for each prompt")
    parser.add_argument("--min_length", type=int, help="The minimum length of tokens to be generated for each prompt.")
    parser.add_argument(
        "--no_repeat_ngram_size", type=int, help=" If set to int > 0, all ngrams of that size can only occur once."
    )
    parser.add_argument("--num_beams", type=int, help="Number of beams for beam search. 1 means no beam search.")
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        help="The number of independently computed returned sequences for each element in the batch.",
    )
    parser.add_argument(
        "--past_present_share_buffer",
        type=bool,
        help="The past/present kv tensors are shared and allocated once to max_length (cuda only)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        help="Repetition penalty to sample with. The parameter for repetition penalty. 1.0 means no penalty.",
    )
    parser.add_argument("--temperature", type=float, help="The value used to module the next token probabilities.")
    parser.add_argument(
        "--top_k", type=int, help="The number of highest probability vocabulary tokens to keep for top-k-filtering."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
    )
    args = parser.parse_args()

    print("Loading model...")
    model = og.Model(f"{args.model}")

    print("Creating tokenizer ...")
    tokenizer = og.Tokenizer(model)

    print("Loading genai_config.json ...")
    genai_config_filepath = os.path.join(args.model, "genai_config.json")
    with open(genai_config_filepath) as strm:
        genai_config = json.load(strm)

    print("Evaluating generator params and search options ...")
    params = og.GeneratorParams(model)

    search_options: dict = copy.deepcopy(genai_config["search"])
    search_options.update(
        {
            name: getattr(args, name)
            for name in [
                "diversity_penalty",
                "do_sample",
                "early_stopping",
                "length_penalty",
                "max_length",
                "min_length",
                "no_repeat_ngram_size",
                "num_beams",
                "num_return_sequences",
                "past_present_share_buffer",
                "repetition_penalty",
                "temperature",
                "top_k",
                "top_p",
            ]
            if name in args and getattr(args, name)
        }
    )
    if version.parse(og.__version__) > version.parse("0.1.0"):
        params.set_search_options(**search_options)
    else:
        params.set_generator_params(search_options)

    print("Encoding prompts ...")
    if args.prompts is not None:
        prompts = args.prompts
    else:
        prompts = ["I like walking my cute dog", "What is the best restaurant in town?", "Hello, how are you today?"]
    params.input_ids = tokenizer.encode_batch(prompts)

    print("Generating tokens ...")
    start_time = time.time()
    output_tokens = model.generate(params)
    run_time = time.time() - start_time

    print("Decoding generated tokens ...")
    print()
    output_token_count = 0
    for i, prompt in enumerate(prompts):
        print(f"Prompt #{i:02d}: {prompt}")
        print()
        print(tokenizer.decode(output_tokens[i]))
        print()

        output_token_count += len(output_tokens[i])

    print()
    print(f"Tokens: {output_token_count}, Time: {run_time:.2f}, Tokens per second: {output_token_count / run_time:.2f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(_main())
