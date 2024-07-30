# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json

from olive.workflows import run as olive_run


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Llama2 optimization using Generative AI")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name, currently only supports llama2 7B/13B",
    )
    parser.add_argument(
        "--metadata-only", action="store_true", required=False, help="Whether to use gpu for optimization."
    )
    parser.add_argument("--tempdir", type=str, help="Root directory for tempfile directories and files", required=False)

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)
    model_name = args.model_name

    input_template = "llama2_model_builder_template.json"
    with open(input_template) as f:
        template_json_str = f.read()

    # update model name
    template_json_str = template_json_str.replace("<model_name_placeholder>", model_name)
    template_json = json.loads(template_json_str)

    # add pass flows
    if args.metadata_only:
        template_json["pass_flows"] = [["conversion", "metadata"]]
    else:
        template_json["pass_flows"] = [["builder", "perf_tuning"]]
    template_json["output_dir"] = f"models/{model_name}"

    # dump config
    output_template = "llama2_model_builder.json"
    with open(output_template, "w") as f:
        json.dump(template_json, f, indent=4)

    olive_run(template_json, tempdir=args.tempdir)  # pylint: disable=not-callable


if __name__ == "__main__":
    main()
