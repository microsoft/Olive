# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from olive.workflows.snpe import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SNPE workflow: Convert and Quantize")
    parser.add_argument("--model", type=str, help="Path to the model", required=True)
    parser.add_argument("--config", type=str, help="Either the path to json config file", required=True)
    parser.add_argument("--data", type=str, help="Path to the data", required=True)
    parser.add_argument(
        "--input_list_file",
        type=str,
        help="Name of the input list file. Optional if 'input_list.txt'",
        required=False,
        default="input_list.txt",
    )

    args = parser.parse_args()

    evaluate(**vars(args))
