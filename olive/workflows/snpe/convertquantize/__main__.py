# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from olive.workflows.snpe import convertquantize

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
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory. Optional if same as model directory",
        required=False,
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="Name of the output model (without extension). Optional if same as model name",
        required=False,
    )

    args = parser.parse_args()

    convertquantize(**vars(args))
