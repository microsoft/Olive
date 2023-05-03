# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import pprint
import sys

import distribute_experts


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-size",
        dest="world_size",
        type=int,
        default=2,
        help="Number of GPU nodes to distribute the model for. Must be greater than 1.",
    )
    parser.add_argument(
        "--input-filepath", dest="input_filepath", required=True, type=str, help="Path to model file to load"
    )
    parser.add_argument(
        "--output-dirpath", dest="output_dirpath", required=True, type=str, help="Save output to this directory"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debugging")
    args = parser.parse_args()

    output_filepaths = distribute_experts.run(args.world_size, args.input_filepath, args.output_dirpath, args.debug)
    pprint.pprint(output_filepaths)

    return 0


if __name__ == "__main__":
    sys.exit(_main())
