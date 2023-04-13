# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from olive.workflows import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: Custom Run")
    parser.add_argument("--config", type=str, default="test", help="Path to json config file")
    parser.add_argument("--clean_cache", action="store_true", help="Clean cache before running")
    # engine results
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output_name", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    run(**vars(args))
