# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import shutil
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dev_mount_path", type=str, help="path to dev mount")

    args, _ = parser.parse_known_args()

    for member in Path(args.dev_mount_path).iterdir():
        if member.is_dir():
            shutil.rmtree(member)
        else:
            member.unlink()
