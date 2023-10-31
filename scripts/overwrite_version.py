# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Overwrite package version in __init__.py")
    parser.add_argument("--version", type=str, required=True, help="Version to overwrite with")
    return parser.parse_args()


def main():
    args = get_args()
    version = args.version

    init_path = Path(__file__).parent.parent.resolve() / "olive" / "__init__.py"
    with open(init_path) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("__version__"):
                lines[i] = f'__version__ = "{version}"\n'
                break

    with open(init_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
