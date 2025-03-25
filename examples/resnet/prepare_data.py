# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import tarfile
import urllib.request
from pathlib import Path

# ruff: noqa: PLW2901, T201


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    return parser.parse_args()


def get_directories():
    current_dir = Path(__file__).resolve().parent

    # data directory for resnet sample
    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def main():
    data_dir = get_directories()

    data_download_path = data_dir / "cifar-10-python.tar.gz"
    urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", data_download_path)
    with tarfile.open(data_download_path) as tar:
        tar.extractall(data_dir)  # lgtm

if __name__ == "__main__":
    main()
