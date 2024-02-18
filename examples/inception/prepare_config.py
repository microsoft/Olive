# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
from pathlib import Path


def resolve_windows_config():
    # pylint: disable=redefined-outer-name

    with Path("inception_config.json").open("r") as f:
        snpe_windows_config = json.load(f)

    del snpe_windows_config["passes"]["snpe_quantization"]
    with Path("inception_config.json").open("w") as f:
        json.dump(snpe_windows_config, f, indent=4)


if __name__ == "__main__":
    if platform.system() == "Windows":
        resolve_windows_config()
