# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path


def get_directories():
    current_dir = Path(__file__).resolve().parent

    # models directory for resnet sample
    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # data directory for resnet sample
    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # cache directory for resnet sample
    cache_dir = current_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, models_dir, data_dir, cache_dir
