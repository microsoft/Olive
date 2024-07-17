# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Optional

from olive.cache import OliveCache
from olive.common.hf.mlflow import get_pretrained_name_or_path, is_mlflow_transformers
from olive.common.utils import hardlink_copy_dir, hash_string

logger = logging.getLogger(__name__)


class MLFlowTransformersMixin:
    def get_mlflow_transformers_path(self) -> Optional[str]:
        if not is_mlflow_transformers(self.model_path):
            return None

        model_dir = get_pretrained_name_or_path(self.model_path, "model")
        config_dir = get_pretrained_name_or_path(self.model_path, "config")
        tokenizer_dir = get_pretrained_name_or_path(self.model_path, "tokenizer")

        # some mlflow models only have model directory
        if config_dir == model_dir and tokenizer_dir == model_dir:
            return model_dir

        # some mlflow models have config and tokenizer directories but model directory also
        # contains the same files
        model_dir_contents = set(Path(model_dir).iterdir())
        if (
            set(Path(config_dir).iterdir()) <= model_dir_contents
            and set(Path(tokenizer_dir).iterdir()) <= model_dir_contents
        ):
            return model_dir

        # have to gather all contents into a single directory
        cache = OliveCache.from_cache_env()
        mlflow_transformers_path = cache.dirs.mlflow / hash_string(str(Path(self.model_path).resolve()))
        if (mlflow_transformers_path / "config.json").exists():
            logger.debug("MLFlow model already exists in cache. Reusing it.")
        else:
            for src_dir in [model_dir, config_dir, tokenizer_dir]:
                hardlink_copy_dir(src_dir, mlflow_transformers_path)
        return str(mlflow_transformers_path)
