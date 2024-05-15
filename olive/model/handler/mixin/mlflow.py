# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path

from olive.common.utils import copy_dir
from olive.constants import ModelFileFormat

logger = logging.getLogger(__name__)


class MLFlowMixin:
    def _get_mlflow_transformers_model_path(self, cache_dir):
        # DO NOT use the model.to_json() to get hash_dict, since it will get hf_config from the model
        # and the operation to get hf_config will use this function to get model_path, which will
        # cause infinite loop
        return str(Path(cache_dir) / "olive_tmp" / "transformers")

    def to_mlflow_transformer_model(self, cache_dir):
        if self.model_file_format != ModelFileFormat.PYTORCH_MLFLOW_MODEL:
            raise ValueError(
                "Model file format is not PyTorch MLFlow model, you cannot get MLFlow transformers model path."
            )
        target_path = self._get_mlflow_transformers_model_path(cache_dir)
        if (Path(target_path) / "config.json").exists():
            logger.debug("Use cached mlflow-transformers models from %s", target_path)
            return target_path
        if (Path(self.model_path) / "data" / "model").exists():
            copy_dir(Path(self.model_path) / "data" / "model", target_path, dirs_exist_ok=True)
            copy_dir(Path(self.model_path) / "data" / "config", target_path, dirs_exist_ok=True)
            copy_dir(Path(self.model_path) / "data" / "tokenizer", target_path, dirs_exist_ok=True)
            return target_path
        return None

    def get_mlflow_model_path_or_name(self, cache_dir):
        # both config.json and model file will be saved under data/model
        mlflow_transformer_model_path = self.to_mlflow_transformer_model(cache_dir)
        if not mlflow_transformer_model_path:
            logger.debug(
                "Model path %s does not exist. Use hf_config.model_name instead.", mlflow_transformer_model_path
            )
            return self.hf_config.model_name
        return str(mlflow_transformer_model_path)
