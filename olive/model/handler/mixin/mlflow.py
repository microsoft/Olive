# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path

import yaml

# from olive.cache import get_cache_sub_dirs
from olive.cache import OliveCache
from olive.common.utils import hardlink_copy_dir, hash_string

logger = logging.getLogger(__name__)


class MLTransformersFlowMixin:
    def maybe_init_mlflow_transformers(self):
        """Initialize MLFlow model if the model is saved in MLFlow format."""
        parent_dir = Path(self.get_resource("model_path")).resolve()

        yaml_path = parent_dir / "MLmodel"
        if not yaml_path.exists():
            return

        with open(yaml_path) as fp:
            mlflow_data = yaml.safe_load(fp)
            # default flavor is "hftransformersv2" from azureml.evaluate.mlflow>=0.0.8
            # "hftransformers" from azureml.evaluate.mlflow<0.0.8
            # TODO(trajep): let user specify flavor name if needed
            # to support other flavors in mlflow not only hftransformers
            flavors = mlflow_data.get("flavors", {})
            if not flavors or not any(flavor.startswith("hftransformers") for flavor in flavors):
                raise ValueError(
                    "Invalid MLFlow model format. Please make sure the input model"
                    " format is same with the result of mlflow.transformers.save_model,"
                    " or aml_mlflow.hftransformers.save_model from azureml.evaluate.mlflow"
                )

        model_dir = parent_dir / "data" / "model"
        assert model_dir.is_dir(), "Model directory does not exist."

        config_dir = parent_dir / "data" / "config"
        tokenizer_dir = parent_dir / "data" / "tokenizer"

        # some mlflow models only have model directory
        if not config_dir.exists() and not tokenizer_dir.exists():
            self.mlflow_transformers_path = str(model_dir)
            return

        # some mlflow models have config and tokenizer directories but model directory also
        # contains the same files
        model_dir_contents = set(model_dir.iterdir())
        if set(config_dir.iterdir()) <= model_dir_contents and set(tokenizer_dir.iterdir()) <= model_dir_contents:
            self.mlflow_transformers_path = str(model_dir)
            return

        # have to gather all contents into a single directory
        cache = OliveCache.from_cache_env()
        mlflow_transformers_path = cache.dirs.mlflow / hash_string(str(parent_dir))
        if (mlflow_transformers_path / "config.json").exists():
            logger.debug("MLFlow model already exists in cache. Reusing it.")
        else:
            for src_dir in [model_dir, config_dir, tokenizer_dir]:
                hardlink_copy_dir(src_dir, mlflow_transformers_path)
        self.mlflow_transformers_path = str(mlflow_transformers_path)
