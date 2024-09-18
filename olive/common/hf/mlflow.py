# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import yaml


def _load_mlflow_model_data(model_name_or_path: str) -> dict:
    yaml_path = Path(model_name_or_path) / "MLmodel"

    if not yaml_path.exists():
        return None

    with open(yaml_path) as fp:
        return yaml.safe_load(fp)


def is_mlflow_transformers(model_name_or_path: str) -> bool:
    mlflow_data = _load_mlflow_model_data(model_name_or_path)

    if not mlflow_data:
        return False

    # default flavor is "hftransformersv2" from azureml.evaluate.mlflow>=0.0.8
    # "hftransformers" from azureml.evaluate.mlflow<0.0.8
    # "transformers" from mlflow.transformers
    # TODO(trajep): let user specify flavor name if needed
    # to support other flavors in mlflow not only hftransformers
    flavors = mlflow_data.get("flavors", {})
    if not flavors or not ({"hftransformers", "hftransformersv2", "transformers"} & flavors.keys()):
        raise ValueError(
            "Invalid MLFlow model format. Please make sure the input model"
            " format is the same as the result of mlflow.transformers.save_model,"
            " or aml_mlflow.hftransformers.save_model from azureml.evaluate.mlflow"
        )

    return True


def get_pretrained_name_or_path(model_name_or_path: str, name: str) -> str:
    if not is_mlflow_transformers(model_name_or_path):
        # assumed to be an hf hub id or a local checkpoint
        return model_name_or_path

    parent_dir = Path(model_name_or_path).resolve()

    mlflow_data = _load_mlflow_model_data(model_name_or_path)
    flavors = mlflow_data.get("flavors", {}).keys()

    if {"hftransformers", "hftransformersv2"} & flavors:
        path = parent_dir / "data" / name
        if path.exists():
            return str(path)

        # some mlflow models only have model directory
        # e.g. https://ml.azure.com/models/Llama-2-7b-chat/version/24/catalog/registry/azureml-meta
        model_dir = parent_dir / "data" / "model"
        if model_dir.exists():
            return str(model_dir)

    elif "transformers" in flavors:
        if name in ("model", "config"):
            path = parent_dir / "model"
            if path.exists():
                return str(path)

        if name == "tokenizer":
            path = parent_dir / "components" / name
            if path.exists():
                return str(path)

    raise ValueError("Invalid MLFlow model format.")
