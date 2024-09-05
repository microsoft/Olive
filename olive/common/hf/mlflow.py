# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import yaml


def load_mlflow_model_data(model_name_or_path: str) -> dict:
    yaml_path = Path(model_name_or_path) / "MLmodel"

    if not yaml_path.exists():
        raise FileNotFoundError(f"No MLmodel file found at path: {yaml_path}")

    with open(yaml_path) as fp:
        return yaml.safe_load(fp)


def is_mlflow_transformers(model_name_or_path: str) -> bool:
    mlflow_data = load_mlflow_model_data(model_name_or_path)

    # TODO(trajep): let user specify flavor name if needed
    # to support other flavors in mlflow not only hftransformers
    flavors = mlflow_data.get("flavors", {})
    if not flavors or not ({"hftransformersv2", "transformers"} & flavors.keys()):
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

    mlflow_data = load_mlflow_model_data(model_name_or_path)
    flavors = mlflow_data.get("flavors", {}).keys()

    if "hftransformersv2" in flavors:
        path = parent_dir / "data" / name
        if path.exists():
            return str(path)
    
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
