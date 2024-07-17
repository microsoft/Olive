# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import yaml


def is_mlflow_transformers(model_name_or_path: str) -> bool:
    yaml_path = Path(model_name_or_path) / "MLmodel"

    if not yaml_path.exists():
        return False

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

    return True


def get_pretrained_name_or_path(model_name_or_path: str, name: str) -> str:
    if not is_mlflow_transformers(model_name_or_path):
        # assumed to be an hf hub id or a local checkpoint
        return model_name_or_path

    parent_dir = Path(model_name_or_path).resolve()

    # assumed to be an mlflow model
    pretrained_path = parent_dir / "data" / name
    if pretrained_path.exists():
        return str(pretrained_path)

    # some mlflow models only have model directory
    model_dir = parent_dir / "data" / "model"
    if model_dir.exists():
        return str(model_dir)

    raise ValueError("Invalid MLFlow model format.")
