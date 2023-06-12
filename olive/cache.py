# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from pathlib import Path
from typing import Union

from olive.common.config_utils import serialize_to_json
from olive.common.utils import hash_dict
from olive.passes import REGISTRY as PASS_REGISTRY
from olive.resource_path import ResourcePath, create_resource_path

logger = logging.getLogger(__name__)


def get_cache_sub_dirs(cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Returns the subdirectories of the cache directory.

    There are three subdirectories: models, runs, and evaluations.
    """
    cache_dir = Path(cache_dir)
    return cache_dir / "models", cache_dir / "runs", cache_dir / "evaluations", cache_dir / "non_local_resources"


def clean_cache(cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Cleans the cache directory by deleting all subdirectories.
    """
    cache_sub_dirs = get_cache_sub_dirs(cache_dir)
    for sub_dir in cache_sub_dirs:
        if sub_dir.exists():
            shutil.rmtree(sub_dir)


def clean_evaluation_cache(cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Cleans the evaluation cache directory.
    """
    evaluation_cache_dir = get_cache_sub_dirs(cache_dir)[2]
    if evaluation_cache_dir.exists():
        shutil.rmtree(evaluation_cache_dir)


def create_cache(cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Creates the cache directory and all subdirectories.
    """
    cache_sub_dirs = get_cache_sub_dirs(cache_dir)
    for sub_dir in cache_sub_dirs:
        sub_dir.mkdir(parents=True, exist_ok=True)


def _delete_model(model_number: str, cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Deletes the model and all associated runs and evaluations.
    """
    model_cache_dir, run_cache_dir, evaluation_cache_dir, _ = get_cache_sub_dirs(cache_dir)
    # delete all model files that start with model_number
    model_files = list(model_cache_dir.glob(f"{model_number}_*"))
    for model_file in model_files:
        if model_file.is_dir():
            shutil.rmtree(model_file, ignore_errors=True)
        elif model_file.is_file():
            model_file.unlink()

    evaluation_jsons = list(evaluation_cache_dir.glob(f"{model_number}_*.json"))
    for evaluation_json in evaluation_jsons:
        evaluation_json.unlink()

    run_jsons = list(run_cache_dir.glob(f"*-{model_number}-*.json"))
    for run_json in run_jsons:
        _delete_run(run_json.stem, cache_dir)


def _delete_run(run_id: str, cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Deletes the run and all associated models and evaluations.
    """
    run_cache_dir = get_cache_sub_dirs(cache_dir)[1]
    run_json = run_cache_dir / f"{run_id}.json"
    try:
        with run_json.open("r") as f:
            run_data = json.load(f)
        # output model and children
        output_model_number = run_data["output_model_id"].split("_")[0]
        _delete_model(output_model_number, cache_dir)
    except Exception as e:
        logger.exception(e)
    finally:
        run_json.unlink()


def clean_pass_run_cache(pass_type: str, cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Clean the cache of runs for a given pass type.

    This function deletes all runs for a given pass type as well as all child models and evaluations.
    """
    assert pass_type.lower() in PASS_REGISTRY, f"Invalid pass type {pass_type}"

    run_cache_dir = get_cache_sub_dirs(cache_dir)[1]

    # cached runs for pass
    run_jsons = list(run_cache_dir.glob(f"{pass_type}-*.json"))
    for run_json in run_jsons:
        _delete_run(run_json.stem, cache_dir)


def get_non_local_resource(resource_path: ResourcePath, cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Returns the path to a non-local resource.

    Non-local resources are stored in the non_local_resources subdirectory of the cache.
    """
    non_local_resource_dir = get_cache_sub_dirs(cache_dir)[3]

    resource_path_hash = hash_dict(resource_path.to_json())
    resource_path_json = non_local_resource_dir / f"{resource_path_hash}.json"

    # check if resource path is cached
    if resource_path_json.exists():
        logger.debug(f"Using cached resource path {resource_path.to_json()}")
        with resource_path_json.open("r") as f:
            resource_path_data = json.load(f)["dest"]
        return create_resource_path(resource_path_data)

    # cache resource path
    save_dir = non_local_resource_dir / resource_path_hash
    # ensure save directory is empty
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # download resource to save directory
    logger.debug(f"Downloading non-local resource {resource_path.to_json()} to {save_dir}")
    local_resource_path = create_resource_path(resource_path.save_to_dir(save_dir))

    # cache resource path
    logger.debug(f"Caching resource path {resource_path}")
    with resource_path_json.open("w") as f:
        data = {"source": resource_path.to_json(), "dest": local_resource_path.to_json()}
        json.dump(data, f, indent=4)

    return local_resource_path


def save_model(
    model_number: str,
    output_dir: Union[str, Path] = None,
    output_name: Union[str, Path] = None,
    overwrite: bool = False,
    cache_dir: Union[str, Path] = ".olive-cache",
):
    """
    Saves a model from the cache to a given path.
    """
    model_number = model_number.split("_")[0]
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = output_name if output_name else "model"

    model_cache_dir = get_cache_sub_dirs(cache_dir)[0]
    model_jsons = list(model_cache_dir.glob(f"{model_number}_*.json"))
    assert len(model_jsons) == 1, f"No model found for {model_number}"

    with model_jsons[0].open("r") as f:
        model_json = serialize_to_json(json.load(f))

    if model_json["type"].lower() == "compositeonnxmodel":
        logger.warning("Saving composite ONNX models is not supported yet.")
        return

    model_path = model_json["config"]["model_path"]
    if model_path:
        # create resource path
        model_resource_path = create_resource_path(model_path)

        # get cached resource path if not local or string name
        if not (model_resource_path.is_local_resource() or model_resource_path.is_string_name()):
            model_resource_path = get_non_local_resource(model_resource_path, cache_dir)

        # save model to output directory
        model_path = model_resource_path.save_to_dir(output_dir, output_name, overwrite)

    # save model json
    model_json["config"]["model_path"] = model_path
    with open(output_dir / f"{output_name}.json", "w") as f:
        json.dump(model_json, f, indent=4)

    return model_json
