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
from olive.passes import REGISTRY as PASS_REGISTRY

logger = logging.getLogger(__name__)


def get_cache_sub_dirs(cache_dir: Union[str, Path] = ".olive-cache"):
    """
    Returns the subdirectories of the cache directory.

    There are three subdirectories: models, runs, and evaluations.
    """
    cache_dir = Path(cache_dir)
    return cache_dir / "models", cache_dir / "runs", cache_dir / "evaluations"


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
    model_cache_dir, run_cache_dir, evaluation_cache_dir = get_cache_sub_dirs(cache_dir)
    model_jsons = list(model_cache_dir.glob(f"{model_number}_*.json"))
    for model_json in model_jsons:
        try:
            model_data = json.load(open(model_json, "r"))
            if model_data != {}:
                model_file = Path(json.load(open(model_json, "r"))["model_path"])
                model_file_number = model_file.stem.split("_")[0]
                if model_file_number == model_number:
                    if model_file.is_dir():
                        shutil.rmtree(model_file, ignore_errors=True)
                    elif model_file.is_file():
                        model_file.unlink()
        except Exception as e:
            logger.exception(e)
        finally:
            model_json = model_json.unlink()

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
        run_data = json.load(open(run_json, "r"))
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


def save_model(
    model_number: str,
    output_dir: Union[str, Path] = None,
    output_name: Union[str, Path] = None,
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

    model_json = serialize_to_json(json.load(open(model_jsons[0], "r")))

    # save model file/folder
    model_path = model_json["config"]["model_path"]
    if model_path is not None and Path(model_path).exists():
        model_path = Path(model_path)
        output_path = (output_dir / output_name).resolve()
        if model_path.is_dir():
            shutil.copytree(model_path, output_path, dirs_exist_ok=True)
        elif model_path.is_file():
            output_path = output_path.with_suffix(model_path.suffix)
            shutil.copy(model_path, output_path)
        output_path = str(output_path)
    else:
        output_path = model_path

    # save model json
    model_json["config"]["model_path"] = output_path
    json.dump(model_json, open(output_dir / f"{output_name}.json", "w"), indent=4)

    return model_json
