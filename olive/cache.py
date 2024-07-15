# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

from olive.common.config_utils import serialize_to_json
from olive.common.utils import hash_dict
from olive.resource_path import ResourcePath, create_resource_path

logger = logging.getLogger(__name__)


def set_cache_dir(cache_dir: str):
    """Set the cache directory."""
    os.environ["OLIVE_CACHE_DIR"] = cache_dir


def get_cache_dir() -> Path:
    """Return the cache directory.

    The cache directory is set by the OLIVE_CACHE_DIR environment variable.
    If the environment variable is not set, the default cache directory is .olive-cache.
    """
    return Path(os.environ.get("OLIVE_CACHE_DIR", ".olive-cache")).resolve()


def get_cache_sub_dirs() -> Dict[str, Path]:
    """Return the subdirectories of the cache directory."""
    cache_dir = get_cache_dir()
    return {
        "models": cache_dir / "models",
        "runs": cache_dir / "runs",
        "evaluations": cache_dir / "evaluations",
        "non_local_resources": cache_dir / "non_local_resources",
        "mlflow": cache_dir / "mlflow",
    }


def clean_cache():
    """Clean the cache directory by deleting all subdirectories."""
    for sub_dir in get_cache_sub_dirs().values():
        if sub_dir.exists():
            shutil.rmtree(sub_dir)


def clean_evaluation_cache():
    """Clean the evaluation cache directory."""
    evaluation_cache_dir = get_cache_sub_dirs()["evaluations"]
    if evaluation_cache_dir.exists():
        shutil.rmtree(evaluation_cache_dir)


def create_cache():
    """Create the cache directory and all subdirectories."""
    for sub_dir in get_cache_sub_dirs().values():
        sub_dir.mkdir(parents=True, exist_ok=True)


def _delete_model(model_number: str):
    """Delete the model and all associated runs and evaluations."""
    cache_sub_dirs = get_cache_sub_dirs()

    # delete all model files that start with model_number
    model_files = list(cache_sub_dirs["models"].glob(f"{model_number}_*"))
    for model_file in model_files:
        if model_file.is_dir():
            shutil.rmtree(model_file, ignore_errors=True)
        elif model_file.is_file():
            model_file.unlink()

    evaluation_jsons = list(cache_sub_dirs["evaluations"].glob(f"{model_number}_*.json"))
    for evaluation_json in evaluation_jsons:
        evaluation_json.unlink()

    run_jsons = list(cache_sub_dirs["runs"].glob(f"*-{model_number}-*.json"))
    for run_json in run_jsons:
        _delete_run(run_json.stem)


def _delete_run(run_id: str):
    """Delete the run and all associated models and evaluations."""
    run_cache_dir = get_cache_sub_dirs()["runs"]
    run_json = run_cache_dir / f"{run_id}.json"
    try:
        with run_json.open("r") as f:
            run_data = json.load(f)
        # output model and children
        output_model_number = run_data["output_model_id"].split("_")[0]
        _delete_model(output_model_number)
    except Exception:
        logger.exception("delete model failed.")
    finally:
        run_json.unlink()


def clean_pass_run_cache(pass_type: str):
    """Clean the cache of runs for a given pass type.

    This function deletes all runs for a given pass type as well as all child models and evaluations.
    """
    from olive.passes import REGISTRY as PASS_REGISTRY

    assert pass_type.lower() in PASS_REGISTRY, f"Invalid pass type {pass_type}"

    run_cache_dir = get_cache_sub_dirs()["runs"]

    # cached runs for pass
    run_jsons = list(run_cache_dir.glob(f"{pass_type}-*.json"))
    for run_json in run_jsons:
        _delete_run(run_json.stem)


def download_resource(resource_path: ResourcePath):
    """Return the path to a non-local resource.

    Non-local resources are stored in the non_local_resources subdirectory of the cache.
    """
    non_local_resource_dir = get_cache_sub_dirs()["non_local_resources"]

    # choose left 8 characters of hash as resource path hash to reduce the risk of length too long
    resource_path_hash = hash_dict(resource_path.to_json())[:8]
    resource_path_json = non_local_resource_dir / f"{resource_path_hash}.json"

    # check if resource path is cached
    if resource_path_json.exists():
        logger.debug("Using cached resource path %s", resource_path.to_json())
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
    logger.debug("Downloading non-local resource %s to %s", resource_path.to_json(), save_dir)
    local_resource_path = create_resource_path(resource_path.save_to_dir(save_dir))

    # cache resource path
    logger.debug("Caching resource path %s", resource_path)
    with resource_path_json.open("w") as f:
        data = {"source": resource_path.to_json(), "dest": local_resource_path.to_json()}
        json.dump(data, f, indent=4)

    return local_resource_path


def get_local_path(resource_path: Optional[ResourcePath]):
    """Return the local path of the any resource path.

    If the resource path is a local resource, the path is returned.
    If the resource path is an AzureML resource, the resource is downloaded to the cache and the path is returned.
    """
    if resource_path is None:
        return None

    if resource_path.is_local_resource_or_string_name():
        return resource_path.get_path()
    elif resource_path.is_azureml_resource():
        return download_resource(resource_path).get_path()
    else:
        return None


def normalize_data_path(data_root: Union[str, Path], data_dir: Union[str, Path, ResourcePath]):
    """Normalize data path, if data_dir is absolute path, return data_dir, else return data_root/data_dir."""
    if isinstance(data_dir, ResourcePath):
        data_dir_str = data_dir.get_path()
    else:
        data_dir_str = data_dir

    data_full_path = None
    if not data_dir_str:
        data_full_path = data_root
    elif Path(data_dir_str).is_absolute():
        data_full_path = data_dir_str
    else:
        if data_root:
            if isinstance(data_dir, ResourcePath) and data_dir.is_azureml_resource():
                raise ValueError("could not append AzureML data to data_root")

            # we cannot use Path to join the path. If the data_root is something like: azureml://, then Path will
            # change the data_root to azureml:/, which is not a valid path
            data_full_path = os.path.join(data_root, data_dir_str).replace("\\", "/")
        else:
            # will keep this as is so that we don't lose information inside ResourcePath
            data_full_path = data_dir

    return create_resource_path(data_full_path)


def get_local_path_from_root(data_root: Union[str, Path], data_dir: Union[str, Path, ResourcePath]):
    data_path = normalize_data_path(data_root, data_dir)
    if data_path:
        return get_local_path(data_path)
    else:
        return None


def save_model(
    model_number: str,
    output_dir: Union[str, Path] = None,
    output_name: Union[str, Path] = None,
    overwrite: bool = False,
) -> Optional[Dict]:
    """Save a model from the cache to a given path."""
    # This function should probably be outside of the cache module
    # just to be safe, import lazily to avoid any future circular imports
    from olive.model import ModelConfig

    model_number = model_number.split("_")[0]
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = output_name if output_name else "model"

    model_cache_dir = get_cache_sub_dirs()["models"]
    model_jsons = list(model_cache_dir.glob(f"{model_number}_*.json"))
    assert len(model_jsons) == 1, f"No model found for {model_number}"

    with model_jsons[0].open("r") as f:
        model_json = serialize_to_json(json.load(f))

    if model_json["type"].lower() in ("compositemodel", "compositepytorchmodel"):
        logger.warning("Saving models of type '%s' is not supported yet.", model_json["type"])
        return None

    # create model object so that we can get the resource paths
    model_config: ModelConfig = ModelConfig.from_json(model_json)
    resource_paths = model_config.get_resource_paths()
    for resource_name, resource_path in resource_paths.items():
        if not resource_path or resource_path.is_string_name():
            # Nothing to do if the path is empty or a string name
            continue
        # get cached resource path if not local or string name
        if not resource_path.is_local_resource():
            local_resource_path = download_resource(resource_path)
        else:
            local_resource_path = resource_path
        # if there are multiple resource paths, we will save them to a subdirectory of output_dir/output_name
        if len(resource_paths) > 1:
            save_dir = (output_dir / output_name).with_suffix("")
            save_name = resource_name.replace("_path", "")
        else:
            save_dir = output_dir
            save_name = output_name

        # save resource to output directory
        model_json["config"][resource_name] = local_resource_path.save_to_dir(save_dir, save_name, overwrite)

    # Copy "additional files" to the output folder
    model_attributes = model_json["config"].get("model_attributes") or {}
    additional_files = model_attributes.get("additional_files", [])

    for i, src_filepath in enumerate(additional_files):
        dst_filepath = Path(output_dir) / output_name / Path(src_filepath).name
        additional_files[i] = str(dst_filepath)

        if not dst_filepath.exists():
            shutil.copy(str(src_filepath), str(dst_filepath))

    if additional_files:
        model_json["config"]["model_attributes"]["additional_files"] = additional_files

    # save model json
    with (output_dir / f"{output_name}.json").open("w") as f:
        json.dump(model_json, f, indent=4)

    return model_json
