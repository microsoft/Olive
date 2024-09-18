# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from olive.common.config_utils import ConfigBase, convert_configs_to_dicts, serialize_to_json, validate_config
from olive.common.constants import DEFAULT_CACHE_DIR, DEFAULT_WORKFLOW_ID
from olive.common.utils import hash_dict, set_nested_dict_value
from olive.resource_path import ResourcePath, create_resource_path, find_all_resources

if TYPE_CHECKING:
    from olive.hardware import AcceleratorSpec

logger = logging.getLogger(__name__)


@dataclass
class CacheSubDirs:
    models: Path
    runs: Path
    evaluations: Path
    resources: Path
    mlflow: Path

    @classmethod
    def from_cache_dir(cls, cache_dir: Path) -> "CacheSubDirs":
        return cls(
            models=cache_dir / "models",
            runs=cache_dir / "runs",
            evaluations=cache_dir / "evaluations",
            resources=cache_dir / "resources",
            mlflow=cache_dir / "mlflow",
        )


class OliveCache:
    def __init__(
        self,
        cache_dir: Union[str, Path],
        clean_cache: bool = False,
        clean_evaluation_cache: bool = False,
    ):
        self.cache_dir = Path(cache_dir).resolve()
        logger.info("Using cache directory: %s", self.cache_dir)
        self.dirs = CacheSubDirs.from_cache_dir(self.cache_dir)

        if clean_evaluation_cache and self.dirs.evaluations.exists():
            shutil.rmtree(self.dirs.evaluations, ignore_errors=True)

        for sub_dir in asdict(self.dirs).values():
            if clean_cache and sub_dir.exists():
                shutil.rmtree(sub_dir)
            sub_dir.mkdir(parents=True, exist_ok=True)

        self.new_model_number = 0
        # model jsons have the format <model_number>_<pass_type>-<source_model>-<pass_config_hash>.json
        # model contents are stored in <model_number>_<pass_type>-<source_model>-<pass_config_hash> folder
        # sometimes the folder is created with contents but the json is not created when the pass fails to run
        # so we check for both when determining the new model number
        model_files = list(self.dirs.models.glob("*_*"))
        if len(model_files) > 0:
            self.new_model_number = max(int(model_file.stem.split("_")[0]) for model_file in model_files) + 1

    def get_cache_dir(self) -> Path:
        """Return the cache directory."""
        return self.cache_dir

    def get_new_model_number(self) -> int:
        """Get a new model number."""
        while True:
            new_model_number = self.new_model_number
            self.new_model_number += 1
            if not list(self.dirs.models.glob(f"{new_model_number}_*")):
                break
        return new_model_number

    def get_model_json_path(self, model_id: str) -> Path:
        """Get the path to the model json file."""
        return self.dirs.models / f"{model_id}.json"

    def get_model_output_path(self, model_id: str) -> Path:
        """Get the path to the model output directory."""
        output_path = self.dirs.models / model_id / "output_model"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def get_evaluation_json_path(self, model_id: str) -> Path:
        """Get the path to the evaluation json."""
        return self.dirs.evaluations / f"{model_id}.json"

    def get_run_json_path(
        self,
        pass_name: int,
        input_model_number: str,
        pass_config: dict,
        accelerator_spec: Optional["AcceleratorSpec"],
    ) -> Path:
        """Get the path to the run json."""
        pass_config_hash = hash_dict(pass_config)[:8]
        if not accelerator_spec:
            run_json_path = self.dirs.runs / f"{pass_name}-{input_model_number}-{pass_config_hash}.json"
        else:
            run_json_path = (
                self.dirs.runs / f"{pass_name}-{input_model_number}-{pass_config_hash}-{accelerator_spec}.json"
            )
        return run_json_path

    def clean_pass_run_cache(self, pass_type: str):
        """Clean the cache of runs for a given pass type.

        This function deletes all runs for a given pass type as well as all child models and evaluations.
        """
        # cached runs for pass
        run_jsons = list(self.dirs.runs.glob(f"{pass_type}-*.json"))
        for run_json in run_jsons:
            self._delete_run(run_json.stem)

    def _delete_model(self, model_number: str):
        """Delete the model and all associated runs and evaluations."""
        # delete all model files that start with model_number
        model_files = list(self.dirs.models.glob(f"{model_number}_*"))
        for model_file in model_files:
            if model_file.is_dir():
                shutil.rmtree(model_file, ignore_errors=True)
            elif model_file.is_file():
                model_file.unlink()

        evaluation_jsons = list(self.dirs.evaluations.glob(f"{model_number}_*.json"))
        for evaluation_json in evaluation_jsons:
            evaluation_json.unlink()

        run_jsons = list(self.dirs.runs.glob(f"*-{model_number}-*.json"))
        for run_json in run_jsons:
            self._delete_run(run_json.stem)

    def _delete_run(self, run_id: str):
        """Delete the run and all associated models and evaluations."""
        run_json = self.dirs.runs / f"{run_id}.json"
        try:
            with run_json.open("r") as f:
                run_data = json.load(f)
            # output model and children
            output_model_number = run_data["output_model_id"].split("_")[0]
            self._delete_model(output_model_number)
        except Exception:
            logger.exception("delete model failed.")
        finally:
            run_json.unlink()

    def get_local_path(self, resource_path: Optional[ResourcePath]) -> Optional[str]:
        """Return the local path of the any resource path as a string.

        If the resource path is a local resource, the path is returned.
        If the resource path is an AzureML resource, the resource is downloaded to the cache and the path is returned.
        """
        if resource_path is None:
            return None

        return self.get_local_path_or_download(resource_path).get_path()

    def get_local_path_or_download(self, resource_path: ResourcePath):
        """Return the path to a local instance of the resource path.

        Non-local resources are downloaded and stored in the non_local_resources subdirectory of the cache.
        """
        if resource_path.is_local_resource_or_string_name():
            return resource_path

        # choose left 8 characters of hash as resource path hash to reduce the risk of length too long
        resource_path_hash = hash_dict(resource_path.to_json())[:8]
        resource_path_json = self.dirs.resources / f"{resource_path_hash}.json"

        # check if resource path is cached
        if resource_path_json.exists():
            logger.debug("Using cached resource path %s", resource_path.to_json())
            with resource_path_json.open("r") as f:
                resource_path_data = json.load(f)["dest"]
            return create_resource_path(resource_path_data)

        # cache resource path
        save_dir = self.dirs.resources / resource_path_hash
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

    def prepare_resources_for_local(self, config: Union[Dict, ConfigBase]) -> Union[Dict, ConfigBase]:
        """Prepare all resources in the config for local execution.

        Download all non-local resources in the config to the cache. All resource paths in the config are replaced with
        strings that represent the local paths of the resources in the cache.
        """
        # keep track of the original config class
        config_class = None
        if isinstance(config, ConfigBase):
            config_class = config.__class__

        # find and download all non-local resources
        config = convert_configs_to_dicts(config)
        all_resources = find_all_resources(config)
        for resource_key, resource_path in all_resources.items():
            set_nested_dict_value(config, resource_key, self.get_local_path(resource_path))

        # validate the config if it was a ConfigBase
        if config_class:
            config = validate_config(config, config_class)

        return config

    def save_model(
        self,
        model_number: str,
        output_dir: Union[str, Path] = None,
        overwrite: bool = False,
        only_cache_files: bool = True,
    ) -> Optional[Dict]:
        """Save a model from the cache to a given path."""
        # This function should probably be outside of the cache module
        # just to be safe, import lazily to avoid any future circular imports
        from olive.model import ModelConfig

        model_number = model_number.split("_")[0]
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        model_jsons = list(self.dirs.models.glob(f"{model_number}_*.json"))
        assert len(model_jsons) == 1, f"No model found for {model_number}"

        with model_jsons[0].open("r") as f:
            model_json = serialize_to_json(json.load(f))

        if model_json["type"].lower() == "compositemodel":
            logger.warning("Saving models of type '%s' is not supported yet.", model_json["type"])
            return None

        # create model object so that we can get the resource paths
        model_config: ModelConfig = ModelConfig.from_json(model_json)
        resource_paths = model_config.get_resource_paths()
        for resource_name, resource_path in resource_paths.items():
            if (
                not resource_path
                or resource_path.is_string_name()
                or (only_cache_files and not resource_path.get_path().startswith(str(self.cache_dir)))
            ):
                # Nothing to do if the path is empty or a string name or if we only want to cache local files
                continue

            # get the path in the cache
            local_resource_path = self.get_local_path_or_download(resource_path)

            # check if path is from non-local resource cache
            local_resource_str = local_resource_path.get_path()
            resource_dir_str = str(self.dirs.resources)
            if only_cache_files and local_resource_str.startswith(resource_dir_str):
                # get the original resource path from the cache
                # resource_path could be "/cache/resources/1234/mlflow_model_folder"
                # load the json file "/cache/resources/1234.json" to get the original resource path
                resource_path_json = (
                    self.dirs.resources / f"{Path(local_resource_str.replace(resource_dir_str, '')).parts[1]}.json"
                )
                with resource_path_json.open("r") as f:
                    resource_json = json.load(f)
                    if create_resource_path(resource_json["dest"]) == local_resource_path:
                        # make sure it is the full resource and not a member of the resource
                        model_json["config"][resource_name] = resource_json["source"]
                        continue

            # save resource to output directory
            model_json["config"][resource_name] = local_resource_path.save_to_dir(
                output_dir, resource_name.replace("_path", ""), overwrite
            )

        # Copy "additional files" to the model folder
        # we only have additional files for onnx models so saving to "model" is safe
        model_attributes = model_json["config"].get("model_attributes") or {}
        additional_files = model_attributes.get("additional_files", [])

        for i, src_filepath in enumerate(additional_files):
            dst_filepath = output_dir / "model" / Path(src_filepath).name
            additional_files[i] = str(dst_filepath)

            if not dst_filepath.exists():
                shutil.copy(str(src_filepath), str(dst_filepath))

        if additional_files:
            model_json["config"]["model_attributes"]["additional_files"] = additional_files

        # save model json
        with (output_dir / "model_config.json").open("w") as f:
            json.dump(model_json, f, indent=4)

        return model_json

    def set_cache_env(self):
        """Set environment variable for the cache directory."""
        os.environ["OLIVE_CACHE_DIR"] = str(self.cache_dir)
        logger.debug("Set OLIVE_CACHE_DIR: %s", self.cache_dir)

    @classmethod
    def from_cache_env(cls) -> "OliveCache":
        """Create an OliveCache object from the cache directory environment variable."""
        cache_dir = os.environ.get("OLIVE_CACHE_DIR")
        if cache_dir is None:
            logger.debug("OLIVE_CACHE_DIR environment variable not set. Using default cache directory.")
            cache_dir = Path(DEFAULT_CACHE_DIR).resolve() / DEFAULT_WORKFLOW_ID
        return cls(cache_dir)
