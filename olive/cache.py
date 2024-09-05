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

        if resource_path.is_local_resource_or_string_name():
            return resource_path.get_path()
        else:
            return self.download_resource(resource_path).get_path()

    def download_resource(self, resource_path: ResourcePath):
        """Return the path to a non-local resource.

        Non-local resources are stored in the non_local_resources subdirectory of the cache.
        """
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
            if not resource_path or resource_path.is_string_name():
                # Nothing to do if the path is empty or a string name
                continue
            # get cached resource path if not local or string name
            if not resource_path.is_local_resource():
                local_resource_path = self.download_resource(resource_path)
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
