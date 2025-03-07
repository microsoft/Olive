# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import io
import json
import logging
import os
import re
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from olive.common.config_utils import ConfigBase, convert_configs_to_dicts, validate_config
from olive.common.constants import DEFAULT_CACHE_DIR, DEFAULT_WORKFLOW_ID
from olive.common.container_client_factory import AzureContainerClientFactory
from olive.common.pydantic_v1 import root_validator, validator
from olive.common.utils import hash_dict, hf_repo_exists, set_nested_dict_value
from olive.model.config.model_config import ModelConfig
from olive.resource_path import ResourcePath, create_resource_path, find_all_resources

if TYPE_CHECKING:
    from olive.hardware import AcceleratorSpec

logger = logging.getLogger(__name__)

SHARED_CACHE_PATTERN = r"https://([^.]+)\.blob\.core\.windows\.net/([^/]+)"


def is_shared_cache_dir(s) -> bool:
    return bool(re.match(SHARED_CACHE_PATTERN, str(s)))


@dataclass
class CacheSubDirs:
    cache_dir: Path
    runs: Path
    evaluations: Path
    resources: Path
    mlflow: Path

    @classmethod
    def from_cache_dir(cls, cache_dir: Path) -> "CacheSubDirs":
        return cls(
            cache_dir=cache_dir,
            runs=cache_dir / "runs",
            evaluations=cache_dir / "evaluations",
            resources=cache_dir / "resources",
            mlflow=cache_dir / "mlflow",
        )


class CacheConfig(ConfigBase):
    cache_dir: Union[str, List[str]] = DEFAULT_CACHE_DIR
    clean_cache: bool = False
    clean_evaluation_cache: bool = False
    account_name: str = None
    container_name: str = None
    enable_shared_cache: bool = False
    update_shared_cache: bool = True

    @validator("cache_dir", pre=True, always=True)
    def validate_cache_dir(cls, v):
        if not v:
            return [DEFAULT_CACHE_DIR]
        if isinstance(v, list):
            if len(v) > 2:
                raise ValueError("Only two cache directories are supported.")
            if len(v) == 2:
                shared_cache_count = sum(is_shared_cache_dir(s) for s in v)
                if shared_cache_count > 1:
                    raise ValueError("Only one shared cache directory is supported.")
                if shared_cache_count == 0:
                    logger.warning("More than one cache directory is provided. Using the first one %s.", v[0])
                    return [v[0]]
                return v
            if len(v) == 1:
                if is_shared_cache_dir(v[0]):
                    v.append(DEFAULT_CACHE_DIR)
                return v
        v = str(v)
        if is_shared_cache_dir(v):
            return [DEFAULT_CACHE_DIR, v]
        return [v]

    @validator("account_name")
    def validate_account_name(cls, v, values):
        if v:
            return v
        match = cls._get_shared_cache_match(values.get("cache_dir"))
        return match.group(1) if match else None

    @validator("container_name")
    def validate_container_name(cls, v, values):
        if v:
            return v
        match = cls._get_shared_cache_match(values.get("cache_dir"))
        return match.group(2) if match else None

    @root_validator()
    def validate_enable_shared_cache(cls, values):
        if values.get("account_name") and values.get("container_name"):
            values["enable_shared_cache"] = True
        elif values.get("enable_shared_cache"):
            values["account_name"] = values.get("account_name") or "olivepublicmodels"
            values["container_name"] = values.get("container_name") or "olivecachemodels"
        else:
            values["enable_shared_cache"] = False
        return values

    @staticmethod
    def _get_shared_cache_match(cache_dir):
        for cache in cache_dir:
            match = re.match(SHARED_CACHE_PATTERN, str(cache))

            if match:
                return match
        return None

    def get_local_cache_dir(self):
        for cache_dir in self.cache_dir:
            if not is_shared_cache_dir(cache_dir):
                return cache_dir
        return DEFAULT_CACHE_DIR

    def create_cache(self, workflow_id: str = DEFAULT_WORKFLOW_ID) -> "OliveCache":
        local_cache_dir = Path(self.get_local_cache_dir()) / workflow_id
        self.cache_dir = [str(local_cache_dir)]
        return OliveCache(self)


class OliveCache:
    def __init__(self, cache_config: Union[CacheConfig, Dict]):
        cache_config = validate_config(cache_config, CacheConfig)
        cache_dir = Path(cache_config.get_local_cache_dir()).resolve()
        logger.info("Using cache directory: %s", cache_dir)
        self.dirs = CacheSubDirs.from_cache_dir(cache_dir)

        if cache_config.clean_cache and cache_dir.exists():
            shutil.rmtree(cache_dir)

        if cache_config.clean_evaluation_cache and self.dirs.evaluations.exists():
            shutil.rmtree(self.dirs.evaluations, ignore_errors=True)

        cache_dir.mkdir(parents=True, exist_ok=True)
        for sub_dir in asdict(self.dirs).values():
            sub_dir.mkdir(parents=True, exist_ok=True)

        self.enable_shared_cache = cache_config.enable_shared_cache
        if self.enable_shared_cache:
            self.shared_cache = SharedCache(cache_config.account_name, cache_config.container_name)
        self.update_shared_cache = cache_config.update_shared_cache

    @staticmethod
    def get_run_json(
        pass_name: str, pass_config: Dict[str, Any], input_model_id: str, accelerator_spec: "AcceleratorSpec"
    ) -> Dict[str, Any]:
        accelerator_spec = str(accelerator_spec) if accelerator_spec else None
        return {
            "input_model_id": input_model_id,
            "pass_name": pass_name,
            "pass_config": pass_config,
            "accelerator_spec": accelerator_spec,
        }

    @classmethod
    def from_cache_env(cls) -> "OliveCache":
        """Create an OliveCache object from the cache directory environment variable."""
        cache_dir = os.environ.get("OLIVE_CACHE_DIR")
        if cache_dir is None:
            logger.debug("OLIVE_CACHE_DIR environment variable not set. Using default cache directory.")
            cache_dir = Path(DEFAULT_CACHE_DIR).resolve() / DEFAULT_WORKFLOW_ID
        return cls(cache_config={"cache_dir": cache_dir})

    def cache_model(self, model_id: str, model_json: Dict):
        model_json_path = self.get_model_json_path(model_id)
        try:
            with model_json_path.open("w") as f:
                json.dump(model_json, f, indent=4)
            logger.debug("Cached model %s to %s", model_id, model_json_path)

        except Exception:
            logger.exception("Failed to cache model to local cache.")

        if self.enable_shared_cache and self.update_shared_cache:
            self.shared_cache.cache_model(model_id, model_json)

    def load_model(self, model_id: str) -> Optional[ModelConfig]:
        """Load the model from the cache directory."""
        model_json_path = self.get_model_json_path(model_id)
        if model_json_path.exists():
            try:
                logger.info("Loading model %s from cache.", model_id)
                with model_json_path.open() as f:
                    return json.load(f)
            except Exception:
                logger.exception("Failed to load model from local cache.")

        if self.enable_shared_cache:
            return self.shared_cache.load_model(model_id, model_json_path)

        return None

    def download_shared_cache_model(self, input_model_config: ModelConfig, input_model_id: str):
        model_saved_path = self.get_model_cache_path(input_model_id)
        return self.shared_cache.download_model(input_model_config, input_model_id, model_saved_path)

    def cache_run(
        self,
        pass_name: int,
        pass_config: dict,
        input_model_id: str,
        output_model_id: str,
        accelerator_spec: "AcceleratorSpec",
    ):
        run_json = self.get_run_json(pass_name, pass_config, input_model_id, accelerator_spec)
        run_json["output_model_id"] = output_model_id
        run_json_path = self.get_run_json_path(output_model_id)
        try:
            with run_json_path.open("w") as f:
                json.dump(run_json, f, indent=4)
            logger.debug("Cached run %s to %s", output_model_id, run_json_path)
        except Exception:
            logger.exception("Failed to cache run to local cache.")

        if self.enable_shared_cache and self.update_shared_cache:
            self.shared_cache.cache_run(output_model_id, run_json_path)

    def load_run_from_model_id(self, model_id: str):
        run_json_path = self.get_run_json_path(model_id)
        if run_json_path.exists():
            try:
                logger.info("Loading run %s from cache.", model_id)
                with run_json_path.open() as f:
                    return json.load(f)
            except Exception:
                logger.exception("Failed to load run from local cache.")
        if self.enable_shared_cache:
            return self.shared_cache.load_run(model_id, run_json_path)
        return {}

    def get_run_path(self, model_id: str) -> Path:
        run_path = self.dirs.runs / model_id
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path

    def get_run_json_path(self, model_id: str):
        run_path = self.get_run_path(model_id)
        return run_path / "run.json"

    def get_model_json_path(self, model_id: str) -> Path:
        return self.get_run_path(model_id) / "model.json"

    def cache_evaluation(self, model_id: str, evaluation_json: Dict):
        evaluation_json_path = self.get_evaluation_json_path(model_id)
        try:
            with evaluation_json_path.open("w") as f:
                json.dump(evaluation_json, f, indent=4)
                logger.debug("Cached evaluation %s to %s", model_id, evaluation_json_path)
        except Exception:
            logger.exception("Failed to cache evaluation")

    def get_evaluation_json_path(self, model_id: str) -> Path:
        """Get the path to the evaluation json."""
        return self.dirs.evaluations / f"{model_id}.json"

    def cache_olive_config(self, olive_config: Dict):
        olive_config_path = self.dirs.cache_dir / "olive_config.json"
        try:
            with olive_config_path.open("w") as f:
                json.dump(olive_config, f, indent=4)
            logger.debug("Cached olive config to %s", olive_config_path)
        except Exception:
            logger.exception("Failed to cache olive config")

    def get_output_model_id(
        self,
        pass_name: str,
        pass_config: Dict[str, Any],
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec" = None,
    ):
        run_json = self.get_run_json(pass_name.lower(), pass_config, input_model_id, accelerator_spec)
        return hash_dict(run_json)[:8]

    def get_cache_dir(self) -> Path:
        """Return the cache directory."""
        return self.dirs.cache_dir

    def set_cache_env(self):
        """Set environment variable for the cache directory."""
        os.environ["OLIVE_CACHE_DIR"] = str(self.dirs.cache_dir)
        logger.debug("Set OLIVE_CACHE_DIR: %s", self.dirs.cache_dir)

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

    def get_resource_cache_path(self):
        return self.dirs.resources

    def get_model_cache_path(self, model_id: str) -> Path:
        """Get the path to the model output directory."""
        run_path = self.get_run_path(model_id)
        output_path = run_path / "models"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def save_model(
        self, model_id: str, output_dir: str = None, overwrite: bool = False, only_cache_files: bool = False
    ):
        """Save a model from the cache to a given path."""
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_json = self.load_model(model_id)
        if model_json["type"].lower() == "compositemodel":
            model_json_config = model_json["config"]
            copied_components = []
            saved_external_files = {}
            for component_name, component in zip(
                model_json_config["model_component_names"], model_json_config["model_components"]
            ):
                if component["type"].lower() != "onnxmodel":
                    # save each component with a prefix
                    # e.g. "component_1" -> "component_1_{resource_name}"
                    copied_components.append(
                        self._save_model(
                            component,
                            output_dir=output_dir,
                            overwrite=overwrite,
                            only_cache_files=only_cache_files,
                            path_prefix=component_name,
                        )
                    )
                else:
                    # save all onnx files into the same directory
                    component_model_json, component_local_resource_names = self._replace_with_local_resources(
                        component, only_cache_files=only_cache_files
                    )

                    for resource_name in component_local_resource_names:
                        if resource_name != "model_path":
                            # this case does not exist in the current code
                            # but we need to handle it for future use
                            component_model_json["config"][resource_name] = component_model_json["config"][
                                resource_name
                            ].save_to_dir(output_dir, resource_name, overwrite)
                        else:
                            from olive.passes.onnx.common import resave_model

                            resave_model(
                                ModelConfig.parse_obj(component_model_json).create_model().model_path,
                                output_dir / "model" / f"{component_name}.onnx",
                                saved_external_files=saved_external_files,
                            )
                            component_model_json["config"][resource_name] = str(output_dir / "model")
                            component_model_json["config"]["onnx_file_name"] = f"{component_name}.onnx"

                    copied_components.append(component_model_json)

            model_json_config["model_components"] = copied_components
            # save additional files
            model_json = self._save_additional_files(model_json, output_dir / "model")
        else:
            model_json = self._save_model(model_json, output_dir, overwrite)

        # save model json
        with (output_dir / "model_config.json").open("w") as f:
            json.dump(model_json, f, indent=4)
        return model_json

    def _save_model(
        self,
        model_json: dict,
        output_dir: str,
        overwrite: bool = False,
        only_cache_files: bool = False,
        path_prefix: str = None,
    ) -> dict:
        # get updated model json with local resources
        model_json, local_resource_names = self._replace_with_local_resources(
            model_json, only_cache_files=only_cache_files
        )

        # save local resources to output directory
        for resource_name in local_resource_names:
            path_name = resource_name.replace("_path", "")
            if path_prefix:
                path_name = f"{path_prefix}_{path_name}"
            # TODO(anyone): consider using hardlink_copy_file/dir instead of copy
            # to avoid copying large files
            model_json["config"][resource_name] = model_json["config"][resource_name].save_to_dir(
                output_dir, path_name, overwrite
            )

        # we only have additional files for onnx models so saving to "model" is safe
        model_path_name = "model"
        if path_prefix:
            model_path_name = f"{path_prefix}_{model_path_name}"
        return self._save_additional_files(model_json, output_dir / model_path_name)

    def _replace_with_local_resources(self, model_json: dict, only_cache_files: bool = False) -> Tuple[dict, List[str]]:
        local_resource_names = []
        # get the resource paths from the model config
        for resource_name, resource_path in ModelConfig.from_json(model_json).get_resource_paths().items():
            if (
                not resource_path
                or resource_path.is_string_name()
                or (only_cache_files and not resource_path.get_path().startswith(str(self.dirs.cache_dir)))
            ):
                # Nothing to do if the path is empty or a string name or if we only want to cache local files
                continue

            # get the path in the cache
            local_resource_path = self.get_local_path_or_download(resource_path)

            # check if path is from non-local resource cache
            local_resource_str = local_resource_path.get_path()
            resource_dir = self.get_resource_cache_path()
            resource_dir_str = str(resource_dir)
            if only_cache_files and local_resource_str.startswith(resource_dir_str):
                # get the original resource path from the cache
                # resource_path could be "/cache/resources/1234/mlflow_model_folder"
                # load the json file "/cache/resources/1234.json" to get the original resource path
                resource_path_json = (
                    resource_dir / f"{Path(local_resource_str.replace(resource_dir_str, '')).parts[1]}.json"
                )
                with resource_path_json.open("r") as f:
                    resource_json = json.load(f)
                    if create_resource_path(resource_json["dest"]) == local_resource_path:
                        # make sure it is the full resource and not a member of the resource
                        model_json["config"][resource_name] = resource_json["source"]
                        continue

            model_json["config"][resource_name] = local_resource_path
            local_resource_names.append(resource_name)

        return model_json, local_resource_names

    def _save_additional_files(self, model_json: dict, output_dir: Path) -> dict:
        # Copy "additional files" to the model folder
        # we only have additional files for onnx models so saving to "model" is safe
        model_attributes = model_json["config"].get("model_attributes") or {}
        additional_files = model_attributes.get("additional_files", [])

        for i, src_filepath in enumerate(additional_files):
            output_dir.mkdir(parents=True, exist_ok=True)
            dst_filepath = output_dir / Path(src_filepath).name
            additional_files[i] = str(dst_filepath)

            if not dst_filepath.exists():
                shutil.copy(str(src_filepath), str(dst_filepath))

        if additional_files:
            model_json["config"]["model_attributes"]["additional_files"] = additional_files

        return model_json

    def disable_shared_cache(self):
        self.enable_shared_cache = False
        self.shared_cache = None


class SharedCache:
    def __init__(self, account_name: str, container_name: str):
        self.container_client_factory = AzureContainerClientFactory(account_name, container_name)

    def cache_run(self, model_id: str, run_json_path: Path) -> None:
        """Cache run json to shared cache."""
        try:
            run_blob = f"{model_id}/run.json"
            self._upload_file_to_blob(run_json_path, run_blob)
            logger.debug("Cached run %s to shared cache.", model_id)
        except Exception:
            logger.exception("Failed to cache run to shared cache.")
            # delete all uploaded files if any upload fails
            try:
                self.container_client_factory.delete_blob(model_id)
            except Exception:
                logger.exception(
                    "Upload model to shared cache failed. There might be some dirty files in the shared cache."
                    "Please manually clean up. %s",
                    model_id,
                )

    def load_run(self, model_id: str, run_json_path: Path) -> Optional[Dict]:
        blob = f"{model_id}/run.json"
        try:
            if not self.exist_in_shared_cache(blob):
                logger.info("Run %s is not found in shared cache.", model_id)
                return {}
            logger.info("Downloading %s to %s", blob, run_json_path)
            self.container_client_factory.download_blob(blob, run_json_path)
            with run_json_path.open() as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to load run from shared cache.")
            return {}

    def cache_model(self, model_id: str, model_json: Dict) -> None:
        """Upload output model to shared cache.

            model path and adapter path (if exists)
                will be uploaded to shared cache to `<model_id>/model_path.json`.
            model files will be uploaded to shared cache to `<model_id>/model/`.
            adapter files will be uploaded to shared cache to `<model_id>/adapter/`.
            HF model with model path as repo name will not be uploaded to shared cache.

        Args:
            model_id (str): Output model id
            model_json (Dict): Output model config json

        """
        if self.exist_in_shared_cache(model_id):
            logger.info("Model is already in shared cache.")
            return

        logger.info("Uploading model %s to shared cache ...", model_id)
        model_path = model_json["config"].get("model_path")

        if model_path is None:
            logger.error("Model path is not found in the output model config. Upload failed.")
            return

        try:
            adapter_path = None

            model_json_copy = deepcopy(model_json)
            model_json_copy["config"]["shared_cache"] = True

            if model_json["config"].get("model_attributes"):
                model_json_copy["config"]["model_attributes"].pop("additional_files", None)

            if model_json["type"].lower() == "hfmodel" and Path(model_path).exists():
                model_json_copy["config"]["model_path"] = Path(model_path).name
            else:
                model_json_copy["config"]["model_path"] = model_path

            if model_json["type"].lower() == "hfmodel" and model_json["config"].get("adapter_path"):
                adapter_path = Path(model_json["config"]["adapter_path"])
                model_json_copy["config"]["adapter_path"] = adapter_path.name

            model_files_blob = f"{model_id}/model"

            # upload model files
            model_blob = f"{model_files_blob}/model/{model_json_copy['config']['model_path']}"
            self.upload_model_files(model_path, model_blob)

            # upload adapter files
            adapter_blob = f"{model_files_blob}/adapter"
            self.upload_model_files(adapter_path, adapter_blob)

            # upload additional files
            if model_json["config"].get("model_attributes") and model_json["config"]["model_attributes"].get(
                "additional_files"
            ):
                additional_files = model_json["config"]["model_attributes"]["additional_files"]
                for file in additional_files:
                    file_path = Path(file)
                    if file_path.exists():
                        self._upload_file_to_blob(file_path, f"{model_files_blob}/additional_files/{file_path.name}")

            # upload model config file
            model_config_bytes = json.dumps(model_json_copy).encode()
            with io.BytesIO(model_config_bytes) as data:
                self.container_client_factory.upload_blob(f"{model_id}/model.json", data)
            logger.info("Model %s is uploaded to shared cache.", model_id)

        except Exception:
            logger.exception("Failed to upload model to shared cache.")
            # delete all uploaded files if any upload fails
            try:
                self.container_client_factory.delete_blob(model_id)
            except Exception:
                logger.exception(
                    "Upload model to shared cache failed. There might be some dirty files in the shared cache."
                    "Please manually clean up. %s",
                    model_id,
                )

    def load_model(self, model_id: str, model_json_path: Path) -> ModelConfig:
        """Get model config from shared cache by model id."""
        model_config_blob = f"{model_id}/model.json"

        if not self.exist_in_shared_cache(model_config_blob):
            logger.info("Model config %s is not found in shared cache.", model_config_blob)
            return None

        try:
            logger.info("Downloading %s to %s", model_config_blob, model_json_path)
            self.container_client_factory.download_blob(model_config_blob, model_json_path)
            with open(model_json_path) as file:
                return json.load(file)
        except Exception:
            logger.exception("Failed to load model from shared cache.")
            return None

    def download_model(self, input_model_config: ModelConfig, input_model_id: str, model_saved_path: str):
        if not self.exist_in_shared_cache(input_model_id):
            logger.error("Model %s is not found in the shared cache.", input_model_id)
            raise ValueError(f"Model {input_model_id} is not found in the shared cache.")

        input_model_config.config.pop("shared_cache")
        model_path = input_model_config.config.get("model_path")
        adapter_path = input_model_config.config.get("adapter_path")
        return self.update_model_config(model_path, adapter_path, input_model_config, input_model_id, model_saved_path)

    def update_model_config(
        self,
        shared_model_path: str,
        shared_adapter_path: str,
        model_config: ModelConfig,
        input_model_id: str,
        output_model_path: str,
    ) -> ModelConfig:
        """Download model files from shared cache and update model config.

        Args:
            shared_model_path (str): Model path stored in shared cache
            shared_adapter_path (str): Adapter path stored in shared cache
            model_config (ModelConfig): Model config to be updated
            input_model_id (str): Input model hash
            output_model_path (Path): Output model path

        Returns:
            ModelConfig: updated model config

        """
        logger.debug("Updating model config with shared model path: %s", shared_model_path)
        output_model_path = Path(output_model_path) / "model"

        model_directory_prefix = f"{input_model_id}/model/model"
        blob_list = self.container_client_factory.get_blob_list(model_directory_prefix)
        self._download_blob_list(blob_list, model_directory_prefix, output_model_path)

        adapter_directory_prefix = f"{input_model_id}/model/adapter"
        blob_list = self.container_client_factory.get_blob_list(adapter_directory_prefix)
        self._download_blob_list(blob_list, adapter_directory_prefix, output_model_path, "adapter")

        additional_files_directory_prefix = f"{input_model_id}/model/additional_files"
        additional_files_blob_list = self.container_client_factory.get_blob_list(additional_files_directory_prefix)
        self._download_blob_list(
            additional_files_blob_list, additional_files_directory_prefix, output_model_path, "additional_files"
        )

        if model_config.type.lower() == "hfmodel" and hf_repo_exists(shared_model_path):
            model_config.config["model_path"] = shared_model_path
        else:
            model_config.config["model_path"] = str(output_model_path / shared_model_path)

        if model_config.type.lower() == "hfmodel" and shared_adapter_path:
            model_config.config["adapter_path"] = str(output_model_path / shared_adapter_path)

        additional_files_path = output_model_path / "additional_files"
        if additional_files_path.exists():
            additional_files = [str(file) for file in additional_files_path.iterdir()]
            model_config.config["model_attributes"]["additional_files"] = additional_files

        return model_config

    def exist_in_shared_cache(self, blob_name: str) -> bool:
        logger.debug("Checking shared cache for: %s", blob_name)
        try:
            return self.container_client_factory.exists(blob_name)
        except Exception:
            logger.exception("Failed to check shared cache for %s.", blob_name)
            return False

    def upload_model_files(self, model_path: str, model_blob: str):
        if model_path:
            model_path = Path(model_path)
            # if HF model, model_path is a repo name, no need to upload
            if model_path.exists():
                if not model_path.is_dir():
                    self._upload_file_to_blob(model_path, model_blob)
                else:
                    self._upload_dir_to_blob(model_path, model_blob)

    def _upload_dir_to_blob(self, dir_path: Path, blob_folder_name: str):
        for item in dir_path.iterdir():
            if item.is_dir():
                self._upload_dir_to_blob(item, f"{blob_folder_name}/{item.name}")
            else:
                blob_name = f"{blob_folder_name}/{item.name}"
                self._upload_file_to_blob(item, blob_name)

    def _upload_file_to_blob(self, file_path: Path, blob_name: str):
        logger.info("Uploading %s to %s", file_path, blob_name)
        with open(file_path, "rb") as data:
            self.container_client_factory.upload_blob(blob_name, data)

    def _download_blob_list(
        self, blob_list, directory_prefix: str, output_model_path: Path, prefix: str = None
    ) -> None:
        for blob in blob_list:
            local_file_path = (
                output_model_path / prefix / blob.name[len(directory_prefix) + 1 :]
                if prefix
                else output_model_path / blob.name[len(directory_prefix) + 1 :]
            )
            logger.info("Downloading %s to %s", blob.name, local_file_path)
            self.container_client_factory.download_blob(blob, local_file_path)
