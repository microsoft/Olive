# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
import shutil
import tempfile
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Type, Union

from pydantic import Field, validator

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.auto_config import AutoConfigClass
from olive.common.config_utils import ConfigBase, ConfigParam, serialize_to_json, validate_config
from olive.common.utils import retry_func

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    LocalFile = "file"
    LocalFolder = "folder"
    StringName = "string_name"
    AzureMLModel = "azureml_model"
    AzureMLRegistryModel = "azureml_registry_model"
    AzureMLDatastore = "azureml_datastore"
    AzureMLJobOutput = "azureml_job_output"

    def __str__(self) -> str:
        return self.value


LOCAL_RESOURCE_TYPES = [ResourceType.LocalFile, ResourceType.LocalFolder]
AZUREML_RESOURCE_TYPES = [
    ResourceType.AzureMLModel,
    ResourceType.AzureMLDatastore,
    ResourceType.AzureMLJobOutput,
]


class ResourcePath(AutoConfigClass):
    registry: ClassVar[Dict[str, Type["ResourcePath"]]] = {}
    name: ResourceType = None

    def __repr__(self) -> str:
        return self.get_path()

    @property
    def type(self) -> ResourceType:  # noqa: A003
        return self.name

    @abstractmethod
    def get_path(self) -> str:
        """Return the resource path as a string."""
        raise NotImplementedError

    @abstractmethod
    def save_to_dir(self, dir_path: Union[Path, str], name: str = None, overwrite: bool = False) -> str:
        """Save the resource to a directory."""
        raise NotImplementedError

    def is_local_resource(self) -> bool:
        """Return True if the resource is a local resource."""
        return self.type in LOCAL_RESOURCE_TYPES

    def is_azureml_resource(self) -> bool:
        """Return True if the resource is an AzureML resource."""
        return self.type in AZUREML_RESOURCE_TYPES

    def is_string_name(self) -> bool:
        """Return True if the resource is a string name."""
        return self.type == ResourceType.StringName

    def is_local_resource_or_string_name(self) -> bool:
        """Return True if the resource is a local resource or a string name."""
        return self.is_local_resource() or self.is_string_name()

    def to_json(self):
        json_data = {"type": self.type, "config": self.config.to_json()}
        return serialize_to_json(json_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourcePath):
            return False
        return self.type == other.type and self.config.to_json() == other.config.to_json()

    def __hash__(self) -> int:
        return hash((self.config.to_json(), self.type))


class ResourcePathConfig(ConfigBase):
    type: ResourceType = Field(..., description="Type of the resource.")  # noqa: A003
    config: ConfigBase = Field(..., description="Config of the resource.")

    @validator("config", pre=True)
    def validate_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type.")

        config_class = ResourcePath.registry[values["type"]].get_config_class()
        return validate_config(v, ConfigBase, config_class)

    def create_resource_path(self) -> ResourcePath:
        return ResourcePath.registry[self.type](self.config)


def create_resource_path(
    resource_path: Optional[Union[str, Path, Dict[str, Any], ResourcePathConfig, ResourcePath]]
) -> Optional[ResourcePath]:
    """Create a resource path from a string or a dict.

    If a string is provided, it is inferred to be a file, folder, or string name.
    If a Path is provided, it is inferred to be a file or folder.
    If a dict is provided, it must have "type" and "config" fields. The "type" field must be one of the
    values in the ResourceType enum. The "config" field must be a dict that can be used to create a resource
    config of the specified type.

    :param resource_path: A string, a Path, or a dict.
    :return: A resource path.
    """
    if resource_path is None:
        return None
    if isinstance(resource_path, ResourcePath):
        return resource_path
    if isinstance(resource_path, list):
        return [create_resource_path(rp) for rp in resource_path]
    if isinstance(resource_path, (ResourcePathConfig, dict)):
        resource_path_config = validate_config(resource_path, ResourcePathConfig)
        return resource_path_config.create_resource_path()

    # check if the resource path is a file, folder, azureml datastore, or a string name
    is_local_file = True
    resource_type: ResourceType = None
    config_key = None
    if Path(resource_path).is_file():
        resource_type = ResourceType.LocalFile
        config_key = "path"
    elif Path(resource_path).is_dir():
        resource_type = ResourceType.LocalFolder
        config_key = "path"
    elif str(resource_path).startswith("azureml://"):
        resource_type = ResourceType.AzureMLDatastore
        config_key = "datastore_url"
        is_local_file = False
    else:
        resource_type = ResourceType.StringName
        config_key = "name"
        is_local_file = False

    if is_local_file and isinstance(resource_path, Path) and not resource_path.exists():
        raise ValueError(f"Resource path {resource_path} of type Path does not exist.")

    logger.debug(f"Resource path {resource_path} is inferred to be of type {resource_type}.")
    return ResourcePathConfig(type=resource_type, config={config_key: resource_path}).create_resource_path()


def _overwrite_helper(new_path: Union[Path, str], overwrite: bool):
    new_path = Path(new_path).resolve()

    # check if the resource already exists
    if new_path.exists():
        if not overwrite:
            # raise an error if the file/folder with same name already exists and overwrite is set to False
            # Olive doesn't know if the existing file/folder is the same as the one being saved
            # or if the user wants to overwrite the existing file/folder
            raise FileExistsError(
                f"Trying to save resource to {new_path} but a file/folder with the same name already exists and"
                " overwrite is set to False. If you want to overwrite the existing file/folder, set overwrite to True."
            )
        else:
            # delete the resource if it already exists
            if new_path.is_file():
                new_path.unlink()
            else:
                shutil.rmtree(new_path)


def _validate_path(v):
    if not Path(v).exists():
        raise ValueError(f"Path {v} does not exist.")
    return Path(v).resolve()


class LocalResourcePath(ResourcePath):
    """Base class for a local resource path."""

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "path": ConfigParam(type_=Union[Path, str], required=True, description="Path to the resource."),
        }

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        return {"validate_path": validator("path", allow_reuse=True)(_validate_path)}

    def get_path(self) -> str:
        return str(self.config.path)

    def save_to_dir(self, dir_path: Union[Path, str], name: str = None, overwrite: bool = False) -> str:
        # directory to save the resource to
        dir_path = Path(dir_path).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        # path to save the resource to
        if name:
            new_path_name = Path(name).with_suffix(self.config.path.suffix).name
        else:
            new_path_name = self.config.path.name
        new_path = dir_path / new_path_name
        _overwrite_helper(new_path, overwrite)

        # is the resource a file or a folder
        is_file = Path(self.config.path).is_file()
        # copy the resource to the new path
        if is_file:
            shutil.copy(self.config.path, new_path)
        else:
            shutil.copytree(self.config.path, new_path)

        return str(new_path)


def _validate_file_path(v):
    path = Path(v)
    if not path.is_file():
        raise ValueError(f"Path {path} is not a file.")
    return path


class LocalFile(LocalResourcePath):
    """Local file resource path."""

    name = ResourceType.LocalFile

    @staticmethod
    def _validators() -> Dict[str, Callable[..., Any]]:
        validators = LocalResourcePath._validators()
        validators.update({"validate_file_path": validator("path", allow_reuse=True)(_validate_file_path)})
        return validators


def _validate_folder_path(v):
    path = Path(v)
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a folder.")
    return path


class LocalFolder(LocalResourcePath):
    """Local folder resource path."""

    name = ResourceType.LocalFolder

    @staticmethod
    def _validators() -> Dict[str, Callable[..., Any]]:
        validators = LocalResourcePath._validators()
        validators.update({"validate_folder_path": validator("path", allow_reuse=True)(_validate_folder_path)})
        return validators


class StringName(ResourcePath):
    """String name resource path."""

    name = ResourceType.StringName

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "name": ConfigParam(type_=str, required=True, description="Name of the resource."),
        }

    def get_path(self) -> str:
        return self.config.name

    def save_to_dir(self, dir_path: Union[Path, str], name: str = None, overwrite: bool = False) -> str:
        logger.debug("Resource is a string name. No need to save to directory.")
        return self.config.name


def _get_azureml_resource_prefix(workspace_config: Dict[str, str]) -> str:
    return (
        f"azureml://subscriptions/{workspace_config['subscription_id']}"
        f"/resourcegroups/{workspace_config['resource_group']}"
        f"/workspaces/{workspace_config['workspace_name']}"
    )


class AzureMLResource(ResourcePath):
    """Base class for AzureML resource paths."""

    @abstractmethod
    def get_path(self) -> str:
        raise NotImplementedError

    def save_to_dir(
        self,
        dir_path: Union[Path, str],
        ml_client,
        azureml_client_config: AzureMLClientConfig,
        overwrite: bool,
        name: str = None,
    ) -> str:
        # directory to save the resource to
        dir_path = Path(dir_path).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        # azureml model
        model = ml_client.models.get(self.config.name, version=self.config.version)
        model_path = Path(model.path)

        # path to save the resource to
        if name:
            new_path_name = Path(name).with_suffix(model_path.suffix).name
        else:
            new_path_name = model_path.name
        new_path = dir_path / new_path_name
        _overwrite_helper(new_path, overwrite)

        # download the resource to the new path
        logger.debug(f"Downloading model {self.config.name} version {self.config.version} to {new_path}.")
        from azure.core.exceptions import ServiceResponseError

        with tempfile.TemporaryDirectory(dir=dir_path, prefix="olive_tmp") as tempdir:
            temp_dir = Path(tempdir)
            retry_func(
                ml_client.models.download,
                [self.config.name],
                {"version": self.config.version, "download_path": temp_dir},
                max_tries=azureml_client_config.max_operation_retries,
                delay=azureml_client_config.operation_retry_interval,
                exceptions=ServiceResponseError,
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_dir / self.config.name / model_path.name, new_path)
        return str(new_path)


class AzureMLModel(AzureMLResource):
    """AzureML Model resource path."""

    name = ResourceType.AzureMLModel

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "azureml_client": ConfigParam(
                type_=AzureMLClientConfig, required=True, description="AzureML client config."
            ),
            "name": ConfigParam(type_=str, required=True, description="Name of the model."),
            "version": ConfigParam(type_=Union[int, str], required=True, description="Version of the model."),
        }

    def get_path(self) -> str:
        return f"azureml:{self.config.name}:{self.config.version}"

    def get_aml_client_config(self) -> AzureMLClientConfig:
        return self.config.azureml_client

    def save_to_dir(self, dir_path: Union[Path, str], name: str = None, overwrite: bool = False) -> str:
        ml_client = self.config.azureml_client.create_client()
        return super().save_to_dir(dir_path, ml_client, self.config.azureml_client, overwrite, name)


class AzureMLRegistryModel(AzureMLResource):
    """AzureML Model resource path."""

    name = ResourceType.AzureMLRegistryModel

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "azureml_client": ConfigParam(
                type_=AzureMLClientConfig, required=False, description="AzureML client config."
            ),
            "registry_name": ConfigParam(type_=str, required=True, description="Name of the registry."),
            "name": ConfigParam(type_=str, required=True, description="Name of the model."),
            "version": ConfigParam(type_=Union[int, str], required=True, description="Version of the model."),
        }

    def get_path(self) -> str:
        return (
            f"azureml://registries/{self.config.registry_name}/models/{self.config.name}/versions/{self.config.version}"
        )

    def save_to_dir(self, dir_path: Union[Path, str], name: str = None, overwrite: bool = False) -> str:
        # azureml client
        azureml_client_config = self.config.azureml_client or AzureMLClientConfig()

        ml_client = azureml_client_config.create_registry_client(self.config.registry_name)
        return super().save_to_dir(dir_path, ml_client, azureml_client_config, overwrite, name)


def _datastore_url_validator(v, values, **kwargs):
    aml_info_ready = all([values.get("azureml_client"), values.get("datastore_name"), values.get("relative_path")])
    if not v and not aml_info_ready:
        raise ValueError(
            "If datastore_url is not specified, then azureml_client, datastore_name, and relative_path "
            "must be specified."
        )
    elif v and aml_info_ready:
        logger.warning("datastore_url is specified. azureml_client, datastore_name, and relative_path are ignored.")

    if v and not v.startswith("azureml:"):
        raise ValueError(f"Datastore URL {v} is not a valid AzureML datastore URL.")
    return v


class AzureMLDatastore(ResourcePath):
    """AzureML DataStore resource path."""

    name = ResourceType.AzureMLDatastore

    @staticmethod
    def _validators() -> Dict[str, Callable[..., Any]]:
        validators = ResourcePath._validators()
        validators.update(
            {
                "validate_datastore_url": validator("datastore_url", allow_reuse=True)(_datastore_url_validator),
            }
        )
        return validators

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "azureml_client": ConfigParam(type_=AzureMLClientConfig, description="AzureML client config."),
            "datastore_name": ConfigParam(type_=str, description="Name of the datastore."),
            "relative_path": ConfigParam(type_=str, description="Relative path to the resource."),
            "datastore_url": ConfigParam(type_=str, description="URL of the datastore."),
        }

    def get_path(self) -> str:
        if self.config.datastore_url:
            return self.config.datastore_url

        workspace_config = self.config.azureml_client.get_workspace_config()
        return (
            f"{_get_azureml_resource_prefix(workspace_config)}"
            f"/datastores/{self.config.datastore_name}/paths/{self.config.relative_path}"
        )

    def is_file(self, fsspec=None) -> bool:
        try:
            from azureml.fsspec import AzureMachineLearningFileSystem
        except ImportError:
            raise ImportError(
                "azureml-fsspec is not installed. Please install azureml-fsspec to use AzureMLDatastore resource path."
            ) from None
        if fsspec is None:
            fsspec = AzureMachineLearningFileSystem(self.get_path())
        return fsspec.info(self.get_relative_path()).get("type") == "file"

    def get_relative_path(self) -> str:
        if self.config.datastore_url:
            return re.split("/datastores/.*/paths/", self.config.datastore_url)[-1]
        return self.config.relative_path

    def get_aml_client_config(self) -> AzureMLClientConfig:
        if self.config.datastore_url:
            subscription_id = re.split("/subscriptions/", self.config.datastore_url)[-1].split("/")[0]
            resource_group = re.split("/resourcegroups/", self.config.datastore_url)[-1].split("/")[0]
            workspace_name = re.split("/workspaces/", self.config.datastore_url)[-1].split("/")[0]
            return AzureMLClientConfig(
                subscription_id=subscription_id,
                resource_group=resource_group,
                workspace_name=workspace_name,
            )
        return self.config.azureml_client

    def save_to_dir(self, dir_path: Union[Path, str], name: str = None, overwrite: bool = False) -> str:
        # there is no direct way to download a file from a datastore
        # so we will use a workaround to download the file by creating a aml model
        # that references the file and downloading the model
        try:
            from azureml.fsspec import AzureMachineLearningFileSystem
        except ImportError:
            raise ImportError(
                "azureml-fsspec is not installed. Please install azureml-fsspec to use AzureMLDatastore resource path."
            ) from None

        azureml_client_config = self.get_aml_client_config()

        # azureml file system
        fs = AzureMachineLearningFileSystem(self.get_path())
        relative_path = Path(self.get_relative_path())
        is_file = self.is_file(fs)
        # path to save the resource to
        if name:
            new_path_name = Path(name).with_suffix("" if not is_file else relative_path.suffix).name
        else:
            new_path_name = relative_path.name

        dir_path = Path(dir_path).resolve()
        new_path = dir_path / new_path_name
        _overwrite_helper(new_path, overwrite)

        dir_path.mkdir(parents=True, exist_ok=True)

        # download artifacts to a temporary directory
        logger.debug(f"Downloading aml resource for datastore {self.config.datastore_name} path {relative_path}.")

        with tempfile.TemporaryDirectory(dir=dir_path, prefix="olive_tmp") as temp_dir:
            retry_func(
                fs.download,
                kwargs={
                    "rpath": self.get_relative_path(),
                    "lpath": temp_dir,
                    "recursive": not is_file,
                },
                max_tries=azureml_client_config.max_operation_retries,
                delay=azureml_client_config.operation_retry_interval,
            )
            downloaded_resource = Path(temp_dir) / relative_path.name
            # only if the resource is a existed file we will move it to the new path
            source_path = downloaded_resource if is_file else temp_dir
            shutil.move(source_path, new_path)

        return str(Path(new_path).resolve())


class AzureMLJobOutput(ResourcePath):
    """AzureML job output resource path."""

    name = ResourceType.AzureMLJobOutput

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "azureml_client": ConfigParam(
                type_=AzureMLClientConfig, required=True, description="AzureML client config."
            ),
            "job_name": ConfigParam(type_=str, required=True, description="Name of the job."),
            "output_name": ConfigParam(type_=str, required=True, description="Name of the output."),
            "relative_path": ConfigParam(type_=str, required=True, description="Relative path to the resource."),
        }

    def get_path(self) -> str:
        return (
            f"azureml://jobs/{self.config.job_name}/outputs/{self.config.output_name}/paths/{self.config.relative_path}"
        )

    def save_to_dir(self, dir_path: Union[Path, str], name: str = None, overwrite: bool = False) -> str:
        # directory to save the resource to
        dir_path = Path(dir_path).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        # path to save the resource to
        if name:
            new_path_name = Path(name).with_suffix(Path(self.config.relative_path).suffix).name
        else:
            new_path_name = Path(self.config.relative_path).name
        new_path = dir_path / new_path_name
        _overwrite_helper(new_path, overwrite)

        # download the resource to the new path
        ml_client = self.config.azureml_client.create_client()
        logger.debug(f"Downloading job output {self.config.job_name} output {self.config.output_name} to {new_path}.")
        from azure.core.exceptions import ServiceResponseError

        with tempfile.TemporaryDirectory(dir=dir_path, prefix="olive_tmp") as tempdir:
            temp_dir = Path(tempdir)
            retry_func(
                ml_client.jobs.download,
                [self.config.job_name],
                {"output_name": self.config.output_name, "download_path": temp_dir},
                max_tries=self.config.azureml_client.max_operation_retries,
                delay=self.config.azureml_client.operation_retry_interval,
                exceptions=ServiceResponseError,
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_dir / "named-outputs" / self.config.output_name / self.config.relative_path, new_path)
        return str(new_path)


OLIVE_RESOURCE_ANNOTATIONS = Optional[Union[str, Path, ResourcePath, ResourcePathConfig]]
