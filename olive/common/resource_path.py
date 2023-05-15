# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
import shutil
import tempfile
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Type, Union
from urllib import request
from urllib.parse import urlparse

from pydantic import Field, validator

from olive.common.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase, validate_config
from olive.common.utils import retry_func

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    LocalFile = "file"
    LocalFolder = "folder"
    URL = "url"
    StringName = "string_name"
    AzureMLModel = "azureml_model"
    AzureMLDatastore = "azureml_datastore"
    AzureMLJobOutput = "azureml_job_output"
    AzureStorageBlob = "azure_blob"

    def __str__(self) -> str:
        return self.value


_resource_config_registry: Dict[ResourceType, Type[ConfigBase]] = {}


class ResourceConfig(ConfigBase):
    _type: ResourceType = None

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """Register the resource config."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and cls._type is not None:
            _resource_config_registry[cls._type] = cls

    @abstractmethod
    def get_path(self) -> str:
        """Return the resource path as a string."""
        raise NotImplementedError

    @abstractmethod
    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        """Save the resource to a directory."""
        raise NotImplementedError


class ResourcePath(ConfigBase):
    """Path to a resource."""

    type: ResourceType = Field(..., description="Type of the resource.")
    config: ResourceConfig = Field(..., description="Config of the resource.")

    @validator("config", pre=True)
    def validate_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type.")

        config_class = _resource_config_registry.get(values["type"])
        return validate_config(v, config_class)

    @classmethod
    def create_resource_path(cls, resource_path: Union[str, Path, Dict[str, Any]]) -> "ResourcePath":
        """
        Create a resource path from a string or a dict.
        If a string is provided, it is inferred to be a file, folder, URL, or string name.
        If a Path is provided, it is inferred to be a file or folder.
        If a dict is provided, it must have "type" and "config" fields. The "type" field must be one of the
        values in the ResourceType enum. The "config" field must be a dict that can be used to create a resource
        config of the specified type.

        :param resource_path: A string, a Path, or a dict.
        :return: A resource path.
        """
        if isinstance(resource_path, dict):
            return cls(**resource_path)

        type: ResourceType = None
        config_key = "path"
        if isinstance(resource_path, Path):
            # check if the resource path is a file or a folder
            if resource_path.is_file():
                type = ResourceType.LocalFile
            elif resource_path.is_dir():
                type = ResourceType.LocalFolder
            else:
                raise ValueError(f"Resource path {resource_path} of type Path is not a file or folder.")
        elif isinstance(resource_path, str):
            # check if the resource path is a file, folder, URL, or a string name
            if Path(resource_path).is_file():
                type = ResourceType.LocalFile
            elif Path(resource_path).is_dir():
                type = ResourceType.LocalFolder
            elif urlparse(resource_path).scheme and urlparse(resource_path).netloc:
                type = ResourceType.URL
                config_key = "url"
            else:
                type = ResourceType.StringName
                config_key = "name"
        else:
            raise ValueError(f"Resource path {resource_path} is not a string, Path or dict.")

        logger.debug(f"Resource path {resource_path} is inferred to be of type {type}.")
        return cls(type=type, config={config_key: resource_path})

    def get_path(self) -> str:
        """Return the resource path as a string."""
        return self.config.get_path()

    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        """Save the resource to a directory."""
        return self.config.save_to_dir(dir_path, overwrite)


def _overwrite_helper(new_path: Union[Path, str], overwrite: bool):
    new_path = Path(new_path).resolve()

    # check if the resource already exists
    if new_path.exists():
        if not overwrite:
            # raise an error if the resource already exists
            raise FileExistsError(f"File {new_path} already exists and overwrite is set to False.")
        else:
            # delete the resource if it already exists
            if new_path.is_file():
                new_path.unlink()
            else:
                shutil.rmtree(new_path)


class LocalResourceConfig(ResourceConfig):
    path: Union[Path, str] = Field(..., description="Path to the resource.")

    def get_path(self) -> str:
        return str(Path(self.path).resolve())

    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        # directory to save the resource to
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # path to save the resource to
        new_path = dir_path / Path(self.path).name
        _overwrite_helper(new_path, overwrite)

        # is the resource a file or a folder
        is_file = Path(self.path).is_file()
        # copy the resource to the new path
        if is_file:
            shutil.copy(self.path, new_path)
        else:
            shutil.copytree(self.path, new_path)

        return str(new_path)

    @validator("path")
    def path_must_exist(cls, v: Union[Path, str]) -> Union[Path, str]:
        """Validate that the path exists."""
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        return path


class LocalFileConfig(LocalResourceConfig):
    _type = ResourceType.LocalFile

    @validator("path")
    def path_must_be_file(cls, v: Union[Path, str]) -> Union[Path, str]:
        """Validate that the path is a file."""
        path = Path(v)
        if not path.is_file():
            raise ValueError(f"Path {path} is not a file.")
        return path


class LocalFolderConfig(LocalResourceConfig):
    _type = ResourceType.LocalFolder

    @validator("path")
    def path_must_be_folder(cls, v: Union[Path, str]) -> Union[Path, str]:
        """Validate that the path is a folder."""
        path = Path(v)
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a folder.")
        return path


class URLConfig(ResourceConfig):
    _type = ResourceType.URL
    url: str = Field(..., description="URL to the resource.")

    def get_path(self) -> str:
        return self.url

    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        # directory to save the resource to
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # path to save the resource to
        new_path = dir_path / Path(self.url).name
        _overwrite_helper(new_path, overwrite)

        # download the resource to the new path
        try:
            logger.debug(f"Downloading resource from {self.url} to {new_path}.")
            request.urlretrieve(self.url, new_path)
        except Exception as e:
            logger.error(f"Failed to download resource from {self.url} to {new_path}.")
            raise e

        return str(new_path)


class StringNameConfig(ResourceConfig):
    _type = ResourceType.StringName
    name: str = Field(..., description="Name of the resource.")

    def get_path(self) -> str:
        return self.name

    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        raise NotImplementedError("Saving a string name resource is not supported.")


def _get_azureml_resource_prefix(workspace_config: Dict[str, str]) -> str:
    return (
        f"azureml://subscriptions/{workspace_config['subscription_id']}"
        f"/resourcegroups/{workspace_config['resource_group']}"
        f"/workspaces/{workspace_config['workspace_name']}"
    )


class AzureMLModelConfig(ResourceConfig):
    _type = ResourceType.AzureMLModel
    aml_client: AzureMLClientConfig = Field(..., description="AzureML client config.")
    name: str = Field(..., description="Name of the model.")
    version: int = Field(..., description="Version of the model.")

    def get_path(self) -> str:
        return f"azureml:{self.name}:{self.version}"

    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        # directory to save the resource to
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # azureml client
        ml_client = self.aml_client.create_client().ml_client

        # azureml model
        model = ml_client.models.get(self.name, self.version)
        model_path = Path(model.path)

        # path to save the resource to
        new_path = dir_path / self.name / str(self.version) / model_path.name
        _overwrite_helper(new_path, overwrite)

        # download the resource to the new path
        try:
            logger.debug(f"Downloading model {self.name} version {self.version} to {new_path}.")
            from azure.core.exceptions import ServiceResponseError

            with tempfile.TemporaryDirectory(dir=dir_path, prefix="olive_tmp") as temp_dir:
                temp_dir = Path(temp_dir)
                retry_func(
                    ml_client.models.download,
                    [self.name],
                    {"version": self.version, "download_path": temp_dir},
                    max_tries=3,
                    delay=5,
                    exceptions=ServiceResponseError,
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(temp_dir / self.name / model_path.name, new_path)
        except Exception as e:
            logger.error(f"Failed to download model {self.name} version {self.version} to {new_path}.")
            raise e

        return str(new_path)


class AzureMLDatastoreConfig(ResourceConfig):
    _type = ResourceType.AzureMLDatastore
    aml_client: AzureMLClientConfig = Field(..., description="AzureML client config.")
    datastore_name: str = Field(..., description="Name of the datastore.")
    relative_path: str = Field(..., description="Relative path to the resource from the datastore root.")

    def get_path(self) -> str:
        workspace_config = self.aml_client.get_workspace_config()
        return (
            f"{_get_azureml_resource_prefix(workspace_config)}"
            f"/datastores/{self.datastore_name}/paths/{self.relative_path}"
        )

    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        raise NotImplementedError("Saving AzureML datastore is not supported.")


class AzureMLJobOutputConfig(ResourceConfig):
    _type = ResourceType.AzureMLJobOutput
    aml_client: AzureMLClientConfig = Field(..., description="AzureML client config.")
    job_name: str = Field(..., description="Name of the job.")
    output_name: str = Field(..., description="Name of the output.")
    relative_path: str = Field(..., description="Relative path to the resource from the output root.")

    def get_path(self) -> str:
        return f"azureml://jobs/{self.job_name}/outputs/{self.output_name}/paths/{self.relative_path}"

    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        # directory to save the resource to
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # path to save the resource to
        new_path = dir_path / self.job_name / self.output_name / self.relative_path
        _overwrite_helper(new_path, overwrite)

        # download the resource to the new path
        ml_client = self.aml_client.create_client().ml_client
        try:
            logger.debug(f"Downloading job output {self.job_name} output {self.output_name} to {new_path}.")
            from azure.core.exceptions import ServiceResponseError

            with tempfile.TemporaryDirectory(dir=dir_path, prefix="olive_tmp") as temp_dir:
                temp_dir = Path(temp_dir)
                retry_func(
                    ml_client.jobs.download,
                    [self.job_name],
                    {"output_name": self.output_name, "download_path": temp_dir},
                    max_tries=3,
                    delay=5,
                    exceptions=ServiceResponseError,
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(temp_dir / "named-outputs" / self.output_name / self.relative_path, new_path)
        except Exception as e:
            logger.error(f"Failed to download job output {self.job_name} output {self.output_name} to {new_path}.")
            raise e

        return str(new_path)


class AzureStorageBlob(ResourceConfig):
    _type = ResourceType.AzureStorageBlob
    account_name: str = Field(..., description="Name of the storage account.")
    container_name: str = Field(..., description="Name of the container.")
    relative_path: str = Field(..., description="Relative path to the resource from the container root.")

    def get_path(self) -> str:
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{self.relative_path}"

    def save_to_dir(self, dir_path: Union[Path, str], overwrite: bool = False) -> str:
        return NotImplementedError("Saving Azure Storage Blob is not supported.")
