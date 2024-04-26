# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from typing import Optional, Union

from olive.common.config_utils import ConfigBase, validate_config
from olive.common.pydantic_v1 import validator


class PackagingType(str, Enum):
    """Output Artifacts type."""

    Zipfile = "Zipfile"
    AzureMLModels = "AzureMLModels"
    AzureMLData = "AzureMLData"
    AzureMLDeployment = "AzureMLDeployment"


class CommonPackagingConfig(ConfigBase):
    export_in_mlflow_format: bool = False


class ZipfilePackagingConfig(CommonPackagingConfig):
    pass


class AzureMLDataPackagingConfig(CommonPackagingConfig):
    version: Union[int, str] = "1"
    description: Optional[str] = None


class AzureMLModelsPackagingConfig(CommonPackagingConfig):
    version: Union[int, str] = "1"
    description: Optional[str] = None


class InferencingServerType(str, Enum):
    AzureMLOnline = "AzureMLOnline"
    AzureMLBatch = "AzureMLBatch"


class InferenceServerConfig(ConfigBase):
    type: InferencingServerType
    code_folder: str
    scoring_script: str


class AzureMLModelModeType(str, Enum):
    download = "download"
    copy = "copy"


class ModelConfigurationConfig(ConfigBase):
    mode: AzureMLModelModeType = AzureMLModelModeType.download
    mount_path: Optional[str] = None  # Relative mount path


class ModelPackageConfig(ConfigBase):
    target_environment: str = "olive-target-environment"
    target_environment_version: Optional[str] = None
    inferencing_server: InferenceServerConfig
    base_environment_id: str
    model_configurations: ModelConfigurationConfig = None
    environment_variables: Optional[dict] = None


class DeploymentConfig(ConfigBase):
    endpoint_name: str = "olive-default-endpoint"
    deployment_name: str = "olive-default-deployment"
    instance_type: Optional[str] = None
    compute: Optional[str] = None
    instance_count: int = 1
    mini_batch_size: int = 10  # AzureMLBatch only
    extra_config: Optional[dict] = None


class AzureMLDeploymentPackagingConfig(CommonPackagingConfig):
    model_name: str = "olive-deployment-model"
    model_version: Union[int, str] = "1"
    model_description: Optional[str] = None
    model_package: ModelPackageConfig
    deployment_config: DeploymentConfig = DeploymentConfig()


_type_to_config = {
    PackagingType.Zipfile: ZipfilePackagingConfig,
    PackagingType.AzureMLModels: AzureMLModelsPackagingConfig,
    PackagingType.AzureMLData: AzureMLDataPackagingConfig,
    PackagingType.AzureMLDeployment: AzureMLDeploymentPackagingConfig,
}


class PackagingConfig(ConfigBase):
    """Olive output artifacts generation config."""

    type: PackagingType = PackagingType.Zipfile
    name: str = "OutputModels"
    config: CommonPackagingConfig = None
    include_runtime_packages: bool = True
    include_sample_code: bool = True

    @validator("config", pre=True, always=True)
    def _validate_config(cls, v, values):
        packaging_type = values.get("type")
        config_class = _type_to_config.get(packaging_type)
        return validate_config(v, config_class)
