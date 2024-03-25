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


_type_to_config = {
    PackagingType.Zipfile: ZipfilePackagingConfig,
    PackagingType.AzureMLModels: AzureMLModelsPackagingConfig,
    PackagingType.AzureMLData: AzureMLDataPackagingConfig,
}


class PackagingConfig(ConfigBase):
    """Olive output artifacts generation config."""

    type: PackagingType = PackagingType.Zipfile
    name: str = "OutputModels"
    config: CommonPackagingConfig = None

    @validator("config", pre=True, always=True)
    def _validate_config(cls, v, values):
        packaging_type = values.get("type")
        config_class = _type_to_config.get(packaging_type)
        return validate_config(v, config_class)
