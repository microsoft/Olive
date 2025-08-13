# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Optional

from olive.common.config_utils import CaseInsensitiveEnum, ConfigBase, NestedConfig, validate_config
from olive.common.constants import BASE_IMAGE
from olive.common.pydantic_v1 import validator


class PackagingType(CaseInsensitiveEnum):
    """Output Artifacts type."""

    Zipfile = "Zipfile"
    Dockerfile = "Dockerfile"


class CommonPackagingConfig(ConfigBase):
    export_in_mlflow_format: bool = False


class ZipfilePackagingConfig(CommonPackagingConfig):
    pass


class DockerfilePackagingConfig(CommonPackagingConfig):
    base_image: str = BASE_IMAGE
    requirements_file: Optional[str] = None


_type_to_config = {
    PackagingType.Zipfile: ZipfilePackagingConfig,
    PackagingType.Dockerfile: DockerfilePackagingConfig,
}


class PackagingConfig(NestedConfig):
    """Olive output artifacts generation config."""

    type: PackagingType = PackagingType.Zipfile
    name: str = "OutputModels"
    config: CommonPackagingConfig = None

    @validator("config", pre=True, always=True)
    def _validate_config(cls, v, values):
        packaging_type = values.get("type")
        config_class = _type_to_config.get(packaging_type)
        return validate_config(v, config_class)
