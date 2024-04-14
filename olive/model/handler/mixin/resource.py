# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, ResourcePath, ResourcePathConfig, create_resource_path

logger = logging.getLogger(__name__)


class ResourceMixin:
    """Provide the resource functionalities for the model handler."""

    resource_keys: Tuple[str, ...] = None

    def add_resources(self, local_args: Dict[str, str]):
        resources = {r: local_args[r] for r in self.resource_keys if r in local_args}
        self._add_resources(resources)

    def set_resource(self, resource_name: str, resource_path: Union[Path, str, ResourcePath, ResourcePathConfig]):
        """Set resource path.

        :param resource_name: name of the resource.
        :param resource_path: resource path.
        """
        if resource_name not in self.resource_paths:
            raise ValueError(f"{resource_name} is not a valid resource name.")
        if self.resource_paths[resource_name]:
            logger.debug(
                "Overriding %s from %s to %s.", resource_name, self.resource_paths[resource_name], resource_path
            )

        if resource_path is not None:
            resolved_resource_path = create_resource_path(resource_path)
            assert (
                resolved_resource_path.is_local_resource_or_string_name()
            ), f"{resource_name} must be local path or string name."
            resource_path = resolved_resource_path.get_path()

        self.resource_paths[resource_name] = resource_path

    def get_resource(self, resource_name: str) -> str:
        """Get local path of a resource.

        :param resource_name: name of the resource.
        :return: local path.
        """
        assert resource_name in self.resource_paths, f"{resource_name} is not a valid resource name."
        resource = self.resource_paths[resource_name]
        assert resource is None or isinstance(resource, str)
        return resource

    @classmethod
    def get_resource_keys(cls) -> Tuple[str]:
        """Get all resource keys.

        :return: all resource keys.
        """
        return cls.resource_keys

    def _add_resources(self, resources: Dict[str, OLIVE_RESOURCE_ANNOTATIONS]):
        for resource_name, resource_path in resources.items():
            if resource_path is not None:
                resolved_resource_path = create_resource_path(resource_path)
                assert (
                    resolved_resource_path.is_local_resource_or_string_name()
                ), f"{resource_name} must be local path or string name."
                self.resource_paths[resource_name] = resolved_resource_path.get_path()
            else:
                self.resource_paths[resource_name] = None
