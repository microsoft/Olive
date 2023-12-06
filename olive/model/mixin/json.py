# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.common.config_utils import serialize_to_json

logger = logging.getLogger(__name__)


class JsonMixin:
    def to_json(self, check_object: bool = False):
        config = {
            "type": self.model_type,
            "config": {
                # serialize resource paths
                resource_name: resource_path if resource_path else None
                for resource_name, resource_path in self.resource_paths.items()
            },
        }
        config["config"].update({"model_attributes": self.model_attributes})
        return serialize_to_json(config, check_object)
