# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Tuple

from olive.common.config_utils import serialize_to_json

logger = logging.getLogger(__name__)


class JsonMixin:
    """Provide the to_json functionality for the model handler.

    Different model handler need to override the behavior to add its own attributes.
    """

    # keys for config parameters that are not part of resource_keys or model_attributes
    # self.{key} must be defined for each key in json_config_keys and should be serializable
    json_config_keys: Tuple[str, ...] = ()

    def to_json(self, check_object: bool = False):
        config = {
            "type": self.model_type,
            "config": {
                # serialize resource paths
                **{
                    resource_name: resource_path if resource_path else None
                    for resource_name, resource_path in self.resource_paths.items()
                },
                # serialize other config attributes
                **{key: getattr(self, key) for key in self.json_config_keys},
                # serialize model attributes
                "model_attributes": self.model_attributes,
            },
        }
        return serialize_to_json(config, check_object)
