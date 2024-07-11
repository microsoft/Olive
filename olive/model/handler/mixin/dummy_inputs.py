# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import abstractmethod

import olive.data.template as data_config_template

logger = logging.getLogger(__name__)


class DummyInputsMixin:
    """Provide the dummy inputs functionality for the model handler.

    the dummy data is used to evaluate the latency if user doesn't provide the data for evaluation.
    """

    def _get_dummy_inputs_from_io_config(self, force_kv_cache: bool = False):
        if not self._io_config:
            return None

        resolved_io_config = self.get_resolved_io_config(self._io_config, force_kv_cache=force_kv_cache) or {}
        if not resolved_io_config.get("input_shapes"):
            return None

        logger.debug("Using io_config.input_shapes to build dummy inputs")
        return (
            data_config_template.dummy_data_config_template(
                input_shapes=resolved_io_config["input_shapes"],
                input_types=resolved_io_config.get("input_types"),
                input_names=resolved_io_config.get("input_names"),
            )
            .to_data_container()
            .get_first_batch()
        )[0]

    @abstractmethod
    def get_new_dummy_inputs(self):
        """Return a dummy input for the model."""
        raise NotImplementedError

    def get_dummy_inputs(self, filter_hook=None, filter_hook_kwargs=None):
        """Return a dummy input for the model."""
        if self.dummy_inputs is not None:
            return self.dummy_inputs

        dummy_inputs = self.get_new_dummy_inputs()

        if dummy_inputs is None:
            raise ValueError("Unable to get dummy inputs for the model.")

        if filter_hook:
            dummy_inputs = filter_hook(dummy_inputs, **(filter_hook_kwargs or {}))
        return dummy_inputs
