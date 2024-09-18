# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import olive.data.template as data_config_template

logger = logging.getLogger(__name__)


class DummyInputsMixin:
    """Provide the dummy inputs functionality for the model handler.

    the dummy data is used to evaluate the latency if user doesn't provide the data for evaluation.
    """

    def _get_dummy_inputs_from_io_config(self, filter_hook=None, filter_hook_kwargs=None):
        if not self._io_config:
            return None

        resolved_io_config = self.io_config
        if not resolved_io_config.get("input_shapes"):
            return None

        logger.debug("Using io_config.input_shapes to build dummy inputs")
        dummy_inputs = (
            data_config_template.dummy_data_config_template(
                input_shapes=resolved_io_config["input_shapes"],
                input_types=resolved_io_config.get("input_types"),
                input_names=resolved_io_config.get("input_names"),
            )
            .to_data_container()
            .get_first_batch()
        )[0]

        if filter_hook:
            dummy_inputs = filter_hook(dummy_inputs, **(filter_hook_kwargs or {}))

        return dummy_inputs
