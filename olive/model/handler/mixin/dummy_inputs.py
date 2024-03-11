# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import olive.data.template as data_config_template
from olive.common.user_module_loader import UserModuleLoader

logger = logging.getLogger(__name__)


class DummyInputsMixin:
    """Provide the dummy inputs functionality for the model handler.

    the dummy data is used to evaluate the latency if user doesn't provide the data for evaluation.
    """

    def get_dummy_inputs(self):
        """Return a dummy input for the model."""
        if self.dummy_inputs is not None:
            return self.dummy_inputs

        # Priority: dummy_inputs_func > io_config.input_shapes > hf_config.dataset > onnx_config
        dummy_inputs = None
        # resolved self.io_config
        # won't use self.get_io_config() since we don't want hf_config to be used
        resolved_io_config = self.get_user_io_config(self.io_config) or {}
        if self.dummy_inputs_func is not None:
            logger.debug("Using dummy_inputs_func to get dummy inputs")
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            dummy_inputs = user_module_loader.call_object(self.dummy_inputs_func, self)
        elif resolved_io_config.get("input_shapes"):
            logger.debug("Using io_config.input_shapes to get dummy inputs")
            dummy_inputs, _ = (
                # input_types is optional
                data_config_template.dummy_data_config_template(
                    input_shapes=resolved_io_config["input_shapes"],
                    input_types=resolved_io_config.get("input_types"),
                )
                .to_data_container()
                .get_first_batch(data_root_path=None)
            )
        elif self.hf_config and self.hf_config.model_name and self.hf_config.task:
            # need both model_name and task to get dummy inputs
            if self.hf_config.dataset:
                logger.debug("Using hf_config.dataset to get dummy inputs")
                dummy_inputs, _ = (
                    data_config_template.huggingface_data_config_template(
                        self.hf_config.model_name,
                        self.hf_config.task,
                        **self.hf_config.dataset,
                    )
                    .to_data_container()
                    .get_first_batch(data_root_path=None)
                )
            elif not self.hf_config.components:
                logger.debug("Using hf onnx_config to get dummy inputs")
                dummy_inputs = self.get_hf_dummy_inputs()

        if dummy_inputs is None:
            raise ValueError(
                "Unable to get dummy inputs. Please provide dummy_inputs_func, io_config.input_shapes,"
                " hf_config.dataset, or hf_config."
            )

        return dummy_inputs
