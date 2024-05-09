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

    def get_dummy_dataloader_from_io_config(self):
        dataloader = None
        # resolved self.io_config
        # won't use self.io_config since we don't want hf_config to be used
        resolved_io_config = self.get_user_io_config(self.io_config) or {}
        if resolved_io_config.get("input_shapes"):
            logger.debug("Using io_config.input_shapes to get dummy dataloader")
            dataloader = (
                # input_types is optional
                data_config_template.dummy_data_config_template(
                    input_shapes=resolved_io_config["input_shapes"],
                    input_types=resolved_io_config.get("input_types"),
                    input_names=resolved_io_config.get("input_names"),
                ).to_data_container()
            )
        return dataloader

    def get_dummy_dataloader_from_hf_config(self):
        dataloader = None
        if self.hf_config and self.hf_config.model_name and self.hf_config.task and self.hf_config.dataset:
            # need both model_name and task to get dummy inputs
            logger.debug("Using hf_config.dataset to get dummy dataloader")
            dataloader = data_config_template.huggingface_data_config_template(
                self.hf_config.model_name,
                self.hf_config.task,
                **self.hf_config.dataset,
            ).to_data_container()
        return dataloader

    def get_dummy_inputs(self, filter_hook=None, filter_hook_kwargs=None):
        """Return a dummy input for the model."""
        if self.dummy_inputs is not None:
            return self.dummy_inputs

        # Priority: dummy_inputs_func > io_config.input_shapes > hf_config.dataset > onnx_config
        dummy_inputs = None

        if self.dummy_inputs_func is not None:
            logger.debug("Using dummy_inputs_func to get dummy inputs")
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            dummy_inputs = user_module_loader.call_object(self.dummy_inputs_func, self)
            # respect user's dummy_inputs_func, no hook
        else:
            dataloader = self.get_dummy_dataloader_from_io_config() or self.get_dummy_dataloader_from_hf_config()
            if dataloader:
                dummy_inputs, _ = dataloader.get_first_batch()
            elif not self.hf_config.components:
                logger.debug("Trying hf onnx_config to get dummy inputs")
                dummy_inputs = self.get_hf_dummy_inputs()
                if dummy_inputs is not None:
                    logger.debug("Got dummy inputs from hf onnx_config")
            if filter_hook:
                dummy_inputs = filter_hook(dummy_inputs, **(filter_hook_kwargs or {}))

        if dummy_inputs is None:
            raise ValueError(
                "Unable to get dummy inputs. Please provide dummy_inputs_func, io_config.input_shapes,"
                " hf_config.dataset, or hf_config."
            )
        return dummy_inputs
