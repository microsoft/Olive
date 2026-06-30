# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class SaveTestModelConfig(Pass):
    """Saves the HuggingFace model config with a reduced layer count to the test_model_path directory.

    When ``test_model_path`` and ``test_model_config`` are set on the input
    ``HfModelHandler``, this pass creates the target directory and writes
    ``config.json`` (with the modified number of hidden layers) plus the
    Olive test-model marker file.  The model weights are *not* written here;
    a subsequent ``ModelBuilder`` (or any other pass that calls
    ``HfModelHandler.load_model``) will generate and persist them on first
    use.

    The pass is a no-op when neither ``test_model_path`` nor
    ``test_model_config`` is set on the model, and it is idempotent — running
    it a second time on a directory that already has the marker file is safe.

    The input model is returned unchanged.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {}

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        from olive.common.hf.utils import save_test_model_config

        test_model_path = model.test_model_path
        test_model_config = model.test_model_config
        if test_model_path and test_model_config:
            logger.info("Saving test model config to %s", test_model_path)
            save_test_model_config(model.model_name_or_path, test_model_config, test_model_path)
        else:
            logger.debug(
                "SaveTestModelConfig: test_model_path=%r, test_model_config=%r — nothing to save.",
                test_model_path,
                test_model_config,
            )
        return model
