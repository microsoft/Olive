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
    """Saves a random-initialised HuggingFace model to the test_model_path directory.

    When ``test_model_path`` and ``test_model_config`` are set on the input
    ``HfModelHandler``, this pass creates the target directory, writes
    ``config.json`` (with the modified number of hidden layers), the Olive
    test-model marker file, *and* the random model weights (safetensors).

    The pass is a no-op when neither ``test_model_path`` nor
    ``test_model_config`` is set on the model, and it is idempotent — running
    it a second time on a directory that already contains both the marker file
    and model weights is safe.

    The input model is returned unchanged.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {}

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        from pathlib import Path

        from olive.common.hf.utils import is_test_model_dir

        test_model_path = model.test_model_path
        test_model_config = model.test_model_config
        if not (test_model_path and test_model_config):
            logger.debug(
                "SaveTestModelConfig: test_model_path=%r, test_model_config=%r — nothing to save.",
                test_model_path,
                test_model_config,
            )
            return model

        test_model_dir = Path(test_model_path)
        _has_weights = is_test_model_dir(test_model_dir) and (
            any(test_model_dir.glob("*.safetensors")) or any(test_model_dir.glob("pytorch_model*.bin"))
        )
        if _has_weights:
            logger.debug("Test model already saved at %s — skipping.", test_model_path)
            return model

        logger.info("Saving test random model to %s", test_model_path)
        # load_model calls load_model_from_task which creates a random-initialised model
        # from the reduced config and persists it (weights + config.json + marker) to
        # test_model_path on the first call.
        model.load_model(cache_model=False)
        return model
