# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Union

from olive.common.config_utils import validate_config
from olive.systems.common import AcceleratorConfig, SystemType

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ModelConfig
    from olive.passes.olive_pass import Pass


logger = logging.getLogger(__name__)


class OliveSystem(ABC):
    system_type: SystemType

    def __init__(
        self,
        accelerators: Union[List[AcceleratorConfig], List[Dict[str, Any]]] = None,
        hf_token: bool = None,
    ):
        # TODO(anyone): Is it possible to expose the arguments to
        # let user set the system environment in Olive config?
        # For example, in some qualcomm cases, the user may need to set
        # SDK root path outside of Olive.
        if accelerators:
            assert all(
                isinstance(validate_config(accelerator, AcceleratorConfig), AcceleratorConfig)
                for accelerator in accelerators
            )

        self.hf_token = hf_token

    @abstractmethod
    def run_pass(self, the_pass: "Pass", model_config: "ModelConfig", output_model_path: str) -> "ModelConfig":
        """Run the pass on the model at a specific point in the search space."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(
        self, model_config: "ModelConfig", evaluator_config: "OliveEvaluatorConfig", accelerator: "AcceleratorSpec"
    ) -> "MetricResult":
        """Evaluate the model."""
        raise NotImplementedError

    @abstractmethod
    def remove(self):
        """Remove the system."""
        raise NotImplementedError
