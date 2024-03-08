# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from olive.common.config_utils import validate_config
from olive.systems.common import AcceleratorConfig, SystemType

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric, MetricResult
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
        # the self._accelerators is not used by now, but it is kept for future use.
        if accelerators:
            self._accelerators = [validate_config(accelerator, AcceleratorConfig) for accelerator in accelerators]
        else:
            self._accelerators = None

        self.hf_token = hf_token

    @abstractmethod
    def run_pass(
        self,
        the_pass: "Pass",
        model_config: "ModelConfig",
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> "ModelConfig":
        """Run the pass on the model at a specific point in the search space."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(
        self, model_config: "ModelConfig", data_root: str, metrics: List["Metric"], accelerator: "AcceleratorSpec"
    ) -> "MetricResult":
        """Evaluate the model."""
        raise NotImplementedError

    @abstractmethod
    def remove(self):
        """Remove the system."""
        raise NotImplementedError
