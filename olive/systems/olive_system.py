# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from olive.evaluator.metric import Metric, MetricResult
from olive.model import OliveModel
from olive.passes.olive_pass import Pass
from olive.systems.common import SystemType

logger = logging.getLogger(__name__)


class OliveSystem(ABC):
    system_type: SystemType

    def __init__(self):
        self.accelerators = []

    @abstractmethod
    def run_pass(
        self,
        the_pass: Pass,
        model: OliveModel,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> OliveModel:
        """
        Run the pass on the model at a specific point in the search space.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_model(self, model: OliveModel, metrics: List[Metric]) -> MetricResult:
        """
        Evaluate the model
        """
        raise NotImplementedError()
