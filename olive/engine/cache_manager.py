# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from typing import Optional, Union

from olive.cache import OliveCache
from olive.engine.config import FAILED_CONFIG
from olive.evaluator.metric_result import MetricResult
from olive.model import ModelConfig

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of models, runs, and evaluations for the Engine."""

    def __init__(self, cache: OliveCache):
        self.cache = cache

    def cache_model(self, model_id: str, model: Union[ModelConfig, str], check_object: bool = True):
        """Cache a model config to the cache directory."""
        # TODO(trajep): move model/pass run/evaluation cache into footprints
        model_json = {} if model == FAILED_CONFIG else model.to_json(check_object=check_object)
        self.cache.cache_model(model_id, model_json)

    def load_model(self, model_id: str) -> Optional[Union[ModelConfig, str]]:
        """Load a model config from the cache directory."""
        model_json = self.cache.load_model(model_id)
        if model_json is None:
            return None

        if model_json == {}:
            return FAILED_CONFIG

        return ModelConfig.from_json(model_json)

    def cache_evaluation(self, model_id: str, signal: MetricResult):
        """Cache the evaluation in the cache directory."""
        evaluation_json = {
            "model_id": model_id,
            "signal": signal.dict(),
        }
        self.cache.cache_evaluation(model_id, evaluation_json)

    def load_evaluation(self, model_id: str) -> Optional[MetricResult]:
        """Load the evaluation from the cache directory."""
        evaluation_json_path = self.cache.get_evaluation_json_path(model_id)
        if evaluation_json_path.exists():
            try:
                with evaluation_json_path.open() as f:
                    evaluation_json = json.load(f)
                signal = evaluation_json["signal"]
                signal = MetricResult(**signal)
            except Exception:
                logger.exception("Failed to load evaluation")
                signal = None
            return signal
        else:
            return None
