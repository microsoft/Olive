# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Union

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Extra
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.strategy.search_strategy import SearchStrategyConfig
from olive.systems.system_config import SystemConfig

# pass search-point was pruned due to failed run
FAILED_CONFIG = "failed-config"
# pass search-point was pruned due to invalid config
INVALID_CONFIG = "invalid-config"
# list of all pruned configs
PRUNED_CONFIGS = (FAILED_CONFIG, INVALID_CONFIG)


class EngineConfig(ConfigBase, extra=Extra.forbid):
    search_strategy: Union[SearchStrategyConfig, bool] = None
    host: SystemConfig = None
    target: SystemConfig = None
    evaluator: OliveEvaluatorConfig = None
    plot_pareto_frontier: bool = False
