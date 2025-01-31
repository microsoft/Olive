# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Union

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Extra, Field
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.passes.pass_config import AbstractPassConfig
from olive.search.search_strategy import SearchStrategyConfig
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
    no_artifacts: bool = False


class RunPassConfig(AbstractPassConfig):
    """Pass configuration for Olive workflow.

    This is the configuration for a single pass in Olive workflow. It includes configurations for pass type, config,
    etc.

    Example:
    .. code-block:: json

        {
            "type": "OlivePass",
            "config": {
                "param1": "value1",
                "param2": "value2"
            }
        }

    """

    host: Union[SystemConfig, str] = Field(
        None,
        description=(
            "Host system for the pass. If it is a string, must refer to a system config under `systems` section. If not"
            " provided, use the engine's host system."
        ),
    )
    evaluator: Union[OliveEvaluatorConfig, str] = Field(
        None,
        description=(
            "Evaluator for the pass. If it is a string, must refer to an evaluator config under `evaluators` section."
            " If not provided, use the engine's evaluator."
        ),
    )
