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
    search_strategy: Union[SearchStrategyConfig, bool] = Field(
        None, description="Search strategy configuration to use to auto optimize the input model."
    )
    host: SystemConfig = Field(
        None,
        description=(
            "The host of the engine. It can be a string or a dictionary. "
            "If it is a string, it is the name of a system in `systems`."
        ),
    )
    target: SystemConfig = Field(
        None,
        description=(
            "The target to run model evaluations on. It can be a string or a dictionary. "
            "If it is a string, it is the name of a system in `systems`. "
            "If it is a dictionary, it contains the system information. If not specified, it is the local system."
        ),
    )
    evaluator: OliveEvaluatorConfig = Field(
        None,
        description=(
            "The evaluator of the engine. It can be a string or a dictionary. "
            "If it is a string, it is the name of an evaluator in `evaluators`. "
            "If it is a dictionary, it contains the evaluator information. "
            "This evaluator will be used to evaluate the input model if needed. "
            "It is also used to evaluate the output models of passes that don't have their own evaluators. "
            "If it is None, skip the evaluation for input model and any output models."
        ),
    )
    plot_pareto_frontier: bool = Field(
        False, description="This decides whether to plot the pareto frontier of the search results."
    )
    no_artifacts: bool = Field(False, description="Set to false to skip generated output artifacts.")


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
