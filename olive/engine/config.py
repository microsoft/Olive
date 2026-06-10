# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Optional, Union

from pydantic import ConfigDict, Field

from olive.common.config_utils import ConfigBase
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

# sentinel key inside `builds` that holds partial defaults applied to all sibling builds
BUILD_DEFAULT_KEY = "_default"


class EngineConfig(ConfigBase):
    model_config = ConfigDict(extra="forbid")
    search_strategy: Optional[Union[SearchStrategyConfig, bool]] = Field(
        None, description="Search strategy configuration to use to auto optimize the input model."
    )
    host: Optional[SystemConfig] = Field(
        None,
        description=(
            "The host of the engine. It can be a string or a dictionary. "
            "If it is a string, it is the name of a system in `systems`."
        ),
    )
    target: Optional[SystemConfig] = Field(
        None,
        description=(
            "The target to run model evaluations on. It can be a string or a dictionary. "
            "If it is a string, it is the name of a system in `systems`. "
            "If it is a dictionary, it contains the system information. If not specified, it is the local system."
        ),
    )
    evaluator: Optional[OliveEvaluatorConfig] = Field(
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

    host: Optional[Union[SystemConfig, str]] = Field(
        None,
        description=(
            "Host system for the pass. If it is a string, must refer to a system config under `systems` section. If not"
            " provided, use the engine's host system."
        ),
    )
    evaluator: Optional[Union[OliveEvaluatorConfig, str]] = Field(
        None,
        description=(
            "Evaluator for the pass. If it is a string, must refer to an evaluator config under `evaluators` section."
            " If not provided, use the engine's evaluator."
        ),
    )


class BuildConfigPartial(ConfigBase):
    """Partial build configuration.

    All fields are optional. Used as the schema for the ``_default`` sentinel inside ``builds``
    and as the unmerged form of every sibling entry before defaults are applied.
    """

    model_config = ConfigDict(extra="forbid")

    components: Optional[list[str]] = Field(
        None,
        description=(
            "Names of input model components this build operates on. Each name must match an entry in the input"
            " model's ``model_component_names``. When omitted, the build runs on the full input model."
            " When a single name is given, the build receives the unwrapped component handler instead of a one-element"
            " composite."
        ),
    )
    pipeline: Optional[list[str]] = Field(
        None,
        description=(
            "Ordered list of pass names (referencing entries in the top-level ``passes`` dict) that form this build's"
            " pipeline."
        ),
    )
    output_dir: Optional[str] = Field(
        None,
        description="Directory where this build's final model artifacts get saved.",
    )
    host: Optional[Union[SystemConfig, str]] = Field(
        None,
        description=(
            "Host system override for this build. If a string, must refer to a system config under ``systems``."
            " If omitted, the engine's host is used."
        ),
    )
    target: Optional[Union[SystemConfig, str]] = Field(
        None,
        description=(
            "Target system override for this build. If a string, must refer to a system config under ``systems``."
            " If omitted, the engine's target is used."
        ),
    )
    evaluator: Optional[Union[OliveEvaluatorConfig, str]] = Field(
        None,
        description=(
            "Evaluator override for this build. If a string, must refer to an evaluator config under ``evaluators``."
            " If omitted, the engine's evaluator is used."
        ),
    )
    search_strategy: Optional[Union[SearchStrategyConfig, bool]] = Field(
        None,
        description="Search strategy override for this build. If omitted, the engine's search strategy is used.",
    )


class BuildConfig(BuildConfigPartial):
    """Full build configuration after defaults have been merged.

    ``pipeline`` and ``output_dir`` are required post-merge; the other fields remain optional and
    fall back to the engine-level configuration when not provided.
    """

    pipeline: list[str] = Field(
        ...,
        description=(
            "Ordered list of pass names (referencing entries in the top-level ``passes`` dict) that form this build's"
            " pipeline."
        ),
    )
    output_dir: str = Field(
        ...,
        description="Directory where this build's final model artifacts get saved.",
    )


def merge_build_default(default_partial: dict, sibling: dict) -> dict:
    """Merge ``_default`` partial values into a sibling build dict.

    Sibling values fully override default values (no deep merge). Returns a new dict; inputs are
    not mutated.
    """
    return {**{k: v for k, v in default_partial.items() if v is not None}, **sibling}
