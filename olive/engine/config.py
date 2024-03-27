# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Optional, Union

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Extra
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.strategy.search_strategy import SearchStrategyConfig
from olive.systems.system_config import SystemConfig

# pass search-point was pruned due to failed run
FAILED_MODEL = "failed-model"
# pass search-point was pruned due to invalid config
INVALID_MODEL = "invalid-model"
# list of all pruned configs
PRUNED_MODELS = (FAILED_MODEL, INVALID_MODEL)


class EngineConfig(ConfigBase, extra=Extra.forbid):
    search_strategy: Union[SearchStrategyConfig, bool] = None
    host: SystemConfig = None
    target: SystemConfig = None
    evaluator: OliveEvaluatorConfig = None
    azureml_client_config: Optional[AzureMLClientConfig] = None
    packaging_config: PackagingConfig = None
    cache_dir: Union[Path, str] = ".olive-cache"
    clean_cache: bool = False
    clean_evaluation_cache: bool = False
    plot_pareto_frontier: bool = False
