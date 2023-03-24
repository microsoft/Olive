# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from olive.cache import clean_cache, clean_evaluation_cache, clean_pass_run_cache, create_cache, get_cache_sub_dirs
from olive.common.config_utils import ConfigBase, validate_config
from olive.common.utils import hash_dict
from olive.evaluator.metric import Metric
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorConfig
from olive.model import ModelConfig, OliveModel
from olive.passes.olive_pass import Pass
from olive.strategy.search_strategy import SearchStrategy, SearchStrategyConfig
from olive.systems.common import SystemType
from olive.systems.local import LocalSystem
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import SystemConfig

logger = logging.getLogger(__name__)

# pass search-point/config was pruned due invalid config or failed run
PRUNED_CONFIG = "pruned-config"


class EngineConfig(ConfigBase):
    search_strategy: SearchStrategyConfig = None
    host: SystemConfig = None
    target: SystemConfig = None
    model_io_config: Dict[str, List] = None
    evaluator: OliveEvaluatorConfig = None
    cache_dir: Union[Path, str] = ".olive-cache"
    clean_cache: bool = False
    clean_evaluation_cache: bool = False


class Engine:
    """
    The engine executes the registered Olive Steps, facilitate evaluation of the output models using
    provided evaluation criteria and produces output model(s).
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], EngineConfig] = None,
        search_strategy: Optional[SearchStrategy] = None,
        host: Optional[OliveSystem] = None,
        evaluator: Optional[OliveEvaluator] = None,
    ):
        self._config = validate_config(config, EngineConfig)

        if search_strategy is not None:
            self._search_strategy = search_strategy
        else:
            assert self._config.search_strategy is not None, "Search strategy must be provided"
            self._search_strategy = SearchStrategy(self._config.search_strategy)

        # default host
        if host is not None:
            self.host = host
        elif self._config.host is not None:
            self.host = self._config.host.create_system()
        else:
            self.host = LocalSystem()

        # Dictionary to keep track of separate hosts for a pass optionally provided by the user.
        self.hosts = {}

        # default evaluator
        self._evaluator = None
        if evaluator is not None:
            self._evaluator = evaluator
        elif self._config.evaluator is not None:
            self._evaluator = self._config.evaluator.create_evaluator()

        # Dictionary to keep track of separate evaluator for a pass optionally provided by the user.
        self._evaluators = {}

        # dictionary of passes
        self._passes = {}
        # list of pass names in the order they were registered
        self._pass_order = []
        self._clean_pass_run_cache = {}

        self._initialized = False

    def initialize(self):
        """
        Initialize engine state. This should be done before running the registered passes.
        """
        cache_dir = self._config.cache_dir
        if self._config.clean_cache:
            clean_cache(cache_dir)
        if self._config.clean_evaluation_cache:
            clean_evaluation_cache(cache_dir)

        self._model_cache_path, self._run_cache_path, self._evaluation_cache_path = get_cache_sub_dirs(cache_dir)
        create_cache(cache_dir)

        # initialize counters
        # we do this before cleaning pass run caches to ensure we don't reuse model numbers even if the model was
        # deleted from the cache
        self._new_model_number = 0
        model_jsons = list(self._model_cache_path.glob("*_*.json"))
        if len(model_jsons) > 0:
            self._new_model_number = max([int(json_file.stem.split("_")[0]) for json_file in model_jsons]) + 1

        # initialize search spaces
        self._pass_search_spaces = []
        for pass_name in self._pass_order:
            p = self._passes[pass_name]
            self._pass_search_spaces.append((pass_name, p.search_space()))
            # clean run cache if requested
            # removes all run cache for pass type and all children elements
            clean_run_cache = self._clean_pass_run_cache[pass_name]
            if clean_run_cache:
                clean_pass_run_cache(p.__class__.__name__, cache_dir)

        self._initialized = True

    def register(
        self,
        p: Pass,
        name: str = None,
        host: OliveSystem = None,
        evaluator: OliveEvaluator = None,
        clean_run_cache: bool = False,
    ):
        """
        Register a pass
        """
        if name is not None:
            assert name not in self._passes, f"Pass with name {name} already registered"
        else:
            id = 0
            while True:
                name = p.__class__.__name__
                if id > 0:
                    name = f"{name}_{id}"
                if name not in self._passes:
                    break

        self._passes[name] = p
        self._pass_order.append(name)
        self.hosts[name] = host
        self._evaluators[name] = evaluator
        self._clean_pass_run_cache[name] = clean_run_cache

    def run(self, input_model: OliveModel, verbose: bool = False):
        """
        Run all the registered Olive passes on the input model and produce one or more candidate models.
        """
        if not self._initialized:
            self.initialize()

        # hash the input model
        input_model_id = self._init_input_model(input_model)

        # get objective_dict
        evaluator = self.evaluator_for_pass(self._pass_order[-1])
        objective_dict = self.resolve_objectives(input_model, input_model_id, evaluator.metrics, verbose)

        # initialize the search strategy
        self._search_strategy.initialize(self._pass_search_spaces, input_model_id, objective_dict)

        # record start time
        start_time = time.time()
        iter_num = 0
        while True:
            iter_num += 1

            # get the next step
            should_prune = False
            next_step = self._search_strategy.next_step()

            # if no more steps, break
            if next_step is None:
                break

            # get the model id of the first input model
            model_id = next_step["model_id"]
            if model_id == input_model_id:
                model = input_model
            else:
                model = self._load_model(model_id)

            if verbose:
                logger.info(f"Step {iter_num} with search point {next_step['search_point']} ...")

            # run all the passes in the step
            model_ids = []
            for pass_id, pass_search_point in next_step["passes"]:
                if verbose:
                    logger.info(f"Running pass {pass_id} with search point {pass_search_point} ...")

                if input_model.is_aml_model and not self.host_for_pass(pass_id).system_type == SystemType.AzureML:
                    error_msg = "Azure ML model only supports AzureMLSystem for Olive Pass"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                model, model_id = self._run_pass(pass_id, pass_search_point, model, model_id, verbose)
                if model == PRUNED_CONFIG:
                    should_prune = True
                    logger.info("Pruned")
                    break
                model_ids.append(model_id)

            signal = {}
            if not should_prune:
                # evaluate the model
                try:
                    signal = self._evaluate_model(model, model_id, self.evaluator_for_pass(pass_id), verbose)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    raise e
                if verbose:
                    logger.info(f"Signal: {signal}")

            # record feedback signal
            self._search_strategy.record_feedback_signal(next_step["search_point"], signal, model_ids, should_prune)

            time_diff = time.time() - start_time
            self._search_strategy.check_exit_criteria(iter_num, time_diff, signal)

        # import json

        # for i, key in enumerate(self._search_strategy._search_results):
        #     json.dump(
        #         self._search_strategy._search_results[key].to_json(), open(f"search_results_{i}.json", "w"), indent=4
        #     )
        return self._search_strategy.get_best_execution()

    def resolve_objectives(
        self, input_model: OliveModel, input_model_id: str, metrics: List[Metric], verbose: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary of objectives and their higher_is_better and goal values.

        {objective_name: {"higher_is_better": bool, "goal": float}}
        """
        goals = self.resolve_goals(input_model, input_model_id, metrics, verbose)
        objective_dict = {
            metric.name: {"higher_is_better": metric.higher_is_better, "goal": goals.get(metric.name)}
            for metric in metrics
        }
        return objective_dict

    def resolve_goals(
        self, input_model: OliveModel, input_model_id: str, metrics: List[Metric], verbose: bool = False
    ) -> Dict[str, float]:
        """
        Resolve the goals of the given metrics into thresholds for the given model.
        """
        goals = {}
        multipliers = {}
        for metric in metrics:
            if metric.goal is not None:
                goals[metric.name] = metric.goal
                multipliers[metric.name] = 1 if metric.higher_is_better else -1
        if verbose and len(goals) > 0:
            logger.info(f"Resolving goals: {goals}")

        # compute baseline for input model if needed
        baseline = {}
        for _, goal in goals.items():
            if goal.type != "threshold":
                assert self._evaluator is not None, "Default evaluator must be provided to resolve goals"
                if verbose:
                    logger.info("Computing baseline for metrics ...")
                baseline = self._evaluate_model(input_model, input_model_id, self._evaluator, verbose=False)
                break
        if verbose and len(baseline) > 0:
            logger.info(f"Baseline: {baseline}")

        # resolve goals to thresholds
        resolved_goals = {}
        for name, goal in goals.items():
            # TODO: make the logic cleaner
            if goal.type == "threshold":
                resolved_goals[name] = goal.value
            elif goal.type == "max-degradation":
                resolved_goals[name] = baseline[name] - multipliers[name] * goal.value
            elif goal.type == "min-improvement":
                resolved_goals[name] = baseline[name] + multipliers[name] * goal.value
            elif goal.type == "percent-max-degradation":
                resolved_goals[name] = baseline[name] * (1 - multipliers[name] * goal.value / 100)
            elif goal.type == "percent-min-improvement":
                resolved_goals[name] = baseline[name] * (1 + multipliers[name] * goal.value / 100)
        if verbose and len(resolved_goals) > 0:
            logger.info(f"Resolved goals: {resolved_goals}")

        return resolved_goals

    def host_for_pass(self, pass_id: str):
        host = self.hosts[pass_id]
        if host is None:
            return self.host
        return host

    def evaluator_for_pass(self, pass_id: str):
        """
        Return evaluator for the given pass.
        """
        e = self._evaluators[pass_id]
        if e is None:
            return self._evaluator
        return e

    def _get_new_model_number(self):
        """
        Get a new model number.
        """
        while True:
            new_model_number = self._new_model_number
            self._new_model_number += 1
            if list(self._model_cache_path.glob(f"{new_model_number}_*.json")) == []:
                break
        return new_model_number

    def _cache_model(self, model: Union[OliveModel, str], model_id: str):
        """
        Cache the model in the cache directory.
        """
        if model == PRUNED_CONFIG:
            model_json = {}
        else:
            model_json = model.to_json()
        model_json_path = self._model_cache_path / f"{model_id}.json"
        try:
            json.dump(model_json, open(model_json_path, "w"))
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")

    def _load_model(self, model_id: str) -> Union[OliveModel, str]:
        """
        Load the model from the cache directory.
        """
        model_json_path = self._model_cache_path / f"{model_id}.json"
        try:
            model_json = json.load(open(model_json_path, "r"))
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

        if model_json == {}:
            return PRUNED_CONFIG

        model = ModelConfig.from_json(model_json).create_model()
        return model

    def _init_input_model(self, input_model: OliveModel):
        """
        Initialize the input model.
        """
        model_hash = hash_dict(input_model.to_json())

        # cache the model
        self._cache_model(input_model, model_hash)

        return model_hash

    def _cache_run(self, pass_name: int, pass_config: dict, input_model_id: str, output_model_id: str):
        """
        Cache the run in the cache directory.
        """
        run_json = {
            "pass_name": pass_name,
            "pass_config": pass_config,
            "input_model_id": input_model_id,
            "output_model_id": output_model_id,
        }
        input_model_number = input_model_id.split("_")[0]
        run_json_path = self._run_cache_path / f"{pass_name}-{input_model_number}-{hash_dict(pass_config)}.json"
        try:
            json.dump(run_json, open(run_json_path, "w"))
        except Exception as e:
            logger.error(f"Failed to cache run: {e}")

    def _load_run(self, input_model_id: str, pass_name: int, pass_config: dict):
        """
        Load the run from the cache directory.
        """
        input_model_number = input_model_id.split("_")[0]
        run_json_path = self._run_cache_path / f"{pass_name}-{input_model_number}-{hash_dict(pass_config)}.json"
        if Path(run_json_path).exists():
            try:
                run_json = json.load(open(run_json_path, "r"))
                output_model_id = run_json["output_model_id"]
            except Exception as e:
                logger.error(f"Failed to load run: {e}")
                output_model_id = None
            return output_model_id
        else:
            return None

    def _run_pass(
        self, pass_id: int, pass_search_point: dict, input_model: OliveModel, input_model_id: str, verbose: bool
    ):
        """
        Run a pass on the input model.
        """
        # pass
        p = self._passes[pass_id]
        pass_name = p.__class__.__name__
        pass_config = p.config_at_search_point(pass_search_point)
        pass_config = p.serialize_config(pass_config)

        # load run from cache if it exists
        output_model_id = self._load_run(input_model_id, pass_name, pass_config)
        if output_model_id is not None:
            if verbose:
                logger.info("Loading model from cache ...")
            output_model = self._load_model(output_model_id)
            if output_model is not None:
                return output_model, output_model_id

        # new model id
        input_model_number = input_model_id.split("_")[0]
        output_model_id = f"{self._get_new_model_number()}_{pass_name}-{input_model_number}-{hash_dict(pass_config)}"
        output_model_path = str(self._model_cache_path / f"{output_model_id}")

        # prune if invalid search_point
        if not p.validate_search_point(pass_search_point):
            output_model = PRUNED_CONFIG
        else:
            # run pass
            try:
                host = self.host_for_pass(pass_id)
                output_model = host.run_pass(p, input_model, output_model_path, pass_search_point)
            except Exception:
                output_model = PRUNED_CONFIG
                # TODO: from the time being, we need to catch all exceptions to make the
                #      search process robust. We need rethrow the exception only when
                #      it is not pass specific. For example, for olive bugs and user errors
                logger.error("Pass run failed.", exc_info=True)

        # cache model
        self._cache_model(output_model, output_model_id)

        # cache run
        self._cache_run(pass_name, pass_config, input_model_id, output_model_id)

        return output_model, output_model_id

    def _cache_evaluation(self, model_id: str, signal: dict):
        """
        Cache the evaluation in the cache directory.
        """
        evaluation_json = {
            "model_id": model_id,
            "signal": signal,
        }
        evaluation_json_path = self._evaluation_cache_path / f"{model_id}.json"
        try:
            json.dump(evaluation_json, open(evaluation_json_path, "w"))
        except Exception as e:
            logger.error(f"Failed to cache evaluation: {e}")

    def _load_evaluation(self, model_id: str):
        """
        Load the evaluation from the cache directory.
        """
        evaluation_json_path = self._evaluation_cache_path / f"{model_id}.json"
        if Path(evaluation_json_path).exists():
            try:
                evaluation_json = json.load(open(evaluation_json_path, "r"))
                signal = evaluation_json["signal"]
            except Exception as e:
                logger.error(f"Failed to load evaluation: {e}")
                signal = None
            return signal
        else:
            return None

    def _evaluate_model(self, model: OliveModel, model_id: str, evaluator: OliveEvaluator, verbose: bool):
        """
        Evaluate a model.
        """
        if verbose:
            logger.info("Evaluating output model ...")
        # load evaluation from cache if it exists
        signal = self._load_evaluation(model_id)
        if signal is not None:
            if verbose:
                logger.info("Loading evaluation from cache ...")
            return signal

        # evaluate model
        signal = evaluator.evaluate(model)

        # cache evaluation
        self._cache_evaluation(model_id, signal)

        return signal
