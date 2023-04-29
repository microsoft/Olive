# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import olive.cache as cache_utils
from olive.common.config_utils import ConfigBase, validate_config
from olive.common.utils import hash_dict
from olive.engine.footprint import Footprint, FootprintNode, FootprintNodeMetric
from olive.evaluator.metric import Metric
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorConfig
from olive.model import ModelConfig, ModelStorageKind, OliveModel
from olive.packaging.packaging_config import PackagingConfig
from olive.packaging.packaging_generator import generate_output_artifacts
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
    search_strategy: Union[SearchStrategyConfig, bool] = None
    host: SystemConfig = None
    target: SystemConfig = None
    evaluator: OliveEvaluatorConfig = None
    packaging_config: PackagingConfig = None
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

        self.no_search = False
        # default search strategy
        self.search_strategy = SearchStrategy({"execution_order": "joint", "search_algorithm": "exhaustive"})
        if search_strategy is not None:
            # if search strategy is provided, use it. It takes precedence
            self.search_strategy = search_strategy
        elif isinstance(self._config.search_strategy, ConfigBase) or isinstance(self._config.search_strategy, dict):
            # if search strategy is provided in config, use it
            self.search_strategy = SearchStrategy(self._config.search_strategy)
        elif not self._config.search_strategy:
            # if search strategy is None or False, disable search
            self.no_search = True

        # default host
        if host is not None:
            self.host = host
        elif self._config.host is not None:
            self.host = self._config.host.create_system()
        else:
            self.host = LocalSystem()

        # default evaluator
        self.evaluator = None
        if evaluator is not None:
            self.evaluator = evaluator
        elif self._config.evaluator is not None:
            self.evaluator = self._config.evaluator.create_evaluator()

        # dictionary of passes
        # {"pass_name": {"pass": pass, "host": host, "evaluator": evaluator, "clean_run_cache": clean_run_cache}}
        self.passes = {}
        # list of pass names in the order they were registered
        self.pass_order = []

        self.footprints = Footprint()

        self._initialized = False

    def initialize(self):
        """
        Initialize engine state. This should be done before running the registered passes.
        """
        cache_dir = self._config.cache_dir
        if self._config.clean_cache:
            cache_utils.clean_cache(cache_dir)
        if self._config.clean_evaluation_cache:
            cache_utils.clean_evaluation_cache(cache_dir)

        self._model_cache_path, self._run_cache_path, self._evaluation_cache_path = cache_utils.get_cache_sub_dirs(
            cache_dir
        )
        cache_utils.create_cache(cache_dir)

        # initialize counters
        # we do this before cleaning pass run caches to ensure we don't reuse model numbers even if the model was
        # deleted from the cache
        self._new_model_number = 0
        model_jsons = list(self._model_cache_path.glob("*_*.json"))
        if len(model_jsons) > 0:
            self._new_model_number = max([int(json_file.stem.split("_")[0]) for json_file in model_jsons]) + 1

        # clean pass run cache if requested
        # removes all run cache for pass type and all children elements
        for pass_name in self.pass_order:
            clean_run_cache = self.passes[pass_name]["clean_run_cache"]
            p = self.passes[pass_name]["pass"]
            if clean_run_cache:
                cache_utils.clean_pass_run_cache(p.__class__.__name__, cache_dir)

        # list of passes starting from the first pass with non-empty search space
        # These passes will be added to the search space
        self.pass_search_spaces = []
        for pass_name in self.pass_order:
            p = self.passes[pass_name]["pass"]
            self.pass_search_spaces.append((pass_name, p.search_space()))

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
            assert name not in self.passes, f"Pass with name {name} already registered"
        else:
            id = 0
            while True:
                name = p.__class__.__name__
                if id > 0:
                    name = f"{name}_{id}"
                id += 1
                if name not in self.passes:
                    break

        if self.no_search and len(p.search_space()) > 0:
            raise ValueError(f"Search strategy is None but pass {name} has search space")

        self.passes[name] = {
            "pass": p,
            "host": host,
            "evaluator": evaluator,
            "clean_run_cache": clean_run_cache,
        }
        self.pass_order.append(name)

    def run(
        self,
        input_model: OliveModel,
        packaging_config: Optional[PackagingConfig] = None,
        verbose: bool = False,
        output_dir: str = None,
        output_name: str = None,
        evaluation_only: bool = False,
    ):
        """
        Run all the registered Olive passes on the input model and produce one or more candidate models.

        if search strategy is None, all passes are run in the order they were registered.
        Save the final model to {output_dir}/{output_name}_model and json file to {output_dir}/{output_name}_model.json
        Save evaluation results of the final model, if any, to {output_dir}/{output_name}_metrics.json
        Return {"model": final_model_json, "metrics": evaluation_results}

        if search strategy is not None, run the search strategy to find candidate models.
        TODO: save the results using updated RunResult

        if evaluation_only is True, run the evaluation on the input model and return the results.
        """
        if not self._initialized:
            self.initialize()

        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix_output_name = f"{output_name}_" if output_name is not None else ""

        if self.no_search:
            for pass_id in self.pass_order:
                if len(self.passes[pass_id]["pass"].search_space()) > 0:
                    raise ValueError(f"Pass {pass_id} has search space but search strategy is None")

        # hash the input model
        input_model_id = self._init_input_model(input_model)
        self.footprints.record(model_id=input_model_id)

        if evaluation_only:
            assert self.evaluator is not None, "Evaluation only is True but no evaluator provided"
            results = self._evaluate_model(input_model, input_model_id, self.evaluator, verbose)
            result_name = f"{prefix_output_name}metrics"
            results_path = output_dir / f"{result_name}.json"
            json.dump(results, open(results_path, "w"), indent=4)
            return results

        # get objective_dict
        evaluator = self.evaluator_for_pass(self.pass_order[-1])
        if self.no_search and evaluator is None:
            # provide dummy objective
            objective_dict = {"dummy": {"higher_is_better": True, "goal": 0}}
        elif evaluator is None:
            raise ValueError("No evaluator provided for the last pass")
        else:
            objective_dict = self.resolve_objectives(input_model, input_model_id, evaluator.metrics, verbose)

        # initialize the search strategy
        self.search_strategy.initialize(self.pass_search_spaces, input_model_id, objective_dict)
        output_model_num = self.search_strategy.get_output_model_num()

        # record start time
        start_time = time.time()
        iter_num = 0
        while True:
            iter_num += 1

            # get the next step
            should_prune = False
            next_step = self.search_strategy.next_step()

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
                    message = f"Running pass {pass_id}"
                    if not self.no_search:
                        message += f" with search point {pass_search_point}"
                    logger.info(message)

                if (
                    input_model.model_storage_kind == ModelStorageKind.AzureMLModel
                    and not self.host_for_pass(pass_id).system_type == SystemType.AzureML
                ):
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
                    evaluator = self.evaluator_for_pass(pass_id)
                    if self.no_search and evaluator is None:
                        # skip evaluation if no search and no evaluator
                        signal = None
                    else:
                        signal = self._evaluate_model(model, model_id, evaluator, verbose)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    raise e
                if verbose:
                    logger.info(f"Signal: {signal}")

            # there is only one step if no search
            if self.no_search:
                break

            # record feedback signal
            self.search_strategy.record_feedback_signal(next_step["search_point"], signal, model_ids, should_prune)

            time_diff = time.time() - start_time
            self.search_strategy.check_exit_criteria(iter_num, time_diff, signal)

        if self.no_search:
            # save the model to output_dir
            output_model_name = f"{prefix_output_name}model"
            output_model_json = cache_utils.save_model(model_id, output_dir, output_model_name, self._config.cache_dir)

            # save the evaluation results to output_dir
            result_name = f"{prefix_output_name}metrics"
            results_path = output_dir / f"{result_name}.json"
            if signal is not None:
                json.dump(signal, open(results_path, "w"), indent=4)

            output = {"model": output_model_json}
            if signal is not None:
                output["metrics"] = signal
            return output

        self.footprints.to_file(output_dir / f"{prefix_output_name}footprints.json")

        pf_footprints = self.footprints.get_pareto_frontier()
        if output_model_num is None or len(pf_footprints.nodes) <= output_model_num or self.no_search:
            logger.info(f"Output all {len(pf_footprints.nodes)} models")
        else:
            top_ranked_nodes = self._get_top_ranked_nodes(evaluator.metrics, pf_footprints, output_model_num)
            logger.info(f"Output top ranked {len(top_ranked_nodes)} models based on metric priorities")
            pf_footprints.update_nodes(top_ranked_nodes)

        pf_footprints.to_file(output_dir / f"{prefix_output_name}pareto_frontier_footprints.json")

        if packaging_config:
            logger.info(f"Package top ranked {len(pf_footprints.nodes)} models as artifacts")
            generate_output_artifacts(packaging_config, self.footprints, pf_footprints, output_dir)
        else:
            logger.info("No packaging config provided, skip packaging artifacts")

        return pf_footprints

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
        self.footprints.record_objective_dict(objective_dict)
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
                assert self.evaluator is not None, "Default evaluator must be provided to resolve goals"
                if verbose:
                    logger.info("Computing baseline for metrics ...")
                baseline = self._evaluate_model(input_model, input_model_id, self.evaluator, verbose=False)
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
        host = self.passes[pass_id]["host"]
        if host is None:
            return self.host
        return host

    def evaluator_for_pass(self, pass_id: str):
        """
        Return evaluator for the given pass.
        """
        e = self.passes[pass_id]["evaluator"]
        if e is None:
            return self.evaluator
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

    def get_model_json_path(self, model_id: str) -> Path:
        """
        Get the path to the model json file.
        """
        return self._model_cache_path / f"{model_id}.json"

    def _cache_model(self, model: Union[OliveModel, str], model_id: str, check_objects: bool = True):
        """
        Cache the model in the cache directory.
        """
        # TODO move model/pass run/evaluation cache into footprints
        if model == PRUNED_CONFIG:
            model_json = {}
        else:
            model_json = model.to_json(check_object=check_objects)
        model_json_path = self.get_model_json_path(model_id)
        try:
            json.dump(model_json, open(model_json_path, "w"), indent=4)
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
        self._cache_model(input_model, model_hash, check_objects=False)

        return model_hash

    def get_run_json_path(self, pass_name: int, input_model_number: str, pass_config: dict):
        """
        Get the path to the run json.
        """
        pass_config_hash = hash_dict(pass_config)
        run_json_path = self._run_cache_path / f"{pass_name}-{input_model_number}-{pass_config_hash}.json"
        return run_json_path

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
        run_json_path = self.get_run_json_path(pass_name, input_model_number, pass_config)
        try:
            json.dump(run_json, open(run_json_path, "w"), indent=4)
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
        p = self.passes[pass_id]["pass"]
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
                # footprint model and run
                self.footprints.record(
                    model_id=output_model_id,
                    model_config=output_model.to_json() if output_model != PRUNED_CONFIG else {"is_pruned": True},
                    parent_model_id=input_model_id,
                    from_pass=pass_name,
                    pass_run_config=pass_config,
                )
                return output_model, output_model_id

        # new model id
        input_model_number = input_model_id.split("_")[0]
        output_model_id = f"{self._get_new_model_number()}_{pass_name}-{input_model_number}-{hash_dict(pass_config)}"
        output_model_path = str(self._model_cache_path / f"{output_model_id}")

        # prune if invalid search_point
        if not p.validate_search_point(pass_search_point) and not self.no_search:
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
                if self.no_search:
                    raise  # rethrow the exception if no search is performed

        # cache model
        self._cache_model(output_model, output_model_id)

        # cache run
        self._cache_run(pass_name, pass_config, input_model_id, output_model_id)

        # footprint model and run
        self.footprints.record(
            model_id=output_model_id,
            model_config=output_model.to_json() if output_model != PRUNED_CONFIG else {"is_pruned": True},
            parent_model_id=input_model_id,
            from_pass=pass_name,
            pass_run_config=pass_config,
        )
        return output_model, output_model_id

    def get_evaluation_json_path(self, model_id: str):
        """
        Get the path to the evaluation json.
        """
        evaluation_json_path = self._evaluation_cache_path / f"{model_id}.json"
        return evaluation_json_path

    def _cache_evaluation(self, model_id: str, signal: dict):
        """
        Cache the evaluation in the cache directory.
        """
        evaluation_json = {
            "model_id": model_id,
            "signal": signal,
        }
        evaluation_json_path = self.get_evaluation_json_path(model_id)
        try:
            json.dump(evaluation_json, open(evaluation_json_path, "w"), indent=4)
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
            logger.info("Evaluating model ...")
        # load evaluation from cache if it exists
        signal = self._load_evaluation(model_id)
        if signal is not None:
            if verbose:
                logger.info("Loading evaluation from cache ...")
            # footprint evaluation
            self.footprints.record(
                model_id=model_id,
                metrics=FootprintNodeMetric(
                    value=signal,
                    is_goals_met=False,
                ),
            )
            return signal

        # evaluate model
        signal = evaluator.evaluate(model)

        # cache evaluation
        self._cache_evaluation(model_id, signal)

        # footprint evaluation
        self.footprints.record(
            model_id=model_id,
            metrics=FootprintNodeMetric(
                value=signal,
                is_goals_met=False,
            ),
        )
        return signal

    def _get_top_ranked_nodes(self, metrics: List[Metric], footprint: Footprint, k: int) -> List[FootprintNode]:
        metric_priority = [metric.name for metric in sorted(metrics, key=lambda x: x.priority_rank)]
        footprint_node_list = footprint.nodes.values()
        sorted_footprint_node_list = sorted(
            footprint_node_list,
            key=lambda x: tuple(
                x.metrics.value[metric] if x.metrics.cmp_direction[metric] == 1 else -x.metrics.value[metric]
                for metric in metric_priority
            ),
            reverse=True,
        )
        selected_footprint_nodes = sorted_footprint_node_list[:k]
        return selected_footprint_nodes
