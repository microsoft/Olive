# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import olive.cache as cache_utils
from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase, validate_config
from olive.common.utils import hash_dict
from olive.engine.footprint import Footprint, FootprintNode, FootprintNodeMetric
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.evaluator.metric import Metric
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig, ModelStorageKind, OliveModel
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
    azureml_client_config: Optional[AzureMLClientConfig] = None
    packaging_config: PackagingConfig = None
    cache_dir: Union[Path, str] = ".olive-cache"
    clean_cache: bool = False
    clean_evaluation_cache: bool = False
    plot_pareto_frontier: bool = False


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
        target: Optional[OliveSystem] = None,
        evaluator_config: Optional[OliveEvaluatorConfig] = None,
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

        # engine target
        if target is not None:
            self.target = target
        elif self._config.target is not None:
            self.target = self._config.target.create_system()
        else:
            self.target = LocalSystem()

        # default evaluator
        self.evaluator_config = None
        if evaluator_config is not None:
            self.evaluator_config = evaluator_config
        elif self._config.evaluator is not None:
            self.evaluator_config = self._config.evaluator

        # dictionary of passes
        self.pass_config = OrderedDict()

        # {"pass_name": {"pass": pass, "host": host, "evaluator": evaluator, "clean_run_cache": clean_run_cache}}
        self.passes = OrderedDict()

        self.footprints = defaultdict(Footprint)

        self.azureml_client_config = self._config.azureml_client_config

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
        # model jsons have the format <model_number>_<pass_type>-<source_model>-<pass_config_hash>.json
        # model contents are stored in <model_number>_<pass_type>-<source_model>-<pass_config_hash> folder
        # sometimes the folder is created with contents but the json is not created when the pass fails to run
        # so we check for both when determining the new model number
        model_files = list(self._model_cache_path.glob("*_*"))
        if len(model_files) > 0:
            self._new_model_number = max([int(model_file.stem.split("_")[0]) for model_file in model_files]) + 1

        # clean pass run cache if requested
        # removes all run cache for pass type and all children elements
        for pass_config in self.pass_config.values():
            clean_run_cache = pass_config["clean_run_cache"]
            if clean_run_cache:
                cache_utils.clean_pass_run_cache(pass_config["type"].__name__, cache_dir)

        self._initialized = True

    def register(
        self,
        pass_type: Type[Pass],
        config: Dict[str, Any] = None,
        disable_search=False,
        name: str = None,
        host: OliveSystem = None,
        evaluator_config: OliveEvaluatorConfig = None,
        clean_run_cache: bool = False,
    ):
        """Register a pass configuration so that it could be instantiated and executed later."""
        if name is not None:
            assert name not in self.passes, f"Pass with name {name} already registered"
        else:
            id = 0
            while True:
                name = pass_type.__name__
                if id > 0:
                    name = f"{name}_{id}"
                id += 1
                if name not in self.pass_config:
                    break

        self.pass_config[name] = {
            "type": pass_type,
            "config": config or {},
            "disable_search": disable_search,
            "host": host,
            "evaluator": evaluator_config,
            "clean_run_cache": clean_run_cache,
        }

    def register_pass(
        self,
        p: Pass,
        name: str = None,
        host: OliveSystem = None,
        evaluator_config: OliveEvaluatorConfig = None,
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
            "evaluator": evaluator_config,
        }

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

        output_dir: Path = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO by myguo: replace the following for loop using the accelerator and excution provider list when adding
        # accelerator support
        outputs = {}
        for i in range(1):
            # generate search space and intialize the passes for each hardware accelerator
            self.setup_passes()

            # hash the input model
            input_model_id = self._init_input_model(input_model)
            self.footprints[i].record(model_id=input_model_id)

            if evaluation_only:
                prefix_output_name = f"{output_name}_{i}_" if output_name is not None else f"{i}_"
                assert self.evaluator_config is not None, "'evaluation_only' is True but no evaluator provided"
                results = self._evaluate_model(input_model, input_model_id, self.evaluator_config, i, verbose)
                result_name = f"{prefix_output_name}metrics"
                results_path = output_dir / f"{result_name}.json"
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)
                outputs[i] = results
            elif self.no_search:
                output = self.run_no_search(
                    input_model, input_model_id, i, packaging_config, verbose, output_dir, output_name
                )
                outputs[i] = output
            else:
                footprint = self.run_search(
                    input_model, input_model_id, i, packaging_config, verbose, output_dir, output_name
                )
                outputs[i] = footprint

        return outputs

    def setup_passes(self):
        # TODO: add the hardware spec later
        # clean the passes
        self.passes.clear()
        for config in self.pass_config.values():
            pass_cls: Type[Pass] = config["type"]
            pass_cfg = config["config"]
            config_class, pass_cfg = pass_cls.generate_search_space(pass_cfg, config["disable_search"])
            p = pass_cls(config_class, pass_cfg)
            self.register_pass(p, host=config["host"], evaluator_config=config["evaluator"])

        # list of passes starting from the first pass with non-empty search space
        # These passes will be added to the search space
        self.pass_search_spaces = []
        for pass_name in self.passes.keys():
            p: Pass = self.passes[pass_name]["pass"]
            self.pass_search_spaces.append((pass_name, p.search_space()))

    def run_no_search(
        self,
        input_model: OliveModel,
        input_model_id: str,
        accelerator_spec: Any,
        packaging_config: Optional[PackagingConfig] = None,
        verbose: bool = False,
        output_dir: str = None,
        output_name: str = None,
    ):
        for pass_item in self.passes.values():
            if len(pass_item["pass"].search_space()) > 0:
                pass_name = pass_item["name"]
                raise ValueError(f"Pass {pass_name} has search space but search strategy is None")

        evaluator_config = self.evaluator_for_pass(list(self.passes.keys())[-1])
        if evaluator_config is None:
            # provide dummy objective
            objective_dict = {"dummy": {"higher_is_better": True, "goal": 0}}
        else:
            objective_dict = self.resolve_objectives(
                input_model, input_model_id, evaluator_config.metrics, accelerator_spec, verbose
            )

        # initialize the search strategy
        self.search_strategy.initialize(self.pass_search_spaces, input_model_id, objective_dict)

        # get the next step
        next_step = self.search_strategy.next_step()
        assert next_step is not None, "Search strategy returned None for the first step"

        # get the model id of the first input model
        model_id = next_step["model_id"]
        if model_id == input_model_id:
            model = input_model
        else:
            model = self._load_model(model_id)

        if verbose:
            logger.info(f"Step no search with search point {next_step['search_point']} ...")

        # run all the passes in the step
        (
            _,
            signal,
            model_ids,
        ) = self._run_passes(next_step["passes"], model, model_id, accelerator_spec, verbose)
        model_id = model_ids[-1]

        prefix_output_name = f"{output_name}_{accelerator_spec}_" if output_name is not None else f"{accelerator_spec}_"
        # save the model to output_dir
        output_model_name = f"{prefix_output_name}model"
        output_model_json = cache_utils.save_model(model_id, output_dir, output_model_name, self._config.cache_dir)

        # save the evaluation results to output_dir
        result_name = f"{prefix_output_name}metrics"
        results_path = output_dir / f"{result_name}.json"
        if signal is not None:
            with open(results_path, "w") as f:
                json.dump(signal, f, indent=4)

        output = {"model": output_model_json}
        if signal is not None:
            output["metrics"] = signal

        # Package output model as artifacts if no search
        if packaging_config:
            logger.info("Package output model as artifacts")
            generate_output_artifacts(
                packaging_config,
                self.footprints[accelerator_spec],
                self.footprints[accelerator_spec].get_last_node(),
                output_dir,
            )

        return output

    def run_search(
        self,
        input_model: OliveModel,
        input_model_id: str,
        accelerator_spec: Any,
        packaging_config: Optional[PackagingConfig] = None,
        verbose: bool = False,
        output_dir: str = None,
        output_name: str = None,
    ):
        """
        Run all the registered Olive passes on the input model and produce one or more candidate models.

        if search strategy is None, all passes are run in the order they were registered.
        Save the final model to {output_dir}/{output_name}_model and json file to {output_dir}/{output_name}_model.json
        Save evaluation results of the final model, if any, to {output_dir}/{output_name}_metrics.json
        Return {"model": final_model_json, "metrics": evaluation_results}

        if search strategy is not None, run the search strategy to find candidate models.
        TODO: save the results using updated RunResult
        """

        prefix_output_name = f"{output_name}_{accelerator_spec}_" if output_name is not None else f"{accelerator_spec}_"

        # get objective_dict
        evaluator_config = self.evaluator_for_pass(list(self.passes.keys())[-1])

        if evaluator_config is None:
            raise ValueError("No evaluator provided for the last pass")
        else:
            objective_dict = self.resolve_objectives(
                input_model, input_model_id, evaluator_config.metrics, accelerator_spec, verbose
            )

        # initialize the search strategy
        self.search_strategy.initialize(self.pass_search_spaces, input_model_id, objective_dict)
        output_model_num = self.search_strategy.get_output_model_num()

        # record start time
        start_time = time.time()
        iter_num = 0
        while True:
            iter_num += 1

            # get the next step
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
            should_prune, signal, model_ids = self._run_passes(
                next_step["passes"], model, model_id, accelerator_spec, verbose
            )

            # record feedback signal
            self.search_strategy.record_feedback_signal(next_step["search_point"], signal, model_ids, should_prune)

            time_diff = time.time() - start_time
            self.search_strategy.check_exit_criteria(iter_num, time_diff, signal)

        self.footprints[accelerator_spec].to_file(output_dir / f"{prefix_output_name}footprints.json")

        pf_footprints = self.footprints[accelerator_spec].get_pareto_frontier()
        if output_model_num is None or len(pf_footprints.nodes) <= output_model_num:
            logger.info(f"Output all {len(pf_footprints.nodes)} models")
        else:
            metrics = evaluator_config.metrics if evaluator_config else []
            top_ranked_nodes = self._get_top_ranked_nodes(metrics, pf_footprints, output_model_num)
            logger.info(f"Output top ranked {len(top_ranked_nodes)} models based on metric priorities")
            pf_footprints.update_nodes(top_ranked_nodes)

        pf_footprints.to_file(output_dir / f"{prefix_output_name}pareto_frontier_footprints.json")

        if self._config.plot_pareto_frontier:
            pf_footprints.plot_pareto_frontier_to_html(
                save_path=output_dir / f"{prefix_output_name}pareto_frontier_footprints_chart.html"
            )

        if packaging_config:
            logger.info(f"Package top ranked {len(pf_footprints.nodes)} models as artifacts")
            generate_output_artifacts(packaging_config, self.footprints[accelerator_spec], pf_footprints, output_dir)
        else:
            logger.info("No packaging config provided, skip packaging artifacts")

        return pf_footprints

    def resolve_objectives(
        self,
        input_model: OliveModel,
        input_model_id: str,
        metrics: List[Metric],
        accelerator_spec: Any,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary of objectives and their higher_is_better and goal values.

        {objective_name: {"higher_is_better": bool, "goal": float}}
        """
        goals = self.resolve_goals(input_model, input_model_id, metrics, accelerator_spec, verbose)
        objective_dict = {
            metric.name: {"higher_is_better": metric.higher_is_better, "goal": goals.get(metric.name)}
            for metric in metrics
        }
        self.footprints[accelerator_spec].record_objective_dict(objective_dict)
        return objective_dict

    def resolve_goals(
        self,
        input_model: OliveModel,
        input_model_id: str,
        metrics: List[Metric],
        accelerator_spec: Any,
        verbose: bool = False,
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
                assert self.evaluator_config is not None, "Default evaluator must be provided to resolve goals"
                if verbose:
                    logger.info("Computing baseline for metrics ...")
                baseline = self._evaluate_model(
                    input_model, input_model_id, self.evaluator_config, accelerator_spec, verbose=False
                )
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
            return self.evaluator_config
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
            with open(model_json_path, "w") as f:
                json.dump(model_json, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")

    def _load_model(self, model_id: str) -> Union[OliveModel, str]:
        """
        Load the model from the cache directory.
        """
        model_json_path = self.get_model_json_path(model_id)
        try:
            with open(model_json_path, "r") as f:
                model_json = json.load(f)
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
            with open(run_json_path, "w") as f:
                json.dump(run_json, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to cache run: {e}")

    def _load_run(self, input_model_id: str, pass_name: int, pass_config: dict):
        """
        Load the run from the cache directory.
        """
        input_model_number = input_model_id.split("_")[0]
        run_json_path = self.get_run_json_path(pass_name, input_model_number, pass_config)
        if run_json_path.exists():
            try:
                with open(run_json_path, "r") as f:
                    run_json = json.load(f)
                output_model_id = run_json["output_model_id"]
            except Exception as e:
                logger.error(f"Failed to load run: {e}")
                output_model_id = None
            return output_model_id
        else:
            return None

    def _run_passes(
        self,
        passes: List[Tuple[str, Dict[str, Any]]],
        model: OliveModel,
        model_id: str,
        accelerator_spec: Any,
        verbose: bool = False,
    ):
        """
        Run all the passes in the order they were registered.
        the passes is the list of (pass_name, pass_search_point) tuples
        """
        should_prune = False
        # run all the passes in the step
        model_ids = []
        for pass_id, pass_search_point in passes:
            if verbose:
                message = f"Running pass {pass_id}"
                logger.info(message)

            if (
                model.model_storage_kind == ModelStorageKind.AzureMLModel
                and not self.host_for_pass(pass_id).system_type == SystemType.AzureML
            ):
                if not self.azureml_client_config:
                    raise ValueError("AzureML client config is required to download the model from AzureML storage")
                model_download_path = self._model_cache_path / "azureml_input_model"
                model_path = model.download_from_azureml(
                    self.azureml_client_config.create_client(), model_download_path
                )
                model.model_path = model_path
                if model_path.is_dir():
                    model.model_storage_kind = ModelStorageKind.LocalFolder
                elif model_path.is_file():
                    model.model_storage_kind = ModelStorageKind.LocalFile
                else:
                    raise ValueError(f"Invalid model path {model_path}")

            model, model_id = self._run_pass(pass_id, pass_search_point, model, model_id, accelerator_spec, verbose)
            if model == PRUNED_CONFIG:
                should_prune = True
                logger.info("Pruned")
                break
            model_ids.append(model_id)

        signal = {}
        if not should_prune:
            # evaluate the model
            try:
                evaluator_config = self.evaluator_for_pass(pass_id)
                if self.no_search and evaluator_config is None:
                    # skip evaluation if no search and no evaluator
                    signal = None
                else:
                    signal = self._evaluate_model(model, model_id, evaluator_config, accelerator_spec, verbose)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                raise e
            if verbose:
                logger.info(f"Signal: {signal}")

        return should_prune, signal, model_ids

    def _run_pass(
        self,
        pass_id: str,
        pass_search_point: Dict[str, Any],
        input_model: OliveModel,
        input_model_id: str,
        accelerator_spec: Any,
        verbose: bool,
    ):
        """
        Run a pass on the input model.
        """
        # pass
        p: Pass = self.passes[pass_id]["pass"]
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
                self.footprints[accelerator_spec].record(
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
        self.footprints[accelerator_spec].record(
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
            with open(evaluation_json_path, "w") as f:
                json.dump(evaluation_json, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to cache evaluation: {e}")

    def _load_evaluation(self, model_id: str):
        """
        Load the evaluation from the cache directory.
        """
        evaluation_json_path = self.get_evaluation_json_path(model_id)
        if evaluation_json_path.exists():
            try:
                with open(evaluation_json_path, "r") as f:
                    evaluation_json = json.load(f)
                signal = evaluation_json["signal"]
            except Exception as e:
                logger.error(f"Failed to load evaluation: {e}")
                signal = None
            return signal
        else:
            return None

    def _evaluate_model(
        self,
        model: OliveModel,
        model_id: str,
        evaluator_config: OliveEvaluatorConfig,
        accelerator_spec: Any,
        verbose: bool,
    ):
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
            self.footprints[accelerator_spec].record(
                model_id=model_id,
                metrics=FootprintNodeMetric(
                    value=signal,
                    is_goals_met=False,
                ),
            )
            return signal

        # TODO: add the accelerator spec to the evaluate
        # evaluate model
        metrics = evaluator_config.metrics if evaluator_config else []
        signal = self.target.evaluate_model(model, metrics)

        # cache evaluation
        self._cache_evaluation(model_id, signal)

        # footprint evaluation
        self.footprints[accelerator_spec].record(
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
