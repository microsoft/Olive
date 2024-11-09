# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

from olive.cache import CacheConfig, OliveCache
from olive.common.config_utils import validate_config
from olive.common.constants import DEFAULT_WORKFLOW_ID, LOCAL_INPUT_MODEL_ID
from olive.engine.config import FAILED_CONFIG, INVALID_CONFIG, PRUNED_CONFIGS
from olive.engine.footprint import Footprint, FootprintNodeMetric
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.evaluator.metric import Metric
from olive.evaluator.metric_result import MetricResult, joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.exception import EXCEPTIONS_TO_RAISE, OlivePassError
from olive.hardware import AcceleratorSpec
from olive.logging import enable_filelog
from olive.model import ModelConfig
from olive.strategy.search_strategy import SearchStrategy, SearchStrategyConfig
from olive.systems.common import SystemType
from olive.systems.system_config import SystemConfig
from olive.systems.utils import create_managed_system_with_cache

if TYPE_CHECKING:
    from olive.engine.packaging.packaging_config import PackagingConfig
    from olive.passes.olive_pass import Pass
    from olive.systems.olive_system import OliveSystem

logger = logging.getLogger(__name__)


class Engine:
    """The engine executes the registered Olive Steps.

    It facilitate evaluation of the output models using provided evaluation criteria and produces output model(s).
    """

    def __init__(
        self,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
        search_strategy: Optional[Union[Dict[str, Any], SearchStrategyConfig]] = None,
        host: Optional[Union[Dict[str, Any], "SystemConfig"]] = None,
        target: Optional[Union[Dict[str, Any], "SystemConfig"]] = None,
        evaluator: Optional[Union[Dict[str, Any], "OliveEvaluatorConfig"]] = None,
        cache_config: Optional[Union[Dict[str, Any], CacheConfig]] = None,
        plot_pareto_frontier: bool = False,
        *,
        azureml_client_config=None,
    ):
        self.workflow_id = workflow_id
        self.search_strategy = SearchStrategy(search_strategy) if search_strategy else None

        # default host
        host = host or {"type": SystemType.Local}
        self.host_config = validate_config(host, SystemConfig)
        self.host = None

        # engine target
        target = target or {"type": SystemType.Local}
        self.target_config = validate_config(target, SystemConfig)
        self.target = None

        # default evaluator
        self.evaluator_config = validate_config(evaluator, OliveEvaluatorConfig) if evaluator else None

        self.cache_config = validate_config(cache_config, CacheConfig) if cache_config else CacheConfig()
        self.cache: OliveCache = self.cache_config.create_cache(workflow_id)

        self.plot_pareto_frontier = plot_pareto_frontier
        self.azureml_client_config = azureml_client_config

        # dictionary of passes
        self.pass_config = OrderedDict()

        # {"pass_name": {"pass": pass, "host": host, "evaluator": evaluator}
        self.passes = OrderedDict()
        self.pass_flows = None
        self.pass_flows_search_spaces = None

        self.footprints = defaultdict(Footprint)

        self._initialized = False

    def initialize(self, log_to_file: bool = False, log_severity_level: int = 1):
        """Initialize engine state. This should be done before running the registered passes."""
        if log_to_file:
            enable_filelog(log_severity_level, self.cache.cache_dir, self.workflow_id)

        # set cache dir environment variables
        # might be used by other parts of olive to cache data
        self.cache.set_cache_env()

        # prepare non-local resources if host/target is not AzureML
        # TODO(anyone): Should the shared cache care about this? If so, the shared cache helper can
        # check for cached non-local resource paths and replace them with the original config
        # during hash calculation.
        if self.target_config.type != SystemType.AzureML:
            if self.evaluator_config:
                self.evaluator_config = self.cache.prepare_resources_for_local(self.evaluator_config)
            for pass_config in self.pass_config.values():
                if pass_config["evaluator"]:
                    pass_config["evaluator"] = self.cache.prepare_resources_for_local(pass_config["evaluator"])

        for pass_config in self.pass_config.values():
            host_type = pass_config["host"].system_type if pass_config["host"] else self.host_config.type
            if host_type == SystemType.AzureML:
                continue
            pass_config["config"] = self.cache.prepare_resources_for_local(pass_config["config"])

        self.set_pass_flows(self.pass_flows)
        self._initialized = True

    def register(
        self,
        pass_type: Type["Pass"],
        config: Dict[str, Any] = None,
        name: str = None,
        host: "OliveSystem" = None,
        evaluator_config: "OliveEvaluatorConfig" = None,
    ):
        """Register a pass configuration so that it could be instantiated and executed later."""
        if name is not None:
            assert name not in self.passes, f"Pass with name {name} already registered"
        else:
            idx = 0
            while True:
                name = pass_type.__name__
                if idx > 0:
                    name = f"{name}_{idx}"
                idx += 1
                if name not in self.pass_config:
                    break

        self.pass_config[name] = {
            "type": pass_type,
            "config": config or {},
            "host": host,
            "evaluator": evaluator_config,
        }

    def register_pass(
        self, p: "Pass", name: str = None, host: "OliveSystem" = None, evaluator_config: "OliveEvaluatorConfig" = None
    ):
        """Register a pass instance."""
        if name is not None:
            assert name not in self.passes, f"Pass with name {name} already registered"
        else:
            idx = 0
            while True:
                name = p.__class__.__name__
                if idx > 0:
                    name = f"{name}_{idx}"
                idx += 1
                if name not in self.passes:
                    break

        if not self.search_strategy and len(p.search_space) > 0:
            raise ValueError(f"Search strategy is None but pass {name} has search space")

        self.passes[name] = {"pass": p, "host": host, "evaluator": evaluator_config}

    def set_pass_flows(self, pass_flows: List[List[str]] = None):
        """Construct pass flows from a list of pass names.

        Args:
            pass_flows: a list of pass names, each pass name is a string.

        """
        if not pass_flows:
            self.pass_flows = [list(self.pass_config.keys())] if self.pass_config else []
        else:
            self.pass_flows = pass_flows

    def run(
        self,
        input_model_config: ModelConfig,
        accelerator_specs: List["AcceleratorSpec"],
        packaging_config: Optional[Union["PackagingConfig", List["PackagingConfig"]]] = None,
        output_dir: str = None,
        evaluate_input_model: bool = True,
        log_to_file: bool = False,
        log_severity_level: int = 1,
    ):
        """Run all the registered Olive passes on the input model and produce one or more candidate models.

        Args:
            input_model_config: input Olive model configuration
            accelerator_specs: list of accelerator specs
            packaging_config: packaging configuration, if packaging_config is provided, the output
                model will be packaged into a zip file.
            output_dir: output directory for the output model
            evaluate_input_model: if evaluate_input_model is True, run the evaluation on the input model.
            log_to_file: if save logs to a file.
            log_severity_level: severity level of the logger.

        Return:
            Search mode:
                1. One accelerator spec:
                    output_dir/footprints.json: footprint of the run
                    output_dir/pareto_frontier_footprints.json: pareto frontier footprints
                    output_dir/run_history.txt: run history
                    output_dir/input_model_metrics.json: evaluation results of the input model

                2. Multiple accelerator specs:
                    output_dir/{acclerator_spec}/...: Same as 1 but for each accelerator spec

            No search mode:
                1. One accelerator spec
                    output_dir/footprints.json: footprint of the run
                    output_dir/run_history.txt: run history
                    output_dir/input_model_metrics.json: evaluation results of the input model
                    output_dir/output_footprints.json: footprint of the output models

                    A. One pass flow:
                        output_dir/output_model/metrics.json: evaluation results of the output model
                        output_dir/output_model/model_config.json: output model configuration
                        output_dir/output_model/...: output model files

                    B. Multiple pass flows:
                        output_dir/output_model/{pass_flow}/...: Same as A but for each pass flow

                2. Multiple accelerator specs
                    output_dir/{acclerator_spec}/...: Same as 1 but for each accelerator spec

        """
        if not accelerator_specs:
            raise ValueError("No accelerator specified")

        if not self._initialized:
            self.initialize(log_to_file, log_severity_level)

        output_dir: Path = (Path(output_dir) if output_dir else Path.cwd()).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}
        output_subdirs = {}
        for accelerator_spec in accelerator_specs:
            logger.info("Running Olive on accelerator: %s", accelerator_spec)
            output_subdirs[accelerator_spec] = accelerator_output_dir = (
                output_dir / str(accelerator_spec) if len(accelerator_specs) > 1 else output_dir
            )
            accelerator_output_dir.mkdir(parents=True, exist_ok=True)
            with self._create_system(accelerator_spec):
                run_result = self.run_accelerator(
                    input_model_config,
                    accelerator_output_dir,
                    evaluate_input_model,
                    accelerator_spec,
                )

                if run_result is None:
                    continue

                outputs[accelerator_spec] = run_result

        for accelerator_spec in self.footprints:
            logger.info("Run history for %s:", accelerator_spec)
            run_history = self.footprints[accelerator_spec].summarize_run_history()
            self.dump_run_history(run_history, output_subdirs[accelerator_spec] / "run_history.txt")

        if packaging_config and self.passes:
            # TODO(trajep): should we support packaging pytorch model?
            logger.info("Package top ranked %d models as artifacts", sum(len(f.nodes) for f in outputs.values()))
            generate_output_artifacts(
                packaging_config,
                self.footprints,
                outputs,
                output_dir,
                self.azureml_client_config,
            )
        else:
            logger.debug("No packaging config provided, skip packaging artifacts")

        return outputs

    def run_accelerator(
        self,
        input_model_config: ModelConfig,
        output_dir: Path,
        evaluate_input_model: bool,
        accelerator_spec: "AcceleratorSpec",
    ):
        # generate search space and initialize the passes for each hardware accelerator
        self.setup_passes(accelerator_spec)
        # hash the input model
        input_model_id = input_model_config.get_model_id()
        if input_model_id == LOCAL_INPUT_MODEL_ID and self.cache.enable_shared_cache:
            logger.warning("Input model has callable attributes, shared cache is disabled.")
            self.cache.disable_shared_cache()

        self.footprints[accelerator_spec].record(model_id=input_model_id)

        # create the output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if evaluate_input_model and not self.evaluator_config:
                logger.debug("evaluate_input_model is True but no evaluator provided. Skipping input model evaluation.")
            elif evaluate_input_model:
                results = self._evaluate_model(
                    input_model_config, input_model_id, self.evaluator_config, accelerator_spec
                )
                logger.info("Input model evaluation results: %s", results)
                results_path = output_dir / "input_model_metrics.json"
                with results_path.open("w") as f:
                    json.dump(results.to_json(), f, indent=4)
                logger.info("Saved evaluation results of input model to %s", results_path)
                if not self.passes:
                    logger.debug("No passes registered, return input model evaluation results.")
                    return results

            if self.search_strategy:
                logger.debug("Running Olive in search mode ...")
                output_footprint = self.run_search(
                    input_model_config,
                    input_model_id,
                    accelerator_spec,
                    output_dir,
                )
            else:
                logger.debug("Running Olive in no-search mode ...")
                output_footprint = self.run_no_search(input_model_config, input_model_id, accelerator_spec, output_dir)
        except EXCEPTIONS_TO_RAISE:
            raise
        except Exception:
            logger.warning("Failed to run Olive on %s.", accelerator_spec, exc_info=True)
            return None

        output_fp_path = output_dir / "footprints.json"
        logger.info("Save footprint to %s.", output_fp_path)
        self.footprints[accelerator_spec].to_file(output_fp_path)
        logger.debug("run_accelerator done")
        return output_footprint

    def get_host_device(self):
        if self.host_config.config.accelerators:
            # for host device, we will always use the first accelerator device
            return self.host_config.config.accelerators[0].device
        else:
            return None

    def setup_passes(self, accelerator_spec: "AcceleratorSpec"):
        host_device = self.get_host_device()
        # clean the passes
        self.passes.clear()
        for name, config in self.pass_config.items():
            pass_cls: Type[Pass] = config["type"]
            pass_cfg = config["config"]
            pass_cfg = pass_cls.generate_search_space(accelerator_spec, pass_cfg, self.search_strategy is None)
            p = pass_cls(accelerator_spec, pass_cfg, host_device)
            self.register_pass(p, name=name, host=config["host"], evaluator_config=config["evaluator"])

        # list of passes starting from the first pass with non-empty search space
        # These passes will be added to the search space
        self.pass_flows_search_spaces = []
        for pass_flow in self.pass_flows:
            pass_search_spaces = []
            for pass_name in pass_flow:
                p: Pass = self.passes[pass_name]["pass"]
                pass_search_spaces.append((pass_name, p.search_space))
            self.pass_flows_search_spaces.append(pass_search_spaces)

    def reset_passes(self):
        """Cleanup the passes."""
        self.passes.clear()
        self.pass_config.clear()
        self.pass_flows = []

    def run_no_search(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec",
        output_dir: Path,
    ):
        """Run all the registered Olive pass flows in no-search mode."""
        for pass_item in self.passes.values():
            if len(pass_item["pass"].search_space) > 0:
                pass_name = pass_item["name"]
                raise ValueError(f"Pass {pass_name} has search space but search strategy is None")

        # output models will be saved in output_dir/output_model
        output_model_dir = Path(output_dir) / "output_model"

        output_model_ids = []
        for pass_flow in self.pass_flows:
            # search point is empty since there is no search
            passes_to_run = [(pass_id, {}) for pass_id in pass_flow]

            # run all the passes in the pass flow
            logger.debug("Running %s with no search ...", pass_flow)
            should_prune, signal, model_ids = self._run_passes(
                passes_to_run,
                input_model_config,
                input_model_id,
                accelerator_spec,
            )

            if should_prune:
                failed_pass = pass_flow[len(model_ids)]
                logger.warning(
                    "Flow %s is pruned due to failed or invalid config for pass '%s'", pass_flow, failed_pass
                )
                continue

            # use output_model_dir if there is only one pass flow
            # else output_model_dir/pass_flow
            flow_output_dir = output_model_dir / "-".join(pass_flow) if len(self.pass_flows) > 1 else output_model_dir
            flow_output_dir.mkdir(parents=True, exist_ok=True)

            if signal is not None:
                results_path = flow_output_dir / "metrics.json"
                with open(results_path, "w") as f:
                    json.dump(signal.to_json(), f, indent=4)
                logger.info("Saved evaluation results of output model to %s", results_path)

            self.cache.save_model(model_id=model_ids[-1], output_dir=flow_output_dir, overwrite=True)
            logger.info("Saved output model to %s", flow_output_dir)

            output_model_ids.append(model_ids[-1])

        output_footprints = self.footprints[accelerator_spec].create_footprints_by_model_ids(output_model_ids)
        output_footprints.to_file(output_dir / "output_footprints.json")
        return output_footprints

    def run_search(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec",
        output_dir: Path,
    ):
        """Run all the registered Olive passes in search model where search strategy is not None."""
        # get objective_dict
        evaluator_config = self.evaluator_for_pass(list(self.passes.keys())[-1])

        if evaluator_config is None:
            raise ValueError("No evaluator provided for the last pass")
        else:
            objective_dict = self.resolve_objectives(
                input_model_config, input_model_id, evaluator_config.metrics, accelerator_spec
            )
            self.footprints[accelerator_spec].record_objective_dict(objective_dict)

        # initialize the search strategy
        self.search_strategy.initialize(self.pass_flows_search_spaces, input_model_id, objective_dict)
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
                model_config = input_model_config
            else:
                model_config = self._load_model(model_id)

            logger.debug("Step %d with search point %s ...", iter_num, next_step["search_point"])

            # run all the passes in the step
            should_prune, signal, model_ids = self._run_passes(
                next_step["passes"],
                model_config,
                model_id,
                accelerator_spec,
            )

            # record feedback signal
            self.search_strategy.record_feedback_signal(next_step["search_point"], signal, model_ids, should_prune)

            time_diff = time.time() - start_time
            self.search_strategy.check_exit_criteria(iter_num, time_diff, signal)

        return self.create_pareto_frontier_footprints(accelerator_spec, output_model_num, output_dir)

    def create_pareto_frontier_footprints(
        self, accelerator_spec: "AcceleratorSpec", output_model_num: int, output_dir: Path
    ):
        pf_footprints = self.footprints[accelerator_spec].create_pareto_frontier(output_model_num)
        if not pf_footprints:
            return None
        pf_footprints.to_file(output_dir / "pareto_frontier_footprints.json")

        if self.plot_pareto_frontier:
            pf_footprints.plot_pareto_frontier_to_html(save_path=output_dir / "pareto_frontier_footprints_chart.html")

        return pf_footprints

    def dump_run_history(self, run_history, output_path: Path):
        if not run_history:
            logger.info("No run history to dump!")
            return
        headers = run_history[0]._fields
        try:
            from tabulate import tabulate

            formatted_rls = tabulate([tuple(rh) for rh in run_history], headers=headers, tablefmt="grid")
            logger.info("run history:\n%s", formatted_rls)
        except ImportError:
            logger.info("Please install tabulate for better run history output")
            formatted_rls = run_history
        with Path(output_path).open("w") as f:
            f.write(f"{formatted_rls}")

    def resolve_objectives(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        metrics: List[Metric],
        accelerator_spec: "AcceleratorSpec",
    ) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary of objectives and their higher_is_better and goal values.

        {objective_name: {"higher_is_better": bool, "goal": float}}
        """
        goals = self.resolve_goals(input_model_config, input_model_id, metrics, accelerator_spec)
        objective_dict = {}
        for metric in metrics:
            for sub_type in metric.sub_types:
                if sub_type.priority <= 0:
                    continue
                metric_key = joint_metric_key(metric.name, sub_type.name)
                objective_dict[metric_key] = {
                    "higher_is_better": sub_type.higher_is_better,
                    "goal": goals.get(metric_key),
                    "priority": sub_type.priority,
                }
        return dict(sorted(objective_dict.items(), key=lambda x: x[1]["priority"]))

    def resolve_goals(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        metrics: List[Metric],
        accelerator_spec: "AcceleratorSpec",
    ) -> Dict[str, float]:
        """Resolve the goals of the given metrics into thresholds for the given model."""
        goals = {}
        multipliers = {}
        for metric in metrics:
            # only resolve sub metrics whose priority > 0
            goals[metric.name] = metric.get_sub_type_info("goal")
            multipliers[metric.name] = metric.get_sub_type_info(
                info_name="higher_is_better",
                callback=lambda x: 1 if x else -1,
            )

        if goals:
            logger.debug("Resolving goals: %s", goals)

        baseline = None
        for goal in goals.values():
            _evaluated = False
            for sub_goal in goal.values():
                if not sub_goal:
                    break
                if sub_goal.type != "threshold":
                    assert self.evaluator_config is not None, "Default evaluator must be provided to resolve goals"
                    logger.debug("Computing baseline for metrics ...")
                    baseline = self._evaluate_model(
                        input_model_config, input_model_id, self.evaluator_config, accelerator_spec
                    )
                    _evaluated = True
                    break
            if _evaluated:
                break
        if not baseline:
            logger.debug("No baseline got as no goal is provided the the goal is threshold")
            return {}

        if baseline:
            logger.debug("Baseline: %s", baseline)

        # resolve goals to thresholds
        resolved_goals = {}
        for metric_name, sub_type_goals in goals.items():
            for sub_type_name, goal in sub_type_goals.items():
                # TODO(trajep): make the logic cleaner
                resolved_goal_value = None
                if goal is not None:
                    baseline_sub_type = baseline.get_value(metric_name, sub_type_name)
                    multiplier = multipliers[metric_name][sub_type_name]
                    if goal.type == "threshold":
                        resolved_goal_value = goal.value
                    elif goal.type == "max-degradation":
                        resolved_goal_value = baseline_sub_type - multiplier * goal.value
                    elif goal.type == "min-improvement":
                        resolved_goal_value = baseline_sub_type + multiplier * goal.value
                    elif goal.type == "percent-max-degradation":
                        resolved_goal_value = baseline_sub_type * (1 - multiplier * goal.value / 100)
                    elif goal.type == "percent-min-improvement":
                        resolved_goal_value = baseline_sub_type * (1 + multiplier * goal.value / 100)

                resolved_goals[joint_metric_key(metric_name, sub_type_name)] = resolved_goal_value
        if len(resolved_goals) > 0:
            logger.debug("Resolved goals: %s", resolved_goals)

        return resolved_goals

    def host_for_pass(self, pass_id: str):
        host = self.passes[pass_id]["host"]
        if host is None:
            return self.host
        return host

    def evaluator_for_pass(self, pass_id: str):
        """Return evaluator for the given pass."""
        e = self.passes[pass_id]["evaluator"]
        if e is None:
            return self.evaluator_config
        return e

    def _cache_model(self, model_id: str, model: Union[ModelConfig, str], check_object: bool = True):
        # TODO(trajep): move model/pass run/evaluation cache into footprints
        model_json = {} if model == FAILED_CONFIG else model.to_json(check_object=check_object)
        self.cache.cache_model(model_id, model_json)

    def _load_model(self, model_id: str) -> Union[ModelConfig, str]:
        model_json = self.cache.load_model(model_id)
        if model_json is None:
            return None

        if model_json == {}:
            return FAILED_CONFIG

        return ModelConfig.from_json(model_json)

    def _run_passes(
        self,
        passes: List[Tuple[str, Dict[str, Any]]],
        model_config: ModelConfig,
        model_id: str,
        accelerator_spec: "AcceleratorSpec",
    ):
        """Run all the passes in the order they were registered.

        the passes is the list of (pass_name, pass_search_point) tuples
        """
        should_prune = False
        # run all the passes in the step
        model_ids = []
        pass_id = None

        for pass_id, pass_search_point in passes:
            model_config, model_id = self._run_pass(
                pass_id,
                pass_search_point,
                model_config,
                model_id,
                accelerator_spec,
            )
            if model_config in PRUNED_CONFIGS:
                should_prune = True
                logger.debug("Pruned for pass %s", pass_id)
                break
            model_ids.append(model_id)

        if model_config not in PRUNED_CONFIGS and model_config.config.get("shared_cache", False):
            model_config = self.cache.download_shared_cache_model(model_config, model_id)

        if not should_prune:
            # evaluate the model
            evaluator_config = self.evaluator_for_pass(pass_id)
            if not self.search_strategy and evaluator_config is None:
                # skip evaluation if no search and no evaluator
                signal = None
            else:
                logger.info("Run model evaluation for the final model...")
                signal = self._evaluate_model(model_config, model_id, evaluator_config, accelerator_spec)
            logger.debug("Signal: %s", signal)
        else:
            signal = None
            logger.warning("Skipping evaluation as model was pruned")

        return should_prune, signal, model_ids

    def _run_pass(
        self,
        pass_id: str,
        pass_search_point: Dict[str, Any],
        input_model_config: ModelConfig,
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec",
    ):
        """Run a pass on the input model."""
        # pass
        run_start_time = datetime.now().timestamp()
        p: Pass = self.passes[pass_id]["pass"]
        pass_name = p.__class__.__name__
        logger.info("Running pass %s:%s %s", pass_id, pass_name, pass_search_point)
        pass_config = p.config_at_search_point(pass_search_point)
        pass_config = p.serialize_config(pass_config)
        output_model_config = None

        # check whether the config is valid
        if not p.validate_search_point(pass_search_point, accelerator_spec, with_fixed_value=True):
            logger.warning("Invalid search point, prune")
            output_model_config = INVALID_CONFIG
            # no need to record in footprint since there was no run and thus no valid/failed model
            # invalid configs are also not cached since the same config can be valid for other accelerator specs
            # a pass can be accelerator agnostic but still have accelerator specific invalid configs
            # this helps reusing cached models for different accelerator specs
            return output_model_config, None

        # load run from cache if it exists
        run_accel = None if p.is_accelerator_agnostic(accelerator_spec) else accelerator_spec
        output_model_id = self.cache.get_output_model_id(pass_name, pass_config, input_model_id, run_accel)
        run_cache = self.cache.load_run_from_model_id(output_model_id)
        if run_cache:
            logger.debug("Loading model from cache ...")
            output_model_config = self._load_model(output_model_id)
            if output_model_config is not None:
                # footprint model and run
                self.footprints[accelerator_spec].record(
                    model_id=output_model_id,
                    model_config=(
                        output_model_config.to_json() if output_model_config != FAILED_CONFIG else {"is_pruned": True}
                    ),
                    parent_model_id=input_model_id,
                    from_pass=pass_name,
                    pass_run_config=pass_config,
                    start_time=run_start_time,
                    end_time=datetime.now().timestamp(),
                )
                logger.info("Loaded model from cache: %s", output_model_id)
                return output_model_config, output_model_id

        output_model_path = str(self.cache.get_model_cache_path(output_model_id))
        if input_model_config.config.get("shared_cache", False):
            input_model_config = self.cache.download_shared_cache_model(input_model_config, input_model_id)

        host = self.host_for_pass(pass_id)
        if host.system_type != SystemType.AzureML:
            input_model_config = self.cache.prepare_resources_for_local(input_model_config)

        try:
            if p.run_on_target:
                if self.target.system_type == SystemType.IsolatedORT:
                    logger.warning(
                        "Cannot run pass %s on IsolatedORT target, will use the host to run the pass.", pass_id
                    )
                else:
                    host = self.target

            output_model_config = host.run_pass(p, input_model_config, output_model_path, pass_search_point)
        except OlivePassError:
            logger.exception("Pass run_pass failed")
            output_model_config = FAILED_CONFIG
        except EXCEPTIONS_TO_RAISE:
            # Don't catch these errors since most of time, it is caused by the user errors and need not retry.
            raise
        except Exception:
            output_model_config = FAILED_CONFIG
            # TODO(jambayk): from the time being, we need to catch all exceptions to make the
            #      search process robust. We need rethrow the exception only when
            #      it is not pass specific. For example, for olive bugs and user errors
            logger.exception("Pass run failed.")
            if not self.search_strategy:
                raise  # rethrow the exception if no search is performed

        run_end_time = datetime.now().timestamp()
        logger.info("Pass %s:%s finished in %f seconds", pass_id, pass_name, run_end_time - run_start_time)

        # cache model
        self._cache_model(output_model_id, output_model_config)

        # cache run
        self.cache.cache_run(pass_name, pass_config, input_model_id, output_model_id, run_accel)

        # footprint model and run
        self.footprints[accelerator_spec].record(
            model_id=output_model_id,
            model_config=output_model_config.to_json() if output_model_config != FAILED_CONFIG else {"is_pruned": True},
            parent_model_id=input_model_id,
            from_pass=pass_name,
            pass_run_config=pass_config,
            start_time=run_start_time,
            end_time=run_end_time,
        )

        return output_model_config, output_model_id

    def _cache_evaluation(self, model_id: str, signal: MetricResult):
        """Cache the evaluation in the cache directory."""
        evaluation_json = {
            "model_id": model_id,
            "signal": signal.dict(),
        }
        self.cache.cache_evaluation(model_id, evaluation_json)

    def _load_evaluation(self, model_id: str):
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

    def _evaluate_model(
        self,
        model_config: ModelConfig,
        model_id: str,
        evaluator_config: "OliveEvaluatorConfig",
        accelerator_spec: "AcceleratorSpec",
    ):
        """Evaluate a model."""
        logger.debug("Evaluating model ...")
        accelerator_suffix = f"-{accelerator_spec}" if accelerator_spec else ""
        if not model_id.endswith(accelerator_suffix):
            # append the suffix if the model is accelerator independent
            model_id_with_accelerator = f"{model_id}{accelerator_suffix}"
        else:
            model_id_with_accelerator = model_id

        # load evaluation from cache if it exists
        signal = self._load_evaluation(model_id_with_accelerator)
        if signal is not None:
            logger.debug("Loading evaluation from cache ...")
            # footprint evaluation
            self.footprints[accelerator_spec].record(
                model_id=model_id,
                metrics=FootprintNodeMetric(
                    value=signal,
                    if_goals_met=False,
                ),
            )
            return signal

        # evaluate model
        if self.target.system_type != SystemType.AzureML:
            model_config = self.cache.prepare_resources_for_local(model_config)
        signal = self.target.evaluate_model(model_config, evaluator_config, accelerator_spec)

        # cache evaluation
        self._cache_evaluation(model_id_with_accelerator, signal)

        # footprint evaluation
        self.footprints[accelerator_spec].record(
            model_id=model_id,
            metrics=FootprintNodeMetric(
                value=signal,
                if_goals_met=False,
            ),
        )
        return signal

    @contextmanager
    def _create_system(self, accelerator_spec):
        def create_system(config: "SystemConfig", accelerator_spec):
            assert config, "System config is not provided"
            if config.olive_managed_env:
                logger.debug(
                    "Creating olive_managed_env %s with EP %s", config.type, accelerator_spec.execution_provider
                )
                return create_managed_system_with_cache(config, accelerator_spec)
            else:
                logger.debug("create native OliveSystem %s", config.type)
                return config.create_system()

        if not self.target:
            logger.info("Creating target system ...")
            target_start_time = time.time()
            self.target = create_system(self.target_config, accelerator_spec)
            logger.info("Target system created in %f seconds", time.time() - target_start_time)
        if not self.host:
            host_accelerators = self.host_config.config.accelerators
            if host_accelerators and host_accelerators[0].execution_providers:
                host_accelerator_spec = AcceleratorSpec(
                    host_accelerators[0].device,
                    host_accelerators[0].execution_providers[0],
                    memory=host_accelerators[0].memory,
                )
            else:
                host_accelerator_spec = None
            logger.info("Creating host system ...")
            host_start_time = time.time()
            self.host = create_system(self.host_config, host_accelerator_spec)
            logger.info("Host system created in %f seconds", time.time() - host_start_time)

        yield

        if self.target_config.olive_managed_env:
            # could we put it under cache system for reusing?
            logger.info("Removing target system ...")
            self.target.remove()
            self.target = None
        if self.host_config.olive_managed_env:
            logger.info("Removing host system ...")
            self.host.remove()
            self.host = None

        create_managed_system_with_cache.cache_clear()
