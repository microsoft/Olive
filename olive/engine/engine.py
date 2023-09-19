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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import olive.cache as cache_utils
from olive.common.config_utils import ConfigBase, validate_config
from olive.common.utils import hash_dict
from olive.engine.config import FAILED_CONFIG, INVALID_CONFIG, PRUNED_CONFIGS, EngineConfig
from olive.engine.footprint import Footprint, FootprintNode, FootprintNodeMetric
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.evaluator.metric import Metric, MetricResult, joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.exception import OlivePassException
from olive.hardware import AcceleratorLookup, AcceleratorSpec, Device
from olive.model import ModelConfig
from olive.passes.olive_pass import Pass
from olive.strategy.search_strategy import SearchStrategy
from olive.systems.common import SystemType
from olive.systems.local import LocalSystem
from olive.systems.olive_system import OliveSystem
from olive.systems.utils import create_new_system_with_cache

logger = logging.getLogger(__name__)

EXCEPTIONS_TO_RAISE = (AssertionError, AttributeError, ImportError, TypeError, ValueError)


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
        execution_providers: Optional[List[str]] = None,
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
            # host accelerator is not used, so no need to specify it
            self.host = LocalSystem()

        # engine target
        if target is not None:
            self.target = target
        elif self._config.target is not None:
            self.target = self._config.target.create_system()
        else:
            self.target = LocalSystem()

        if execution_providers is None:
            execution_providers = self._config.execution_providers

        # verify the AzureML system have specified the execution providers
        # Please note we could not use isinstance(target, AzureMLSystem) since it would import AzureML packages.
        if self.target.system_type == SystemType.AzureML and execution_providers is None:
            raise ValueError("AzureMLSystem requires execution providers to be specified.")
        elif execution_providers is None:
            if self.target.system_type in (SystemType.Local, SystemType.PythonEnvironment):
                execution_providers = self.target.get_supported_execution_providers()
            else:
                # for docker system and python system, we default use CPUExecutionProvider
                execution_providers = ["CPUExecutionProvider"]

        self.execution_providers = execution_providers
        # Flatten the accelerators to list of AcceleratorSpec
        accelerators: List[str] = self.target.accelerators
        if accelerators is None:
            inferred_accelerators = AcceleratorLookup.infer_accelerators_from_execution_provider(
                self.execution_providers
            )
            if not inferred_accelerators:
                logger.warning("Cannot infer the accelerators from the target system. Use CPU as default.")
                accelerators = ["CPU"]
            else:
                logger.debug(
                    f"Use inferred accelerators {inferred_accelerators} "
                    f"from given execution providers {self.execution_providers}."
                )
                accelerators = inferred_accelerators

        ep_to_process = set(self.execution_providers)
        self.accelerator_specs: List[AcceleratorSpec] = []
        is_cpu_available = "cpu" in [accelerator.lower() for accelerator in accelerators]
        for accelerator in accelerators:
            device = Device(accelerator.lower())
            skip_get_available_ep = False
            if (
                self.target.system_type in (SystemType.AzureML, SystemType.Docker, SystemType.PythonEnvironment)
                and self.target.olive_managed_env
            ):
                skip_get_available_ep = True
            supported_eps = AcceleratorLookup.get_execution_providers_for_device(device, skip_get_available_ep)
            for ep in ep_to_process.copy():
                if ep == "CPUExecutionProvider" and device != "cpu" and is_cpu_available:
                    logger.info("ignore the CPUExecutionProvider for non-cpu device")
                elif ep in supported_eps:
                    self.accelerator_specs.append(AcceleratorSpec(device, ep))
                    ep_to_process.remove(ep)

        assert self.accelerator_specs, (
            "No valid accelerator specified for target system. "
            "Please specify the accelerators in the target system or provide valid execution providers. "
            f"Given execution providers: {self.execution_providers}. "
            f"Current accelerators: {accelerators}."
            f"Supported execution providers: {AcceleratorLookup.EXECUTION_PROVIDERS}."
        )
        if ep_to_process:
            logger.warning(
                f"The following execution provider is not supported: {','.join(ep_to_process)}. "
                "Please consider installing an onnxruntime build that contains the relevant execution providers. "
            )

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

        self.pass_flows = None
        self.pass_flows_search_spaces = None

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

        self._model_cache_path, self._run_cache_path, self._evaluation_cache_path, _ = cache_utils.get_cache_sub_dirs(
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

        self.set_pass_flows(self.pass_flows)
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
        output_name: str = None,
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
            "output_name": output_name,
        }

    def register_pass(
        self,
        p: Pass,
        name: str = None,
        host: OliveSystem = None,
        evaluator_config: OliveEvaluatorConfig = None,
        output_name: str = None,
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
        if output_name and not self.no_search:
            # In no-search mode, if output_name is provided, the output model of the pass will be saved to
            # engine's output_dir with the prefix of output_name.
            logger.debug(f"output_name {output_name} for pass {name} will be ignored if search strategy is None")

        self.passes[name] = {
            "pass": p,
            "host": host,
            "evaluator": evaluator_config,
            "output_name": output_name,
        }

    def set_pass_flows(self, pass_flows: List[List[str]] = None):
        """
        Construct pass flows from a list of pass names.
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
        data_root: str = None,
        packaging_config: Optional[PackagingConfig] = None,
        output_dir: str = None,
        output_name: str = None,
        evaluate_input_model: bool = True,
    ):
        """
        Run all the registered Olive passes on the input model and produce one or more candidate models.

        Args:
            input_model_config: input Olive model configuration
            packaging_config: packaging configuration, if packaging_config is provided, the output
                model will be packaged into a zip file.
            output_dir: output directory for the output model
            output_name: output name for the output model, if output_name is provided, the output
                model will be saved to engine's output_dir with the prefix of output_name.
            evaluate_input_model: if evaluate_input_model is True, run the evaluation on the input model.

        Return:
            if search strategy is None, all passes are run in the order they were registered.
                1. Final model -> {output_dir}/{output_name}_{AcceleratorSpec}_model.onnx
                2. JSON file -> {output_dir}/{output_name}_{AcceleratorSpec}_model.json
                3. Evaluation results of the final model -> {output_dir}/{output_name}_{AcceleratorSpec}_metrics.json

            Return footprint/zip(packaging_config) of the final model and evaluation results of the final model.

            if search strategy is not None, run the search strategy to find candidate models.
            Return footprint/zip(packaging_config) of candidate models and evaluation results.
        """
        if not self._initialized:
            self.initialize()

        output_dir: Path = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        for accelerator_spec in self.accelerator_specs:
            with self.create_managed_environment(accelerator_spec):
                run_result = self.run_accelerator(
                    input_model_config, data_root, output_dir, output_name, evaluate_input_model, accelerator_spec
                )

                if run_result is None:
                    continue

                outputs[accelerator_spec] = run_result

        for accelerator_spec in self.footprints.keys():
            logger.info(f"Run history for {accelerator_spec}:")
            run_history = self.footprints[accelerator_spec].summarize_run_history()
            self.dump_run_history(run_history, output_dir / f"run_history_{accelerator_spec}.txt")

        if packaging_config and self.passes:
            # TODO: should we support package input model?
            # TODO: do you support packaging pytorch models?
            logger.info(f"Package top ranked {sum([len(f.nodes) for f in outputs.values()])} models as artifacts")
            generate_output_artifacts(
                packaging_config,
                self.footprints,
                outputs,
                output_dir,
            )
        else:
            logger.info("No packaging config provided, skip packaging artifacts")

        return outputs

    def run_accelerator(
        self,
        input_model_config: ModelConfig,
        data_root: str,
        output_dir: str,
        output_name: str,
        evaluate_input_model: bool,
        accelerator_spec: AcceleratorSpec,
    ):
        # generate search space and initialize the passes for each hardware accelerator
        self.setup_passes(accelerator_spec)

        # hash the input model
        input_model_id = self._init_input_model(input_model_config)
        self.footprints[accelerator_spec].record(model_id=input_model_id)

        try:
            if evaluate_input_model:
                prefix_output_name = (
                    f"{output_name}_{accelerator_spec}_" if output_name is not None else f"{accelerator_spec}"
                )
                assert self.evaluator_config is not None, "evaluate_input_model is True but no evaluator provided"
                results = self._evaluate_model(
                    input_model_config, input_model_id, data_root, self.evaluator_config, accelerator_spec
                )
                logger.info(f"Input model evaluation results: {results}")
                result_name = f"{prefix_output_name}_input_model_metrics"
                results_path = output_dir / f"{result_name}.json"
                with open(results_path, "w") as f:
                    json.dump(results.to_json(), f, indent=4)
                logger.info(f"Saved evaluation results of input model to {results_path}")
                if not self.passes:
                    logger.debug("No passes registered, return input model evaluation results.")
                    return results

            if self.no_search:
                return self.run_no_search(
                    input_model_config,
                    input_model_id,
                    data_root,
                    accelerator_spec,
                    output_dir,
                    output_name,
                )
            else:
                return self.run_search(
                    input_model_config,
                    input_model_id,
                    data_root,
                    accelerator_spec,
                    output_dir,
                    output_name,
                )
        except EXCEPTIONS_TO_RAISE:
            raise
        except Exception as e:
            logger.warning(f"Failed to run Olive on {accelerator_spec}: {e}", exc_info=True)
            return None

    def setup_passes(self, accelerator_spec: AcceleratorSpec):
        # clean the passes
        self.passes.clear()
        for name, config in self.pass_config.items():
            pass_cls: Type[Pass] = config["type"]
            pass_cfg = config["config"]
            pass_cfg = pass_cls.generate_search_space(accelerator_spec, pass_cfg, config["disable_search"])
            p = pass_cls(accelerator_spec, pass_cfg, config["disable_search"])
            self.register_pass(
                p,
                name=name,
                host=config["host"],
                evaluator_config=config["evaluator"],
                output_name=config["output_name"],
            )

        # list of passes starting from the first pass with non-empty search space
        # These passes will be added to the search space
        self.pass_flows_search_spaces = []
        for pass_flow in self.pass_flows:
            self.pass_search_spaces = []
            for pass_name in pass_flow:
                p: Pass = self.passes[pass_name]["pass"]
                self.pass_search_spaces.append((pass_name, p.search_space()))
            self.pass_flows_search_spaces.append(self.pass_search_spaces)

    def run_no_search(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        data_root: str,
        accelerator_spec: AcceleratorSpec,
        output_dir: str = None,
        output_name: str = None,
    ):
        """
        Run all the registered Olive passes in no-search model where search strategy is None.
        """
        assert (
            self.search_strategy._config.execution_order == "joint"
        ), "run_no_search only supports default joint execution order"

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
                input_model_config, input_model_id, data_root, evaluator_config.metrics, accelerator_spec
            )

        # initialize the search strategy
        self.search_strategy.initialize(self.pass_flows_search_spaces, input_model_id, objective_dict)

        iter_num = 0
        output_models = {}
        while True:
            iter_num += 1

            # get the next step
            next_step = self.search_strategy.next_step()

            if iter_num == 1:
                assert next_step is not None, "Search strategy returned None for the first step"
            # if no more steps, break
            if next_step is None:
                break

            assert iter_num <= len(self.pass_flows), "No more pass flows to run"

            # get the model id of the first input model
            model_id = next_step["model_id"]
            if model_id == input_model_id:
                model_config = input_model_config
            else:
                model_config = self._load_model(model_id)

            logger.debug(f"Step no search with search point {next_step['search_point']} ...")

            # run all the passes in the step
            should_prune, signal, model_ids = self._run_passes(
                next_step["passes"], model_config, model_id, data_root, accelerator_spec
            )

            pass_flow = self.pass_flows[iter_num - 1]
            if should_prune:
                failed_pass = pass_flow[len(model_ids)]
                logger.warning(f"Flow {pass_flow} is pruned due to failed or invalid config for pass '{failed_pass}'")
                continue

            # names of the output models of the passes
            pass_output_names = [self.passes[pass_name]["output_name"] for pass_name, _ in next_step["passes"]]
            pass_output_names = [f"{name}_{accelerator_spec}" if name else None for name in pass_output_names]

            # output dir with pass flow
            output_dir_with_pf = Path(output_dir) / "-".join(pass_flow)

            final_output_name = pass_output_names[-1]
            if output_name:
                # override the output name of the last pass
                logger.debug("Engine output_name is provided. Will ignore output_name for final pass")
                final_output_name = f"{output_name}_{accelerator_spec}"
            elif not final_output_name:
                # use the default output name
                final_output_name = f"{accelerator_spec}"
            pass_output_names[-1] = final_output_name

            output_model_json = None
            for pass_output_name, pass_output_model_id in zip(pass_output_names, model_ids):
                if not pass_output_name:
                    continue
                output_model_json = cache_utils.save_model(
                    model_number=pass_output_model_id,
                    output_dir=output_dir_with_pf,
                    output_name=f"{pass_output_name}_model",
                    overwrite=True,
                    cache_dir=self._config.cache_dir,
                )
                output_models[pass_output_model_id] = output_model_json

            # save the evaluation results to output_dir
            if signal is not None:
                results_path = output_dir_with_pf / f"{final_output_name}_metrics.json"
                with open(results_path, "w") as f:
                    json.dump(signal.to_json(), f, indent=4)

        output_model_ids = list(output_models.keys())
        fp_outputs = self.footprints[accelerator_spec].create_footprints_by_model_ids(output_model_ids)
        # update the output model config
        for model_id, model_config in output_models.items():
            fp_outputs.nodes[model_id].model_config = model_config

        return fp_outputs

    def run_search(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        data_root: str,
        accelerator_spec: AcceleratorSpec,
        output_dir: str = None,
        output_name: str = None,
    ):
        """
        Run all the registered Olive passes in search model where search strategy is not None.
        """

        prefix_output_name = f"{output_name}_{accelerator_spec}_" if output_name is not None else f"{accelerator_spec}_"

        # get objective_dict
        evaluator_config = self.evaluator_for_pass(list(self.passes.keys())[-1])

        if evaluator_config is None:
            raise ValueError("No evaluator provided for the last pass")
        else:
            objective_dict = self.resolve_objectives(
                input_model_config, input_model_id, data_root, evaluator_config.metrics, accelerator_spec
            )

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

            logger.debug(f"Step {iter_num} with search point {next_step['search_point']} ...")

            # run all the passes in the step
            should_prune, signal, model_ids = self._run_passes(
                next_step["passes"], model_config, model_id, data_root, accelerator_spec
            )

            # record feedback signal
            self.search_strategy.record_feedback_signal(next_step["search_point"], signal, model_ids, should_prune)

            time_diff = time.time() - start_time
            self.search_strategy.check_exit_criteria(iter_num, time_diff, signal)

        self.footprints[accelerator_spec].to_file(output_dir / f"{prefix_output_name}footprints.json")

        return self.create_pareto_frontier_footprints(
            accelerator_spec, output_model_num, objective_dict, output_dir, prefix_output_name
        )

    def create_pareto_frontier_footprints(
        self, accelerator_spec, output_model_num, objective_dict, output_dir, prefix_output_name
    ):
        pf_footprints = self.footprints[accelerator_spec].create_pareto_frontier()
        if output_model_num is None or len(pf_footprints.nodes) <= output_model_num:
            logger.info(f"Output all {len(pf_footprints.nodes)} models")
        else:
            top_ranked_nodes = self._get_top_ranked_nodes(objective_dict, pf_footprints, output_model_num)
            logger.info(f"Output top ranked {len(top_ranked_nodes)} models based on metric priorities")
            pf_footprints.update_nodes(top_ranked_nodes)

        pf_footprints.to_file(output_dir / f"{prefix_output_name}pareto_frontier_footprints.json")

        if self._config.plot_pareto_frontier:
            pf_footprints.plot_pareto_frontier_to_html(
                save_path=output_dir / f"{prefix_output_name}pareto_frontier_footprints_chart.html"
            )

        return pf_footprints

    def dump_run_history(self, run_history, output_path: str = None):
        if not run_history:
            logger.info("No run history to dump!")
            return
        headers = run_history[0]._fields
        try:
            from tabulate import tabulate

            formatted_rls = tabulate([tuple(rh) for rh in run_history], headers=headers, tablefmt="grid")
            logger.info(f"run history:\n{formatted_rls}")
        except ImportError:
            logger.info("Please install tabulate for better run history output")
            formatted_rls = run_history
        with open(output_path, "w") as f:
            f.write(f"{formatted_rls}")

    def resolve_objectives(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        data_root: str,
        metrics: List[Metric],
        accelerator_spec: AcceleratorSpec,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary of objectives and their higher_is_better and goal values.

        {objective_name: {"higher_is_better": bool, "goal": float}}
        """
        goals = self.resolve_goals(input_model_config, input_model_id, data_root, metrics, accelerator_spec)
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
        self.footprints[accelerator_spec].record_objective_dict(objective_dict)
        ranked_objective_dict = dict(sorted(objective_dict.items(), key=lambda x: x[1]["priority"]))
        return ranked_objective_dict

    def resolve_goals(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        data_root: str,
        metrics: List[Metric],
        accelerator_spec: AcceleratorSpec,
    ) -> Dict[str, float]:
        """
        Resolve the goals of the given metrics into thresholds for the given model.
        """
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
            logger.debug(f"Resolving goals: {goals}")

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
                        input_model_config, input_model_id, data_root, self.evaluator_config, accelerator_spec
                    )
                    _evaluated = True
                    break
            if _evaluated:
                break
        if not baseline:
            logger.debug("No baseline got as no goal is provided the the goal is threshold")
            return {}

        if baseline:
            logger.debug(f"Baseline: {baseline}")

        # resolve goals to thresholds
        resolved_goals = {}
        for metric_name, sub_type_goals in goals.items():
            for sub_type_name, goal in sub_type_goals.items():
                # TODO: make the logic cleaner
                resolved_goal_value = None
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
            logger.debug(f"Resolved goals: {resolved_goals}")

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
            if list(self._model_cache_path.glob(f"{new_model_number}_*")) == []:
                break
        return new_model_number

    def get_model_json_path(self, model_id: str) -> Path:
        """
        Get the path to the model json file.
        """
        return self._model_cache_path / f"{model_id}.json"

    def _cache_model(self, model: Union[ModelConfig, str], model_id: str, check_object: bool = True):
        """
        Cache the model in the cache directory.
        """
        # TODO move model/pass run/evaluation cache into footprints
        if model == FAILED_CONFIG:
            model_json = {}
        else:
            model_json = model.to_json(check_object=check_object)
        model_json_path = self.get_model_json_path(model_id)
        try:
            with open(model_json_path, "w") as f:
                json.dump(model_json, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to cache model: {e}", exc_info=True)

    def _load_model(self, model_id: str) -> Union[ModelConfig, str]:
        """
        Load the model from the cache directory.
        """
        model_json_path = self.get_model_json_path(model_id)
        try:
            with open(model_json_path, "r") as f:
                model_json = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return None

        if model_json == {}:
            return FAILED_CONFIG

        model = ModelConfig.from_json(model_json)
        return model

    def _prepare_non_local_model(self, model_config: ModelConfig) -> ModelConfig:
        """
        Prepare models with non-local model path for local run by downloading the model resources to cache
        """
        # TODO: maybe we can move this method into OliveSystem?
        resource_paths = model_config.get_resource_paths()
        for resource_name, resource_path in resource_paths.items():
            if not resource_path or resource_path.is_local_resource_or_string_name():
                continue
            downloaded_resource_path = cache_utils.download_resource(resource_path, self._config.cache_dir)
            if downloaded_resource_path:
                # set local resource path
                model_config.config[resource_name] = downloaded_resource_path

        return model_config

    def _init_input_model(self, input_model_config: ModelConfig):
        """
        Initialize the input model.
        """
        model_hash = hash_dict(input_model_config.to_json())

        # cache the model
        self._cache_model(input_model_config, model_hash, check_object=False)

        return model_hash

    def get_run_json_path(
        self,
        pass_name: int,
        input_model_number: str,
        pass_config: dict,
        accelerator_spec: AcceleratorSpec,
    ):
        """
        Get the path to the run json.
        """
        pass_config_hash = hash_dict(pass_config)
        if not accelerator_spec:
            run_json_path = self._run_cache_path / f"{pass_name}-{input_model_number}-{pass_config_hash}.json"
        else:
            run_json_path = (
                self._run_cache_path / f"{pass_name}-{input_model_number}-{pass_config_hash}-{accelerator_spec}.json"
            )
        return run_json_path

    def _cache_run(
        self,
        pass_name: int,
        pass_config: dict,
        input_model_id: str,
        output_model_id: str,
        accelerator_spec: AcceleratorSpec,
        run_start_time: float = 0,
        run_end_time: float = 0,
    ):
        """
        Cache the run in the cache directory.
        """
        run_json = {
            "pass_name": pass_name,
            "pass_config": pass_config,
            "input_model_id": input_model_id,
            "output_model_id": output_model_id,
            "run_start_time": run_start_time,
            "run_end_time": run_end_time,
        }
        input_model_number = input_model_id.split("_")[0]
        run_json_path = self.get_run_json_path(pass_name, input_model_number, pass_config, accelerator_spec)
        try:
            with open(run_json_path, "w") as f:
                json.dump(run_json, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to cache run: {e}", exc_info=True)

    def _load_run(self, input_model_id: str, pass_name: int, pass_config: dict, accelerator_spec: AcceleratorSpec):
        """
        Load the run from the cache directory.
        """
        input_model_number = input_model_id.split("_")[0]
        run_json_path = self.get_run_json_path(pass_name, input_model_number, pass_config, accelerator_spec)
        run_json = {}
        if run_json_path.exists():
            try:
                with open(run_json_path, "r") as f:
                    run_json = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load run: {e}", exc_info=True)
                run_json = {}
        return run_json

    def _run_passes(
        self,
        passes: List[Tuple[str, Dict[str, Any]]],
        model_config: ModelConfig,
        model_id: str,
        data_root: str,
        accelerator_spec: AcceleratorSpec,
    ):
        """
        Run all the passes in the order they were registered.
        the passes is the list of (pass_name, pass_search_point) tuples
        """
        should_prune = False
        # run all the passes in the step
        model_ids = []
        for pass_id, pass_search_point in passes:
            model_config, model_id = self._run_pass(
                pass_id, pass_search_point, model_config, model_id, data_root, accelerator_spec
            )
            if model_config in PRUNED_CONFIGS:
                should_prune = True
                logger.debug(f"Pruned for pass {pass_id}")
                break
            model_ids.append(model_id)

        if not should_prune:
            # evaluate the model
            evaluator_config = self.evaluator_for_pass(pass_id)
            if self.no_search and evaluator_config is None:
                # skip evaluation if no search and no evaluator
                signal = None
            else:
                signal = self._evaluate_model(model_config, model_id, data_root, evaluator_config, accelerator_spec)
            logger.debug(f"Signal: {signal}")
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
        data_root: str,
        accelerator_spec: AcceleratorSpec,
    ):
        """
        Run a pass on the input model.
        """
        # pass
        p: Pass = self.passes[pass_id]["pass"]
        pass_name = p.__class__.__name__
        logger.info(f"Running pass {pass_id}:{pass_name}")
        pass_config = p.config_at_search_point(pass_search_point)
        pass_config = p.serialize_config(pass_config)

        # check whether the config is valid
        if not p.validate_search_point(pass_search_point, accelerator_spec, with_fixed_value=True):
            logger.debug("Invalid search point, prune")
            output_model_config = INVALID_CONFIG
            # no need to record in footprint since there was no run and thus no valid/failed model
            # invalid configs are also not cached since the same config can be valid for other accelerator specs
            # a pass can be accelerator agnostic but still have accelerator specific invalid configs
            # this helps reusing cached models for different accelerator specs
            return output_model_config, None

        # load run from cache if it exists
        run_accel = None if p.is_accelerator_agnostic(accelerator_spec) else accelerator_spec
        run_cache = self._load_run(input_model_id, pass_name, pass_config, run_accel)
        output_model_id = run_cache.get("output_model_id", None)
        if output_model_id is not None:
            logger.debug("Loading model from cache ...")
            output_model_config = self._load_model(output_model_id)
            if output_model_config is not None:
                # footprint model and run
                self.footprints[accelerator_spec].record(
                    model_id=output_model_id,
                    model_config=output_model_config.to_json()
                    if output_model_config != FAILED_CONFIG
                    else {"is_pruned": True},
                    parent_model_id=input_model_id,
                    from_pass=pass_name,
                    pass_run_config=pass_config,
                    start_time=run_cache.get("run_start_time", 0),
                    end_time=run_cache.get("run_end_time", 0),
                )
                return output_model_config, output_model_id

        # new model id
        input_model_number = input_model_id.split("_")[0]
        # Note: the final output model id need contains the accelerator information
        # if the output model is accelerator dependent.
        output_model_id_parts = [
            f"{self._get_new_model_number()}_{pass_name}",
            input_model_number,
            hash_dict(pass_config),
        ]

        if not p.is_accelerator_agnostic(accelerator_spec):
            output_model_id_parts.append(f"{accelerator_spec}")

        output_model_id = "-".join(map(str, output_model_id_parts))
        output_model_path = self._model_cache_path / f"{output_model_id}" / "output_model"
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        output_model_path = str(output_model_path)

        # run pass
        host = self.host_for_pass(pass_id)
        if host.system_type != SystemType.AzureML:
            input_model_config = self._prepare_non_local_model(input_model_config)

        run_start_time = datetime.now().timestamp()
        try:
            output_model_config = host.run_pass(p, input_model_config, data_root, output_model_path, pass_search_point)
        except OlivePassException as e:
            logger.error(f"Pass run_pass failed: {e}", exc_info=True)
            output_model_config = FAILED_CONFIG
        except EXCEPTIONS_TO_RAISE:
            # Don't catch these errors since most of time, it is caused by the user errors and need not retry.
            raise
        except Exception:
            output_model_config = FAILED_CONFIG
            # TODO: from the time being, we need to catch all exceptions to make the
            #      search process robust. We need rethrow the exception only when
            #      it is not pass specific. For example, for olive bugs and user errors
            logger.error("Pass run failed.", exc_info=True)
            if self.no_search:
                raise  # rethrow the exception if no search is performed

        run_end_time = datetime.now().timestamp()
        # cache model
        self._cache_model(output_model_config, output_model_id)

        # cache run
        self._cache_run(
            pass_name, pass_config, input_model_id, output_model_id, run_accel, run_start_time, run_end_time
        )

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

    def get_evaluation_json_path(self, model_id: str):
        """
        Get the path to the evaluation json.
        """
        evaluation_json_path = self._evaluation_cache_path / f"{model_id}.json"
        return evaluation_json_path

    def _cache_evaluation(self, model_id: str, signal: MetricResult):
        """
        Cache the evaluation in the cache directory.
        """
        evaluation_json = {
            "model_id": model_id,
            "signal": signal.dict(),
        }
        evaluation_json_path = self.get_evaluation_json_path(model_id)
        try:
            with open(evaluation_json_path, "w") as f:
                json.dump(evaluation_json, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to cache evaluation: {e}", exc_info=True)

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
                signal = MetricResult(**signal)
            except Exception as e:
                logger.error(f"Failed to load evaluation: {e}", exc_info=True)
                signal = None
            return signal
        else:
            return None

    def _evaluate_model(
        self,
        model_config: ModelConfig,
        model_id: str,
        data_root: str,
        evaluator_config: OliveEvaluatorConfig,
        accelerator_spec: AcceleratorSpec,
    ):
        """
        Evaluate a model.
        """
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
                    is_goals_met=False,
                ),
            )
            return signal

        # evaluate model
        metrics = evaluator_config.metrics if evaluator_config else []
        if self.target.system_type != SystemType.AzureML:
            model_config = self._prepare_non_local_model(model_config)
        signal = self.target.evaluate_model(model_config, data_root, metrics, accelerator_spec)

        # cache evaluation
        self._cache_evaluation(model_id_with_accelerator, signal)

        # footprint evaluation
        self.footprints[accelerator_spec].record(
            model_id=model_id,
            metrics=FootprintNodeMetric(
                value=signal,
                is_goals_met=False,
            ),
        )
        return signal

    def _get_top_ranked_nodes(
        self, objective_dict: Dict[str, Any], footprint: Footprint, k: int
    ) -> List[FootprintNode]:
        footprint_node_list = footprint.nodes.values()
        sorted_footprint_node_list = sorted(
            footprint_node_list,
            key=lambda x: tuple(
                x.metrics.value[metric].value
                if x.metrics.cmp_direction[metric] == 1
                else -x.metrics.value[metric].value
                for metric in objective_dict.keys()
            ),
            reverse=True,
        )
        selected_footprint_nodes = sorted_footprint_node_list[:k]
        return selected_footprint_nodes

    @contextmanager
    def create_managed_environment(self, accelerator_spec):
        origin_target = self.target
        origin_host = self.host
        if origin_target.olive_managed_env:
            self.target = create_new_system_with_cache(origin_target, accelerator_spec)
        if origin_host.olive_managed_env:
            self.host = create_new_system_with_cache(origin_host, accelerator_spec)

        yield

        if origin_host.olive_managed_env:
            self.host.remove()
            self.host = origin_host
        if origin_target.olive_managed_env:
            self.target.remove()
            self.target = origin_target

        create_new_system_with_cache.cache_clear()
