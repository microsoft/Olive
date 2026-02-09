# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from olive.common.constants import LOCAL_INPUT_MODEL_ID
from olive.engine.cache_manager import CacheManager
from olive.engine.footprint import Footprint
from olive.engine.pass_runner import PassRunner
from olive.evaluator.metric import Metric
from olive.evaluator.metric_result import joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.exception import EXCEPTIONS_TO_RAISE
from olive.model import ModelConfig
from olive.search.search_strategy import SearchStrategy

if TYPE_CHECKING:
    from olive.cache import OliveCache
    from olive.hardware import AcceleratorSpec

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    """Orchestrates search-based and no-search optimization workflows."""

    def __init__(
        self,
        search_strategy: Optional[SearchStrategy],
        pass_runner: PassRunner,
        cache_manager: CacheManager,
        cache: "OliveCache",
        footprint: Footprint,
        evaluator_config: Optional[OliveEvaluatorConfig],
        input_passes_configs: dict,
        skip_saving_artifacts: bool = False,
        plot_pareto_frontier: bool = False,
    ):
        self.search_strategy = search_strategy
        self.pass_runner = pass_runner
        self.cache_manager = cache_manager
        self.cache = cache
        self.footprint = footprint
        self.evaluator_config = evaluator_config
        self.input_passes_configs = input_passes_configs
        self.skip_saving_artifacts = skip_saving_artifacts
        self.plot_pareto_frontier = plot_pareto_frontier

    def run(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec",
        artifacts_dir: Path,
        evaluate_input_model: bool = True,
    ):
        """Run the optimization workflow (search or no-search mode).

        Handles input model evaluation, pass execution, and result saving.
        """
        if input_model_id == LOCAL_INPUT_MODEL_ID and self.cache.enable_shared_cache:
            logger.warning("Input model has callable attributes, shared cache is disabled.")
            self.cache.disable_shared_cache()

        self.footprint.record(is_input_model=True, model_id=input_model_id)

        try:
            if evaluate_input_model and not self.evaluator_config:
                logger.debug("evaluate_input_model is True but no evaluator provided. Skipping input model evaluation.")

            elif evaluate_input_model:
                results = self.pass_runner.evaluate_model(
                    input_model_config, input_model_id, self.evaluator_config, accelerator_spec
                )
                logger.info("Input model evaluation results: %s", results)

                if not self.skip_saving_artifacts:
                    results_path = artifacts_dir / "input_model_metrics.json"
                    with results_path.open("w") as f:
                        json.dump(results.to_json(), f, indent=4)
                    logger.info("Saved evaluation results of input model to %s", results_path)

                if not self.input_passes_configs:
                    logger.debug("No passes registered.")
                    return

            if self.search_strategy:
                logger.debug("Running Olive in search mode ...")
                self._run_search(input_model_config, input_model_id, accelerator_spec, artifacts_dir)
            else:
                logger.debug("Running Olive in no-search mode ...")
                self._run_no_search(input_model_config, input_model_id, accelerator_spec, artifacts_dir)
        except EXCEPTIONS_TO_RAISE:
            raise
        except Exception:
            logger.warning("Failed to run Olive on %s.", accelerator_spec, exc_info=True)
            return

        if not self.skip_saving_artifacts:
            output_fp_path = artifacts_dir / "footprint.json"
            logger.info("Save footprint to %s.", output_fp_path)
            self.footprint.to_file(output_fp_path)
        logger.debug("run_accelerator done")

    def _run_no_search(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec",
        artifacts_dir: Path,
    ):
        """Run all the registered Olive pass flows in no-search mode."""
        self._get_search_space_objectives(input_model_config, input_model_id, accelerator_spec)

        # Compute pass configs
        self.pass_runner.compute_no_search_pass_configs(accelerator_spec)

        # run all the passes in the pass flow
        pass_flow = list(self.pass_runner.computed_passes_configs.keys())
        logger.debug("Running %s with no search ...", pass_flow)
        should_prune, signal, model_ids = self.pass_runner.run_passes(
            input_model_config, input_model_id, accelerator_spec
        )

        if should_prune:
            failed_pass = pass_flow[len(model_ids)]
            logger.warning("Flow %s is pruned due to failed or invalid config for pass '%s'", pass_flow, failed_pass)
            return

        if signal is not None and not self.skip_saving_artifacts:
            results_path = artifacts_dir / "metrics.json"
            with open(results_path, "w") as f:
                json.dump(signal.to_json(), f, indent=4)
            logger.info("Saved evaluation results of output model to %s", results_path)

        self.footprint.set_output_model_ids([model_ids[-1]])
        if not self.skip_saving_artifacts:
            self.footprint.to_file(artifacts_dir / "output_footprint.json")

    def _run_search(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec",
        artifacts_dir: Path,
    ):
        """Run all the registered Olive passes in search mode where search strategy is not None."""
        # initialize the search strategy
        search_space_config = self.pass_runner.get_search_space_config(accelerator_spec)
        search_space_objectives = self._get_search_space_objectives(
            input_model_config, input_model_id, accelerator_spec
        )
        self.search_strategy.initialize(search_space_config, input_model_id, search_space_objectives)

        for sample in self.search_strategy:  # pylint: disable=not-an-iterable
            self.pass_runner.compute_search_pass_configs(accelerator_spec, sample)

            should_prune, signal, model_ids = True, None, []
            if self.pass_runner.computed_passes_configs:
                # get the model id of the first input model
                model_id = sample.model_ids[0]
                model_config = (
                    input_model_config if model_id == input_model_id else self.cache_manager.load_model(model_id)
                )

                logger.info(
                    "Step %d with search point %s ...", self.search_strategy.iteration_count, sample.search_point
                )

                try:
                    # run all the passes in the step
                    should_prune, signal, model_ids = self.pass_runner.run_passes(
                        model_config, model_id, accelerator_spec
                    )
                except Exception:
                    logger.warning(
                        "Step %d search point %s ... FAILED.",
                        self.search_strategy.iteration_count,
                        sample.search_point,
                        exc_info=True,
                    )

            # record feedback signal
            self.search_strategy.record_feedback_signal(sample.search_point.index, signal, model_ids, should_prune)

        self._create_pareto_frontier_footprint(artifacts_dir)

    def _get_search_space_objectives(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec",
    ) -> dict[str, dict[str, dict[str, Any]]]:
        # NOTE: Olive config doesn't easily lend itself to enforcing one evaluator across
        # multiple pass run configs since each can have its own. That freedom creates some
        # bad unexpected scenarios for search. If two or more pass run configs in the same
        # pass group dictates different objectives (and thus different goals), there is no
        # way to resolve them. To keep things simple for the time being, the objectives
        # across all pass run configs within a pass group are merged by name (so the last
        # one) in the group will win.
        objectives_by_pass_name: dict[str, dict[str, dict[str, Any]]] = {}
        objectives_by_evaluator_name: dict[str, dict[str, Any]] = {}
        for pass_name, passes_configs in self.input_passes_configs.items():
            objectives_by_pass_name[pass_name] = passes_objectives = {}
            for pass_config in passes_configs:
                evaluator_config = pass_config.evaluator or self.evaluator_config
                if evaluator_config:
                    if evaluator_config.name not in objectives_by_evaluator_name:
                        objectives_by_evaluator_name[evaluator_config.name] = self.resolve_objectives(
                            input_model_config, input_model_id, evaluator_config.metrics, accelerator_spec
                        )
                    passes_objectives.update(objectives_by_evaluator_name[evaluator_config.name])

        accelerator_objectives: dict[str, Any] = {}
        for objectives in objectives_by_evaluator_name.values():
            accelerator_objectives.update(objectives)
        self.footprint.record_objective_dict(accelerator_objectives)
        return objectives_by_pass_name

    def resolve_objectives(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        metrics: list[Metric],
        accelerator_spec: "AcceleratorSpec",
    ) -> dict[str, dict[str, Any]]:
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
        return OrderedDict(sorted(objective_dict.items(), key=lambda x: x[1]["priority"]))

    def resolve_goals(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        metrics: list[Metric],
        accelerator_spec: "AcceleratorSpec",
    ) -> dict[str, float]:
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
                    baseline = self.pass_runner.evaluate_model(
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

    def _create_pareto_frontier_footprint(self, artifacts_dir: Path):
        self.footprint.create_pareto_frontier()
        if not self.footprint.output_model_ids:
            return
        if self.plot_pareto_frontier:
            self.footprint.plot_pareto_frontier_to_html(
                save_path=artifacts_dir / "pareto_frontier_footprint_chart.html"
            )

    def dump_run_history(self, run_history, output_path: Path):
        from olive.logging import get_verbosity, set_verbosity

        def _dump_run_history_internal():
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
            if not self.skip_saving_artifacts:
                with Path(output_path).open("w") as f:
                    f.write(f"{formatted_rls}")

        verbosity = get_verbosity()
        set_verbosity(logging.INFO)
        _dump_run_history_internal()
        set_verbosity(verbosity)
