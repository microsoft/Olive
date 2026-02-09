# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from olive.cache import CacheConfig, OliveCache
from olive.common.config_utils import validate_config
from olive.common.constants import DEFAULT_WORKFLOW_ID
from olive.engine.cache_manager import CacheManager
from olive.engine.config import RunPassConfig
from olive.engine.footprint import Footprint
from olive.engine.output import WorkflowOutput
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.engine.pass_runner import PassRunner
from olive.engine.search_orchestrator import SearchOrchestrator
from olive.evaluator.metric import Metric
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.logging import enable_filelog
from olive.model import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.search.search_strategy import SearchStrategy, SearchStrategyConfig
from olive.systems.common import SystemType
from olive.systems.system_config import SystemConfig

if TYPE_CHECKING:
    from olive.engine.packaging.packaging_config import PackagingConfig
    from olive.hardware import AcceleratorSpec
    from olive.passes.olive_pass import Pass

logger = logging.getLogger(__name__)


class Engine:
    """The engine executes the registered Olive Steps.

    It facilitate evaluation of the output models using provided evaluation criteria and produces output model(s).
    """

    def __init__(
        self,
        olive_config: OlivePackageConfig = None,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
        search_strategy: Optional[Union[dict[str, Any], SearchStrategyConfig]] = None,
        host: Optional[Union[dict[str, Any], "SystemConfig"]] = None,
        target: Optional[Union[dict[str, Any], "SystemConfig"]] = None,
        evaluator: Optional[Union[dict[str, Any], OliveEvaluatorConfig]] = None,
        cache_config: Optional[Union[dict[str, Any], CacheConfig]] = None,
        plot_pareto_frontier: bool = False,
        no_artifacts: bool = False,
    ):
        self.olive_config = olive_config or OlivePackageConfig.load_default_config()
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
        self.cache_manager = CacheManager(self.cache)

        self.plot_pareto_frontier = plot_pareto_frontier
        self.skip_saving_artifacts = no_artifacts

        self.input_passes_configs: dict[str, list[RunPassConfig]] = OrderedDict()
        self.computed_passes_configs: dict[str, RunPassConfig] = OrderedDict()
        self.footprint: Footprint = Footprint()

        self._initialized = False
        self.pass_runner: Optional[PassRunner] = None
        self.search_orchestrator: Optional[SearchOrchestrator] = None

    def initialize(self, log_to_file: bool = False, log_severity_level: int = 1):
        """Initialize engine state. This should be done before running the registered passes."""
        if log_to_file:
            enable_filelog(log_severity_level, self.cache.dirs.cache_dir, self.workflow_id)

        # set cache dir environment variables
        # might be used by other parts of olive to cache data
        self.cache.set_cache_env()

        # prepare non-local resources
        # TODO(anyone): Should the shared cache care about this? If so, the shared cache helper can
        # check for cached non-local resource paths and replace them with the original config
        # during hash calculation.
        if self.evaluator_config:
            self.evaluator_config = self.cache.prepare_resources_for_local(self.evaluator_config)

        for passes_configs in self.input_passes_configs.values():
            for pass_config in passes_configs:
                if pass_config.evaluator:
                    pass_config.evaluator = self.cache.prepare_resources_for_local(pass_config.evaluator)

        for passes_configs in self.input_passes_configs.values():
            for pass_config in passes_configs:
                pass_config.config = self.cache.prepare_resources_for_local(pass_config.config)

        self._initialized = True

        self.pass_runner = PassRunner(
            olive_config=self.olive_config,
            cache=self.cache,
            cache_manager=self.cache_manager,
            footprint=self.footprint,
            host_config=self.host_config,
            target_config=self.target_config,
            evaluator_config=self.evaluator_config,
            input_passes_configs=self.input_passes_configs,
            host=self.host,
            target=self.target,
            search_strategy=self.search_strategy,
        )

        self.search_orchestrator = SearchOrchestrator(
            search_strategy=self.search_strategy,
            pass_runner=self.pass_runner,
            cache_manager=self.cache_manager,
            cache=self.cache,
            footprint=self.footprint,
            evaluator_config=self.evaluator_config,
            input_passes_configs=self.input_passes_configs,
            skip_saving_artifacts=self.skip_saving_artifacts,
            plot_pareto_frontier=self.plot_pareto_frontier,
        )

    def register(
        self,
        pass_type: Union[type["Pass"], str],
        config: dict[str, Any] = None,
        name: str = None,
        host: SystemConfig = None,
        evaluator_config: OliveEvaluatorConfig = None,
    ):
        """Register a pass configuration so that it could be instantiated and executed later."""
        if name:
            assert name not in self.input_passes_configs, f"Pass with name {name} already registered"
        else:
            idx = 0
            while True:
                name = pass_type.__name__
                if idx > 0:
                    name = f"{name}_{idx}"
                idx += 1
                if name not in self.input_passes_configs:
                    break

        pass_type_name = pass_type if isinstance(pass_type, str) else pass_type.__name__
        logger.debug("Registering pass %s:%s", name, pass_type_name)
        self.input_passes_configs[name] = [
            RunPassConfig(
                type=pass_type_name,
                config=config or {},
                host=host,
                evaluator=evaluator_config,
            )
        ]

    def set_input_passes_configs(self, pass_configs: dict[str, list[RunPassConfig]]):
        self.input_passes_configs = pass_configs

    def run(
        self,
        input_model_config: ModelConfig,
        accelerator_spec: "AcceleratorSpec",
        packaging_config: Optional[Union["PackagingConfig", list["PackagingConfig"]]] = None,
        output_dir: str = None,
        evaluate_input_model: bool = True,
        log_to_file: bool = False,
        log_severity_level: int = 1,
    ):
        """Run all the registered Olive passes on the input model and produce one or more candidate models.

        Args:
            input_model_config: input Olive model configuration
            accelerator_spec: accelerator spec
            packaging_config: packaging configuration, if packaging_config is provided, the output
                model will be packaged into a zip file.
            output_dir: output directory for the output model
            evaluate_input_model: if evaluate_input_model is True, run the evaluation on the input model.
            log_to_file: if save logs to a file.
            log_severity_level: severity level of the logger.

        Return:
            Search mode:
                output_dir/footprint.json: footprint of the run
                output_dir/pareto_frontier_footprint.json: pareto frontier footprint
                output_dir/run_history.txt: run history
                output_dir/input_model_metrics.json: evaluation results of the input model
                output_dir/...: output model files

            No search mode:
                output_dir/footprint.json: footprint of the run
                output_dir/run_history.txt: run history
                output_dir/input_model_metrics.json: evaluation results of the input model
                output_dir/output_footprint.json: footprint of the output models
                output_dir/...: output model files

        """
        if not accelerator_spec:
            raise ValueError("No accelerator specified")

        if not self._initialized:
            self.initialize(log_to_file, log_severity_level)

        output_dir: Path = (Path(output_dir) if output_dir else Path.cwd()).resolve()
        if output_dir.suffix:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Determine the directory for artifacts (run_history, etc.)
        # If output_dir is a file path (has suffix), use parent directory
        # Otherwise use output_dir itself
        artifacts_dir = output_dir.parent if output_dir.suffix else output_dir

        logger.info("Running Olive on accelerator: %s", accelerator_spec)
        with self._create_system():
            self.run_accelerator(
                input_model_config,
                output_dir,
                evaluate_input_model,
                accelerator_spec,
            )

        logger.info("Run history for %s:", accelerator_spec)
        run_history = self.footprint.summarize_run_history()
        self.search_orchestrator.dump_run_history(run_history, artifacts_dir / "run_history.txt")

        workflow_output = WorkflowOutput(accelerator_spec, self.footprint)
        if self.input_passes_configs and workflow_output.has_output_model():
            if packaging_config:
                # TODO(trajep): should we support packaging pytorch model?
                logger.info("Package top ranked %d models as artifacts", len(workflow_output.get_output_models()))
                generate_output_artifacts(
                    packaging_config,
                    workflow_output,
                    output_dir,
                )
            else:
                logger.debug("No packaging config provided, skip packaging artifacts")
                best_node = workflow_output.get_best_candidate()
                model_json = self.cache.save_model(model_id=best_node.model_id, output_dir=output_dir, overwrite=True)
                best_node._update_with_model_config(model_json)  # pylint: disable=W0212
                logger.info("Saved output model to %s", output_dir)
        else:
            logger.warning("No output model produced. Please check the log for details.")

        return workflow_output

    def run_accelerator(
        self,
        input_model_config: ModelConfig,
        output_dir: Path,
        evaluate_input_model: bool,
        accelerator_spec: "AcceleratorSpec",
    ):
        artifacts_dir = output_dir.parent if output_dir.suffix else output_dir
        input_model_id = input_model_config.get_model_id()
        self.search_orchestrator.run(
            input_model_config, input_model_id, accelerator_spec, artifacts_dir, evaluate_input_model
        )

    def get_host_device(self):
        """Get the device from the host config."""
        return self.pass_runner.get_host_device()

    def host_for_pass(self, pass_name: str) -> SystemConfig:
        return self.pass_runner.host_for_pass(pass_name)

    def evaluator_for_pass(self, pass_name: str) -> OliveEvaluatorConfig:
        """Return evaluator for the given pass."""
        return self.pass_runner.evaluator_for_pass(pass_name)

    def resolve_objectives(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        metrics: list[Metric],
        accelerator_spec: "AcceleratorSpec",
    ) -> dict[str, dict[str, Any]]:
        """Return a dictionary of objectives and their higher_is_better and goal values."""
        return self.search_orchestrator.resolve_objectives(
            input_model_config, input_model_id, metrics, accelerator_spec
        )

    def resolve_goals(
        self,
        input_model_config: ModelConfig,
        input_model_id: str,
        metrics: list[Metric],
        accelerator_spec: "AcceleratorSpec",
    ) -> dict[str, float]:
        """Resolve the goals of the given metrics into thresholds for the given model."""
        return self.search_orchestrator.resolve_goals(input_model_config, input_model_id, metrics, accelerator_spec)

    @contextmanager
    def _create_system(self):
        def create_system(config: "SystemConfig"):
            assert config, "System config is not provided"
            logger.debug("create native OliveSystem %s", config.type)
            return config.create_system()

        if not self.target:
            logger.info("Creating target system ...")
            target_start_time = time.time()
            self.target = create_system(self.target_config)
            logger.info("Target system created in %f seconds", time.time() - target_start_time)

        if not self.host:
            logger.info("Creating host system ...")
            host_start_time = time.time()
            self.host = create_system(self.host_config)
            logger.info("Host system created in %f seconds", time.time() - host_start_time)

        # Update pass_runner with created systems
        if self.pass_runner:
            self.pass_runner.host = self.host
            self.pass_runner.target = self.target

        yield
