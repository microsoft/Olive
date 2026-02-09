# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from olive.cache import OliveCache
from olive.common.config_utils import validate_config
from olive.engine.cache_manager import CacheManager
from olive.engine.config import FAILED_CONFIG, INVALID_CONFIG, PRUNED_CONFIGS, RunPassConfig
from olive.engine.footprint import Footprint, FootprintNodeMetric
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.exception import EXCEPTIONS_TO_RAISE, OlivePassError
from olive.model import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.search.search_sample import SearchSample
from olive.systems.system_config import SystemConfig

if TYPE_CHECKING:
    from olive.hardware import AcceleratorSpec
    from olive.passes.olive_pass import Pass
    from olive.search.search_parameter import SearchParameter
    from olive.search.search_strategy import SearchStrategy

logger = logging.getLogger(__name__)


class PassRunner:
    """Orchestrates execution of optimization passes on models."""

    def __init__(
        self,
        olive_config: OlivePackageConfig,
        cache: OliveCache,
        cache_manager: CacheManager,
        footprint: Footprint,
        host_config: SystemConfig,
        target_config: SystemConfig,
        evaluator_config: Optional[OliveEvaluatorConfig],
        input_passes_configs: dict[str, list[RunPassConfig]],
        host=None,
        target=None,
        search_strategy: Optional["SearchStrategy"] = None,
    ):
        self.olive_config = olive_config
        self.cache = cache
        self.cache_manager = cache_manager
        self.footprint = footprint
        self.host_config = host_config
        self.target_config = target_config
        self.evaluator_config = evaluator_config
        self.input_passes_configs = input_passes_configs
        self.host = host
        self.target = target
        self.search_strategy = search_strategy
        self.computed_passes_configs: dict[str, RunPassConfig] = OrderedDict()

    def get_host_device(self):
        """Get the device from the host config."""
        return self.host_config.config.accelerators[0].device if self.host_config.config.accelerators else None

    def host_for_pass(self, pass_name: str) -> SystemConfig:
        host: SystemConfig = self.computed_passes_configs[pass_name].host
        return host.create_system() if host else self.host

    def evaluator_for_pass(self, pass_name: str) -> OliveEvaluatorConfig:
        """Return evaluator for the given pass."""
        return self.computed_passes_configs[pass_name].evaluator or self.evaluator_config

    def compute_no_search_pass_configs(self, accelerator_spec: "AcceleratorSpec"):
        self.computed_passes_configs.clear()
        for name, passes_configs in self.input_passes_configs.items():
            pass_config = validate_config(passes_configs[0].dict(), RunPassConfig)

            pass_cls: type[Pass] = self.olive_config.import_pass_module(pass_config.type)
            pass_config.config = pass_cls.generate_config(accelerator_spec, pass_config.config, {}, True)
            self.computed_passes_configs[name] = pass_config

    def compute_search_pass_configs(self, accelerator_spec: "AcceleratorSpec", sample: SearchSample):
        self.computed_passes_configs.clear()
        sample_passes_configs = sample.passes_configs
        if not sample_passes_configs:
            return

        disable_pass_params_search = not self.search_strategy.config.include_pass_params
        for pass_name, passes_configs in self.input_passes_configs.items():
            if pass_name in sample_passes_configs:
                sample_pass_config = sample_passes_configs[pass_name]
                pass_config = passes_configs[sample_pass_config["index"]]
                pass_config = validate_config(pass_config.dict(), RunPassConfig)

                pass_cls = self.olive_config.import_pass_module(pass_config.type)
                pass_config.config = pass_cls.generate_config(
                    accelerator_spec,
                    pass_config.config,
                    sample_pass_config["params"],
                    disable_pass_params_search,
                )
                self.computed_passes_configs[pass_name] = pass_config

    def get_search_space_config(
        self, accelerator_spec: "AcceleratorSpec"
    ) -> dict[str, list[dict[str, "SearchParameter"]]]:
        space_config: dict[str, list[dict[str, SearchParameter]]] = OrderedDict()
        for pass_name, passes_configs in self.input_passes_configs.items():
            space_config[pass_name] = pass_params_config = []
            for pass_config in passes_configs:
                pass_cls = self.olive_config.import_pass_module(pass_config.type)
                _, _, search_params = pass_cls.get_config_params(accelerator_spec, pass_config.config, False)
                pass_params_config.append(search_params)
        return space_config

    def run_passes(
        self,
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
        pass_name = None

        for pass_name in self.computed_passes_configs:
            model_config, model_id = self.run_single_pass(
                pass_name,
                model_config,
                model_id,
                accelerator_spec,
            )
            if model_config in PRUNED_CONFIGS:
                should_prune = True
                logger.debug("Pruned for pass %s", pass_name)
                break
            model_ids.append(model_id)

        if model_config not in PRUNED_CONFIGS and model_config.config.get("shared_cache", False):
            model_config = self.cache.download_shared_cache_model(model_config, model_id)

        if not should_prune:
            # evaluate the model
            evaluator_config = self.evaluator_for_pass(pass_name)
            if not self.search_strategy and evaluator_config is None:
                # skip evaluation if no search and no evaluator
                signal = None
            else:
                logger.info("Run model evaluation for the final model...")
                signal = self.evaluate_model(model_config, model_id, evaluator_config, accelerator_spec)
            logger.debug("Signal: %s, %s", signal, model_ids)
        else:
            signal = None
            logger.warning("Skipping evaluation as model was pruned")

        return should_prune, signal, model_ids

    def run_single_pass(
        self,
        pass_name: str,
        input_model_config: ModelConfig,
        input_model_id: str,
        accelerator_spec: "AcceleratorSpec",
    ):
        """Run a pass on the input model."""
        run_start_time = datetime.now().timestamp()

        pass_config: RunPassConfig = self.computed_passes_configs[pass_name]
        pass_type_name = pass_config.type

        logger.info("Running pass %s:%s", pass_name, pass_type_name)

        # check whether the config is valid
        pass_cls: type[Pass] = self.olive_config.import_pass_module(pass_config.type)
        if not pass_cls.validate_config(pass_config.config, accelerator_spec):
            logger.warning("Invalid config, pruned.")
            logger.debug(pass_config)
            return INVALID_CONFIG, None

        p: Pass = pass_cls(accelerator_spec, pass_config.config, self.get_host_device())
        pass_config = p.config.to_json()
        output_model_config = None

        # load run from cache if it exists
        run_accel = None if p.is_accelerator_agnostic(accelerator_spec) else accelerator_spec
        output_model_id = self.cache.get_output_model_id(pass_type_name, pass_config, input_model_id, run_accel)
        run_cache = self.cache.load_run_from_model_id(output_model_id)
        if run_cache:
            logger.debug("Loading model from cache ...")
            output_model_config = self.cache_manager.load_model(output_model_id)
            if output_model_config is not None:
                # footprint model and run
                self.footprint.record(
                    model_id=output_model_id,
                    model_config=(
                        output_model_config.to_json() if output_model_config != FAILED_CONFIG else {"is_pruned": True}
                    ),
                    parent_model_id=input_model_id,
                    from_pass=pass_type_name,
                    pass_run_config=pass_config,
                    start_time=run_start_time,
                    end_time=datetime.now().timestamp(),
                )
                logger.info("Loaded model from cache: %s", output_model_id)
                return output_model_config, output_model_id

        output_model_path = str(self.cache.get_model_cache_path(output_model_id))
        if input_model_config.config.get("shared_cache", False):
            input_model_config = self.cache.download_shared_cache_model(input_model_config, input_model_id)

        host = self.host_for_pass(pass_name)
        input_model_config = self.cache.prepare_resources_for_local(input_model_config)

        try:
            if p.run_on_target:
                host = self.target

            output_model_config = host.run_pass(p, input_model_config, output_model_path)
        except OlivePassError:
            logger.exception("Pass run_pass failed")
            output_model_config = FAILED_CONFIG
        except EXCEPTIONS_TO_RAISE:
            raise
        except Exception:
            output_model_config = FAILED_CONFIG
            logger.exception("Pass run failed.")
            if not self.search_strategy:
                raise  # rethrow the exception if no search is performed

        run_end_time = datetime.now().timestamp()
        logger.info("Pass %s:%s finished in %f seconds", pass_name, pass_type_name, run_end_time - run_start_time)

        # cache model
        self.cache_manager.cache_model(output_model_id, output_model_config)

        # cache run
        self.cache.cache_run(pass_type_name, pass_config, input_model_id, output_model_id, run_accel)

        # footprint model and run
        self.footprint.record(
            model_id=output_model_id,
            model_config=(
                output_model_config.to_json() if output_model_config != FAILED_CONFIG else {"is_pruned": True}
            ),
            parent_model_id=input_model_id,
            from_pass=pass_type_name,
            pass_run_config=pass_config,
            start_time=run_start_time,
            end_time=run_end_time,
        )

        return output_model_config, output_model_id

    def evaluate_model(
        self,
        model_config: ModelConfig,
        model_id: str,
        evaluator_config: OliveEvaluatorConfig,
        accelerator_spec: "AcceleratorSpec",
    ):
        """Evaluate a model."""
        logger.debug("Evaluating model ...")
        accelerator_suffix = f"-{accelerator_spec}" if accelerator_spec else ""
        if not model_id.endswith(accelerator_suffix):
            model_id_with_accelerator = f"{model_id}{accelerator_suffix}"
        else:
            model_id_with_accelerator = model_id

        # load evaluation from cache if it exists
        signal = self.cache_manager.load_evaluation(model_id_with_accelerator)
        if signal is not None:
            logger.debug("Loading evaluation from cache ...")
            self.footprint.record(
                model_id=model_id,
                metrics=FootprintNodeMetric(
                    value=signal,
                    if_goals_met=False,
                ),
            )
            return signal

        # evaluate model
        model_config = self.cache.prepare_resources_for_local(model_config)
        signal = self.target.evaluate_model(model_config, evaluator_config, accelerator_spec)

        # cache evaluation
        self.cache_manager.cache_evaluation(model_id_with_accelerator, signal)

        # footprint evaluation
        self.footprint.record(
            model_id=model_id,
            metrics=FootprintNodeMetric(
                value=signal,
                if_goals_met=False,
            ),
        )
        return signal
