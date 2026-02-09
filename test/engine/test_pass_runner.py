# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from unittest.mock import MagicMock

from olive.engine.config import FAILED_CONFIG
from olive.engine.footprint import Footprint
from olive.engine.pass_runner import PassRunner
from olive.evaluator.metric_result import MetricResult, joint_metric_key
from olive.systems.common import SystemType
from olive.systems.system_config import LocalTargetUserConfig, SystemConfig

# pylint: disable=protected-access


def _make_system_config():
    return SystemConfig(
        type=SystemType.Local,
        config=LocalTargetUserConfig(accelerators=[{"device": "CPU", "execution_providers": None}]),
    )


def _make_metric_result(value=0.95):
    return MetricResult.parse_obj(
        {
            joint_metric_key("accuracy", "accuracy_score"): {
                "value": value,
                "priority": 1,
                "higher_is_better": True,
            }
        }
    )


def _make_pass_runner(
    cache=None,
    cache_manager=None,
    footprint=None,
    host=None,
    target=None,
    evaluator_config=None,
    input_passes_configs=None,
    search_strategy=None,
):
    olive_config = MagicMock()
    return PassRunner(
        olive_config=olive_config,
        cache=cache or MagicMock(),
        cache_manager=cache_manager or MagicMock(),
        footprint=footprint or Footprint(),
        host_config=_make_system_config(),
        target_config=_make_system_config(),
        evaluator_config=evaluator_config,
        input_passes_configs=input_passes_configs or OrderedDict(),
        host=host,
        target=target,
        search_strategy=search_strategy,
    )


class TestPassRunnerInit:
    def test_init_sets_all_attributes(self):
        pr = _make_pass_runner()
        assert pr.olive_config is not None
        assert pr.cache is not None
        assert pr.cache_manager is not None
        assert pr.footprint is not None
        assert isinstance(pr.computed_passes_configs, OrderedDict)

    def test_get_host_device_returns_cpu(self):
        pr = _make_pass_runner()
        assert pr.get_host_device() == "CPU"

    def test_get_host_device_returns_none_when_no_accelerators(self):
        pr = _make_pass_runner()
        pr.host_config = SystemConfig(type=SystemType.Local, config=LocalTargetUserConfig(accelerators=[]))
        assert pr.get_host_device() is None


class TestPassRunnerHostEvaluator:
    def test_host_for_pass_returns_default_host(self):
        pr = _make_pass_runner(host="mock_host")
        pass_config = MagicMock()
        pass_config.host = None
        pr.computed_passes_configs["test_pass"] = pass_config

        result = pr.host_for_pass("test_pass")
        assert result == "mock_host"

    def test_host_for_pass_returns_per_pass_host(self):
        pr = _make_pass_runner(host="default_host")
        pass_config = MagicMock()
        pass_host = MagicMock()
        pass_host.create_system.return_value = "custom_host"
        pass_config.host = pass_host
        pr.computed_passes_configs["test_pass"] = pass_config

        result = pr.host_for_pass("test_pass")
        assert result == "custom_host"

    def test_evaluator_for_pass_returns_default(self):
        eval_config = MagicMock()
        pr = _make_pass_runner(evaluator_config=eval_config)
        pass_config = MagicMock()
        pass_config.evaluator = None
        pr.computed_passes_configs["test_pass"] = pass_config

        result = pr.evaluator_for_pass("test_pass")
        assert result is eval_config

    def test_evaluator_for_pass_returns_per_pass_evaluator(self):
        pr = _make_pass_runner(evaluator_config=MagicMock())
        pass_eval = MagicMock()
        pass_config = MagicMock()
        pass_config.evaluator = pass_eval
        pr.computed_passes_configs["test_pass"] = pass_config

        result = pr.evaluator_for_pass("test_pass")
        assert result is pass_eval


class TestPassRunnerEvaluateModel:
    def test_evaluate_model_uses_cache(self):
        cache_manager = MagicMock()
        cached_signal = _make_metric_result()
        cache_manager.load_evaluation.return_value = cached_signal

        pr = _make_pass_runner(cache_manager=cache_manager)
        accelerator_spec = MagicMock()
        accelerator_spec.__str__ = MagicMock(return_value="cpu-CPUExecutionProvider")

        result = pr.evaluate_model(MagicMock(), "model_id", MagicMock(), accelerator_spec)
        assert result is cached_signal
        # Should not call target.evaluate_model when cache hit
        assert pr.target is None or not getattr(pr.target, "evaluate_model", MagicMock()).called

    def test_evaluate_model_calls_target_on_cache_miss(self):
        cache_manager = MagicMock()
        cache_manager.load_evaluation.return_value = None
        target = MagicMock()
        expected_signal = _make_metric_result()
        target.evaluate_model.return_value = expected_signal
        cache = MagicMock()
        cache.prepare_resources_for_local.side_effect = lambda x: x

        pr = _make_pass_runner(cache=cache, cache_manager=cache_manager, target=target)
        accelerator_spec = MagicMock()
        accelerator_spec.__str__ = MagicMock(return_value="cpu-CPUExecutionProvider")

        result = pr.evaluate_model(MagicMock(), "model_id", MagicMock(), accelerator_spec)
        assert result is expected_signal
        target.evaluate_model.assert_called_once()
        cache_manager.cache_evaluation.assert_called_once()


class TestPassRunnerRunPasses:
    def test_run_passes_prunes_on_failed_config(self):
        pr = _make_pass_runner(search_strategy=None)
        pass_config = MagicMock()
        pass_config.evaluator = None
        pr.computed_passes_configs = OrderedDict({"pass1": pass_config})

        # Mock run_single_pass to return FAILED_CONFIG
        pr.run_single_pass = MagicMock(return_value=(FAILED_CONFIG, None))

        should_prune, signal, model_ids = pr.run_passes(MagicMock(), "input_id", MagicMock())
        assert should_prune is True
        assert signal is None
        assert model_ids == []

    def test_run_passes_evaluates_on_success(self):
        eval_config = MagicMock()
        pr = _make_pass_runner(evaluator_config=eval_config, search_strategy=MagicMock())
        pass_config = MagicMock()
        pass_config.evaluator = None
        pr.computed_passes_configs = OrderedDict({"pass1": pass_config})

        output_model = MagicMock()
        output_model.config.get.return_value = False  # no shared_cache
        pr.run_single_pass = MagicMock(return_value=(output_model, "output_id"))
        expected_signal = _make_metric_result()
        pr.evaluate_model = MagicMock(return_value=expected_signal)

        should_prune, signal, model_ids = pr.run_passes(MagicMock(), "input_id", MagicMock())
        assert should_prune is False
        assert signal is expected_signal
        assert model_ids == ["output_id"]

    def test_run_passes_skips_eval_when_no_search_and_no_evaluator(self):
        pr = _make_pass_runner(evaluator_config=None, search_strategy=None)
        pass_config = MagicMock()
        pass_config.evaluator = None
        pr.computed_passes_configs = OrderedDict({"pass1": pass_config})

        output_model = MagicMock()
        output_model.config.get.return_value = False
        pr.run_single_pass = MagicMock(return_value=(output_model, "output_id"))

        should_prune, signal, model_ids = pr.run_passes(MagicMock(), "input_id", MagicMock())
        assert should_prune is False
        assert signal is None
        assert model_ids == ["output_id"]
