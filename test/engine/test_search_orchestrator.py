# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock

from olive.engine.footprint import Footprint
from olive.engine.search_orchestrator import SearchOrchestrator
from olive.evaluator.metric_result import MetricResult, joint_metric_key

# pylint: disable=protected-access


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


def _make_orchestrator(
    search_strategy=None,
    pass_runner=None,
    cache_manager=None,
    cache=None,
    footprint=None,
    evaluator_config=None,
    input_passes_configs=None,
    skip_saving_artifacts=False,
    plot_pareto_frontier=False,
):
    return SearchOrchestrator(
        search_strategy=search_strategy,
        pass_runner=pass_runner or MagicMock(),
        cache_manager=cache_manager or MagicMock(),
        cache=cache or MagicMock(),
        footprint=footprint or Footprint(),
        evaluator_config=evaluator_config,
        input_passes_configs=input_passes_configs or OrderedDict(),
        skip_saving_artifacts=skip_saving_artifacts,
        plot_pareto_frontier=plot_pareto_frontier,
    )


class TestSearchOrchestratorInit:
    def test_init_stores_all_attributes(self):
        orch = _make_orchestrator(skip_saving_artifacts=True, plot_pareto_frontier=True)
        assert orch.skip_saving_artifacts is True
        assert orch.plot_pareto_frontier is True
        assert orch.search_strategy is None
        assert orch.footprint is not None


class TestSearchOrchestratorRunNoSearch:
    def test_run_no_search_saves_metrics(self, tmp_path):
        pass_runner = MagicMock()
        pass_runner.computed_passes_configs = OrderedDict({"pass1": MagicMock()})
        signal = _make_metric_result()
        pass_runner.run_passes.return_value = (False, signal, ["output_model_id"])

        footprint = MagicMock()
        orch = _make_orchestrator(
            pass_runner=pass_runner,
            footprint=footprint,
            skip_saving_artifacts=False,
        )

        orch._run_no_search(MagicMock(), "input_id", MagicMock(), tmp_path)

        # Verify metrics saved
        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()
        with metrics_path.open() as f:
            saved = json.load(f)
        key = joint_metric_key("accuracy", "accuracy_score")
        assert key in saved

        # Verify output model ids set
        footprint.set_output_model_ids.assert_called_once_with(["output_model_id"])

    def test_run_no_search_handles_pruning(self, tmp_path):
        pass_runner = MagicMock()
        pass_runner.computed_passes_configs = OrderedDict({"pass1": MagicMock()})
        pass_runner.run_passes.return_value = (True, None, [])

        footprint = MagicMock()
        orch = _make_orchestrator(pass_runner=pass_runner, footprint=footprint)

        orch._run_no_search(MagicMock(), "input_id", MagicMock(), tmp_path)

        # Should NOT set output model ids when pruned
        footprint.set_output_model_ids.assert_not_called()

    def test_run_no_search_skip_artifacts(self, tmp_path):
        pass_runner = MagicMock()
        pass_runner.computed_passes_configs = OrderedDict({"pass1": MagicMock()})
        signal = _make_metric_result()
        pass_runner.run_passes.return_value = (False, signal, ["out_id"])

        orch = _make_orchestrator(pass_runner=pass_runner, skip_saving_artifacts=True)

        orch._run_no_search(MagicMock(), "input_id", MagicMock(), tmp_path)

        # No metrics file should be created
        assert not (tmp_path / "metrics.json").exists()


class TestSearchOrchestratorRunSearch:
    def test_run_search_iterates_and_records_feedback(self):
        # Setup search strategy that yields one sample
        sample = MagicMock()
        sample.model_ids = ["input_id"]
        sample.search_point.index = 0
        search_strategy = MagicMock()
        search_strategy.__iter__ = MagicMock(return_value=iter([sample]))
        search_strategy.iteration_count = 1

        pass_runner = MagicMock()
        pass_runner.computed_passes_configs = OrderedDict({"pass1": MagicMock()})
        signal = _make_metric_result()
        pass_runner.run_passes.return_value = (False, signal, ["output_id"])

        footprint = MagicMock()
        orch = _make_orchestrator(
            search_strategy=search_strategy,
            pass_runner=pass_runner,
            footprint=footprint,
        )

        orch._run_search(MagicMock(), "input_id", MagicMock(), Path("/tmp"))

        # Verify feedback was recorded
        search_strategy.record_feedback_signal.assert_called_once_with(0, signal, ["output_id"], False)
        # Verify pareto frontier created
        footprint.create_pareto_frontier.assert_called_once()

    def test_run_search_handles_pass_failure(self):
        sample = MagicMock()
        sample.model_ids = ["input_id"]
        sample.search_point.index = 0
        search_strategy = MagicMock()
        search_strategy.__iter__ = MagicMock(return_value=iter([sample]))
        search_strategy.iteration_count = 1

        pass_runner = MagicMock()
        pass_runner.computed_passes_configs = OrderedDict({"pass1": MagicMock()})
        pass_runner.run_passes.side_effect = RuntimeError("Pass failed")

        footprint = MagicMock()
        orch = _make_orchestrator(
            search_strategy=search_strategy,
            pass_runner=pass_runner,
            footprint=footprint,
        )

        # Should not raise — failures are caught and reported
        orch._run_search(MagicMock(), "input_id", MagicMock(), Path("/tmp"))

        # Feedback should still be recorded with should_prune=True
        search_strategy.record_feedback_signal.assert_called_once_with(0, None, [], True)


class TestSearchOrchestratorRun:
    def test_run_delegates_to_no_search(self, tmp_path):
        pass_runner = MagicMock()
        pass_runner.computed_passes_configs = OrderedDict({"pass1": MagicMock()})
        pass_runner.run_passes.return_value = (False, None, ["out_id"])
        pass_runner.evaluate_model.return_value = _make_metric_result()

        cache = MagicMock()
        cache.enable_shared_cache = False

        orch = _make_orchestrator(
            search_strategy=None,
            pass_runner=pass_runner,
            cache=cache,
            skip_saving_artifacts=True,
        )

        orch.run(MagicMock(), "input_id", MagicMock(), tmp_path, evaluate_input_model=False)

        pass_runner.run_passes.assert_called_once()

    def test_run_evaluates_input_model(self, tmp_path):
        pass_runner = MagicMock()
        pass_runner.computed_passes_configs = OrderedDict()
        signal = _make_metric_result()
        pass_runner.evaluate_model.return_value = signal

        eval_config = MagicMock()
        cache = MagicMock()
        cache.enable_shared_cache = False

        orch = _make_orchestrator(
            evaluator_config=eval_config,
            pass_runner=pass_runner,
            cache=cache,
            input_passes_configs=OrderedDict(),
            skip_saving_artifacts=False,
        )

        orch.run(MagicMock(), "input_id", MagicMock(), tmp_path, evaluate_input_model=True)

        pass_runner.evaluate_model.assert_called_once()
        assert (tmp_path / "input_model_metrics.json").exists()

    def test_run_disables_shared_cache_for_local_model(self, tmp_path):
        cache = MagicMock()
        cache.enable_shared_cache = True

        pass_runner = MagicMock()
        pass_runner.run_passes.return_value = (False, None, [])

        orch = _make_orchestrator(cache=cache, pass_runner=pass_runner, skip_saving_artifacts=True)

        from olive.common.constants import LOCAL_INPUT_MODEL_ID

        orch.run(MagicMock(), LOCAL_INPUT_MODEL_ID, MagicMock(), tmp_path, evaluate_input_model=False)

        cache.disable_shared_cache.assert_called_once()


class TestSearchOrchestratorDumpRunHistory:
    def test_dump_run_history_writes_file(self, tmp_path):
        orch = _make_orchestrator(skip_saving_artifacts=False)

        # Create mock run history with named tuples
        from typing import NamedTuple

        class RunHistoryEntry(NamedTuple):
            pass_name: str
            status: str

        run_history = [RunHistoryEntry("onnx_conversion", "success")]

        output_path = tmp_path / "run_history.txt"
        orch.dump_run_history(run_history, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "onnx_conversion" in content

    def test_dump_run_history_empty(self, tmp_path):
        orch = _make_orchestrator()
        output_path = tmp_path / "run_history.txt"

        orch.dump_run_history([], output_path)

        # With empty history, file should not be created
        assert not output_path.exists()
