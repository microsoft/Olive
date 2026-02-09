# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from unittest.mock import MagicMock, patch

from olive.engine.cache_manager import CacheManager
from olive.engine.config import FAILED_CONFIG
from olive.evaluator.metric_result import MetricResult, joint_metric_key

# pylint: disable=protected-access


class TestCacheManager:
    """Tests for CacheManager."""

    def _make_metric_result(self):
        """Create a simple MetricResult for testing."""
        return MetricResult.parse_obj(
            {
                joint_metric_key("accuracy", "accuracy_score"): {
                    "value": 0.95,
                    "priority": 1,
                    "higher_is_better": True,
                }
            }
        )

    def test_cache_model_delegates_to_olive_cache(self):
        mock_cache = MagicMock()
        manager = CacheManager(mock_cache)
        model = MagicMock()
        model.to_json.return_value = {"type": "ONNXModel"}

        manager.cache_model("model_123", model)

        mock_cache.cache_model.assert_called_once_with("model_123", {"type": "ONNXModel"})

    def test_cache_model_with_failed_config(self):
        mock_cache = MagicMock()
        manager = CacheManager(mock_cache)

        manager.cache_model("model_123", FAILED_CONFIG)

        mock_cache.cache_model.assert_called_once_with("model_123", {})

    def test_cache_model_respects_check_object(self):
        mock_cache = MagicMock()
        manager = CacheManager(mock_cache)
        model = MagicMock()

        manager.cache_model("model_123", model, check_object=False)

        model.to_json.assert_called_once_with(check_object=False)

    def test_load_model_returns_none_when_not_found(self):
        mock_cache = MagicMock()
        mock_cache.load_model.return_value = None
        manager = CacheManager(mock_cache)

        result = manager.load_model("nonexistent")

        assert result is None

    def test_load_model_returns_failed_config_for_empty_json(self):
        mock_cache = MagicMock()
        mock_cache.load_model.return_value = {}
        manager = CacheManager(mock_cache)

        result = manager.load_model("model_123")

        assert result == FAILED_CONFIG

    @patch("olive.engine.cache_manager.ModelConfig")
    def test_load_model_returns_model_config(self, mock_model_config_cls):
        mock_cache = MagicMock()
        model_json = {"type": "ONNXModel", "config": {"model_path": "/some/path"}}
        mock_cache.load_model.return_value = model_json
        expected = MagicMock()
        mock_model_config_cls.from_json.return_value = expected
        manager = CacheManager(mock_cache)

        result = manager.load_model("model_123")

        mock_model_config_cls.from_json.assert_called_once_with(model_json)
        assert result is expected

    def test_cache_evaluation(self):
        mock_cache = MagicMock()
        manager = CacheManager(mock_cache)
        signal = self._make_metric_result()

        manager.cache_evaluation("model_123", signal)

        mock_cache.cache_evaluation.assert_called_once()
        call_args = mock_cache.cache_evaluation.call_args
        assert call_args[0][0] == "model_123"
        evaluation_json = call_args[0][1]
        assert evaluation_json["model_id"] == "model_123"
        assert "signal" in evaluation_json

    def test_load_evaluation_returns_none_when_not_found(self):
        mock_cache = MagicMock()
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_cache.get_evaluation_json_path.return_value = mock_path
        manager = CacheManager(mock_cache)

        result = manager.load_evaluation("model_123")

        assert result is None

    def test_load_evaluation_returns_metric_result(self, tmp_path):
        mock_cache = MagicMock()
        signal = self._make_metric_result()
        eval_json = {"model_id": "model_123", "signal": signal.dict()}
        eval_path = tmp_path / "evaluation.json"
        with eval_path.open("w") as f:
            json.dump(eval_json, f)
        mock_cache.get_evaluation_json_path.return_value = eval_path
        manager = CacheManager(mock_cache)

        result = manager.load_evaluation("model_123")

        assert result is not None
        key = joint_metric_key("accuracy", "accuracy_score")
        assert result[key].value == 0.95

    def test_load_evaluation_returns_none_on_corrupt_file(self, tmp_path):
        mock_cache = MagicMock()
        eval_path = tmp_path / "evaluation.json"
        eval_path.write_text("not valid json {{{")
        mock_cache.get_evaluation_json_path.return_value = eval_path
        manager = CacheManager(mock_cache)

        result = manager.load_evaluation("model_123")

        assert result is None
