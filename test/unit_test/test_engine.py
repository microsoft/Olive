# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from test.unit_test.utils import (
    get_accuracy_metric,
    get_onnx_model,
    get_onnxconversion_pass,
    get_pytorch_model,
    pytorch_model_loader,
)
from unittest.mock import patch

from olive.common.utils import hash_dict
from olive.engine import Engine
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.model import PyTorchModel
from olive.systems.local import LocalSystem


# Please not your test case could still "pass" even if it throws exception to fail.
# Please check log message to make sure your test case passes.
class TestEngine:
    def test_register(self):
        # setup
        p = get_onnxconversion_pass()
        name = p.__class__.__name__
        system = LocalSystem()
        evaluator = OliveEvaluator(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)], target=system)

        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "random",
            },
        }
        engine = Engine(options)

        # execute
        engine.register(p, host=system, evaluator=evaluator)

        # assert
        assert (name, p) in engine._passes.items()
        assert name in engine._pass_order
        assert (name, system) in engine.hosts.items()
        assert (name, evaluator) in engine._evaluators.items()
        assert (name, False) in engine._clean_pass_run_cache.items()

    @patch("olive.engine.LocalSystem")
    def test_run(self, mock_local_system):
        # setup
        pytorch_model = get_pytorch_model()
        p = get_onnxconversion_pass()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator = OliveEvaluator(metrics=[metric], target=mock_local_system)
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "random",
            },
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=mock_local_system, evaluator=evaluator)
        engine.register(p, clean_run_cache=True)
        onnx_model = get_onnx_model()
        mock_local_system.run_pass.return_value = onnx_model
        mock_local_system.evaluate_model.return_value = {metric.name: 0.998}
        model_id = f"0_{p.__class__.__name__}-{hash_dict(pytorch_model.to_json())}"
        expected_res = {
            "search_points": {"OnnxConversion": {}},
            "metric": [0.998],
        }

        # execute
        actual_res = engine.run(pytorch_model)

        # assert
        assert expected_res.items() < actual_res.items()
        assert len(actual_res["model_ids"]) == 1
        assert model_id in actual_res["model_ids"][0]
        mock_local_system.run_pass.assert_called_once()
        mock_local_system.evaluate_model.assert_called_once_with(onnx_model, [metric])

    def test_pass_exception(self, caplog):
        # Need explicitly set the propagate to allow the message to be logged into caplog
        # setup
        logger = logging.getLogger("olive")
        logger.propagate = True

        with patch("olive.passes.onnx.conversion.OnnxConversion.run") as mock_run:
            mock_run.side_effect = Exception("test")
            system = LocalSystem()
            evaluator = OliveEvaluator(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)], target=system)
            options = {
                "cache_dir": "./cache",
                "clean_cache": True,
                "search_strategy": {
                    "execution_order": "joint",
                    "search_algorithm": "random",
                },
            }
            engine = Engine(options, evaluator=evaluator, host=system)
            onnx_conversion_pass = get_onnxconversion_pass()
            engine.register(onnx_conversion_pass, clean_run_cache=True)
            model = PyTorchModel(model_loader=pytorch_model_loader, model_path=None)

            # execute
            engine.run(model)

            # assert
            assert "Exception: test" in caplog.text
