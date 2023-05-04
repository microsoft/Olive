# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from pathlib import Path
from test.unit_test.utils import (
    get_accuracy_metric,
    get_onnx_dynamic_quantization_pass,
    get_onnx_model,
    get_onnxconversion_pass,
    get_pytorch_model,
    pytorch_model_loader,
)
from unittest.mock import patch

import pytest

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
        evaluator = OliveEvaluator(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)])

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
        assert name in engine.passes
        assert name in engine.pass_order
        assert engine.passes[name]["pass"] == p
        assert engine.passes[name]["host"] == system
        assert engine.passes[name]["evaluator"] == evaluator
        assert engine.passes[name]["clean_run_cache"] is False

    def test_register_no_search(self):
        # setup
        p = get_onnx_dynamic_quantization_pass(disable_search=True)
        name = p.__class__.__name__

        options = {
            "search_strategy": None,
        }
        engine = Engine(options)

        # execute
        engine.register(p)

        # assert
        assert name in engine.passes
        assert name in engine.pass_order

    def test_register_no_search_fail(self):
        # setup
        p = get_onnx_dynamic_quantization_pass(disable_search=False)
        name = p.__class__.__name__

        options = {
            "search_strategy": None,
        }
        engine = Engine(options)

        # execute
        with pytest.raises(ValueError) as exc_info:
            engine.register(p)

        assert str(exc_info.value) == f"Search strategy is None but pass {name} has search space"

    @patch("olive.engine.engine.LocalSystem")
    def test_run(self, mock_local_system):
        # setup
        pytorch_model = get_pytorch_model()
        input_model_id = hash_dict(pytorch_model.to_json())
        p, pass_config = get_onnxconversion_pass(ignore_pass_config=False)
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator = OliveEvaluator(metrics=[metric])
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "random",
            },
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=mock_local_system, target=mock_local_system, evaluator=evaluator)
        engine.register(p, clean_run_cache=True)
        onnx_model = get_onnx_model()
        mock_local_system.run_pass.return_value = onnx_model
        mock_local_system.evaluate_model.return_value = {metric.name: 0.998}
        model_id = f"0_{p.__class__.__name__}-{input_model_id}-{hash_dict(pass_config)}"
        expected_res = {
            model_id: {
                "model_id": model_id,
                "parent_model_id": input_model_id,
                "metrics": {
                    "value": {metric.name: 0.998},
                    "cmp_direction": {metric.name: 1},
                    "is_goals_met": True,
                },
            }
        }

        # execute
        actual_res = engine.run(pytorch_model)

        # make sure the input model always be in engine.footprints
        assert input_model_id in engine.footprints.nodes
        # make sure the input model always not in engine's pareto frontier
        assert input_model_id not in actual_res.nodes

        # assert
        assert len(actual_res.nodes) == 1
        assert model_id in actual_res.nodes
        assert actual_res.nodes[model_id].model_id == model_id
        for k, v in expected_res[model_id].items():
            if k == "metrics":
                assert getattr(actual_res.nodes[model_id].metrics, "is_goals_met")
            assert getattr(actual_res.nodes[model_id], k) == v
        assert engine.get_model_json_path(actual_res.nodes[model_id].model_id).exists()
        mock_local_system.run_pass.assert_called_once()
        mock_local_system.evaluate_model.assert_called_once_with(onnx_model, [metric])

    @patch("olive.engine.engine.LocalSystem")
    def test_run_no_search(self, mock_local_system):
        # setup
        pytorch_model = get_pytorch_model()
        p = get_onnxconversion_pass()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator = OliveEvaluator(metrics=[metric])
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=mock_local_system, target=mock_local_system, evaluator=evaluator)
        engine.register(p, clean_run_cache=True)
        onnx_model = get_onnx_model()
        mock_local_system.run_pass.return_value = onnx_model
        mock_local_system.evaluate_model.return_value = {metric.name: 0.998}

        # output model to output_dir
        output_dir = Path("cache") / "output"
        shutil.rmtree(output_dir, ignore_errors=True)

        expected_res = {"model": onnx_model.to_json(), "metrics": {metric.name: 0.998}}
        expected_res["model"]["config"]["model_path"] = str(Path(output_dir / "model.onnx").resolve())

        # execute
        actual_res = engine.run(pytorch_model, output_dir=output_dir)

        assert expected_res == actual_res
        assert Path(actual_res["model"]["config"]["model_path"]).is_file()
        model_json_path = Path(output_dir / "model.json")
        assert model_json_path.is_file()
        assert json.load(open(model_json_path, "r")) == actual_res["model"]
        result_json_path = Path(output_dir / "metrics.json")
        assert result_json_path.is_file()
        assert json.load(open(result_json_path, "r")) == actual_res["metrics"]

        # clean up
        shutil.rmtree(output_dir, ignore_errors=True)

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

    @patch("olive.engine.engine.LocalSystem")
    def test_run_evaluation_only(self, mock_local_system):
        # setup
        pytorch_model = get_pytorch_model()
        p = get_onnxconversion_pass()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator = OliveEvaluator(metrics=[metric])
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=mock_local_system, target=mock_local_system, evaluator=evaluator)
        engine.register(p, clean_run_cache=True)
        onnx_model = get_onnx_model()
        mock_local_system.run_pass.return_value = onnx_model
        mock_local_system.evaluate_model.return_value = {metric.name: 0.998}

        # output model to output_dir
        output_dir = Path("cache") / "output"
        shutil.rmtree(output_dir, ignore_errors=True)

        expected_res = {metric.name: 0.998}

        # execute
        actual_res = engine.run(pytorch_model, output_dir=output_dir, evaluation_only=True)

        assert expected_res == actual_res
        result_json_path = Path(output_dir / "metrics.json")
        assert result_json_path.is_file()
        assert json.load(open(result_json_path, "r")) == actual_res

        # clean up
        shutil.rmtree(output_dir, ignore_errors=True)
