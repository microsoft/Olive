# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
import tempfile
from pathlib import Path
from test.unit_test.utils import (
    get_accuracy_metric,
    get_onnx_model,
    get_onnxconversion_pass,
    get_pytorch_model,
    pytorch_model_loader,
)
from unittest.mock import Mock, patch

import pytest

from olive.common.utils import hash_dict
from olive.engine import Engine
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import PyTorchModel
from olive.passes.onnx import OnnxConversion, OnnxDynamicQuantization
from olive.systems.local import LocalSystem


# Please not your test case could still "pass" even if it throws exception to fail.
# Please check log message to make sure your test case passes.
class TestEngine:
    def test_register(self):
        # setup
        p = get_onnxconversion_pass()
        name = p.__class__.__name__
        system = LocalSystem()
        evaluator_config = OliveEvaluatorConfig(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)])

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
        engine.register(OnnxConversion, host=system, evaluator_config=evaluator_config)

        # assert
        assert name in engine.pass_config
        assert engine.pass_config[name]["type"] == OnnxConversion
        assert engine.pass_config[name]["host"] == system
        assert engine.pass_config[name]["evaluator"] == evaluator_config
        assert engine.pass_config[name]["clean_run_cache"] is False

    def test_register_no_search(self):
        # setup
        options = {
            "search_strategy": None,
        }
        engine = Engine(options)

        # execute
        engine.register(OnnxDynamicQuantization, disable_search=True)

        # assert
        assert "OnnxDynamicQuantization" in engine.pass_config

    def test_register_no_search_fail(self):
        name = "OnnxDynamicQuantization"
        # setup
        pytorch_model = get_pytorch_model()

        options = {
            "search_strategy": None,
        }
        engine = Engine(options)

        # execute
        engine.register(OnnxDynamicQuantization)
        with pytest.raises(ValueError) as exc_info:
            engine.run(pytorch_model)

        assert str(exc_info.value) == f"Search strategy is None but pass {name} has search space"

    @patch("olive.engine.engine.LocalSystem")
    def test_run(self, mock_local_system):
        # setup
        pytorch_model = get_pytorch_model()
        input_model_id = hash_dict(pytorch_model.to_json())
        p, pass_config = get_onnxconversion_pass(ignore_pass_config=False)
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "random",
            },
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=mock_local_system, target=mock_local_system, evaluator_config=evaluator_config)
        engine.register(OnnxConversion, clean_run_cache=True)
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
        output_dir = Path(tempfile.TemporaryDirectory().name)
        actual_res = engine.run(pytorch_model, output_dir=output_dir)
        accelerator_spec = 0
        actual_res = actual_res[accelerator_spec]

        # make sure the input model always be in engine.footprints
        footprint = engine.footprints[accelerator_spec]
        assert input_model_id in footprint.nodes
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
        shutil.rmtree(output_dir, ignore_errors=True)

    @patch("olive.engine.engine.LocalSystem")
    def test_run_no_search(self, mock_local_system):
        # setup
        pytorch_model = get_pytorch_model()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=mock_local_system, target=mock_local_system, evaluator_config=evaluator_config)
        engine.register(OnnxConversion, disable_search=True, clean_run_cache=True)
        onnx_model = get_onnx_model()
        mock_local_system.run_pass.return_value = onnx_model
        mock_local_system.evaluate_model.return_value = {metric.name: 0.998}

        # output model to output_dir
        output_dir = Path(tempfile.TemporaryDirectory().name)

        # TODO: replace with the real accelerator spec
        accelerator_spec = 0
        expected_res = {"model": onnx_model.to_json(), "metrics": {metric.name: 0.998}}
        expected_res["model"]["config"]["model_path"] = str(
            Path(output_dir / f"{accelerator_spec}_model.onnx").resolve()
        )

        # execute
        actual_res = engine.run(pytorch_model, output_dir=output_dir)
        actual_res = list(actual_res.values())[0]

        assert expected_res == actual_res
        assert Path(actual_res["model"]["config"]["model_path"]).is_file()
        model_json_path = Path(output_dir / f"{accelerator_spec}_model.json")
        assert model_json_path.is_file()
        assert json.load(open(model_json_path, "r")) == actual_res["model"]
        result_json_path = Path(output_dir / f"{accelerator_spec}_metrics.json")
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
            evaluator_config = OliveEvaluatorConfig(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)])
            options = {
                "cache_dir": "./cache",
                "clean_cache": True,
                "search_strategy": {
                    "execution_order": "joint",
                    "search_algorithm": "random",
                },
            }
            engine = Engine(options, evaluator_config=evaluator_config, host=system, target=system)
            engine.register(OnnxConversion, clean_run_cache=True)
            model = PyTorchModel(model_loader=pytorch_model_loader, model_path=None)

            # execute
            output_dir = Path(tempfile.TemporaryDirectory().name)
            engine.run(model, output_dir=output_dir)

            # assert
            assert "Exception: test" in caplog.text
            # clean up
            shutil.rmtree(output_dir, ignore_errors=True)

    @patch("olive.engine.engine.LocalSystem")
    def test_run_evaluation_only(self, mock_local_system):
        # setup
        pytorch_model = get_pytorch_model()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=mock_local_system, target=mock_local_system, evaluator_config=evaluator_config)
        engine.register(OnnxConversion, clean_run_cache=True)
        onnx_model = get_onnx_model()
        mock_local_system.run_pass.return_value = onnx_model
        mock_local_system.evaluate_model.return_value = {metric.name: 0.998}

        # output model to output_dir
        output_dir = Path(tempfile.TemporaryDirectory().name)

        expected_res = {metric.name: 0.998}

        # execute
        actual_res = engine.run(pytorch_model, output_dir=output_dir, evaluation_only=True)
        actual_res = list(actual_res.values())[0]

        # TODO: replace with the real accelerator spec
        accelerator_spec = 0
        assert expected_res == actual_res
        result_json_path = Path(output_dir / f"{accelerator_spec}_metrics.json")
        assert result_json_path.is_file()
        assert json.load(open(result_json_path, "r")) == actual_res

        # clean up
        shutil.rmtree(output_dir, ignore_errors=True)

    @patch.object(Path, "glob", return_value=[Path("cache") / "output" / "100_model.json"])
    @patch.object(Path, "unlink")
    def test_model_path_suffix(self, mock_glob, mock_unlink: Mock):
        # setup
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=LocalSystem(), target=LocalSystem(), evaluator_config=evaluator_config)
        engine.register(OnnxConversion, clean_run_cache=True)

        engine.initialize()

        assert engine._new_model_number == 101
        assert mock_unlink.called

    def test_model_path_suffix_with_exception(self):
        # setup
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": "./cache",
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
        }
        engine = Engine(options, host=LocalSystem(), target=LocalSystem(), evaluator_config=evaluator_config)
        engine.register(OnnxConversion, clean_run_cache=True)
        with patch.object(Path, "glob"):
            Path.glob.return_value = [Path("cache") / "output" / "435d_0.json"]

            with pytest.raises(ValueError) as exc_info:
                engine.initialize()
                assert str(exc_info.value) == "ValueError: invalid literal for int() with base 10: '435d'"
