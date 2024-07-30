# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from test.unit_test.utils import (
    get_accuracy_metric,
    get_composite_onnx_model_config,
    get_onnx_model_config,
    get_onnxconversion_pass,
    get_pytorch_model_config,
    get_pytorch_model_io_config,
)
from unittest.mock import MagicMock, patch

import pytest

from olive.common.utils import hash_dict
from olive.data.config import DataComponentConfig, DataConfig
from olive.engine import Engine
from olive.engine.cloud_cache_helper import CloudCacheConfig
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.metric_result import MetricResult, joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.optimum_conversion import OptimumConversion
from olive.passes.onnx.quantization import OnnxDynamicQuantization, OnnxStaticQuantization
from olive.systems.accelerator_creator import create_accelerators
from olive.systems.common import SystemType
from olive.systems.local import LocalSystem
from olive.systems.system_config import LocalTargetUserConfig, SystemConfig

# pylint: disable=protected-access


# Please note your test case could still "pass" even if it throws exception to fail.
# Please check log message to make sure your test case passes.
class TestEngine:
    def test_register(self, tmpdir):
        # setup
        p = get_onnxconversion_pass()
        name = p.__class__.__name__
        system = LocalSystem()
        evaluator_config = OliveEvaluatorConfig(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)])

        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "random",
            },
        }
        engine = Engine(**options)

        # execute
        engine.register(OnnxConversion, host=system, evaluator_config=evaluator_config)

        # assert
        assert name in engine.pass_config
        assert engine.pass_config[name]["type"] == OnnxConversion
        assert engine.pass_config[name]["host"] == system
        assert engine.pass_config[name]["evaluator"] == evaluator_config
        assert engine.pass_config[name]["clean_run_cache"] is False

    def test_register_no_search(self, tmpdir):
        # setup
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": None,
        }
        engine = Engine(**options)

        # execute
        engine.register(OnnxDynamicQuantization, disable_search=True)

        # assert
        assert "OnnxDynamicQuantization" in engine.pass_config

    def test_register_no_search_fail(self, tmpdir):
        name = "OnnxDynamicQuantization"
        # setup
        model_config = get_onnx_model_config()

        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": None,
        }
        engine = Engine(**options)

        # execute
        engine.register(OnnxDynamicQuantization)
        with pytest.raises(ValueError) as exc_info:  # noqa: PT011
            engine.run(
                model_config, [DEFAULT_CPU_ACCELERATOR], cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False)
            )

        assert str(exc_info.value) == f"Search strategy is None but pass {name} has search space"

    def test_default_engine_run(self, tmpdir):
        # setup
        model_config = get_pytorch_model_config()
        engine = Engine(cache_dir=tmpdir)
        assert engine.no_search, "Expect no_search to be True by default"

        engine.register(OnnxConversion, name="converter_13", config={"target_opset": 13}, clean_run_cache=True)
        outputs = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=tmpdir,
            cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
        )

        assert outputs
        for fp_nodes in outputs.values():
            for node in fp_nodes.nodes.values():
                assert node.model_config
                assert node.from_pass == "OnnxConversion"
                assert node.metrics is None, "Should not evaluate input/output model by default"

    @patch("olive.systems.local.LocalSystem")
    def test_run(self, mock_local_system, tmpdir):
        # setup
        model_config = get_pytorch_model_config()
        input_model_id = hash_dict(model_config.to_json())[:8]
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "random",
            },
            "clean_evaluation_cache": True,
            "evaluator": evaluator_config,
        }
        metric_result_dict = {
            joint_metric_key(metric.name, sub_metric.name): {
                "value": 0.998,
                "priority": sub_metric.priority,
                "higher_is_better": sub_metric.higher_is_better,
            }
            for sub_metric in metric.sub_types
        }
        onnx_model_config = get_onnx_model_config()
        system_object = MagicMock()
        mock_local_system.return_value = system_object
        system_object.system_type = SystemType.Local
        system_object.run_pass.return_value = onnx_model_config
        system_object.evaluate_model.return_value = MetricResult.parse_obj(metric_result_dict)
        system_object.get_supported_execution_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        system_object.olive_managed_env = False

        engine = Engine(**options)
        engine.register(OnnxConversion, name="converter_13", config={"target_opset": 13}, clean_run_cache=True)
        engine.register(OnnxConversion, name="converter_14", config={"target_opset": 14}, clean_run_cache=True)
        engine.set_pass_flows([["converter_13"], ["converter_14"]])
        p1, pass_config1 = get_onnxconversion_pass(ignore_pass_config=False, target_opset=13)
        p2, pass_config2 = get_onnxconversion_pass(ignore_pass_config=False, target_opset=14)
        model_ids = [
            f"0_{p1.__class__.__name__}-{input_model_id}-{hash_dict(pass_config1)[:8]}",
            f"1_{p2.__class__.__name__}-{input_model_id}-{hash_dict(pass_config2)[:8]}",
        ]
        expected_res = {
            model_id: {
                "model_id": model_id,
                "parent_model_id": input_model_id,
                "metrics": {
                    "value": metric_result_dict,
                    "cmp_direction": {},
                    "if_goals_met": True,
                },
            }
            for model_id in model_ids
        }

        # execute
        output_dir = Path(tmpdir)
        actual_res = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=output_dir,
            cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
        )
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        actual_res = actual_res[accelerator_spec]

        # make sure the input model always be in engine.footprints
        footprint = engine.footprints[accelerator_spec]
        assert input_model_id in footprint.nodes
        # make sure the input model always not in engine's pareto frontier
        assert input_model_id not in actual_res.nodes

        # assert
        assert len(actual_res.nodes) == 2
        assert model_ids == list(actual_res.nodes.keys())

        assert actual_res.nodes[model_ids[0]].model_id != actual_res.nodes[model_ids[1]].model_id

        for model_id, result in expected_res.items():
            # ensure two converted models are from the same input model
            assert actual_res.nodes[model_id].parent_model_id == input_model_id

            assert engine.cache.get_model_json_path(actual_res.nodes[model_id].model_id).exists()
            for k, v in result.items():
                if k == "metrics":
                    assert actual_res.nodes[model_id].metrics.if_goals_met
                else:
                    assert getattr(actual_res.nodes[model_id], k) == v

        assert system_object.run_pass.call_count == 2
        assert system_object.evaluate_model.call_count == 3
        system_object.evaluate_model.assert_called_with(onnx_model_config.to_json(), [metric], accelerator_spec)

    @patch("olive.systems.local.LocalSystem")
    def test_run_no_search_model_components(self, mock_local_system_init, tmpdir):
        model_config = get_pytorch_model_config()
        composite_onnx_model_config = get_composite_onnx_model_config()

        mock_local_system = MagicMock()
        mock_local_system_init.return_value = mock_local_system
        mock_local_system.system_type = SystemType.Local
        mock_local_system.run_pass.return_value = composite_onnx_model_config
        mock_local_system.accelerators = ["CPU"]
        mock_local_system.get_supported_execution_providers.return_value = ["CPUExecutionProvider"]
        mock_local_system.olive_managed_env = False

        engine = Engine(cache_dir=tmpdir)
        engine.register(OptimumConversion, disable_search=True, clean_run_cache=True)
        engine.set_pass_flows()
        # output model to output_dir
        output_dir = Path(tmpdir)

        # execute
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        _actual_res = engine.run(
            model_config,
            [accelerator_spec],
            output_dir=output_dir,
            cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
        )

        # assert
        actual_res = next(iter(_actual_res[accelerator_spec].nodes.values()))
        assert composite_onnx_model_config.to_json() == actual_res.model_config

    @patch("olive.systems.local.LocalSystem")
    def test_run_no_search(self, mock_local_system_init, tmpdir):
        # setup
        model_config = get_pytorch_model_config()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
            "evaluator": evaluator_config,
        }
        metric_result_dict = {
            joint_metric_key(metric.name, sub_metric.name): {
                "value": 0.998,
                "priority": sub_metric.priority,
                "higher_is_better": sub_metric.higher_is_better,
            }
            for sub_metric in metric.sub_types
        }
        onnx_model_config = get_onnx_model_config()
        mock_local_system = MagicMock()
        mock_local_system_init.return_value = mock_local_system
        mock_local_system.system_type = SystemType.Local
        mock_local_system.run_pass.return_value = onnx_model_config
        mock_local_system.evaluate_model.return_value = MetricResult.parse_obj(metric_result_dict)
        mock_local_system.get_supported_execution_providers.return_value = ["CPUExecutionProvider"]
        mock_local_system.olive_managed_env = False

        engine = Engine(**options)
        engine.register(OnnxConversion, disable_search=True, clean_run_cache=True)
        engine.set_pass_flows()
        # output model to output_dir
        output_dir = Path(tmpdir)
        expected_output_dir = output_dir / "-".join(engine.pass_flows[0])

        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        output_prefix = f"{accelerator_spec}"
        expected_res = {"model": onnx_model_config.to_json(), "metrics": MetricResult.parse_obj(metric_result_dict)}
        expected_res["model"]["config"]["model_path"] = str(
            Path(expected_output_dir / f"{output_prefix}_model.onnx").resolve()
        )

        # execute
        _actual_res = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=output_dir,
            cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
        )
        actual_res = next(iter(_actual_res[accelerator_spec].nodes.values()))

        assert expected_res["model"] == actual_res.model_config
        assert expected_res["metrics"] == actual_res.metrics.value
        assert Path(actual_res.model_config["config"]["model_path"]).is_file()
        model_json_path = Path(expected_output_dir / f"{output_prefix}_model.json")
        assert model_json_path.is_file()
        with model_json_path.open() as f:
            assert json.load(f) == actual_res.model_config
        result_json_path = Path(expected_output_dir / f"{output_prefix}_metrics.json")
        assert result_json_path.is_file()
        with result_json_path.open() as f:
            assert json.load(f) == actual_res.metrics.value.__root__

    def test_pass_exception(self, caplog, tmpdir):
        # Need explicitly set the propagate to allow the message to be logged into caplog
        # setup
        logger = logging.getLogger("olive")
        logger.propagate = True

        with patch("olive.passes.onnx.conversion.OnnxConversion.run") as mock_run:
            mock_run.side_effect = Exception("test")
            evaluator_config = OliveEvaluatorConfig(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)])
            options = {
                "cache_dir": tmpdir,
                "clean_cache": True,
                "search_strategy": {
                    "execution_order": "joint",
                    "search_algorithm": "random",
                },
                "evaluator": evaluator_config,
            }
            engine = Engine(**options)
            engine.register(OnnxConversion, clean_run_cache=True)

            model_config = get_pytorch_model_config()

            # execute
            output_dir = Path(tmpdir)
            engine.run(
                model_config,
                [DEFAULT_CPU_ACCELERATOR],
                output_dir=output_dir,
                cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
            )

            # assert
            assert "Exception: test" in caplog.text

    @patch("olive.systems.local.LocalSystem")
    def test_run_evaluate_input_model(self, mock_local_system_init, tmpdir):
        # setup
        model_config = get_pytorch_model_config()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
            "evaluator": evaluator_config,
        }
        metric_result_dict = {
            joint_metric_key(metric.name, sub_metric.name): {
                "value": 0.998,
                "priority": sub_metric.priority,
                "higher_is_better": sub_metric.higher_is_better,
            }
            for sub_metric in metric.sub_types
        }
        mock_local_system = MagicMock()
        mock_local_system.run_pass.return_value = get_onnx_model_config()
        mock_local_system.get_supported_execution_providers.return_value = ["CPUExecutionProvider"]
        mock_local_system.evaluate_model.return_value = MetricResult.parse_obj(metric_result_dict)
        mock_local_system.system_type = SystemType.Local
        mock_local_system.olive_managed_env = False
        mock_local_system_init.return_value = mock_local_system

        engine = Engine(**options)
        engine.register(OnnxConversion, clean_run_cache=True)

        # output model to output_dir
        output_dir = Path(tmpdir)
        expected_res = MetricResult.parse_obj(metric_result_dict)

        # execute
        actual_res = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=output_dir,
            evaluate_input_model=True,
            cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
        )
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        actual_res = next(iter(actual_res[accelerator_spec].nodes.values())).metrics.value

        assert expected_res == actual_res
        result_json_path = Path(output_dir / f"{accelerator_spec}_input_model_metrics.json")
        assert result_json_path.is_file()
        assert MetricResult.parse_file(result_json_path) == actual_res

    @patch("olive.systems.local.LocalSystem")
    def test_run_no_pass(self, mock_local_system_init, tmpdir):
        # setup
        model_config = get_pytorch_model_config()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
            "evaluator": evaluator_config,
        }
        metric_result_dict = {
            joint_metric_key(metric.name, sub_metric.name): {
                "value": 0.998,
                "priority": sub_metric.priority,
                "higher_is_better": sub_metric.higher_is_better,
            }
            for sub_metric in metric.sub_types
        }
        mock_local_system = MagicMock()
        mock_local_system.evaluate_model.return_value = MetricResult.parse_obj(metric_result_dict)
        mock_local_system.olive_managed_env = False
        mock_local_system.system_type = SystemType.Local
        mock_local_system.get_supported_execution_providers.return_value = ["CPUExecutionProvider"]
        mock_local_system_init.return_value = mock_local_system

        engine = Engine(**options)

        # output model to output_dir
        output_dir = Path(tmpdir)
        expected_res = MetricResult.parse_obj(metric_result_dict)

        # execute
        actual_res = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=output_dir,
            evaluate_input_model=True,
            cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
        )
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        actual_res = actual_res[accelerator_spec]

        assert expected_res == actual_res
        result_json_path = Path(output_dir / f"{accelerator_spec}_input_model_metrics.json")
        assert result_json_path.is_file()
        assert MetricResult.parse_file(result_json_path) == actual_res

    @patch.object(Path, "glob", return_value=[Path("cache") / "output" / "100_model.json"])
    @patch.object(Path, "unlink")
    def test_model_path_suffix(self, mock_unlink, mock_glob, tmpdir):
        # setup
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
            "evaluator": evaluator_config,
        }
        engine = Engine(**options)
        engine.register(OnnxConversion, clean_run_cache=True)

        engine.initialize()

        assert engine.cache.new_model_number == 101
        assert mock_unlink.call_count == 1
        assert mock_glob.call_count == 2

    def test_model_path_suffix_with_exception(self, tmpdir):
        # setup
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": None,
            "clean_evaluation_cache": True,
            "evaluator": evaluator_config,
        }
        with patch.object(Path, "glob"):
            Path.glob.return_value = [Path("cache") / "output" / "435d_0.json"]

            with pytest.raises(ValueError) as exc_info:  # noqa: PT011
                Engine(**options)
            assert str(exc_info.value) == "invalid literal for int() with base 10: '435d'"

    @patch("olive.systems.local.LocalSystem")
    @patch("onnxruntime.get_available_providers")
    def test_pass_cache_reuse(self, mock_get_available_providers, mock_local_system_init, caplog, tmpdir):
        logger = logging.getLogger("olive")
        logger.propagate = True

        # setup
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "random",
            },
            "clean_evaluation_cache": True,
            "evaluator": evaluator_config,
        }
        mock_local_system = MagicMock()
        mock_local_system.system_type = SystemType.Local
        mock_local_system.get_supported_execution_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_local_system.run_pass.return_value = get_onnx_model_config()
        mock_local_system.olive_managed_env = False
        metric_result_dict = {
            joint_metric_key(metric.name, sub_metric.name): {
                "value": 0.998,
                "priority": sub_metric.priority,
                "higher_is_better": sub_metric.higher_is_better,
            }
            for sub_metric in metric.sub_types
        }
        mock_local_system.evaluate_model.return_value = MetricResult.parse_obj(metric_result_dict)
        mock_local_system_init.return_value = mock_local_system

        engine = Engine(**options)
        system_config = SystemConfig(
            type=SystemType.Local,
            config=LocalTargetUserConfig(
                accelerators=[{"device": "GPU", "execution_providers": None}, {"device": "CPU"}],
            ),
        )
        accelerator_specs = create_accelerators(system_config)
        assert len(accelerator_specs) == 2
        engine.register(OnnxConversion, clean_run_cache=True)

        model_config = get_pytorch_model_config()
        output_dir = Path(tmpdir)
        _ = engine.run(
            model_config,
            accelerator_specs,
            output_dir=output_dir,
            cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
        )

        mock_local_system.run_pass.assert_called_once()

    @patch("olive.systems.local.LocalSystem")
    @patch("onnxruntime.get_available_providers")
    def test_pass_cache(self, mock_get_available_providers, mock_local_system_init, tmpdir):
        # setup
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_dir": tmpdir,
            "clean_cache": True,
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "random",
            },
            "clean_evaluation_cache": True,
            "evaluator": evaluator_config,
        }
        mock_local_system = MagicMock()
        mock_local_system.system_type = SystemType.Local
        mock_local_system.get_supported_execution_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_local_system.run_pass.return_value = get_onnx_model_config()
        mock_local_system.olive_managed_env = False
        metric_result_dict = {
            joint_metric_key(metric.name, sub_metric.name): {
                "value": 0.998,
                "priority": sub_metric.priority,
                "higher_is_better": sub_metric.higher_is_better,
            }
            for sub_metric in metric.sub_types
        }
        mock_local_system.evaluate_model.return_value = MetricResult.parse_obj(metric_result_dict)
        mock_local_system_init.return_value = mock_local_system

        engine = Engine(**options)
        system_config = SystemConfig(
            type=SystemType.Local,
            config=LocalTargetUserConfig(
                accelerators=[{"device": "GPU"}, {"device": "CPU", "execution_providers": None}],
            ),
        )
        accelerator_specs = create_accelerators(system_config)
        engine.register(OnnxConversion, clean_run_cache=True)

        model_config = get_pytorch_model_config()
        output_dir = Path(tmpdir)

        with patch(
            "olive.passes.onnx.conversion.OnnxConversion.is_accelerator_agnostic"
        ) as is_accelerator_agnostic_mock:
            is_accelerator_agnostic_mock.return_value = False
            _ = engine.run(
                model_config,
                accelerator_specs,
                output_dir=output_dir,
                cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
            )
            assert mock_local_system.run_pass.call_count == 2

    def test_pass_value_error(self, caplog, tmpdir):
        # Need explicitly set the propagate to allow the message to be logged into caplog
        # setup
        logger = logging.getLogger("olive")
        logger.propagate = True

        with patch("olive.passes.onnx.conversion.OnnxConversion.run") as mock_run:
            mock_run.side_effect = ValueError("test")
            evaluator_config = OliveEvaluatorConfig(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)])
            options = {
                "cache_dir": tmpdir,
                "clean_cache": True,
                "search_strategy": {
                    "execution_order": "joint",
                    "search_algorithm": "random",
                },
                "evaluator": evaluator_config,
            }
            engine = Engine(**options)
            engine.register(OnnxConversion, clean_run_cache=True)
            model_config = get_pytorch_model_config()
            # execute
            output_dir = Path(tmpdir)
            with pytest.raises(ValueError):  # noqa: PT011
                engine.run(
                    model_config,
                    [DEFAULT_CPU_ACCELERATOR],
                    output_dir=output_dir,
                    cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
                )

    @pytest.mark.parametrize("is_search", [True, False])
    def test_pass_quantization_error(self, is_search, caplog, tmpdir):
        # Need explicitly set the propagate to allow the message to be logged into caplog
        # setup
        logger = logging.getLogger("olive")
        logger.propagate = True

        onnx_model_config = get_onnx_model_config()
        # output model to output_dir
        output_dir = Path(tmpdir)

        # setup
        if is_search:
            metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
            evaluator_config = OliveEvaluatorConfig(metrics=[metric])
            options = {
                "cache_dir": tmpdir,
                "clean_cache": True,
                "search_strategy": {
                    "execution_order": "joint",
                    "search_algorithm": "random",
                },
                "evaluator": evaluator_config,
            }
            engine = Engine(**options)
            io_config = get_pytorch_model_io_config()
            engine.register(
                OnnxStaticQuantization,
                {
                    "data_config": DataConfig(
                        name="quant_data_config",
                        type="DummyDataContainer",
                        load_dataset_config=DataComponentConfig(
                            params={"input_names": io_config["input_names"], "input_shapes": io_config["input_shapes"]}
                        ),
                    )
                },
            )
            with patch("onnxruntime.quantization.quantize_static") as mock_quantize_static:
                mock_quantize_static.side_effect = AttributeError("test")
                actual_res = engine.run(
                    onnx_model_config,
                    [DEFAULT_CPU_ACCELERATOR],
                    output_dir=output_dir,
                    cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
                )
                assert not actual_res, "Expect empty dict when quantization fails"
        else:
            options = {
                "cache_dir": tmpdir,
                "clean_cache": True,
                "search_strategy": None,
            }
            engine = Engine(**options)
            engine.register(OnnxDynamicQuantization, disable_search=True)
            with patch("onnxruntime.quantization.quantize_dynamic") as mock_quantize_dynamic:
                mock_quantize_dynamic.side_effect = AttributeError("test")
                actual_res = engine.run(
                    onnx_model_config,
                    [DEFAULT_CPU_ACCELERATOR],
                    output_dir=output_dir,
                    evaluate_input_model=False,
                    cloud_cache_config=CloudCacheConfig(enable_cloud_cache=False),
                )
                assert not actual_res[DEFAULT_CPU_ACCELERATOR].nodes, "Expect empty dict when quantization fails"
