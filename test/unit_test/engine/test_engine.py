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

from olive.data.config import DataComponentConfig, DataConfig
from olive.engine import Engine
from olive.engine.config import RunPassConfig
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.metric_result import MetricResult, joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.optimum_conversion import OptimumConversion
from olive.passes.onnx.quantization import OnnxDynamicQuantization, OnnxStaticQuantization
from olive.systems.accelerator_creator import create_accelerators
from olive.systems.common import SystemType
from olive.systems.system_config import LocalTargetUserConfig, SystemConfig

# pylint: disable=protected-access


# Please note your test case could still "pass" even if it throws exception to fail.
# Please check log message to make sure your test case passes.


class TestEngine:
    def test_register(self, tmpdir):
        # setup
        name = OnnxConversion.__name__
        host = SystemConfig(type=SystemType.Local)
        evaluator_config = OliveEvaluatorConfig(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)])

        options = {
            "cache_config": {
                "cache_dir": tmpdir,
                "clean_cache": True,
            },
            "search_strategy": {
                "execution_order": "joint",
                "sampler": "random",
            },
        }
        engine = Engine(**options)

        # execute
        engine.register(OnnxConversion, host=host, evaluator_config=evaluator_config)

        # assert
        assert name in engine.input_passes_configs
        assert len(engine.input_passes_configs[name]) == 1
        assert engine.input_passes_configs[name][0].type == OnnxConversion.__name__.lower()
        assert engine.input_passes_configs[name][0].host == host
        assert engine.input_passes_configs[name][0].evaluator == evaluator_config

    def test_register_no_search(self, tmpdir):
        # setup
        options = {
            "cache_config": {
                "cache_dir": tmpdir,
                "clean_cache": True,
            },
            "search_strategy": None,
        }
        engine = Engine(**options)

        # execute
        engine.register(OnnxDynamicQuantization)

        # assert
        assert "OnnxDynamicQuantization" in engine.input_passes_configs

    def test_default_engine_run(self, tmpdir):
        # setup
        model_config = get_pytorch_model_config()
        engine = Engine(cache_config={"cache_dir": tmpdir})

        engine.register(OnnxConversion, name="converter_13", config={"target_opset": 13})
        outputs = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=tmpdir,
        )

        assert outputs
        for fp_nodes in outputs.values():
            for node in fp_nodes.nodes.values():
                assert node.model_config
                assert node.from_pass == "onnxconversion"
                assert node.metrics is None, "Should not evaluate input/output model by default"

    @patch("olive.systems.local.LocalSystem")
    def test_run(self, mock_local_system, tmp_path):
        # setup
        model_config = get_pytorch_model_config()
        input_model_id = model_config.get_model_id()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_config": {
                "cache_dir": tmp_path,
                "clean_cache": True,
                "clean_evaluation_cache": True,
            },
            "search_strategy": {
                "execution_order": "joint",
                "sampler": "random",
            },
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
        p_name = "converter"
        p1: OnnxConversion = get_onnxconversion_pass(target_opset=13)
        p2: OnnxConversion = get_onnxconversion_pass(target_opset=14)
        engine.set_input_passes_configs(
            {
                p_name: [
                    RunPassConfig.from_json(p1.to_json(check_object=True)),
                    RunPassConfig.from_json(p2.to_json(check_object=True)),
                ]
            }
        )
        model_ids = [
            engine.cache.get_output_model_id(p1.__class__.__name__, p1.config.to_json(), input_model_id),
            engine.cache.get_output_model_id(p2.__class__.__name__, p2.config.to_json(), input_model_id),
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
        output_dir = Path(tmp_path)
        actual_res = engine.run(model_config, [DEFAULT_CPU_ACCELERATOR], output_dir=output_dir)
        actual_res = actual_res[DEFAULT_CPU_ACCELERATOR]

        # make sure the input model always be in engine.footprints
        footprint = engine.footprints[DEFAULT_CPU_ACCELERATOR]
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
        system_object.evaluate_model.assert_called_with(onnx_model_config, evaluator_config, DEFAULT_CPU_ACCELERATOR)

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

        engine = Engine(cache_config={"cache_dir": tmpdir})
        engine.register(OptimumConversion)
        # output model to output_dir
        output_dir = Path(tmpdir)

        # execute
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        _actual_res = engine.run(
            model_config,
            [accelerator_spec],
            output_dir=output_dir,
        )

        # assert
        actual_res = next(iter(_actual_res[accelerator_spec].nodes.values()))
        assert composite_onnx_model_config.to_json() == actual_res.model_config

    @patch("olive.systems.local.LocalSystem")
    def test_run_no_search(self, mock_local_system_init, tmp_path):
        # setup
        model_config = get_pytorch_model_config()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_config": {
                "cache_dir": tmp_path,
                "clean_cache": True,
                "clean_evaluation_cache": True,
            },
            "search_strategy": None,
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

        engine = Engine(**options)
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        p_config = OnnxConversion.generate_config(accelerator_spec, {"target_opset": 13}).dict()
        engine.register(OnnxConversion, config=p_config)

        output_model_id = engine.cache.get_output_model_id(
            "OnnxConversion", p_config, model_config.get_model_id(), accelerator_spec
        )
        model_path = engine.cache.get_model_cache_path(output_model_id) / "model.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w") as f:
            f.write("dummy-onnx-model")
        assert model_path.is_file()
        onnx_model_config = get_onnx_model_config(model_path=model_path)

        mock_local_system = MagicMock()
        mock_local_system_init.return_value = mock_local_system
        mock_local_system.system_type = SystemType.Local
        mock_local_system.run_pass.return_value = onnx_model_config
        mock_local_system.evaluate_model.return_value = MetricResult.parse_obj(metric_result_dict)
        mock_local_system.get_supported_execution_providers.return_value = ["CPUExecutionProvider"]
        mock_local_system.olive_managed_env = False

        # output model to output_dir
        output_dir = tmp_path / "output_dir"
        expected_metrics = MetricResult.parse_obj(metric_result_dict)
        expected_saved_model_config = get_onnx_model_config(model_path=output_dir / "model.onnx")

        # execute
        footprint = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=output_dir,
        )
        output_node = next(iter(footprint[accelerator_spec].nodes.values()))

        # assert
        assert output_node.model_config == onnx_model_config
        assert expected_metrics == output_node.metrics.value

        model_json_path = output_dir / "model_config.json"
        assert model_json_path.is_file()
        with model_json_path.open() as f:
            assert json.load(f) == expected_saved_model_config.to_json()

        result_json_path = output_dir / "metrics.json"
        assert result_json_path.is_file()
        with result_json_path.open() as f:
            assert json.load(f) == expected_metrics.__root__

    @pytest.mark.parametrize(
        "search_strategy",
        [
            {
                "execution_order": "joint",
                "sampler": "random",
            },
            None,
        ],
    )
    def test_run_output_model(self, search_strategy, tmp_path):
        # setup
        model_config = get_pytorch_model_config()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_config": {
                "cache_dir": tmp_path,
                "clean_cache": True,
                "clean_evaluation_cache": True,
            },
            "search_strategy": search_strategy,
            "evaluator": evaluator_config,
        }
        engine = Engine(**options)
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        p_config = OnnxConversion.generate_config(accelerator_spec, {"target_opset": 13}).dict()
        engine.register(OnnxConversion, config=p_config)
        # output model to output_dir
        output_dir = tmp_path / "output_dir"

        # execute
        engine.run(
            model_config,
            [accelerator_spec],
            output_dir=output_dir,
        )

        # assert
        assert Path(output_dir / "model.onnx").is_file()

    def test_pass_exception(self, caplog, tmpdir):
        # Need explicitly set the propagate to allow the message to be logged into caplog
        # setup
        logger = logging.getLogger("olive")
        logger.propagate = True

        with patch("olive.passes.onnx.conversion.OnnxConversion.run") as mock_run:
            mock_run.side_effect = Exception("test")
            evaluator_config = OliveEvaluatorConfig(metrics=[get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)])
            options = {
                "cache_config": {
                    "cache_dir": tmpdir,
                    "clean_cache": True,
                },
                "search_strategy": {
                    "execution_order": "joint",
                    "sampler": "random",
                },
                "evaluator": evaluator_config,
            }
            engine = Engine(**options)
            engine.register(OnnxConversion)

            model_config = get_pytorch_model_config()

            # execute
            output_dir = Path(tmpdir)
            engine.run(
                model_config,
                [DEFAULT_CPU_ACCELERATOR],
                output_dir=output_dir,
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
            "cache_config": {
                "cache_dir": tmpdir,
                "clean_cache": True,
                "clean_evaluation_cache": True,
            },
            "search_strategy": None,
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
        engine.register(OnnxConversion)

        # output model to output_dir
        output_dir = Path(tmpdir)
        expected_res = MetricResult.parse_obj(metric_result_dict)

        # execute
        actual_res = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=output_dir,
            evaluate_input_model=True,
        )
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        actual_res = next(iter(actual_res[accelerator_spec].nodes.values())).metrics.value

        assert expected_res == actual_res
        result_json_path = Path(output_dir / "input_model_metrics.json")
        assert result_json_path.is_file()
        assert MetricResult.parse_file(result_json_path) == actual_res

    @patch("olive.systems.local.LocalSystem")
    def test_run_no_pass(self, mock_local_system_init, tmp_path):
        # setup
        model_config = get_pytorch_model_config()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_config": {
                "cache_dir": tmp_path,
                "clean_cache": True,
                "clean_evaluation_cache": True,
            },
            "search_strategy": None,
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
        output_dir = tmp_path
        expected_res = MetricResult.parse_obj(metric_result_dict)

        # execute
        actual_res = engine.run(
            model_config,
            [DEFAULT_CPU_ACCELERATOR],
            output_dir=output_dir,
            evaluate_input_model=True,
        )
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        actual_res = actual_res[accelerator_spec]

        assert expected_res == actual_res
        result_json_path = output_dir / "input_model_metrics.json"
        assert result_json_path.is_file()
        assert MetricResult.parse_file(result_json_path) == actual_res

    @patch("olive.systems.local.LocalSystem")
    @patch("onnxruntime.get_available_providers")
    def test_pass_cache(self, mock_get_available_providers, mock_local_system_init, tmpdir):
        # setup
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {
            "cache_config": {
                "cache_dir": tmpdir,
                "clean_cache": True,
                "clean_evaluation_cache": True,
            },
            "search_strategy": {
                "execution_order": "joint",
                "sampler": "random",
            },
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
        engine.register(OnnxConversion)

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
                "cache_config": {
                    "cache_dir": tmpdir,
                    "clean_cache": True,
                },
                "search_strategy": {
                    "execution_order": "joint",
                    "sampler": "random",
                },
                "evaluator": evaluator_config,
            }
            engine = Engine(**options)
            engine.register(OnnxConversion)
            model_config = get_pytorch_model_config()
            # execute
            output_dir = Path(tmpdir)
            with pytest.raises(ValueError):  # noqa: PT011
                engine.run(
                    model_config,
                    [DEFAULT_CPU_ACCELERATOR],
                    output_dir=output_dir,
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
                "cache_config": {
                    "cache_dir": tmpdir,
                    "clean_cache": True,
                },
                "search_strategy": {
                    "execution_order": "joint",
                    "sampler": "random",
                    "max_samples": 1,
                    "seed": 1,
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
                )
                assert not actual_res, "Expect empty dict when quantization fails"
        else:
            options = {
                "cache_config": {
                    "cache_dir": tmpdir,
                    "clean_cache": True,
                },
                "search_strategy": None,
            }
            engine = Engine(**options)
            engine.register(OnnxDynamicQuantization)
            with patch("onnxruntime.quantization.quantize_dynamic") as mock_quantize_dynamic:
                mock_quantize_dynamic.side_effect = AttributeError("test")
                actual_res = engine.run(
                    onnx_model_config,
                    [DEFAULT_CPU_ACCELERATOR],
                    output_dir=output_dir,
                    evaluate_input_model=False,
                )
                assert not actual_res[DEFAULT_CPU_ACCELERATOR].nodes, "Expect empty dict when quantization fails"
