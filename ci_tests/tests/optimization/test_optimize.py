import os
import shutil

import pytest

from olive.optimization_config import OptimizationConfig
from olive.optimize import optimize

ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "onnx_mnist", "model.onnx")
SAMPLE_INPUT_DATA_PATH = os.path.join(os.path.dirname(__file__), "onnx_mnist", "sample_input_data.npz")


@pytest.mark.parametrize('quantization_enabled', [False, True])
def test_optimize_quantization(quantization_enabled):
    model_path = os.path.join(os.path.dirname(__file__), "other_models", "TFBertForQuestionAnswering.onnx")
    result_path = "quantization_opt_{}".format(quantization_enabled)
    inputs_spec = {"attention_mask": [1, 7], "input_ids": [1, 7], "token_type_ids": [1, 7]}
    opt_config = OptimizationConfig(model_path=model_path, inputs_spec=inputs_spec,
                                    quantization_enabled=quantization_enabled, result_path=result_path)
    optimize(opt_config)
    assert os.path.exists(result_path)
    shutil.rmtree(result_path)


@pytest.mark.parametrize('transformer_enabled', [False, True])
def test_optimize_transformer(transformer_enabled):
    model_path = os.path.join(os.path.dirname(__file__), "other_models", "TFBertForQuestionAnswering.onnx")
    result_path = "transformer_opt_{}".format(transformer_enabled)
    inputs_spec = {"attention_mask": [1, 7],"input_ids": [1, 7], "token_type_ids": [1, 7]}
    if transformer_enabled:
        transformer_args = "--model_type bert --num_heads 12"
        opt_config = OptimizationConfig(model_path=model_path, inputs_spec=inputs_spec, result_path=result_path,
                                        transformer_enabled=transformer_enabled, transformer_args=transformer_args)
        optimize(opt_config)
    else:
        opt_config = OptimizationConfig(model_path=model_path, inputs_spec=inputs_spec, result_path=result_path)
        optimize(opt_config)
    assert os.path.exists(result_path)
    shutil.rmtree(result_path)


@pytest.mark.parametrize('providers_list', [None, ["cpu"]])
def test_optimize_providers(providers_list):
    result_path = "ep_opt_{}".format(providers_list)
    opt_config = OptimizationConfig(model_path=ONNX_MODEL_PATH, providers_list=providers_list, result_path=result_path)
    optimize(opt_config)
    assert os.path.exists(result_path)
    shutil.rmtree(result_path)


@pytest.mark.parametrize('concurrency_num', [2])
def test_optimize_concurrency(concurrency_num):
    result_path = "concurrency_opt_{}".format(concurrency_num)
    opt_config = OptimizationConfig(model_path=ONNX_MODEL_PATH, concurrency_num=concurrency_num,
                                    result_path=result_path)
    optimize(opt_config)
    shutil.rmtree(result_path)


@pytest.mark.parametrize('sample_input_data_path', [SAMPLE_INPUT_DATA_PATH])
def test_optimize_sample_data(sample_input_data_path):
    result_path = "sample_data_opt"
    opt_config = OptimizationConfig(model_path=ONNX_MODEL_PATH, sample_input_data_path=sample_input_data_path,
                                    result_path=result_path)
    optimize(opt_config)
    shutil.rmtree(result_path)


@pytest.mark.parametrize('dynamic_batching_size', [1, 4])
def test_throughput_tuning(dynamic_batching_size):
    result_path = "throughput_tuning_res"
    model_path = os.path.join(os.path.dirname(__file__), "other_models", "TFBertForQuestionAnswering.onnx")

    opt_config = OptimizationConfig(model_path=model_path,
                                    inputs_spec={"attention_mask": [-1, 7], "input_ids": [-1, 7], "token_type_ids": [-1, 7]},
                                    throughput_tuning_enabled=True,
                                    max_latency_percentile=0.95,
                                    max_latency=0.1,
                                    threads_num=1,
                                    dynamic_batching_size=dynamic_batching_size,
                                    result_path=result_path,
                                    min_duration_sec=1)
    optimize(opt_config)
    shutil.rmtree(result_path)
