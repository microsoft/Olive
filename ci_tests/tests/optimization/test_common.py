import os

import onnxruntime as ort

from olive.optimization_config import OptimizationConfig
from olive.optimization.tuning_process import generate_test_name, create_inference_session
from olive.util import generate_npz_files

ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "onnx_mnist", "model.onnx")
DEFAULT_EP = ["CPUExecutionProvider"]

def test_optimization_config_providers_1():
    opt_config = OptimizationConfig(model_path=ONNX_MODEL_PATH)
    assert sorted(opt_config.providers_list) == sorted(ort.get_available_providers())


def test_optimization_config_providers_2():
    opt_config = OptimizationConfig(model_path=ONNX_MODEL_PATH, providers_list=["cpu"])
    assert opt_config.providers_list == ["CPUExecutionProvider"]


def test_optimization_config_providers_3():
    try:
        opt_config = OptimizationConfig(model_path=ONNX_MODEL_PATH, providers_list=["cuda"])
    except ValueError as e:
        assert str(e) == "No providers available for test"


def test_optimization_config_model_path_1():
    model_path = "model_not_exist.onnx"
    try:
        opt_config = OptimizationConfig(model_path=model_path)
    except FileNotFoundError as e:
        assert str(e) == "Can't find the model file, please check the model_path"


def test_optimization_config_model_path_2():
    model_path = os.path.join(os.path.dirname(__file__), "other_models", "pytorch_model.pth")
    try:
        opt_config = OptimizationConfig(model_path=model_path)
    except ValueError as e:
        assert str(e) == "File ends with .onnx is required for ONNX model"


def test_optimization_config_model_path_3():
    model_path = ONNX_MODEL_PATH
    opt_config = OptimizationConfig(model_path=model_path)
    assert opt_config.model_path == os.path.join(opt_config.result_path, "optimized_model.onnx")


def test_generate_test_name():
    test_params = {"execution_provider": "CPUExecutionProvider",
                   "execution_mode": ort.ExecutionMode.ORT_SEQUENTIAL,
                   "inter_op_num_threads": 2}
    test_name = generate_test_name(test_params)
    assert test_name == "execution_provider_CPUExecutionProvider_execution_mode_ExecutionMode.ORT_SEQUENTIAL_inter_op_num_threads_2"


def test_create_inference_session_1():
    model_path = ONNX_MODEL_PATH
    _, session_name = create_inference_session(model_path)
    assert session_name == "pretuning"


def test_create_inference_session_2():
    model_path = ONNX_MODEL_PATH
    test_params = {"execution_provider": "CPUExecutionProvider",
                   "execution_mode": ort.ExecutionMode.ORT_SEQUENTIAL,
                   "inter_op_num_threads": 2}
    onnx_session, session_name = create_inference_session(model_path, test_params)
    assert session_name == "execution_provider_CPUExecutionProvider_execution_mode_ExecutionMode.ORT_SEQUENTIAL_inter_op_num_threads_2"
    assert onnx_session.get_session_options().inter_op_num_threads == 2
    assert onnx_session.get_session_options().execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert onnx_session.get_providers() == ["CPUExecutionProvider"]


def test_test_optimization_config_input_data_1():
    model_path = ONNX_MODEL_PATH
    inputs_spec = {"Input3": [1, 1, 28, 28]}
    opt_config = OptimizationConfig(model_path=model_path, inputs_spec=inputs_spec)
    input_dict = opt_config.inference_input_dict
    onnx_session = ort.InferenceSession(opt_config.model_path, providers=DEFAULT_EP)
    output_names = [o.name for o in onnx_session.get_outputs()]
    onnx_session.run(output_names, input_dict)


def test_test_optimization_config_input_data_2():
    model_path = ONNX_MODEL_PATH
    sample_input_data_path = os.path.join(os.path.dirname(__file__), "onnx_mnist", "sample_input_data.npz")
    inputs_spec = {"Input3": [1, 1, 28, 28]}
    opt_config_1 = OptimizationConfig(model_path=model_path, inputs_spec=inputs_spec)
    input_dict_1 = opt_config_1.inference_input_dict
    # save random generated input data into sample_input_data.npz
    generate_npz_files(output_npz_path=sample_input_data_path, name_list=["Input3"], value_list=[input_dict_1.get("Input3")])
    # test gegerate_input_data_when sample_input_data_path is given
    opt_config_2 = OptimizationConfig(model_path=model_path, sample_input_data_path=sample_input_data_path)
    input_dict_2 = opt_config_2.inference_input_dict
    onnx_session = ort.InferenceSession(model_path, providers=DEFAULT_EP)
    output_names = [o.name for o in onnx_session.get_outputs()]
    result_1 = onnx_session.run(output_names, input_dict_1)
    result_2 = onnx_session.run(output_names, input_dict_2)
    assert len(result_1) == len(result_2)
    for i in range(len(result_1)):
        assert result_1[i].all() == result_2[i].all()
