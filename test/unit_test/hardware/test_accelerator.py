# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.common.config_utils import validate_config
from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec, create_accelerators
from olive.systems.common import AcceleratorConfig, SystemType
from olive.systems.python_environment.python_environment_system import PythonEnvironmentSystem
from olive.systems.system_config import SystemConfig


@pytest.mark.parametrize(
    "execution_providers_test",
    [
        (["CPUExecutionProvider"], None),
        (["AzureMLExecutionProvider"], None),
        (["OpenVINOExecutionProvider"], None),
        (["CUDAExecutionProvider"], ["gpu"]),
        (["CPUExecutionProvider", "CUDAExecutionProvider"], ["gpu"]),
        (["DmlExecutionProvider", "CUDAExecutionProvider"], ["gpu"]),
        (["QNNExecutionProvider", "CUDAExecutionProvider"], ["npu", "gpu"]),
    ],
)
def test_infer_accelerators_from_execution_provider(execution_providers_test):
    execution_providers, expected_accelerators = execution_providers_test
    actual_rls = AcceleratorLookup.infer_accelerators_from_execution_providers(execution_providers)
    assert actual_rls == expected_accelerators


@pytest.mark.parametrize(
    ("system_config", "expected_acc_specs", "available_providers"),
    [
        # LocalSystem
        (
            {"type": "LocalSystem"},
            [("cpu", "CPUExecutionProvider")],
            ["AzureExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            # openvino need device to be specified since it exists in both GPU and CPU
            {"type": "LocalSystem", "config": {"accelerators": [{"device": "cpu"}]}},
            [("CPU", "OpenVINOExecutionProvider"), ("cpu", "CPUExecutionProvider")],
            ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            {"type": "LocalSystem"},
            [("gpu", "TensorrtExecutionProvider"), ("gpu", "CUDAExecutionProvider"), ("gpu", "CPUExecutionProvider")],
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}]},
            },
            [("cpu", "CPUExecutionProvider")],
            ["CPUExecutionProvider"],
        ),
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"execution_providers": ["CPUExecutionProvider"]}]},
            },
            [("cpu", "CPUExecutionProvider")],
            ["CPUExecutionProvider"],
        ),
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}]
                },
            },
            [("gpu", "CUDAExecutionProvider"), ("gpu", "CPUExecutionProvider")],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        # PythonEnvironment
        (
            {"type": "PythonEnvironment", "config": {"python_environment_path": Path(sys.executable).parent}},
            [("cpu", "CPUExecutionProvider")],
            ["AzureExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            {
                "type": "PythonEnvironment",
                "config": {"accelerators": [{"device": "cpu"}], "python_environment_path": Path(sys.executable).parent},
            },
            [("CPU", "OpenVINOExecutionProvider"), ("cpu", "CPUExecutionProvider")],
            ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            {"type": "PythonEnvironment", "config": {"python_environment_path": Path(sys.executable).parent}},
            [("gpu", "TensorrtExecutionProvider"), ("gpu", "CUDAExecutionProvider"), ("gpu", "CPUExecutionProvider")],
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            {
                "type": "PythonEnvironment",
                "config": {
                    "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
                    "python_environment_path": Path(sys.executable).parent,
                },
            },
            [("cpu", "CPUExecutionProvider")],
            ["CPUExecutionProvider"],
        ),
        # AzureML system
        (
            {
                "type": "AzureML",
                "config": {
                    "aml_compute": "aml_compute",
                    "olive_managed_env": True,
                    "accelerators": [{"device": "gpu", "execution_providers": ["CUDAExecutionProvider"]}],
                },
            },
            [("gpu", "CUDAExecutionProvider")],
            None,
        ),
        (
            {
                "type": "AzureML",
                "config": {
                    "aml_compute": "aml_compute",
                    "olive_managed_env": False,
                    "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
                },
            },
            [("cpu", "CPUExecutionProvider")],
            None,
        ),
        # Docker system
        (
            {
                "type": "Docker",
                "config": {
                    "local_docker_config": {
                        "image_name": "olive-image",
                        "build_context_path": "docker",
                        "dockerfile": "Dockerfile",
                    },
                    "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
                },
            },
            [("cpu", "CPUExecutionProvider")],
            ["CPUExecutionProvider"],
        ),
    ],
)
@patch("onnxruntime.get_available_providers")
def test_create_accelerators(get_available_providers_mock, system_config, expected_acc_specs, available_providers):
    system_config = validate_config(system_config, SystemConfig)
    python_mock = None
    if system_config.type == SystemType.Local:
        get_available_providers_mock.return_value = available_providers
    elif system_config.type == SystemType.PythonEnvironment:
        python_mock = patch.object(
            PythonEnvironmentSystem, "get_supported_execution_providers", return_value=available_providers
        )
        python_mock.start()

    expected_accelerator_specs = [
        AcceleratorSpec(accelerator_type=acc_spec[0].lower(), execution_provider=acc_spec[1])
        for acc_spec in expected_acc_specs
    ]

    accelerators = create_accelerators(system_config, skip_supported_eps_check=False)
    assert accelerators == expected_accelerator_specs
    if python_mock:
        python_mock.stop()


@pytest.mark.parametrize(
    ("system_config", "available_providers", "exception", "error_message"),
    [
        (
            {"type": "LocalSystem"},
            ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
            AssertionError,
            "Cannot infer the accelerators from the execution providers",
        ),
        (
            {"type": "PythonEnvironment", "config": {"olive_managed_env": True}},
            None,
            ValueError,
            "Managed environment requires accelerators to be specified.",
        ),
        (
            {
                "type": "AzureML",
                "config": {
                    "aml_compute": "aml_compute",
                    "olive_managed_env": True,
                    "accelerators": [{"device": "cpu"}],
                },
            },
            None,
            ValueError,
            "Managed environment requires execution providers to be specified for cpu",
        ),
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"execution_providers": ["OpenVINOExecutionProvider", "CPUExecutionProvider"]}]
                },
            },
            ["CPUExecutionProvider"],
            AssertionError,
            (
                "Cannot infer the accelerators from the execution providers "
                "['OpenVINOExecutionProvider', 'CPUExecutionProvider']. "
                "Please specify the accelerator in the accelerator configs."
            ),
        ),
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"execution_providers": ["QNNExecutionProvider", "CUDAExecutionProvider"]}]
                },
            },
            ["CPUExecutionProvider"],
            AssertionError,
            (
                "Cannot infer the accelerators from the execution providers "
                "['QNNExecutionProvider', 'CUDAExecutionProvider']. Multiple accelerators are inferred: ['npu', 'gpu']."
                " Please specify the accelerator in the accelerator configs."
            ),
        ),
    ],
)
@patch("onnxruntime.get_available_providers")
def test_create_accelerator_with_error(
    get_available_providers_mock, system_config, available_providers, exception, error_message
):
    system_config = validate_config(system_config, SystemConfig)
    get_available_providers_mock.return_value = available_providers

    with pytest.raises(exception) as exp:
        create_accelerators(system_config)
    if error_message:
        assert error_message in str(exp.value)


def test_accelerator_config():
    acc_cfg1 = AcceleratorConfig.parse_obj({"device": "cpu"})
    assert acc_cfg1.execution_providers is None
    acc_cfg2 = AcceleratorConfig.parse_obj({"execution_providers": ["CPUExecutionProvider"]})
    assert acc_cfg2.device is None
    with pytest.raises(ValueError, match="Either device or execution_providers must be provided"):
        _ = AcceleratorConfig.parse_obj({})
