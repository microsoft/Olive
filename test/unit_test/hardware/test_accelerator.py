# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.common.config_utils import validate_config
from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec
from olive.systems.accelerator_creator import AcceleratorNormalizer, create_accelerators
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
        (["DmlExecutionProvider", "CUDAExecutionProvider"], None),
        (["QNNExecutionProvider", "CUDAExecutionProvider"], ["npu", "gpu"]),
    ],
)
def test_infer_accelerators_from_execution_provider(execution_providers_test):
    execution_providers, expected_accelerators = execution_providers_test
    actual_rls = AcceleratorLookup.infer_devices_from_execution_providers(execution_providers)
    assert actual_rls == expected_accelerators


@pytest.mark.parametrize(
    ("system_config", "expected_acc_specs", "available_providers"),
    [
        # LocalSystem
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}]},
            },
            [("cpu", "CPUExecutionProvider")],
            ["CPUExecutionProvider"],
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
        # LocalSystem with memory
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"], "memory": 1234}]
                },
            },
            [("cpu", "CPUExecutionProvider", 1234)],
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
        AcceleratorSpec(
            accelerator_type=acc_spec[0].lower(),
            execution_provider=acc_spec[1],
            memory=acc_spec[2] if len(acc_spec) == 3 else None,
        )
        for acc_spec in expected_acc_specs
    ]

    accelerators = create_accelerators(system_config, skip_supported_eps_check=False)
    assert accelerators == expected_accelerator_specs
    if python_mock:
        python_mock.stop()


@pytest.mark.parametrize(
    ("system_config", "expected_accs", "expected_logs", "available_providers"),
    [
        (
            # fill both device and ep.
            {"type": "LocalSystem"},
            [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
            [
                "There is no any accelerator specified. Inferred accelerators: "
                "[AcceleratorConfig(device='cpu', execution_providers=['CPUExecutionProvider'], memory=None)]"
            ],
            ["AzureExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            # fill both device and ep but with GPU
            {"type": "LocalSystem"},
            [
                {
                    "device": "gpu",
                    "execution_providers": [
                        "TensorrtExecutionProvider",
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                }
            ],
            [],
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            # fill the EPs.
            {"type": "LocalSystem", "config": {"accelerators": [{"device": "cpu"}]}},
            [{"device": "cpu", "execution_providers": ["OpenVINOExecutionProvider", "CPUExecutionProvider"]}],
            [
                "The following execution providers are filtered: DmlExecutionProvider.",
                (
                    "The accelerator execution providers is not specified for cpu. Use the inferred ones. "
                    "['OpenVINOExecutionProvider', 'CPUExecutionProvider']"
                ),
            ],
            ["OpenVINOExecutionProvider", "CPUExecutionProvider", "DmlExecutionProvider"],
        ),
        (
            # user only specify EPs
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"execution_providers": ["CPUExecutionProvider"]}]},
            },
            [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
            ["the accelerator device is not specified. Inferred device: cpu."],
            ["CPUExecutionProvider"],
        ),
        (
            # deduce device for GPU
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}]
                },
            },
            [
                {
                    "device": "gpu",
                    "execution_providers": [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                }
            ],
            [],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            # doesn't fill both device and ep.
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "cpu", "execution_providers": ["OpenVINOExecutionProvider"]}]},
            },
            [{"device": "cpu", "execution_providers": ["OpenVINOExecutionProvider"]}],
            ["The accelerator device and execution providers are specified, skipping deduce"],
            ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            # user specify invalid EPs.
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "execution_providers": [
                                "ROCMExecutionProvider",
                                "CUDAExecutionProvider",
                                "CPUExecutionProvider",
                            ]
                        }
                    ]
                },
            },
            [
                {
                    "device": "gpu",
                    "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                }
            ],
            ["The following execution providers are not supported: 'ROCMExecutionProvider'"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "device": "gpu",
                            "execution_providers": [
                                "ROCMExecutionProvider",
                                "CUDAExecutionProvider",
                                "CPUExecutionProvider",
                            ],
                        }
                    ]
                },
            },
            [
                {
                    "device": "gpu",
                    "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                }
            ],
            ["The following execution providers are not supported: 'ROCMExecutionProvider'"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            # fill the EPs. memory is provided.
            {"type": "LocalSystem", "config": {"accelerators": [{"device": "cpu", "memory": "10kB"}]}},
            [
                {
                    "device": "cpu",
                    "execution_providers": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
                    "memory": 10000,
                }
            ],
            [
                "The following execution providers are filtered: DmlExecutionProvider.",
                (
                    "The accelerator execution providers is not specified for cpu. Use the inferred ones. "
                    "['OpenVINOExecutionProvider', 'CPUExecutionProvider']"
                ),
            ],
            ["OpenVINOExecutionProvider", "CPUExecutionProvider", "DmlExecutionProvider"],
        ),
    ],
)
@patch("onnxruntime.get_available_providers")
def test_normalize_accelerators(
    get_available_providers_mock,
    caplog,
    system_config,
    expected_accs,
    expected_logs,
    available_providers,
):
    # capture logging
    logger = logging.getLogger("olive")
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger="olive")

    system_config = validate_config(system_config, SystemConfig)
    python_mock = None
    if system_config.type == SystemType.Local:
        get_available_providers_mock.return_value = available_providers
    elif system_config.type == SystemType.PythonEnvironment:
        python_mock = patch.object(
            PythonEnvironmentSystem, "get_supported_execution_providers", return_value=available_providers
        )
        python_mock.start()

    normalized_accs = AcceleratorNormalizer(system_config, skip_supported_eps_check=False).normalize()
    assert len(normalized_accs.config.accelerators) == len(expected_accs)
    for i, acc in enumerate(expected_accs):
        assert normalized_accs.config.accelerators[i].device == acc["device"]
        assert normalized_accs.config.accelerators[i].execution_providers == acc["execution_providers"]
        if "memory" in acc:
            assert normalized_accs.config.accelerators[i].memory == acc["memory"]

    if expected_logs:
        for log in expected_logs:
            assert log in caplog.text

    if python_mock:
        python_mock.stop()


@pytest.mark.parametrize(
    ("system_config", "expected_acc"),
    [
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "cpu", "execution_providers": ["CUDAExecutionProvider"]}]},
            },
            ("cpu", ["CPUExecutionProvider"]),
        ),
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"execution_providers": ["QNNExecutionProvider"]}]},
            },
            ("npu", ["QNNExecutionProvider"]),
        ),
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"execution_providers": ["QNNExecutionProvider"], "memory": "1gb"}]},
            },
            ("npu", ["QNNExecutionProvider"], 1e9),
        ),
    ],
)
def test_normalize_accelerators_skip_ep_check(system_config, expected_acc):
    system_config = validate_config(system_config, SystemConfig)
    normalized_accs = AcceleratorNormalizer(system_config, skip_supported_eps_check=True).normalize()
    assert normalized_accs.config.accelerators[0].device == expected_acc[0]
    assert normalized_accs.config.accelerators[0].execution_providers == expected_acc[1]
    if len(expected_acc) == 3:
        assert normalized_accs.config.accelerators[0].memory == expected_acc[2]


@pytest.mark.parametrize(
    ("system_config", "available_providers", "exception", "error_message"),
    [
        (
            {"type": "LocalSystem"},
            ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
            AssertionError,
            "Cannot infer the devices from the execution providers",
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
                "Cannot infer the devices from the execution providers "
                "['OpenVINOExecutionProvider', 'CPUExecutionProvider']."
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
                "Cannot infer the devices from the execution providers "
                "['QNNExecutionProvider', 'CUDAExecutionProvider']. Multiple devices are inferred: ['npu', 'gpu']."
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


@pytest.mark.parametrize(
    ("system_config", "expected_acc_specs"),
    [
        # LocalSystem
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}]},
            },
            [("cpu", "CPUExecutionProvider")],
        ),
        # doesn't specify the accelerator
        (
            {
                "type": "LocalSystem",
            },
            [("cpu", None)],
        ),
        # only specify the device
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "gpu"}]},
            },
            [("gpu", None)],
        ),
        # only specify the EP
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"execution_providers": ["CPUExecutionProvider"]}]},
            },
            [("cpu", None)],
        ),
        (
            {
                "type": "AzureML",
                "config": {
                    "aml_compute": "aml_compute",
                    "olive_managed_env": False,
                    "accelerators": [{"device": "gpu"}],
                },
            },
            [("gpu", None)],
        ),
        # LocalSystem with memory
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"], "memory": 1234}]
                },
            },
            [("cpu", "CPUExecutionProvider", 1234)],
        ),
    ],
)
def test_create_accelerator_without_ep(system_config, expected_acc_specs):
    system_config = validate_config(system_config, SystemConfig)
    expected_accelerator_specs = [
        AcceleratorSpec(
            accelerator_type=acc_spec[0].lower(),
            execution_provider=acc_spec[1],
            memory=acc_spec[2] if len(acc_spec) == 3 else None,
        )
        for acc_spec in expected_acc_specs
    ]
    accelerators = create_accelerators(system_config, skip_supported_eps_check=False, is_ep_required=False)
    assert accelerators == expected_accelerator_specs


def test_accelerator_config():
    acc_cfg1 = AcceleratorConfig.parse_obj({"device": "cpu"})
    assert acc_cfg1.execution_providers is None
    acc_cfg2 = AcceleratorConfig.parse_obj({"execution_providers": ["CPUExecutionProvider"]})
    assert acc_cfg2.device is None
    with pytest.raises(ValueError, match="Either device or execution_providers must be provided"):
        _ = AcceleratorConfig.parse_obj({})
    acc_cfg3 = AcceleratorConfig.parse_obj({"device": "cpu", "memory": "1MB"})
    assert acc_cfg3.memory == 1e6
