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
from olive.hardware.constants import ExecutionProvider
from olive.systems.accelerator_creator import AcceleratorNormalizer, create_accelerators
from olive.systems.common import AcceleratorConfig, SystemType
from olive.systems.python_environment.python_environment_system import PythonEnvironmentSystem
from olive.systems.system_config import SystemConfig


@pytest.mark.parametrize(
    "execution_providers_test",
    [
        ([ExecutionProvider.CPUExecutionProvider], None),
        ([ExecutionProvider.OpenVINOExecutionProvider], None),
        ([ExecutionProvider.CUDAExecutionProvider], ["gpu"]),
        ([ExecutionProvider.CPUExecutionProvider, ExecutionProvider.CUDAExecutionProvider], ["gpu"]),
        ([ExecutionProvider.DmlExecutionProvider, ExecutionProvider.CUDAExecutionProvider], None),
        ([ExecutionProvider.VitisAIExecutionProvider, ExecutionProvider.CUDAExecutionProvider], ["npu", "gpu"]),
    ],
)
def test_infer_accelerators_from_execution_provider(execution_providers_test):
    execution_providers, expected_accelerators = execution_providers_test
    actual_rls = AcceleratorLookup.infer_devices_from_execution_providers(execution_providers)
    assert actual_rls == expected_accelerators


# NOTE: Use PythonEnvironmentSystem test cases when using EPs that are not CPU EP
# The @patch("onnxruntime.get_available_providers") doesn't seem to work on Windows in CI
@pytest.mark.parametrize(
    ("system_config", "expected_acc_specs", "available_providers"),
    [
        # LocalSystem
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"device": "cpu", "execution_providers": [ExecutionProvider.CPUExecutionProvider]}]
                },
            },
            [("cpu", ExecutionProvider.CPUExecutionProvider)],
            [ExecutionProvider.CPUExecutionProvider],
        ),
        # PythonEnvironment
        # no accelerators provided
        (
            {"type": "PythonEnvironment", "config": {"python_environment_path": Path(sys.executable).parent}},
            [("cpu", ExecutionProvider.CPUExecutionProvider)],
            ["AzureExecutionProvider", ExecutionProvider.CPUExecutionProvider],
        ),
        (
            {"type": "PythonEnvironment", "config": {"python_environment_path": Path(sys.executable).parent}},
            [("cpu", ExecutionProvider.CPUExecutionProvider)],
            [
                ExecutionProvider.TensorrtExecutionProvider,
                ExecutionProvider.CUDAExecutionProvider,
                ExecutionProvider.CPUExecutionProvider,
            ],
        ),
        # only device provided
        (
            {
                "type": "PythonEnvironment",
                "config": {"accelerators": [{"device": "cpu"}], "python_environment_path": Path(sys.executable).parent},
            },
            [("cpu", ExecutionProvider.CPUExecutionProvider)],
            [ExecutionProvider.CPUExecutionProvider],
        ),
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "gpu"}]},
            },
            [("gpu", ExecutionProvider.CUDAExecutionProvider)],
            [ExecutionProvider.CUDAExecutionProvider],
        ),
        (
            {
                "type": "PythonEnvironment",
                "config": {"accelerators": [{"device": "gpu"}], "python_environment_path": Path(sys.executable).parent},
            },
            [("gpu", ExecutionProvider.CUDAExecutionProvider)],
            [ExecutionProvider.CUDAExecutionProvider],
        ),
        # only EP provided
        (
            {
                "type": "PythonEnvironment",
                "config": {
                    "accelerators": [{"execution_providers": [ExecutionProvider.CPUExecutionProvider]}],
                    "python_environment_path": Path(sys.executable).parent,
                },
            },
            [("cpu", ExecutionProvider.CPUExecutionProvider)],
            [ExecutionProvider.CPUExecutionProvider],
        ),
        # for qnn, if only EP provided, we map it to npu device
        (
            {
                "type": "PythonEnvironment",
                "config": {
                    "accelerators": [{"execution_providers": [ExecutionProvider.QNNExecutionProvider]}],
                    "python_environment_path": Path(sys.executable).parent,
                },
            },
            [("npu", ExecutionProvider.QNNExecutionProvider)],
            [ExecutionProvider.QNNExecutionProvider],
        ),
        # both device and EP provided
        (
            {
                "type": "PythonEnvironment",
                "config": {
                    "accelerators": [
                        {"device": "gpu", "execution_providers": [ExecutionProvider.CUDAExecutionProvider]}
                    ],
                    "python_environment_path": Path(sys.executable).parent,
                },
            },
            [("gpu", ExecutionProvider.CUDAExecutionProvider)],
            [
                ExecutionProvider.TensorrtExecutionProvider,
                ExecutionProvider.CUDAExecutionProvider,
                ExecutionProvider.CPUExecutionProvider,
            ],
        ),
        # LocalSystem with memory
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "device": "cpu",
                            "execution_providers": [ExecutionProvider.CPUExecutionProvider],
                            "memory": 1234,
                        }
                    ]
                },
            },
            [("cpu", ExecutionProvider.CPUExecutionProvider, 1234)],
            [ExecutionProvider.CPUExecutionProvider],
        ),
    ],
)
@patch("olive.systems.local.get_ort_available_providers")
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
            [{"device": "cpu", "execution_providers": [ExecutionProvider.CPUExecutionProvider]}],
            ["No accelerators specified. Defaulting to cpu."],
            ["AzureExecutionProvider", ExecutionProvider.CPUExecutionProvider],
        ),
        (
            # fill both device and ep
            {"type": "LocalSystem"},
            [{"device": "cpu", "execution_providers": [ExecutionProvider.CPUExecutionProvider]}],
            ["No accelerators specified. Defaulting to cpu."],
            [
                ExecutionProvider.TensorrtExecutionProvider,
                ExecutionProvider.CUDAExecutionProvider,
                ExecutionProvider.CPUExecutionProvider,
            ],
        ),
        (
            # fill the EPs.
            {"type": "LocalSystem", "config": {"accelerators": [{"device": "cpu"}]}},
            [{"device": "cpu", "execution_providers": [ExecutionProvider.CPUExecutionProvider]}],
            [
                "The following execution providers are filtered: OpenVINOExecutionProvider,DmlExecutionProvider.",
                (
                    "The accelerator execution providers is not specified for cpu. Use the inferred ones. "
                    "['CPUExecutionProvider']"
                ),
            ],
            [
                ExecutionProvider.OpenVINOExecutionProvider,
                ExecutionProvider.CPUExecutionProvider,
                ExecutionProvider.DmlExecutionProvider,
            ],
        ),
        (
            # fill the EPs.
            {"type": "LocalSystem", "config": {"accelerators": [{"device": "gpu"}]}},
            [{"device": "gpu", "execution_providers": [ExecutionProvider.CUDAExecutionProvider]}],
            [
                "The following execution providers are filtered: CPUExecutionProvider.",
                (
                    "The accelerator execution providers is not specified for gpu. Use the inferred ones. "
                    "['CUDAExecutionProvider']"
                ),
            ],
            [ExecutionProvider.CPUExecutionProvider, ExecutionProvider.CUDAExecutionProvider],
        ),
        (
            # user only specify EPs
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"execution_providers": [ExecutionProvider.CPUExecutionProvider]}]},
            },
            [{"device": "cpu", "execution_providers": [ExecutionProvider.CPUExecutionProvider]}],
            ["the accelerator device is not specified. Inferred device: cpu."],
            [ExecutionProvider.CPUExecutionProvider],
        ),
        (
            # deduce device for GPU
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "execution_providers": [
                                ExecutionProvider.CUDAExecutionProvider,
                                ExecutionProvider.CPUExecutionProvider,
                            ]
                        }
                    ]
                },
            },
            [
                {
                    "device": "gpu",
                    "execution_providers": [
                        ExecutionProvider.CUDAExecutionProvider,
                        ExecutionProvider.CPUExecutionProvider,
                    ],
                }
            ],
            [],
            [ExecutionProvider.CUDAExecutionProvider, ExecutionProvider.CPUExecutionProvider],
        ),
        (
            # deduce device for GPU
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "execution_providers": [
                                (ExecutionProvider.CUDAExecutionProvider, "onnxruntime_providers_cuda.dll"),
                            ]
                        }
                    ]
                },
            },
            [
                {
                    "device": "gpu",
                    "execution_providers": [
                        (ExecutionProvider.CUDAExecutionProvider, "onnxruntime_providers_cuda.dll")
                    ],
                }
            ],
            [],
            [ExecutionProvider.CUDAExecutionProvider, ExecutionProvider.CPUExecutionProvider],
        ),
        (
            # doesn't fill both device and ep.
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {"device": "cpu", "execution_providers": [ExecutionProvider.OpenVINOExecutionProvider]}
                    ]
                },
            },
            [{"device": "cpu", "execution_providers": [ExecutionProvider.OpenVINOExecutionProvider]}],
            ["The accelerator device and execution providers are specified, skipping deduce"],
            [ExecutionProvider.OpenVINOExecutionProvider, ExecutionProvider.CPUExecutionProvider],
        ),
        (
            # doesn't fill both device and ep.
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "device": "cpu",
                            "execution_providers": [
                                (ExecutionProvider.OpenVINOExecutionProvider, "onnxruntime_providers_openvino.dll")
                            ],
                        }
                    ]
                },
            },
            [
                {
                    "device": "cpu",
                    "execution_providers": [
                        (ExecutionProvider.OpenVINOExecutionProvider, "onnxruntime_providers_openvino.dll")
                    ],
                }
            ],
            ["The accelerator device and execution providers are specified, skipping deduce"],
            [ExecutionProvider.OpenVINOExecutionProvider, ExecutionProvider.CPUExecutionProvider],
        ),
        (
            # user specify invalid EPs.
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "execution_providers": [
                                ExecutionProvider.ROCMExecutionProvider,
                                ExecutionProvider.CUDAExecutionProvider,
                                ExecutionProvider.CPUExecutionProvider,
                            ]
                        }
                    ]
                },
            },
            [
                {
                    "device": "gpu",
                    "execution_providers": [
                        ExecutionProvider.CUDAExecutionProvider,
                        ExecutionProvider.CPUExecutionProvider,
                    ],
                }
            ],
            ["The following execution providers are not supported: 'ROCMExecutionProvider'"],
            [ExecutionProvider.CUDAExecutionProvider, ExecutionProvider.CPUExecutionProvider],
        ),
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "device": "gpu",
                            "execution_providers": [
                                ExecutionProvider.ROCMExecutionProvider,
                                ExecutionProvider.CUDAExecutionProvider,
                                ExecutionProvider.CPUExecutionProvider,
                            ],
                        }
                    ]
                },
            },
            [
                {
                    "device": "gpu",
                    "execution_providers": [
                        ExecutionProvider.CUDAExecutionProvider,
                        ExecutionProvider.CPUExecutionProvider,
                    ],
                }
            ],
            ["The following execution providers are not supported: 'ROCMExecutionProvider'"],
            [ExecutionProvider.CUDAExecutionProvider, ExecutionProvider.CPUExecutionProvider],
        ),
        (
            # fill the EPs. memory is provided.
            {"type": "LocalSystem", "config": {"accelerators": [{"device": "cpu", "memory": "10kB"}]}},
            [
                {
                    "device": "cpu",
                    "execution_providers": [ExecutionProvider.CPUExecutionProvider],
                    "memory": 10000,
                }
            ],
            [
                "The following execution providers are filtered: OpenVINOExecutionProvider,DmlExecutionProvider.",
                (
                    "The accelerator execution providers is not specified for cpu. Use the inferred ones. "
                    "['CPUExecutionProvider']"
                ),
            ],
            [
                ExecutionProvider.OpenVINOExecutionProvider,
                ExecutionProvider.CPUExecutionProvider,
                ExecutionProvider.DmlExecutionProvider,
            ],
        ),
    ],
)
@patch("olive.systems.local.get_ort_available_providers")
@patch("olive.systems.local.maybe_register_ep_libraries")
def test_normalize_accelerators(
    maybe_register_ep_libraries_mock,
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
    has_accelerators = system_config.config.accelerators is not None
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
    if system_config.type == SystemType.Local and has_accelerators:
        assert maybe_register_ep_libraries_mock.call_count == 1

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
                "config": {
                    "accelerators": [
                        {"device": "cpu", "execution_providers": [ExecutionProvider.CUDAExecutionProvider]}
                    ]
                },
            },
            ("cpu", [ExecutionProvider.CPUExecutionProvider]),
        ),
        (
            {
                "type": "LocalSystem",
                "config": {"accelerators": [{"execution_providers": [ExecutionProvider.QNNExecutionProvider]}]},
            },
            ("npu", [ExecutionProvider.QNNExecutionProvider]),
        ),
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"execution_providers": [ExecutionProvider.QNNExecutionProvider], "memory": "1gb"}]
                },
            },
            ("npu", [ExecutionProvider.QNNExecutionProvider], 1e9),
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
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "execution_providers": [
                                ExecutionProvider.OpenVINOExecutionProvider,
                                ExecutionProvider.CPUExecutionProvider,
                            ]
                        }
                    ]
                },
            },
            [ExecutionProvider.CPUExecutionProvider],
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
                    "accelerators": [
                        {
                            "execution_providers": [
                                ExecutionProvider.VitisAIExecutionProvider,
                                ExecutionProvider.CUDAExecutionProvider,
                            ]
                        }
                    ]
                },
            },
            [ExecutionProvider.CPUExecutionProvider],
            AssertionError,
            (
                "Cannot infer the devices from the execution providers "
                "['VitisAIExecutionProvider', 'CUDAExecutionProvider']. Multiple devices are inferred: ['npu', 'gpu']."
            ),
        ),
    ],
)
@patch("olive.systems.local.get_ort_available_providers")
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
                "config": {
                    "accelerators": [{"device": "cpu", "execution_providers": [ExecutionProvider.CPUExecutionProvider]}]
                },
            },
            [("cpu", ExecutionProvider.CPUExecutionProvider)],
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
                "config": {"accelerators": [{"execution_providers": [ExecutionProvider.CPUExecutionProvider]}]},
            },
            [("cpu", None)],
        ),
        # LocalSystem with memory
        (
            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [
                        {
                            "device": "cpu",
                            "execution_providers": [ExecutionProvider.CPUExecutionProvider],
                            "memory": 1234,
                        }
                    ]
                },
            },
            [("cpu", ExecutionProvider.CPUExecutionProvider, 1234)],
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
    # only device
    acc_cfg1 = AcceleratorConfig.parse_obj({"device": "cpu"})
    assert acc_cfg1.execution_providers is None
    # only ep
    acc_cfg2 = AcceleratorConfig.parse_obj({"execution_providers": [ExecutionProvider.CPUExecutionProvider]})
    assert acc_cfg2.device is None
    # neither device nor ep
    with pytest.raises(ValueError, match="Either device or execution_providers must be provided"):
        _ = AcceleratorConfig.parse_obj({})
    # device and memory
    acc_cfg3 = AcceleratorConfig.parse_obj({"device": "cpu", "memory": "1MB"})
    assert acc_cfg3.memory == 1e6
    # with ep library path
    acc_cfg4 = AcceleratorConfig.parse_obj(
        {
            "device": "gpu",
            "execution_providers": [(ExecutionProvider.CUDAExecutionProvider, "onnxruntime_providers_cuda.dll")],
        }
    )
    assert acc_cfg4.device == "gpu"
    assert acc_cfg4.execution_providers == [(ExecutionProvider.CUDAExecutionProvider, "onnxruntime_providers_cuda.dll")]
    assert acc_cfg4.get_ep_strs() == [ExecutionProvider.CUDAExecutionProvider]
    assert acc_cfg4.get_ep_path_map() == {ExecutionProvider.CUDAExecutionProvider: "onnxruntime_providers_cuda.dll"}
