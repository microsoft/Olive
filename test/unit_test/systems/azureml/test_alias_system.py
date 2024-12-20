import pytest

from olive.systems.system_config import SystemConfig


@pytest.mark.parametrize(
    ("accelerators", "expected_accelerators"),
    [
        (
            [
                {"device": "gpu", "execution_providers": ["CUDAExecutionProvider"]},
                {"device": "cpu", "execution_providers": ["CPUExecutionProvider"]},
            ],
            [{"device": "gpu", "execution_providers": ["CUDAExecutionProvider"], "memory": None}],
        ),
        (
            [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
            None,
        ),
    ],
)
def test_system_alias(accelerators, expected_accelerators):
    config = {
        "type": "AzureNDV2System",
        "config": {
            "aml_compute": "gpu-cluster",
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
                "conda_file_path": "conda.yaml",
            },
        },
    }
    config["config"]["accelerators"] = accelerators
    system_config = SystemConfig.parse_obj(config)
    assert system_config.type == "AzureML"
    assert system_config.config.aml_compute == "gpu-cluster"
    assert system_config.config.aml_docker_config.base_image == "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
    assert system_config.config.accelerators == expected_accelerators


def test_system_alias_no_accelerators():
    config = {
        "type": "AzureNDV2System",
        "config": {
            "aml_compute": "gpu-cluster",
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
                "conda_file_path": "conda.yaml",
            },
        },
    }
    with pytest.raises(ValueError, match="accelerators is required for AzureML system"):
        _ = SystemConfig.parse_obj(config)


def test_surface_system_alias():
    config = {"type": "SurfaceProSystem1796"}
    system_config = SystemConfig.parse_obj(config)
    assert system_config.type == "LocalSystem"
    assert system_config.config.accelerators[0].device == "GPU"
    assert system_config.config.accelerators[0].execution_providers is None
