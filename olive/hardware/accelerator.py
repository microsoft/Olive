# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from dataclasses import dataclass
from enum import Enum


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    INTEL_MYRIAD = "intel_myriad"


@dataclass
class AcceleratorSpec:
    accelerator_type: Device
    execution_provider: str
    vender: str = None
    version: str = None
    memory: int = None
    num_cores: int = None


class AcceleratorLookup:
    EXECUTION_PROVIDERS = {
        "cpu": ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
        "gpu": [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ],
        "npu": ["QNNExecutionProvider", "CPUExecutionProvider"],
    }

    @staticmethod
    def get_execution_providers_for_device(device: Device):
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        return AcceleratorLookup.get_execution_providers_for_device_by_available_providers(device, available_providers)

    @staticmethod
    def get_execution_providers_for_device_by_available_providers(device: Device, available_providers):
        eps_per_device = AcceleratorLookup.EXECUTION_PROVIDERS.get(device)
        return AcceleratorLookup.get_execution_providers(eps_per_device, available_providers)

    @staticmethod
    def get_execution_providers(execution_providers, available_providers):
        eps = AcceleratorLookup.filter_execution_providers(execution_providers, available_providers)
        return eps or available_providers

    @staticmethod
    def filter_execution_providers(execution_providers, available_providers):
        if not execution_providers:
            return execution_providers

        assert isinstance(execution_providers, list)
        assert isinstance(available_providers, list)

        return [ep for ep in execution_providers if ep in available_providers]
