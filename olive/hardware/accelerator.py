# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Union

logger = logging.getLogger(__name__)


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    INTEL_MYRIAD = "intel_myriad"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, eq=True)
class AcceleratorSpec:
    """Accelerator specification is the concept of a hardware device that be used to optimize or evaluate a model."""

    accelerator_type: Union[str, Device]
    execution_provider: str
    vender: str = None
    version: str = None
    memory: int = None
    num_cores: int = None

    def __str__(self) -> str:
        # remove the suffix "ExecutionProvider", len("ExecutionProvider") = 17
        ep = self.execution_provider[:-17] or self.execution_provider
        return f"{str(self.accelerator_type).lower()}-{ep.lower()}"

    def to_json(self):
        return {
            "accelerator_type": str(self.accelerator_type),
            "execution_provider": self.execution_provider,
        }


DEFAULT_CPU_ACCELERATOR = AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="CPUExecutionProvider")
DEFAULT_GPU_CUDA_ACCELERATOR = AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider")
DEFAULT_GPU_TRT_ACCELERATOR = AcceleratorSpec(
    accelerator_type=Device.GPU, execution_provider="TensorrtExecutionProvider"
)


class AcceleratorLookup:
    EXECUTION_PROVIDERS: ClassVar[dict] = {
        "cpu": ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
        "gpu": [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
            "OpenVINOExecutionProvider",
        ],
        "npu": ["QNNExecutionProvider", "CPUExecutionProvider"],
    }

    @staticmethod
    def get_managed_supported_execution_providers(device: Device):
        return AcceleratorLookup.EXECUTION_PROVIDERS.get(device)

    @staticmethod
    def get_execution_providers_for_device(device: Device):
        import onnxruntime

        return AcceleratorLookup.get_execution_providers_for_device_by_available_providers(
            device, onnxruntime.get_available_providers()
        )

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

    @staticmethod
    def infer_accelerators_from_execution_provider(execution_provider: List[str]):
        """Infer the device from the execution provider name.

        If all the execution provider is uniquely mapped to a device, return the device list.
        Otherwise, return None.
        For example:
            execution_provider = ["CPUExecutionProvider", "CUDAExecutionProvider"]
            return None (CPUExecutionProvider is mapped to CPU and GPU, Olive cannot infer the device)
            execution_provider = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
            return ["gpu"]
        """
        if not execution_provider:
            return None

        is_unique_inferring = True
        accelerators = []
        for idx, ep in enumerate(execution_provider):
            accelerators.append([])
            for accelerator, eps in AcceleratorLookup.EXECUTION_PROVIDERS.items():
                if ep in eps:
                    accelerators[idx].append(accelerator)
                    if len(accelerators[idx]) > 1:
                        logger.warning(
                            f"Execution provider {ep} is mapped to multiple accelerators {accelerators[idx]}. "
                            "Olive cannot infer the device which may cause unexpected behavior. "
                            "Please specify the accelerator in the accelerator configs"
                        )
                        is_unique_inferring = False

        if is_unique_inferring:
            return list({accelerator[0] for accelerator in accelerators})
        return None
