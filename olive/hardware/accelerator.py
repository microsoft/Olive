# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from olive.hardware.constants import DEVICE_TO_EXECUTION_PROVIDERS

logger = logging.getLogger(__name__)


class Device(str, Enum):
    CPU = "cpu"
    CPU_SPR = "cpu_spr"
    GPU = "gpu"
    NPU = "npu"
    VPU = "vpu"
    INTEL_MYRIAD = "intel_myriad"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, eq=True)
class AcceleratorSpec:
    """Accelerator specification is the concept of a hardware device that be used to optimize or evaluate a model."""

    accelerator_type: Union[str, Device]
    execution_provider: Optional[str] = None
    vender: str = None
    version: str = None
    memory: int = None
    num_cores: int = None

    def __str__(self) -> str:
        if self.execution_provider:
            # remove the suffix "ExecutionProvider", len("ExecutionProvider") = 17
            ep = self.execution_provider[:-17]
            return f"{str(self.accelerator_type).lower()}-{ep.lower()}"
        else:
            return str(self.accelerator_type).lower()

    def to_json(self):
        json_data = {"accelerator_type": str(self.accelerator_type)}
        if self.execution_provider:
            json_data["execution_provider"] = self.execution_provider

        return json_data


DEFAULT_CPU_ACCELERATOR = AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="CPUExecutionProvider")
DEFAULT_GPU_CUDA_ACCELERATOR = AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider")
DEFAULT_GPU_TRT_ACCELERATOR = AcceleratorSpec(
    accelerator_type=Device.GPU, execution_provider="TensorrtExecutionProvider"
)


class AcceleratorLookup:
    @staticmethod
    def get_managed_supported_execution_providers(device: Device):
        return [*DEVICE_TO_EXECUTION_PROVIDERS.get(device), "CPUExecutionProvider"]

    @staticmethod
    def get_execution_providers_for_device(device: Device):
        import onnxruntime

        return AcceleratorLookup.get_execution_providers_for_device_by_available_providers(
            device, onnxruntime.get_available_providers()
        )

    @staticmethod
    def get_execution_providers_for_device_by_available_providers(device: Device, available_providers):
        eps_per_device = AcceleratorLookup.get_managed_supported_execution_providers(device)
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

        return [ep for ep in available_providers if ep in execution_providers]

    @staticmethod
    def infer_devices_from_execution_providers(execution_providers: List[str]):
        """Infer the device from the execution provider name.

        If all the execution provider is uniquely mapped to a device, return the device list.
        Otherwise, return None.
        Please note that the CPUExecutionProvider is skipped for device infer. And only other ORT EPs are considered.
        For example:
            execution_provider = ["CPUExecutionProvider", "CUDAExecutionProvider"]
            return ["gpu"]
            execution_provider = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
            return ["gpu"]
        """
        if not execution_providers:
            return None

        ep_to_devices = {}
        for ep in execution_providers:
            if ep == "CPUExecutionProvider":
                # cannot infer device for CPUExecutionProvider since all ORT EP supports CPU
                continue

            inferered_devices = []
            for device, eps in DEVICE_TO_EXECUTION_PROVIDERS.items():
                if ep in eps:
                    inferered_devices.append(device)
            if inferered_devices:
                ep_to_devices[ep] = inferered_devices
            else:
                ep_to_devices[ep] = None

        mapped_devices = []
        for ep, inferred_device in ep_to_devices.items():
            if inferred_device is None:
                logger.warning(
                    "Execution provider %s is not able to be mapped to any device. "
                    "Olive cannot infer the device which may cause unexpected behavior. "
                    "Please specify the accelerator in the accelerator configs",
                    ep,
                )
                return None
            elif len(inferred_device) > 1:
                logger.warning(
                    "Execution provider %s is mapped to multiple devices %s. "
                    "Olive cannot infer the device which may cause unexpected behavior. "
                    "Please specify the accelerator in the accelerator configs",
                    ep,
                    inferred_device,
                )
                return None
            else:
                if inferred_device[0] not in mapped_devices:
                    mapped_devices.append(inferred_device[0])
        return mapped_devices if mapped_devices else None

    @staticmethod
    def infer_single_device_from_execution_providers(execution_providers: List[str]) -> str:
        if not execution_providers:
            return None

        if execution_providers == ["CPUExecutionProvider"]:
            inferred_devices = ["cpu"]
        else:
            inferred_devices = AcceleratorLookup.infer_devices_from_execution_providers(execution_providers)
            assert inferred_devices, (
                f"Cannot infer the devices from the execution providers {execution_providers}."
                " Please specify the device in the accelerator configs."
            )
            assert len(inferred_devices) == 1, (
                f"Cannot infer the devices from the execution providers {execution_providers}. "
                f"Multiple devices are inferred: {inferred_devices}."
                " Please specify the device in the accelerator configs."
            )

        return inferred_devices[0]
