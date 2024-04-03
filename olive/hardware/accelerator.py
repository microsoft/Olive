# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Union

from olive.hardware.constants import DEVICE_TO_EXECUTION_PROVIDERS

if TYPE_CHECKING:
    from olive.systems.system_config import SystemConfig


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
        eps_per_device = [*DEVICE_TO_EXECUTION_PROVIDERS.get(device), "CPUExecutionProvider"]
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


def normalize_accelerators(system_config: "SystemConfig", skip_supported_eps_check: bool = True) -> "SystemConfig":
    """Normalize the accelerators in the system config.

    * the accelerators is not specified, infer the device/ep based on the installed ORT in case of local/python system.
    * only device is specified, infer the execution providers based on the installed ORT in case of local/python system.
    * only EP is specified, infer the device based on the installed ORT in case of local/python system.
    * For AzureML and Docker system, the accelerators and execution providers must be specified.
    """
    from olive.systems.common import SystemType

    if system_config.olive_managed_env:
        if not system_config.config.accelerators:
            raise ValueError("Managed environment requires accelerators to be specified.")

        for accelerator in system_config.config.accelerators:
            if not accelerator.execution_providers:
                raise ValueError(
                    f"Managed environment requires execution providers to be specified for {accelerator.device}"
                )
    else:
        if system_config.type in (SystemType.Local, SystemType.PythonEnvironment, SystemType.IsolatedORT):
            target = system_config.create_system()
            # TODO(myguo): Handle the ORT not installed scenario. In this case, the call will raise ImportError.
            # and the system_supported_eps will be None.
            system_supported_eps = target.get_supported_execution_providers()
            # Remove the AzureMLExecutionProvider
            if "AzureExecutionProvider" in system_supported_eps:
                system_supported_eps.remove("AzureExecutionProvider")

            assert system_supported_eps, "No supported execution providers found for the target system."

            if not system_config.config.accelerators:
                # User does not specify the accelerators.
                inferred_device = AcceleratorLookup.infer_single_device_from_execution_providers(system_supported_eps)
                # here the pydantic validate_assignment will initialize the accelerator instances
                system_config.config.accelerators = [
                    {"device": inferred_device, "execution_providers": system_supported_eps}
                ]
                logger.info(
                    "There is no any accelerator specified. Inferred accelerators: %s",
                    system_config.config.accelerators,
                )
            else:
                for accelerator in system_config.config.accelerators:
                    if not accelerator.device:
                        # User does not specify the device but providing the execution providers
                        assert accelerator.execution_providers, "The execution providers are not specified."
                        inferred_device = AcceleratorLookup.infer_single_device_from_execution_providers(
                            accelerator.execution_providers
                        )
                        logger.info("the accelerator device is not specified. Inferred device: %s.", inferred_device)
                        accelerator.device = inferred_device
                    elif not accelerator.execution_providers:
                        # User specify the device but missing the execution providers
                        execution_providers = (
                            AcceleratorLookup.get_execution_providers_for_device_by_available_providers(
                                accelerator.device.lower(), system_supported_eps
                            )
                        )
                        accelerator.execution_providers = execution_providers
                        filtered_eps = [ep for ep in system_supported_eps if ep not in execution_providers]
                        if filtered_eps:
                            logger.warning(
                                "The following execution providers are filtered: %s. "
                                "Please raise issue in Olive site since it might be a bug. ",
                                ",".join(filtered_eps),
                            )

                        logger.info(
                            "The accelerator execution providers is not specified for %s. Use the inferred ones. %s",
                            accelerator.device,
                            accelerator.execution_providers,
                        )
                    else:
                        logger.debug("The accelerator device and execution providers are specified, skipping deduce.")
        else:
            # for AzureML and Docker System
            if not system_config.config.accelerators:
                raise ValueError("AzureML and Docker system requires accelerators to be specified.")
            for accelerator in system_config.config.accelerators:
                if not accelerator.device or not accelerator.execution_providers:
                    raise ValueError(
                        "AzureML and Docker system requires device and execution providers to be specified explicitly."
                    )

    ep_not_supported = []
    for accelerator in system_config.config.accelerators:
        device = Device(accelerator.device.lower())

        if system_config.olive_managed_env:
            available_eps = AcceleratorLookup.get_managed_supported_execution_providers(device)
        elif (
            system_config.type in (SystemType.Local, SystemType.PythonEnvironment, SystemType.IsolatedORT)
            and not skip_supported_eps_check
        ):
            # don't need to check the supported execution providers if there is no evaluation
            # target is only used for evaluation
            available_eps = system_supported_eps
        else:
            available_eps = accelerator.execution_providers

        supported_eps = AcceleratorLookup.get_execution_providers_for_device_by_available_providers(
            device, available_eps
        )
        logger.debug("Supported execution providers for device %s: %s", device, supported_eps)

        eps = []
        for ep in accelerator.execution_providers:
            if ep not in supported_eps:
                ep_not_supported.append(ep)
            else:
                eps.append(ep)

        # remove the unsupported execution providers
        accelerator.execution_providers = eps

    if ep_not_supported:
        logger.warning(
            "The following execution providers are not supported: %s. "
            "Please consider installing an onnxruntime build that contains the relevant execution providers. ",
            ",".join(ep_not_supported),
        )
    return system_config


def create_accelerators(system_config: "SystemConfig", skip_supported_eps_check: bool = True) -> List[AcceleratorSpec]:
    system_config = normalize_accelerators(system_config, skip_supported_eps_check)

    device_to_eps = {}
    for accelerator in system_config.config.accelerators:
        device_to_eps[accelerator.device] = accelerator.execution_providers
    logger.debug("Initial accelerators and execution providers: %s", device_to_eps)

    # Flatten the accelerators to list of AcceleratorSpec
    accelerator_specs: List[AcceleratorSpec] = []
    is_cpu_available = "cpu" in [accelerator.lower() for accelerator in device_to_eps]
    for accelerator in system_config.config.accelerators:
        device = Device(accelerator.device.lower())
        for ep in accelerator.execution_providers:
            if ep == "CPUExecutionProvider" and device != "cpu" and is_cpu_available:
                logger.warning(
                    "Ignore the CPUExecutionProvider for non-cpu device since cpu accelerator is also present."
                )
            else:
                accelerator_specs.append(AcceleratorSpec(device, ep))

    assert accelerator_specs, (
        "No valid accelerator specified for target system. "
        "Please specify the accelerators in the target system or provide valid execution providers. "
        f"Given execution providers: {device_to_eps.values()}. "
        f"Current accelerators: {device_to_eps.keys()}."
        f"Supported execution providers: {DEVICE_TO_EXECUTION_PROVIDERS}."
    )
    logger.info("Running workflow on accelerator specs: %s", ",".join([str(spec) for spec in accelerator_specs]))
    return accelerator_specs
