# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, List, Union

if TYPE_CHECKING:
    from olive.systems.olive_system import OliveSystem


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
    EXECUTION_PROVIDERS: ClassVar[dict] = {
        "cpu": ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
        "gpu": [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
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

        return [ep for ep in available_providers if ep in execution_providers]

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


def create_accelerators(target: "OliveSystem", execution_providers):
    from olive.systems.common import SystemType

    if not execution_providers:
        if target.olive_managed_env:
            raise ValueError("Managed environment requires execution providers to be specified.")
        elif target.system_type == SystemType.AzureML:
            # verify the AzureML system have specified the execution providers
            # Please note we could not use isinstance(target, AzureMLSystem) since it would import AzureML packages.
            raise ValueError("AzureMLSystem requires execution providers to be specified.")
        elif target.system_type in (SystemType.Local, SystemType.PythonEnvironment):
            execution_providers = target.get_supported_execution_providers()
        elif target.system_type == SystemType.Docker:
            # for docker system we default use CPUExecutionProvider
            execution_providers = ["CPUExecutionProvider"]
    logger.debug(f"Initial execution providers: {execution_providers}")

    accelerators: List[str] = target.accelerators
    if accelerators is None:
        inferred_accelerators = AcceleratorLookup.infer_accelerators_from_execution_provider(execution_providers)
        if not inferred_accelerators:
            logger.warning("Cannot infer the accelerators from the target system. Use CPU as default.")
            accelerators = ["CPU"]
        else:
            logger.debug(
                "User inferred accelerators %s from given execution providers %s.", accelerators, execution_providers
            )
            accelerators = inferred_accelerators
    logger.debug(f"Initial accelerators: {accelerators}")

    ep_to_process = set(execution_providers)
    # Flatten the accelerators to list of AcceleratorSpec
    accelerator_specs: List[AcceleratorSpec] = []
    is_cpu_available = "cpu" in [accelerator.lower() for accelerator in accelerators]
    for accelerator in accelerators:
        device = Device(accelerator.lower())
        if target.olive_managed_env:
            available_eps = AcceleratorLookup.get_managed_supported_execution_providers(device)
        elif target.system_type in (SystemType.Local, SystemType.PythonEnvironment):
            available_eps = target.get_supported_execution_providers()
        elif target.system_type == SystemType.Docker:
            # TODO(myguo): do we need allow docker system support other execution providers?
            available_eps = ["CPUExecutionProvider"]
        else:
            available_eps = execution_providers

        supported_eps = AcceleratorLookup.get_execution_providers_for_device_by_available_providers(
            device, available_eps
        )
        logger.debug(f"Supported execution providers for device {device}: {supported_eps}")
        for ep in ep_to_process.copy():
            if ep == "CPUExecutionProvider" and device != "cpu" and is_cpu_available:
                logger.info("Ignore the CPUExecutionProvider for non-cpu device since cpu accelerator is also present.")
            elif ep in supported_eps:
                accelerator_specs.append(AcceleratorSpec(device, ep))
                ep_to_process.remove(ep)

    assert accelerator_specs, (
        "No valid accelerator specified for target system. "
        "Please specify the accelerators in the target system or provide valid execution providers. "
        f"Given execution providers: {execution_providers}. "
        f"Current accelerators: {accelerators}."
        f"Supported execution providers: {AcceleratorLookup.EXECUTION_PROVIDERS}."
    )
    logger.info(f"Running workflow on accelerator specs: {','.join([str(spec) for spec in accelerator_specs])}")
    if ep_to_process:
        logger.warning(
            f"The following execution provider is not supported: {','.join(ep_to_process)}. "
            "Please consider installing an onnxruntime build that contains the relevant execution providers. "
        )

    return accelerator_specs
