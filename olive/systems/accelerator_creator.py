# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import TYPE_CHECKING, List

from olive.hardware.accelerator import DEVICE_TO_EXECUTION_PROVIDERS, AcceleratorLookup, AcceleratorSpec, Device
from olive.systems.common import SystemType

if TYPE_CHECKING:
    from olive.systems.system_config import SystemConfig

logger = logging.getLogger(__name__)


class AcceleratorNormalizer:
    """Fill the device, execution providers and check the compatibility between the device and execution providers."""

    def __init__(
        self, system_config: "SystemConfig", skip_supported_eps_check: bool = True, is_ep_required: bool = True
    ) -> None:
        self.system_config = system_config
        self.skip_supported_eps_check = skip_supported_eps_check
        self.is_ep_required = is_ep_required
        self.system_supported_eps = None

    def normalize(self) -> "SystemConfig":
        if self.system_config.olive_managed_env:
            if not self.system_config.config.accelerators:
                raise ValueError("Managed environment requires accelerators to be specified.")

            for accelerator in self.system_config.config.accelerators:
                if not accelerator.execution_providers:
                    raise ValueError(
                        f"Managed environment requires execution providers to be specified for {accelerator.device}"
                    )
        else:
            if self.system_config.type in (SystemType.Local, SystemType.PythonEnvironment, SystemType.IsolatedORT):
                if self.is_ep_required:
                    target = self.system_config.create_system()
                    # TODO(myguo): Handle the ORT not installed scenario. In this case, the call will raise ImportError.
                    # and the system_supported_eps will be None.
                    self.system_supported_eps = target.get_supported_execution_providers()
                    # Remove the AzureMLExecutionProvider
                    if "AzureExecutionProvider" in self.system_supported_eps:
                        self.system_supported_eps.remove("AzureExecutionProvider")

                    assert self.system_supported_eps, "No supported execution providers found for the target system."

                    self._fill_accelerators()
                else:
                    self._fill_device()
            else:
                # for AzureML and Docker System
                if not self.system_config.config.accelerators:
                    raise ValueError("AzureML and Docker system requires accelerators to be specified.")
                for accelerator in self.system_config.config.accelerators:
                    if not accelerator.device or (not accelerator.execution_providers and self.is_ep_required):
                        raise ValueError(
                            "AzureML and Docker system requires device and execution providers to be specified "
                            "explicitly."
                        )

            if self.is_ep_required:
                self._check_execution_providers()

        return self.system_config

    def _fill_device(self):
        """Fill only the device in the system config accelerators and leave the execution providers None."""
        if not self.system_config.config.accelerators:
            self.system_config.config.accelerators = [{"device": "cpu"}]
        else:
            for accelerator in self.system_config.config.accelerators:
                if not accelerator.device:
                    accelerator.device = "cpu"
                    accelerator.execution_providers = None

    def _fill_accelerators(self):
        """Fill the accelerators including device and execution providers in the system config.

        * If the accelerators are not specified, fill the device and execution providers based on the installed ORT for
        local/python system.
        * If the device is specified but the execution providers are not, fill the execution providers based on the
        installed ORT for local/python system.
        * If the execution providers are specified but the device is not, fill the device based on the installed ORT.
        """
        if not self.system_config.config.accelerators:
            # User does not specify the accelerators.
            inferred_device = AcceleratorLookup.infer_single_device_from_execution_providers(self.system_supported_eps)
            # here the pydantic validate_assignment will initialize the accelerator instances
            self.system_config.config.accelerators = [
                {"device": inferred_device, "execution_providers": self.system_supported_eps}
            ]
            logger.info(
                "There is no any accelerator specified. Inferred accelerators: %s",
                self.system_config.config.accelerators,
            )
        else:
            for accelerator in self.system_config.config.accelerators:
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
                    execution_providers = AcceleratorLookup.get_execution_providers_for_device_by_available_providers(
                        accelerator.device.lower(), self.system_supported_eps
                    )
                    accelerator.execution_providers = execution_providers
                    filtered_eps = [ep for ep in self.system_supported_eps if ep not in execution_providers]
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

    def _check_execution_providers(self):
        """Check the execution providers are supported by the device and remove the unsupported ones.

        If the skip_supported_eps_check is True, the check will be skipped and the accelerators will be filtered against
        the device.
        """
        # check the execution providers are supported
        # TODO(myguo): should we cleanup the EPs if ep is not used?
        ep_not_supported = []
        for accelerator in self.system_config.config.accelerators:
            device = Device(accelerator.device.lower())
            eps_per_device = AcceleratorLookup.get_managed_supported_execution_providers(device)

            if self.system_config.olive_managed_env:
                available_eps = eps_per_device
            elif (
                self.system_config.type in (SystemType.Local, SystemType.PythonEnvironment, SystemType.IsolatedORT)
                and not self.skip_supported_eps_check
            ):
                # skip_supported_eps_check is False here
                # target is used so we need to check that the system supported eps are compatible with the accelerators
                available_eps = self.system_supported_eps
            else:
                # AzureML and Docker system: These are required to be specified by the user.
                # Local, PythonEnvironment, IsolatedORT: skip_supported_eps_check is True
                # the target is not used so no need to check the compatibility between the system supported eps and
                # the accelerators (available_eps == accelerator.execution_providers, the check will always pass)
                # Example scenario: to run optimization workflow for qnn-ep on x86 machine, the pass (onnxquantization)
                # needs to know qnn-ep is the target ep, but ort-qnn is not available on x86 machine.
                # we can still run the workflow using cpu ORT package as the target is not used for evaluation or
                # pass runs (= no inference sesion is created). The ort tools don't need the ep to be available.
                eps = AcceleratorLookup.filter_execution_providers(accelerator.execution_providers, eps_per_device)
                available_eps = eps or ["CPUExecutionProvider"]

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
            if not self.skip_supported_eps_check and not eps:
                raise ValueError(
                    f"None of the execution providers {accelerator.execution_providers} cannot be found in the target"
                    " system but at least one is required to run the workflow."
                )
            accelerator.execution_providers = eps or ["CPUExecutionProvider"]

        if ep_not_supported:
            logger.warning(
                "The following execution providers are not supported: '%s' by the device: '%s' and will be ignored. "
                "Please consider installing an onnxruntime build that contains the relevant execution providers. ",
                ",".join(ep_not_supported),
                ",".join([accelerator.device for accelerator in self.system_config.config.accelerators]),
            )


def create_accelerators(
    system_config: "SystemConfig", skip_supported_eps_check: bool = True, is_ep_required=True
) -> List[AcceleratorSpec]:
    normalizer = AcceleratorNormalizer(system_config, skip_supported_eps_check, is_ep_required)
    system_config = normalizer.normalize()

    device_to_eps = {
        accelerator.device: accelerator.execution_providers for accelerator in system_config.config.accelerators
    }
    logger.debug("Initial accelerators and execution providers: %s", device_to_eps)

    # Flatten the accelerators to list of AcceleratorSpec
    accelerator_specs: List[AcceleratorSpec] = []
    is_cpu_available = "cpu" in [accelerator.lower() for accelerator in device_to_eps]
    for accelerator in system_config.config.accelerators:
        device = Device(accelerator.device.lower())
        if accelerator.execution_providers:
            for ep in accelerator.execution_providers:
                if ep == "CPUExecutionProvider" and device != "cpu" and is_cpu_available:
                    logger.warning(
                        "Ignore the CPUExecutionProvider for non-cpu device since cpu accelerator is also present."
                    )
                else:
                    accelerator_specs.append(AcceleratorSpec(device, ep, memory=accelerator.memory))
        else:
            accelerator_specs.append(AcceleratorSpec(device, memory=accelerator.memory))

    assert accelerator_specs, (
        "No valid accelerator specified for target system. "
        "Please specify the accelerators in the target system or provide valid execution providers. "
        f"Given execution providers: {device_to_eps.values()}. "
        f"Current accelerators: {device_to_eps.keys()}."
        f"Supported execution providers: {DEVICE_TO_EXECUTION_PROVIDERS}."
    )
    logger.info("Running workflow on accelerator specs: %s", ",".join([str(spec) for spec in accelerator_specs]))
    return accelerator_specs
