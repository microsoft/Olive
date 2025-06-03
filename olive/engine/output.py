# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Optional, Union

from olive.engine.footprint import Footprint, FootprintNode
from olive.hardware.accelerator import AcceleratorSpec, Device, ExecutionProvider

logger = logging.getLogger(__name__)


class ModelOutput:
    """Represents the output of a specific model with its metrics and configuration.

    This class stores the model_path, metrics, model_id, and hardware information for a
    specific model output from the Olive optimization workflow.
    """

    def __init__(
        self, device: Union[Device, str], execution_provider: Union[ExecutionProvider, str], node: FootprintNode
    ):
        """Initialize a ModelOutput instance.

        Args:
            device: The hardware device used for this model
            execution_provider: The execution provider used (e.g., 'CPUExecutionProvider')
            node: The FootprintNode object containing model information and metrics

        """
        if not node:
            raise ValueError("FootprintNode cannot be None.")

        self._model_node = node
        self._device = device
        self._execution_provider = execution_provider
        self._init_with_model_config(self._model_node.model_config)

    def _init_with_model_config(self, model_config: dict[str, Any]):
        """Initialize the model config."""
        self._olive_model_config = model_config
        self._model_config = self._get_model_config()
        self._model_path = self._model_config.get("model_path", None)

    @property
    def metrics(self) -> dict[str, Any]:
        """Get the model metrics dictionary."""
        if self._model_node.metrics:
            return self._model_node.metrics.to_json()
        return None

    @property
    def metrics_value(self) -> dict[str, Any]:
        """Get the model metrics value."""
        if self._model_node.metrics:
            return self._model_node.metrics.value.to_json()
        return None

    @property
    def model_path(self) -> str:
        """Get the path to the model file."""
        return self._model_path

    @property
    def model_id(self) -> str:
        """Get the model id."""
        return self._model_node.model_id

    @property
    def olive_model_config(self) -> dict[str, Any]:
        """Get the Olive model configuration."""
        return self._olive_model_config

    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self._olive_model_config.get("type", None)

    @property
    def model_config(self) -> dict[str, Any]:
        """Get the model configuration."""
        return self._model_config

    def from_device(self) -> Device:
        """Get the device used for this model."""
        return str(self._device)

    def from_execution_provider(self) -> str:
        """Get the execution provider used for this model."""
        return str(self._execution_provider)

    def from_pass(self) -> str:
        """Get the pass used for this model."""
        return self._model_node.from_pass

    def get_parent_model_id(self) -> str:
        """Get the parent model ID."""
        return self._model_node.parent_model_id

    def use_ort_extension(self) -> bool:
        """Check if the model uses the ORT extension."""
        return self._model_config.get("use_ort_extension", False)

    def get_inference_config(self) -> dict[str, Any]:
        """Get the model inference configuration."""
        return self._model_config.get("inference_settings") or {}

    def _update_with_model_config(self, model_config: dict[str, Any]):
        """Update the model config."""
        self._init_with_model_config(model_config)

    def _get_model_config(self) -> dict[str, Any]:
        """Get the model config."""
        if not self._olive_model_config:
            return {}
        return self._olive_model_config.get("config", {})


class BaseRankedOutput:
    """Base class for outputs that can be ranked based on metrics.

    Provides common functionality for sorting and ranking model outputs
    based on their performance metrics.
    """

    def _sort_by_metrics(self, items: list[ModelOutput], objective_dict: dict[str, Any]) -> list[ModelOutput]:
        """Sort items based on metrics according to objective direction.

        Args:
            items: List of ModelOutput objects to sort
            objective_dict: Dictionary of objectives with their priorities

        Returns:
            List of ModelOutput objects sorted by their metrics

        """
        if not items:
            return []

        if not objective_dict:
            return items

        try:
            return sorted(
                items,
                key=lambda x: tuple(
                    x.metrics["value"][metric]["value"]
                    if x.metrics["cmp_direction"][metric] == 1
                    else -x.metrics["value"][metric]["value"]
                    for metric in objective_dict
                ),
                reverse=True,
            )
        except (KeyError, TypeError) as e:
            # Handle case where metrics might be missing or incorrectly formatted
            logger.warning("Warning: Error during metric sorting: %s", e)
            return items


class DeviceOutput(BaseRankedOutput):
    """Groups model outputs for a specific accelerator device.

    Contains model outputs for different execution providers on the same
    hardware accelerator type, with methods to retrieve and compare them.
    """

    def __init__(
        self,
        device: Union[Device, str],
        ep_footprint_map: dict[str, Footprint],
        objective_dict: dict[str, Any],
    ):
        """Initialize a DeviceOutput instance.

        Args:
            device: The hardware device for these outputs
            ep_footprint_map: Dictionary mapping execution providers to their Footprint
            objective_dict: Dictionary of objectives with their priorities

        """
        self._device = device
        self._ep_model_map: dict[str, ModelOutput] = {}
        self._objective_dict = objective_dict

        for ep, footprint in ep_footprint_map.items():
            if ep not in self._ep_model_map:
                self._ep_model_map[ep] = []
            if not footprint.check_empty_nodes():
                for node in footprint.nodes.values():
                    self._ep_model_map[ep].append(ModelOutput(device, ep, node))

        self._best_candidate = self._get_best_candidate()

    @property
    def device(self) -> str:
        """Get the device type for this accelerator output."""
        return str(self._device)

    def has_output_model(self) -> bool:
        """Check if any model outputs are available."""
        return any(self._ep_model_map.values())

    def get_output_models(self) -> list[ModelOutput]:
        """Get all model outputs for this accelerator output."""
        return [model for models in self._ep_model_map.values() for model in models]

    def get_best_candidate(self) -> Optional[ModelOutput]:
        """Get the best model output for this accelerator based on metrics."""
        return self._best_candidate

    def get_best_candidate_by_execution_provider(
        self, execution_provider: Union[ExecutionProvider, str]
    ) -> Optional[ModelOutput]:
        """Get the best model output for this accelerator based on metrics."""
        model_outputs = self._ep_model_map.get(execution_provider)
        return self._sort_by_metrics(model_outputs, self._objective_dict)[0] if model_outputs else None

    def __getitem__(self, key: str) -> list[ModelOutput]:
        """Get model outputs by execution provider name.

        Args:
            key: The execution provider to retrieve

        Returns:
            The ModelOutput for the specified execution provider or None if not found

        """
        return self._ep_model_map.get(key)

    def _get_best_candidate(self) -> Optional[ModelOutput]:
        """Find the best model output for this accelerator based on metrics.

        Returns:
            The best ModelOutput or None if no outputs are available

        """
        if not self._ep_model_map:
            return None

        model_outputs = []
        for outputs in self._ep_model_map.values():
            model_outputs.extend(outputs)

        sorted_outputs = self._sort_by_metrics(model_outputs, self._objective_dict)
        return sorted_outputs[0] if sorted_outputs else None


class WorkflowOutput(BaseRankedOutput):
    """Organized output from an Olive workflow with device-specific results.

    Provides a structured way to access and compare optimization results across
    different hardware devices and execution providers.
    """

    def __init__(
        self,
        output_acc_footprint_map: dict[AcceleratorSpec, Footprint],
        all_footprints: dict[AcceleratorSpec, Footprint],
    ):
        """Initialize a WorkflowOutput instance.

        Args:
            output_acc_footprint_map: Dictionary mapping accelerator specs to their Footprint
            all_footprints: All footprints in the workflow

        """
        self._device_outputs: dict[str, DeviceOutput] = {}
        self._all_footprints = all_footprints
        self._input_model_id = all_footprints[
            AcceleratorSpec(Device.CPU, ExecutionProvider.CPUExecutionProvider)
        ].input_model_id

        if not output_acc_footprint_map:
            self._objective_dict = {}
            return

        # Store objective dict for ranking
        no_search_footprint = next(iter(output_acc_footprint_map.values()))
        self._objective_dict = no_search_footprint.objective_dict if output_acc_footprint_map else {}

        # Group outputs by device type
        for acc_spec, footprint in output_acc_footprint_map.items():
            device_type = str(acc_spec.accelerator_type).lower()
            if device_type not in self._device_outputs:
                self._device_outputs[device_type] = {}
            self._device_outputs[device_type][acc_spec.execution_provider] = footprint

        # Create DeviceOutput objects for each device type
        for device_type in list(self._device_outputs.keys()):
            self._device_outputs[device_type] = DeviceOutput(
                device_type, self._device_outputs[device_type], self._objective_dict
            )

        # Set device attributes for easy access
        for device in Device:
            setattr(self, device.value, self._device_outputs.get(device.value))

        self._best_candidate = self._get_best_candidate()

    def _get_device_output_case_insensitive(self, name: str) -> Optional[DeviceOutput]:
        if name in self._device_outputs:
            return self._device_outputs[name]

        for device_name, device_output in self._device_outputs.items():
            if device_name.lower() == name.lower():
                return device_output

        return None

    def __getattr__(self, name):
        """Handle case-insensitive device attribute access (e.g., .CPU or .cpu)."""
        return self._get_device_output_case_insensitive(name)

    def __getitem__(self, key: str) -> Optional[DeviceOutput]:
        """Get accelerator output by device type.

        Args:
            key: The device type to retrieve (case-insensitive)

        Returns:
            The DeviceOutput for the specified device or None if not found

        """
        return self._get_device_output_case_insensitive(key)

    def get_input_model_metrics(
        self, device: Union[Device, str] = None, execution_provider: Union[ExecutionProvider, str] = None
    ) -> dict[str, Any]:
        """Get the metrics for the input model."""
        device = device or Device.CPU
        execution_provider = execution_provider or ExecutionProvider.CPUExecutionProvider
        input_node = self._all_footprints[AcceleratorSpec(device, execution_provider)].nodes[self._input_model_id]
        return input_node.metrics.value.to_json() if input_node.metrics else None

    def get_available_devices(self) -> list[str]:
        """Get a list of available device types in this workflow output.

        Returns:
            List of device type strings

        """
        return list(self._device_outputs.keys())

    def has_output_model(self) -> bool:
        """Check if any model outputs are available.

        Returns:
            True if there are any model outputs, False otherwise

        """
        return any(self.get_output_models())

    def get_output_models_by_device(self, device: Union[Device, str]) -> Optional[list[ModelOutput]]:
        """Get the output model for a specific device.

        Args:
            device: The device to retrieve the output model for

        """
        device_type = str(device).lower()
        if device_type not in self._device_outputs:
            return None
        return self._device_outputs.get(device_type).get_output_models()

    def get_output_model_by_id(self, model_id: str) -> Optional[ModelOutput]:
        """Get the output model by its ID.

        Args:
            model_id: The ID of the model to retrieve

        """
        for model in self.get_output_models():
            if model.model_id == model_id:
                return model
        return None

    def get_output_models(self) -> list[ModelOutput]:
        """Get all model outputs from the workflow sorted by metrics.

        Returns:
            List of ModelOutput objects

        """
        output_models = [
            model for acc_output in self._device_outputs.values() for model in acc_output.get_output_models()
        ]
        return self._sort_by_metrics(output_models, self._objective_dict)

    def get_best_candidate_by_device(self, device: Union[Device, str]) -> Optional[ModelOutput]:
        """Get the best candidate for a specific device.

        Args:
            device: The device to retrieve the best candidate for

        """
        device_type = str(device).lower()
        return self._device_outputs.get(device_type).get_best_candidate()

    def get_best_candidate(self) -> Optional[ModelOutput]:
        """Get the best candidate across all devices based on metrics."""
        return self._best_candidate

    def trace_back_run_history(self, model_id: str) -> dict[str, Any]:
        """Trace back the run history for a specific model."""
        model_output = self.get_output_model_by_id(model_id)
        if not model_output:
            return {}
        device = model_output.from_device()
        execution_provider = model_output.from_execution_provider()
        return self._all_footprints[AcceleratorSpec(device, execution_provider)].trace_back_run_history(model_id)

    def _get_best_candidate(self) -> Optional[ModelOutput]:
        """Get the best candidate across all devices based on metrics.

        Returns:
            The best ModelOutput across all devices or None if no outputs available

        """
        if not self._device_outputs:
            return None

        model_outputs = [
            acc_output.get_best_candidate()
            for acc_output in self._device_outputs.values()
            if acc_output and acc_output.get_best_candidate()
        ]

        if not model_outputs:
            return None

        sorted_outputs = self._sort_by_metrics(model_outputs, self._objective_dict)
        return sorted_outputs[0] if sorted_outputs else None
