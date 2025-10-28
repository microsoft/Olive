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
    def metrics(self) -> Optional[dict[str, Any]]:
        """Get the model metrics dictionary."""
        if self._model_node.metrics:
            return self._model_node.metrics.to_json()
        return None

    @property
    def metrics_value(self) -> Optional[dict[str, Any]]:
        """Get the model metrics value."""
        if self._model_node.metrics:
            return self._model_node.metrics.value.to_json()
        return None

    @property
    def model_path(self) -> Optional[str]:
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
    def model_type(self) -> Optional[str]:
        """Get the model type."""
        return self._olive_model_config.get("type", None)

    @property
    def model_config(self) -> dict[str, Any]:
        """Get the model configuration."""
        return self._model_config

    def from_device(self) -> str:
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


class WorkflowOutput(BaseRankedOutput):
    """Organized output from an Olive workflow with device-specific results.

    Provides a structured way to access and compare optimization results across
    different hardware devices and execution providers.
    """

    def __init__(self, accelerator_spec: AcceleratorSpec, footprint: Footprint):
        """Initialize a WorkflowOutput instance.

        Args:
            accelerator_spec: The accelerator specification used in the workflow
            footprint: The Footprint object containing all workflow results

        """
        self.accelerator_spec = accelerator_spec
        self.footprint = footprint
        self._input_model_id = footprint.input_model_id
        self.output_models = (
            [
                ModelOutput(
                    accelerator_spec.accelerator_type,
                    accelerator_spec.execution_provider,
                    footprint.nodes[model_id],
                )
                for model_id in self.footprint.output_model_ids
            ]
            if self.footprint.output_model_ids
            else []
        )

        self._objective_dict = footprint.objective_dict
        self._best_candidate = self._get_best_candidate()

    def from_device(self) -> str:
        """Get the device used for this model."""
        return str(self.accelerator_spec.accelerator_type)

    def from_execution_provider(self) -> str:
        """Get the execution provider used for this model."""
        return str(self.accelerator_spec.execution_provider)

    def get_input_model_metrics(self) -> Optional[dict[str, Any]]:
        """Get the metrics for the input model."""
        input_node = self.footprint.nodes[self._input_model_id]
        return input_node.metrics.value.to_json() if input_node.metrics else None

    def has_output_model(self) -> bool:
        """Check if any model outputs are available.

        Returns:
            True if there are any model outputs, False otherwise

        """
        return len(self.output_models) > 0

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
        return self._sort_by_metrics(self.output_models, self._objective_dict)

    def get_best_candidate(self) -> Optional[ModelOutput]:
        """Get the best candidate across all devices based on metrics."""
        return self._best_candidate

    def trace_back_run_history(self, model_id: str) -> dict[str, Any]:
        """Trace back the run history for a specific model."""
        model_output = self.get_output_model_by_id(model_id)
        if not model_output:
            return {}
        return self.footprint.trace_back_run_history(model_id)

    def _get_best_candidate(self) -> Optional[ModelOutput]:
        """Get the best candidate across all devices based on metrics.

        Returns:
            The best ModelOutput across all devices or None if no outputs available

        """
        sorted_outputs = self._sort_by_metrics(self.output_models, self._objective_dict)
        return sorted_outputs[0] if sorted_outputs else None
