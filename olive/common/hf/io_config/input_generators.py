# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from olive.common.hf.io_config.io_resolver import (
    get_default_shapes,
    get_diffusers_component_config,
    get_task_template,
    resolve_alias,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class DummyInputGenerator(ABC):
    """Generate dummy inputs for ONNX export."""

    supported_input_names = ()

    def supports_input(self, input_name: str) -> bool:
        """Check whether the generator supports the requested input."""
        return any(input_name.startswith(name) for name in self.supported_input_names)

    @abstractmethod
    def generate(self, input_name: str):
        """Generate the dummy input tensor."""
        raise NotImplementedError

    @staticmethod
    def zeros_int(shape: list[int]):
        """Generate a zero int64 tensor."""
        return torch.zeros(shape, dtype=torch.int64)

    @staticmethod
    def zeros_float(shape: list[int]):
        """Generate a zero float32 tensor."""
        return torch.zeros(shape, dtype=torch.float32)

    @staticmethod
    def ones_int(shape: list[int]):
        """Generate an int64 tensor of ones (for masks)."""
        return torch.ones(shape, dtype=torch.int64)

    @staticmethod
    def _to_int(value) -> int | None:
        """Convert a config value to int, handling list/tuple types."""
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return int(value[0])
        return None


# ============================================================================
# Unified Diffusers Dummy Input Generator
# ============================================================================


class DiffusersDummyInputGenerator(DummyInputGenerator):
    """Dummy input generator for diffusers components.

    Reads input specifications from diffusers.yaml.
    """

    def __init__(self, component_name: str, config: PretrainedConfig):
        self.component_name = component_name
        self.config = config

        self.component_spec = get_diffusers_component_config(component_name)
        if self.component_spec is None:
            raise ValueError(f"Unknown diffusers component: {component_name}")

        self.defaults = get_default_shapes()

        self._supported_inputs = set()
        for section in ["inputs", "sdxl_inputs", "optional_inputs"]:
            if section in self.component_spec:
                self._supported_inputs.update(self.component_spec[section].keys())

    @property
    def supported_input_names(self):
        return tuple(self._supported_inputs)

    def _get_dim_value(self, dim_name: str) -> int:
        """Get the actual value for a dimension name."""
        # Try config with aliases
        value = self._to_int(resolve_alias(self.config, dim_name))
        if value is not None:
            return value

        # Handle computed dimensions
        if dim_name == "height_latent":
            return self._get_dim_value("height") // 8
        if dim_name == "width_latent":
            return self._get_dim_value("width") // 8
        if dim_name == "packed_height_width":
            return (self._get_dim_value("height") * self._get_dim_value("width")) // 4

        # Fall back to defaults
        if dim_name in self.defaults:
            return self.defaults[dim_name]

        raise ValueError(f"Cannot resolve dimension: {dim_name}")

    def _resolve_shape(self, shape_template: list[str]) -> list[int]:
        """Resolve a shape template to actual dimensions."""
        return [dim if isinstance(dim, int) else self._get_dim_value(dim) for dim in shape_template]

    def _get_input_spec(self, input_name: str) -> dict | None:
        """Get the specification for an input."""
        for section in ["inputs", "sdxl_inputs", "optional_inputs"]:
            if section in self.component_spec and input_name in self.component_spec[section]:
                return self.component_spec[section][input_name]
        return None

    def generate(self, input_name: str):
        """Generate a dummy input tensor."""
        spec = self._get_input_spec(input_name)
        if spec is None:
            raise ValueError(f"Unsupported input name: {input_name}")

        # Check condition for optional inputs
        if "condition" in spec and not getattr(self.config, spec["condition"], False):
            return None

        shape_template = spec["shape"]
        shape = self._resolve_shape(shape_template)
        dtype = spec.get("dtype", "float")

        if dtype in ["int64", "int32"]:
            if "mask" in input_name:
                return self.ones_int(shape)
            return self.zeros_int(shape)
        return self.zeros_float(shape)


def generate_diffusers_dummy_inputs(
    component_name: str,
    config: PretrainedConfig,
) -> dict[str, Any]:
    """Create all dummy inputs for a diffusers component.

    Args:
        component_name: Name of the diffusers component (e.g., "unet", "vae_encoder")
        config: The component's config object

    Returns:
        Dict of input_name -> tensor

    """
    generator = DiffusersDummyInputGenerator(component_name, config)

    component_spec = get_diffusers_component_config(component_name)
    if component_spec is None:
        raise ValueError(f"Unknown diffusers component: {component_name}")

    dummy_inputs = {}

    # Generate required inputs
    for input_name in component_spec.get("inputs", {}):
        dummy_inputs[input_name] = generator.generate(input_name)

    # Generate optional inputs
    for input_name, spec in component_spec.get("optional_inputs", {}).items():
        if "condition" in spec and getattr(config, spec["condition"], False):
            result = generator.generate(input_name)
            if result is not None:
                dummy_inputs[input_name] = result
        else:
            try:
                result = generator.generate(input_name)
                if result is not None:
                    dummy_inputs[input_name] = result
            except ValueError:
                # Optional input not supported by this config
                pass

    return dummy_inputs


# ============================================================================
# Unified Task Dummy Input Generator
# ============================================================================


class TaskDummyInputGenerator(DummyInputGenerator):
    """Dummy input generator for HuggingFace tasks.

    Reads input specifications from tasks.yaml.
    """

    def __init__(self, task: str, config: PretrainedConfig, use_past_in_inputs: bool = False):
        self.task = task
        self.config = config
        self.use_past_in_inputs = use_past_in_inputs

        self.task_spec = get_task_template(task)
        if self.task_spec is None:
            raise ValueError(f"Unknown task: {task}")

        self.defaults = get_default_shapes()
        self._supported_inputs = set(self.task_spec.get("inputs", {}).keys())

    @property
    def supported_input_names(self):
        return tuple(self._supported_inputs)

    def _get_dim_value(self, dim_name: str) -> int:
        """Get the actual value for a dimension name."""
        # Handle computed dimensions like "past_sequence_length + sequence_length"
        if "+" in dim_name:
            parts = [p.strip() for p in dim_name.split("+")]
            return sum(self._get_dim_value(p) for p in parts)

        # Try config with aliases
        value = self._to_int(resolve_alias(self.config, dim_name))
        if value is not None:
            return value

        # Fall back to defaults
        if dim_name in self.defaults:
            return self.defaults[dim_name]

        raise ValueError(f"Cannot resolve dimension: {dim_name}")

    def _resolve_shape(self, shape_template: list[str]) -> list[int]:
        """Resolve a shape template to actual dimensions."""
        return [dim if isinstance(dim, int) else self._get_dim_value(dim) for dim in shape_template]

    def _get_input_spec(self, input_name: str) -> dict | None:
        """Get the specification for an input."""
        return self.task_spec.get("inputs", {}).get(input_name)

    def generate(self, input_name: str):
        """Generate a dummy input tensor."""
        spec = self._get_input_spec(input_name)
        if spec is None:
            raise ValueError(f"Unsupported input name: {input_name}")

        # Get shape from spec - check for with_past override first
        if self.use_past_in_inputs and "with_past" in self.task_spec:
            with_past_spec = self.task_spec["with_past"]
            if input_name in with_past_spec:
                spec = with_past_spec[input_name]

        shape_template = spec["shape"]
        shape = self._resolve_shape(shape_template)
        dtype = spec.get("dtype", "int64")

        if dtype in ["int64", "int32"]:
            if "mask" in input_name:
                return self.ones_int(shape)
            if "position" in input_name:
                # Position ids need sequential values
                seq_len = shape[-1]
                pos = torch.arange(seq_len).unsqueeze(0)
                if len(shape) > 2:
                    pos = pos.unsqueeze(1).expand(shape)
                else:
                    pos = pos.expand(shape)
                return pos
            return self.zeros_int(shape)
        return self.zeros_float(shape)


def generate_task_dummy_inputs(
    task: str,
    config: PretrainedConfig,
    model=None,
    use_past: bool = False,
    use_past_in_inputs: bool = False,
) -> dict[str, Any]:
    """Create all dummy inputs for a task.

    Args:
        task: Task name (e.g., "text-generation", "text-classification")
        config: The model's config object
        model: Optional model for forward signature inspection
        use_past: Whether to use past key values
        use_past_in_inputs: Whether past key values are inputs

    Returns:
        Dict of input_name -> tensor

    """
    import inspect

    generator = TaskDummyInputGenerator(task, config, use_past_in_inputs)

    task_spec = get_task_template(task)
    if task_spec is None:
        raise ValueError(f"Unknown task: {task}")

    # Get model forward signature if available
    model_params = set()
    if model is not None:
        try:
            sig = inspect.signature(model.forward)
            model_params = set(sig.parameters.keys())
        except (ValueError, TypeError):
            # Model may not have inspectable forward signature
            pass

    dummy_inputs = {}

    for input_name, spec in task_spec.get("inputs", {}).items():
        is_optional = spec.get("optional", False)
        if is_optional and model_params and input_name not in model_params:
            continue

        try:
            dummy_inputs[input_name] = generator.generate(input_name)
        except ValueError:
            if not is_optional:
                raise

    if use_past and use_past_in_inputs:
        dummy_inputs["past_key_values"] = _generate_past_key_values(config)

    return dummy_inputs


def _generate_past_key_values(config: PretrainedConfig) -> list[tuple]:
    """Generate past_key_values dummy input."""
    defaults = get_default_shapes()
    batch_size = defaults["batch_size"]
    past_sequence_length = defaults["sequence_length"]

    num_layers = resolve_alias(config, "num_layers") or 12
    num_heads = resolve_alias(config, "num_attention_heads") or 12
    hidden_size = resolve_alias(config, "hidden_size") or 768
    head_dim = resolve_alias(config, "head_dim") or hidden_size // num_heads

    # Auto-detect num_kv_heads (MHA/GQA/MQA)
    num_kv_heads = resolve_alias(config, "num_kv_heads") or num_heads
    if getattr(config, "multi_query", False):
        num_kv_heads = 1

    # Gemma: special head_dim calculation
    query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", None)
    if query_pre_attn_scalar is not None:
        head_dim = int(query_pre_attn_scalar**0.5)

    shape = (batch_size, num_kv_heads, past_sequence_length, head_dim)

    return [(torch.zeros(shape), torch.zeros(shape)) for _ in range(num_layers)]
