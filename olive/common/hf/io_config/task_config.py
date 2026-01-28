# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations

import inspect
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from olive.common.hf.io_config.io_resolver import (
    get_diffusers_component_config,
    get_task_template,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


# =============================================================================
# IO Config Generation
# =============================================================================


def get_io_config(
    model_name_or_config: str | PretrainedConfig,
    task: str,
    model: PreTrainedModel | None = None,
    use_past: bool = False,
    use_past_in_inputs: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Get IO configuration for ONNX export.

    Args:
        model_name_or_config: HuggingFace model name/path or PretrainedConfig.
        task: Task type (e.g., "text-generation", "text-classification").
        model: Optional loaded model for input signature inspection.
        use_past: Whether to use past key values (KV cache).
        use_past_in_inputs: Whether past key values are inputs.
        **kwargs: Additional arguments.

    Returns:
        Dict containing:
            - input_names: List of input names
            - output_names: List of output names
            - dynamic_axes: Dict mapping names to axis definitions
            - dynamic_shapes: Dict with nested structure for dynamo export

    """
    from transformers import AutoConfig

    from olive.common.hf.io_config.io_resolver import resolve_alias

    # Load config if needed
    if isinstance(model_name_or_config, str):
        config = AutoConfig.from_pretrained(model_name_or_config, **kwargs)
    else:
        config = model_name_or_config

    # Get task template
    task_template = get_task_template(task)
    if task_template is None:
        raise ValueError(f"Unsupported task: {task}")

    # Build inputs (auto-detects from model.forward signature)
    inputs = _build_inputs(task_template, model)

    # Build outputs
    outputs = _build_outputs(task_template)

    # Build dynamic_axes from inputs/outputs
    dynamic_axes = {**inputs, **outputs}

    # Add past_key_values inputs if needed
    if use_past_in_inputs:
        _add_past_key_values_inputs(dynamic_axes, config)

    # Add present outputs if use_past is enabled (KV cache)
    if use_past:
        _add_present_outputs(dynamic_axes, config)

    # Order inputs according to model forward signature
    ordered_inputs = _order_inputs(dynamic_axes, model, set(outputs.keys()))

    # Separate input and output names
    input_names = [name for name in ordered_inputs if not name.startswith("present.")]
    output_names = list(outputs.keys())
    if use_past:
        # Add present outputs in order
        num_layers = resolve_alias(config, "num_layers") or 12
        for i in range(num_layers):
            output_names.append(f"present.{i}.key")
            output_names.append(f"present.{i}.value")

    # Build dynamic_shapes for dynamo export (nested structure)
    dynamic_shapes = _build_dynamic_shapes(ordered_inputs)

    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "dynamic_shapes": dynamic_shapes,
    }


def _build_inputs(
    task_template: dict,
    model: PreTrainedModel | None,
) -> OrderedDict:
    """Build input specification from task template.

    Uses model.forward signature to auto-detect which optional inputs are needed.
    """
    inputs = OrderedDict()
    template_inputs = task_template.get("inputs", {})

    # Get model forward signature if available
    model_params = set()
    if model is not None:
        try:
            sig = inspect.signature(model.forward)
            model_params = set(sig.parameters.keys())
        except (ValueError, TypeError):
            # Model may not have inspectable forward signature
            pass

    for input_name, input_spec in template_inputs.items():
        is_optional = input_spec.get("optional", False)
        if is_optional and model_params and input_name not in model_params:
            continue

        axes = input_spec.get("axes", {})
        inputs[input_name] = {int(k): v for k, v in axes.items()}

    return inputs


def _build_outputs(task_template: dict) -> OrderedDict:
    """Build output specification from task template."""
    outputs = OrderedDict()
    template_outputs = task_template.get("outputs", {})

    for output_name, output_spec in template_outputs.items():
        axes = output_spec.get("axes", {})
        outputs[output_name] = {int(k): v for k, v in axes.items()}

    return outputs


def _add_past_key_values_inputs(
    dynamic_axes: dict,
    config: PretrainedConfig,
) -> None:
    """Add past_key_values inputs for decoder models with KV cache.

    Args:
        dynamic_axes: Dict to add the dynamic axes to (modified in place).
        config: Model config to get num_layers.

    """
    from olive.common.hf.io_config.io_resolver import resolve_alias

    # Update attention_mask for past context
    if "attention_mask" in dynamic_axes:
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "past_sequence_length + sequence_length"}

    # Get number of layers from config
    num_layers = resolve_alias(config, "num_layers") or 12

    # Add flattened past_key_values inputs
    for i in range(num_layers):
        dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch_size", 2: "past_sequence_length"}
        dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch_size", 2: "past_sequence_length"}


def _add_present_outputs(
    dynamic_axes: dict,
    config: PretrainedConfig,
) -> None:
    """Add present (KV cache) outputs for decoder models.

    Args:
        dynamic_axes: Dict to add the dynamic axes to (modified in place).
        config: Model config to get num_layers.

    """
    from olive.common.hf.io_config.io_resolver import resolve_alias

    # Get number of layers from config
    num_layers = resolve_alias(config, "num_layers") or 12

    # Add present outputs
    for i in range(num_layers):
        dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "past_sequence_length + sequence_length"}
        dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "past_sequence_length + sequence_length"}


def _order_inputs(
    dynamic_axes: dict,
    model: PreTrainedModel | None,
    output_names: set[str] | None = None,
) -> OrderedDict:
    """Order inputs according to model forward signature.

    This ensures input_names order matches torch.onnx.export's flattening order.

    Args:
        dynamic_axes: Dict of all dynamic axes (inputs and outputs).
        model: Optional model for forward signature inspection.
        output_names: Set of output names to exclude from input ordering.

    Returns:
        OrderedDict of input names to dynamic axes, ordered by forward signature.

    """
    import re

    if output_names is None:
        output_names = set()

    # Filter to only input names (not outputs like present.* or explicit output names)
    input_axes = OrderedDict()
    for name, axes in dynamic_axes.items():
        if name.startswith("present.") or name in output_names:
            continue
        input_axes[name] = axes

    if model is None:
        return input_axes

    # Get model forward signature
    try:
        sig = inspect.signature(model.forward)
    except (ValueError, TypeError):
        return input_axes

    # Reorder inputs according to forward signature
    ordered = OrderedDict()
    for param in sig.parameters:
        # Match param and param.* (e.g., past_key_values matches past_key_values.0.key)
        param_regex = re.compile(rf"^{re.escape(param)}(\..*)?$")
        for name, axes in input_axes.items():
            if re.match(param_regex, name) and name not in ordered:
                ordered[name] = axes

    # Add any remaining inputs not in signature
    for name, axes in input_axes.items():
        if name not in ordered:
            ordered[name] = axes

    return ordered


def _build_dynamic_shapes(
    ordered_inputs: OrderedDict,
) -> dict:
    """Build dynamic_shapes for dynamo export.

    Converts flattened past_key_values to nested structure:
    {"past_key_values.0.key": ..., "past_key_values.0.value": ...}
    -> {"past_key_values": [[key_shape, value_shape], ...]}

    Args:
        ordered_inputs: OrderedDict of input names to dynamic axes.

    Returns:
        Dict with nested structure for dynamic_shapes.

    """
    # Find max index of past_key_values
    max_idx = -1
    pkv_count = 0
    for name in ordered_inputs:
        if name.startswith("past_key_values."):
            parts = name.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1])
                max_idx = max(max_idx, idx)
                pkv_count += 1

    # No past_key_values, return as-is
    if max_idx == -1:
        return dict(ordered_inputs)

    # Validate count (should be 2 per layer: key + value)
    expected_count = 2 * (max_idx + 1)
    if pkv_count != expected_count:
        logger.warning(
            "Expected %d past_key_values entries, but found %d. Using dynamic_axes instead.",
            expected_count,
            pkv_count,
        )
        return {}

    # Build nested structure (exclude flattened past_key_values)
    dynamic_shapes = {name: axes for name, axes in ordered_inputs.items() if not name.startswith("past_key_values.")}

    # Generate nested past_key_values
    dynamic_shapes["past_key_values"] = [
        [
            ordered_inputs.get(f"past_key_values.{i}.key", {}),
            ordered_inputs.get(f"past_key_values.{i}.value", {}),
        ]
        for i in range(max_idx + 1)
    ]

    return dynamic_shapes


# =============================================================================
# Dummy Input Generation
# =============================================================================


def generate_dummy_inputs(
    model_name_or_config: str | PretrainedConfig,
    task: str,
    model: PreTrainedModel | None = None,
    use_past: bool = False,
    use_past_in_inputs: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Generate dummy inputs for ONNX export.

    Args:
        model_name_or_config: HuggingFace model name/path or PretrainedConfig.
        task: Task type.
        model: Optional loaded model.
        use_past: Whether to generate past_key_values.
        use_past_in_inputs: Whether past_key_values are inputs.
        **kwargs: Additional arguments.

    Returns:
        Dict of dummy input tensors.

    """
    from transformers import AutoConfig

    from olive.common.hf.io_config.input_generators import generate_task_dummy_inputs

    # Load config if needed
    if isinstance(model_name_or_config, str):
        config = AutoConfig.from_pretrained(model_name_or_config, **kwargs)
    else:
        config = model_name_or_config

    return generate_task_dummy_inputs(
        task=task,
        config=config,
        model=model,
        use_past=use_past,
        use_past_in_inputs=use_past_in_inputs,
    )


# =============================================================================
# Diffusers Support
# =============================================================================


def get_diffusers_io_config(
    component_name: str,
    config: PretrainedConfig,
    **kwargs,
) -> dict[str, Any]:
    """Get IO configuration for a diffusers component.

    Args:
        component_name: Component name (e.g., "text_encoder", "unet").
        config: Component's config.
        **kwargs: Additional arguments (e.g., is_sdxl).

    Returns:
        Dict containing input_names, output_names, dynamic_axes.

    """
    component_config = get_diffusers_component_config(component_name)
    if component_config is None:
        raise ValueError(f"Unknown diffusers component: {component_name}")

    inputs = OrderedDict()
    outputs = OrderedDict()

    # Build inputs
    for name, spec in component_config.get("inputs", {}).items():
        axes = spec.get("axes", {})
        inputs[name] = {int(k): v for k, v in axes.items()}

    # Add SDXL-specific inputs if needed
    is_sdxl = kwargs.get("is_sdxl", False) or getattr(config, "addition_embed_type", None) == "text_time"
    if is_sdxl and "sdxl_inputs" in component_config:
        for name, spec in component_config["sdxl_inputs"].items():
            axes = spec.get("axes", {})
            inputs[name] = {int(k): v for k, v in axes.items()}

    # Add optional inputs based on config
    if "optional_inputs" in component_config:
        for name, spec in component_config["optional_inputs"].items():
            if (name == "timestep_cond" and getattr(config, "time_cond_proj_dim", None)) or (
                name == "guidance" and getattr(config, "guidance_embeds", False)
            ):
                axes = spec.get("axes", {})
                inputs[name] = {int(k): v for k, v in axes.items()}

    # Build outputs
    for name, spec in component_config.get("outputs", {}).items():
        axes = spec.get("axes", {})
        outputs[name] = {int(k): v for k, v in axes.items()}

    dynamic_axes = {**inputs, **outputs}

    return {
        "input_names": list(inputs.keys()),
        "output_names": list(outputs.keys()),
        "dynamic_axes": dynamic_axes,
    }
