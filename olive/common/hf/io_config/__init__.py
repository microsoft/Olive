# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING

from olive.common.hf.io_config.base import OnnxConfig, OnnxConfigWithPast
from olive.common.hf.io_config.config import (
    AudioOnnxConfig,
    AudioToTextOnnxConfig,
    TextAndVisionOnnxConfig,
    TextDecoderOnnxConfig,
    TextDecoderWithPositionIdsOnnxConfig,
    TextEncoderOnnxConfig,
    TextSeq2SeqOnnxConfig,
    VisionOnnxConfig,
)
from olive.common.hf.io_config.input_generators import DEFAULT_DUMMY_SHAPES, DummyInputGenerator
from olive.common.hf.io_config.model_configs import (
    _ONNX_CONFIG_REGISTRY,
    DcaeDecoderOnnxConfig,
    DcaeEncoderOnnxConfig,
    DiffusersGemma2TextEncoderOnnxConfig,
    DiffusersT5EncoderOnnxConfig,
    # Diffusers OnnxConfig classes
    DiffusersTextEncoderOnnxConfig,
    DiffusersTextEncoderWithProjectionOnnxConfig,
    FluxTransformerOnnxConfig,
    SanaTransformerOnnxConfig,
    SD3TransformerOnnxConfig,
    UNetOnnxConfig,
    VaeDecoderOnnxConfig,
    VaeEncoderOnnxConfig,
    get_onnx_config_class,
    get_supported_model_types,
    get_supported_tasks_for_model,
)
from olive.common.hf.io_config.normalized_config import (
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedEncoderDecoderConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextConfig,
    NormalizedTextConfigWithGQA,
    NormalizedVisionConfig,
)
from olive.common.hf.io_config.tasks import (
    COMMON_TEXT2TEXT_GENERATION_TASKS,
    COMMON_TEXT_GENERATION_TASKS,
    COMMON_TEXT_TASKS,
    TaskType,
    map_task_synonym,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def get_onnx_config(
    model_type: str,
    task: str,
    config: "PretrainedConfig",
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    use_past: bool = False,
    use_past_in_inputs: bool = False,
    **kwargs,
) -> OnnxConfig:
    """Get an ONNX config instance for a model type and task.

    Args:
        model_type: The model type (e.g., "bert", "llama", "gpt2").
        task: The task type (e.g., "feature-extraction", "text-generation").
        config: The model's PretrainedConfig.
        int_dtype: The integer data type for inputs (default: "int64").
        float_dtype: The float data type for inputs (default: "fp32").
        use_past: Whether to use past key values (for decoder models).
        use_past_in_inputs: Whether past key values are inputs.
        **kwargs: Additional arguments passed to the config constructor.

    Returns:
        An OnnxConfig instance configured for the model and task.

    Raises:
        KeyError: If the model type or task is not supported.

    """
    config_class = get_onnx_config_class(model_type, task)

    # Build constructor kwargs based on what the config class accepts
    constructor_kwargs = {
        "config": config,
        "task": task,
        "int_dtype": int_dtype,
        "float_dtype": float_dtype,
    }

    # Add use_past and use_past_in_inputs for decoder configs
    if issubclass(config_class, OnnxConfigWithPast):
        constructor_kwargs["use_past"] = use_past
        constructor_kwargs["use_past_in_inputs"] = use_past_in_inputs

    constructor_kwargs.update(kwargs)

    return config_class(**constructor_kwargs)


def is_model_supported(model_type: str) -> bool:
    """Check if a model type is supported."""
    return model_type in _ONNX_CONFIG_REGISTRY


def is_task_supported(model_type: str, task: str) -> bool:
    """Check if a task is supported for a model type."""
    if model_type not in _ONNX_CONFIG_REGISTRY:
        return False
    return task in _ONNX_CONFIG_REGISTRY[model_type]


# ============================================================================
# Diffusers pipeline component to OnnxConfig mapping
# ============================================================================

# Mapping: (pipeline_type, component_name) -> OnnxConfig class
_DIFFUSERS_CONFIG_REGISTRY: dict[tuple[str, str], type[OnnxConfig]] = {
    # SD 1.5
    ("sd", "text_encoder"): DiffusersTextEncoderOnnxConfig,
    ("sd", "unet"): UNetOnnxConfig,
    ("sd", "vae_encoder"): VaeEncoderOnnxConfig,
    ("sd", "vae_decoder"): VaeDecoderOnnxConfig,
    # SDXL
    ("sdxl", "text_encoder"): DiffusersTextEncoderOnnxConfig,
    ("sdxl", "text_encoder_2"): DiffusersTextEncoderWithProjectionOnnxConfig,
    ("sdxl", "unet"): UNetOnnxConfig,
    ("sdxl", "vae_encoder"): VaeEncoderOnnxConfig,
    ("sdxl", "vae_decoder"): VaeDecoderOnnxConfig,
    # SD3
    ("sd3", "text_encoder"): DiffusersTextEncoderOnnxConfig,
    ("sd3", "text_encoder_2"): DiffusersTextEncoderWithProjectionOnnxConfig,
    ("sd3", "text_encoder_3"): DiffusersT5EncoderOnnxConfig,
    ("sd3", "transformer"): SD3TransformerOnnxConfig,
    ("sd3", "vae_encoder"): VaeEncoderOnnxConfig,
    ("sd3", "vae_decoder"): VaeDecoderOnnxConfig,
    # Flux
    ("flux", "text_encoder"): DiffusersTextEncoderOnnxConfig,
    ("flux", "text_encoder_2"): DiffusersT5EncoderOnnxConfig,
    ("flux", "transformer"): FluxTransformerOnnxConfig,
    ("flux", "vae_encoder"): VaeEncoderOnnxConfig,
    ("flux", "vae_decoder"): VaeDecoderOnnxConfig,
    # Sana
    ("sana", "text_encoder"): DiffusersGemma2TextEncoderOnnxConfig,
    ("sana", "transformer"): SanaTransformerOnnxConfig,
    ("sana", "vae_encoder"): DcaeEncoderOnnxConfig,
    ("sana", "vae_decoder"): DcaeDecoderOnnxConfig,
}


def get_diffusers_onnx_config(
    pipeline_type: str,
    component_name: str,
    config: "PretrainedConfig",
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    **kwargs,
) -> OnnxConfig:
    """Get an ONNX config instance for a diffusers pipeline component.

    Args:
        pipeline_type: The diffusers pipeline type (e.g., "sd", "sdxl", "sd3", "flux", "sana").
        component_name: The component name (e.g., "text_encoder", "unet", "vae_encoder").
        config: The component's config.
        int_dtype: The integer data type for inputs (default: "int64").
        float_dtype: The float data type for inputs (default: "fp32").
        **kwargs: Additional arguments passed to the config constructor.

    Returns:
        An OnnxConfig instance configured for the component.

    Raises:
        KeyError: If the pipeline type and component combination is not supported.

    """
    key = (pipeline_type.lower(), component_name.lower())
    if key not in _DIFFUSERS_CONFIG_REGISTRY:
        supported = [f"{p}:{c}" for p, c in _DIFFUSERS_CONFIG_REGISTRY]
        raise KeyError(
            f"Pipeline type '{pipeline_type}' with component '{component_name}' is not supported. "
            f"Supported combinations: {supported}"
        )

    config_class = _DIFFUSERS_CONFIG_REGISTRY[key]
    return config_class(
        config=config,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        **kwargs,
    )


def get_supported_diffusers_pipelines() -> list[str]:
    """Get list of supported diffusers pipeline types."""
    return list({p for p, _ in _DIFFUSERS_CONFIG_REGISTRY})


def get_supported_components_for_pipeline(pipeline_type: str) -> list[str]:
    """Get list of supported components for a diffusers pipeline type."""
    pipeline_type = pipeline_type.lower()
    return [c for p, c in _DIFFUSERS_CONFIG_REGISTRY if p == pipeline_type]


__all__ = [
    "COMMON_TEXT2TEXT_GENERATION_TASKS",
    "COMMON_TEXT_GENERATION_TASKS",
    "COMMON_TEXT_TASKS",
    "DEFAULT_DUMMY_SHAPES",
    "AudioOnnxConfig",
    "AudioToTextOnnxConfig",
    "DcaeDecoderOnnxConfig",
    "DcaeEncoderOnnxConfig",
    "DiffusersGemma2TextEncoderOnnxConfig",
    "DiffusersT5EncoderOnnxConfig",
    "DiffusersTextEncoderOnnxConfig",
    "DiffusersTextEncoderWithProjectionOnnxConfig",
    "DummyInputGenerator",
    "FluxTransformerOnnxConfig",
    "NormalizedConfig",
    "NormalizedConfigManager",
    "NormalizedEncoderDecoderConfig",
    "NormalizedSeq2SeqConfig",
    "NormalizedTextConfig",
    "NormalizedTextConfigWithGQA",
    "NormalizedVisionConfig",
    "OnnxConfig",
    "OnnxConfigWithPast",
    "SD3TransformerOnnxConfig",
    "SanaTransformerOnnxConfig",
    "TaskType",
    "TextAndVisionOnnxConfig",
    "TextDecoderOnnxConfig",
    "TextDecoderWithPositionIdsOnnxConfig",
    "TextEncoderOnnxConfig",
    "TextSeq2SeqOnnxConfig",
    "UNetOnnxConfig",
    "VaeDecoderOnnxConfig",
    "VaeEncoderOnnxConfig",
    "VisionOnnxConfig",
    "get_diffusers_onnx_config",
    "get_onnx_config",
    "get_onnx_config_class",
    "get_supported_components_for_pipeline",
    "get_supported_diffusers_pipelines",
    "get_supported_model_types",
    "get_supported_tasks_for_model",
    "is_model_supported",
    "is_task_supported",
    "map_task_synonym",
]
