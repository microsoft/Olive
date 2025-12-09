# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from olive.common.utils import StrEnumBase
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


class DiffusersModelType(StrEnumBase):
    """Diffusion model types."""

    AUTO = "auto"
    SD15 = "sd15"
    SDXL = "sdxl"
    FLUX = "flux"


@model_handler_registry("DiffusersModel")
class DiffusersModelHandler(OliveModelHandler):
    """Model handler for diffusers models (Stable Diffusion, SDXL, Flux, etc.).

    This handler is designed for diffusion models from the diffusers library.

    Example usage:
        model = DiffusersModelHandler(
            model_path="runwayml/stable-diffusion-v1-5",
            model_type="sd15",  # optional: sd15, sdxl, flux, or auto
        )
    """

    resource_keys: tuple[str, ...] = ("model_path", "adapter_path")
    json_config_keys: tuple[str, ...] = ("model_type", "load_kwargs")

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
        model_type: Union[str, DiffusersModelType] = DiffusersModelType.AUTO,
        load_kwargs: Optional[dict[str, Any]] = None,
        adapter_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[dict[str, Any]] = None,
    ):
        """Initialize DiffusersModelHandler.

        Args:
            model_path: Path to diffusion model (local path or HuggingFace model ID).
            model_type: Model type: 'sd15', 'sdxl', 'flux', or 'auto' for auto-detection.
            load_kwargs: Additional kwargs for loading the model (e.g., torch_dtype, variant).
            adapter_path: Path to LoRA adapter weights.
            model_attributes: Additional model attributes.

        """
        super().__init__(
            framework=Framework.PYTORCH,
            model_file_format=ModelFileFormat.PYTORCH_ENTIRE_MODEL,
            model_path=model_path,
            model_attributes=model_attributes or {},
        )
        self.add_resources(locals())

        self.model_type = DiffusersModelType(model_type)
        self.load_kwargs = load_kwargs or {}
        self._pipeline = None

    @property
    def adapter_path(self) -> Optional[str]:
        """Return the path to the LoRA adapter."""
        return self.get_resource("adapter_path")

    @property
    def detected_model_type(self) -> DiffusersModelType:
        """Detect the diffusion model type."""
        if self.model_type != DiffusersModelType.AUTO:
            return self.model_type

        model_path_lower = self.model_path.lower() if self.model_path else ""

        # Check for Flux
        if "flux" in model_path_lower:
            return DiffusersModelType.FLUX

        # Try to detect from model config
        try:
            from diffusers import FluxTransformer2DModel

            FluxTransformer2DModel.load_config(self.model_path, subfolder="transformer")
            return DiffusersModelType.FLUX
        except Exception as exc:
            logger.debug("Error detecting Flux model type with FluxTransformer2DModel: %s", exc)

        try:
            from diffusers import UNet2DConditionModel

            unet_config = UNet2DConditionModel.load_config(self.model_path, subfolder="unet")
            if unet_config.get("cross_attention_dim", 768) >= 2048:
                return DiffusersModelType.SDXL
        except Exception as exc:
            logger.debug("Error detecting SDXL model type with UNet2DConditionModel: %s", exc)

        # Check model name patterns
        if "xl" in model_path_lower or "sdxl" in model_path_lower:
            return DiffusersModelType.SDXL
        if "sd" in model_path_lower or "stable-diffusion" in model_path_lower:
            return DiffusersModelType.SD15

        raise ValueError(
            f"Cannot detect model type from '{self.model_path}'. "
            "Please specify model_type explicitly: 'sd15', 'sdxl', or 'flux'."
        )

    def load_model(self, rank: int = None, cache_model: bool = True) -> "DiffusionPipeline":
        """Load the diffusion pipeline.

        Args:
            rank: GPU rank for distributed training.
            cache_model: Whether to cache the loaded model.

        Returns:
            DiffusionPipeline instance.

        """
        if self._pipeline is not None:
            return self._pipeline

        from diffusers import DiffusionPipeline

        logger.info("Loading diffusion model from %s", self.model_path)

        pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            **self.load_kwargs,
        )

        # Load LoRA adapter if provided
        if self.adapter_path:
            logger.info("Loading LoRA adapter from %s", self.adapter_path)
            pipeline.load_lora_weights(self.adapter_path)

        if cache_model:
            self._pipeline = pipeline

        return pipeline

    def prepare_session(
        self,
        inference_settings: Optional[dict[str, Any]] = None,
        device: Device = Device.GPU,
        execution_providers: Union[str, list[str]] = None,
        rank: Optional[int] = None,
    ) -> "DiffusionPipeline":
        """Prepare the pipeline for inference.

        Args:
            inference_settings: Additional inference settings.
            device: Device to run on (GPU recommended).
            execution_providers: Not used for diffusers.
            rank: GPU rank for distributed inference.

        Returns:
            DiffusionPipeline ready for inference.

        """
        import torch

        pipeline = self.load_model(rank)

        # Move to device
        device_str = "cuda" if device == Device.GPU and torch.cuda.is_available() else "cpu"
        return pipeline.to(device_str)

    def run_session(
        self,
        session: Any = None,
        inputs: Union[dict[str, Any], list[Any], tuple[Any, ...]] = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        """Run inference on the pipeline.

        Args:
            session: The pipeline from prepare_session.
            inputs: Inputs for the pipeline (prompt, etc.).
            **kwargs: Additional arguments for the pipeline.

        Returns:
            Pipeline output (usually images).

        """
        if session is None:
            session = self.load_model()

        if isinstance(inputs, dict):
            return session(**inputs, **kwargs)
        elif isinstance(inputs, (list, tuple)):
            return session(*inputs, **kwargs)
        else:
            return session(inputs, **kwargs)

    def get_component(self, component_name: str) -> Any:
        """Get a specific component from the pipeline.

        Args:
            component_name: Name of the component (e.g., 'unet', 'vae', 'text_encoder').

        Returns:
            The requested component.

        """
        pipeline = self.load_model()
        if hasattr(pipeline, component_name):
            return getattr(pipeline, component_name)
        raise ValueError(f"Component '{component_name}' not found in pipeline")
