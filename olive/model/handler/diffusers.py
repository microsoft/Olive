# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from olive.constants import DiffusersComponent as DC  # noqa: N817
from olive.constants import DiffusersModelVariant, Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.model.utils.diffusers_utils import is_valid_diffusers_model
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


@model_handler_registry("DiffusersModel")
class DiffusersModelHandler(OliveModelHandler):
    """Model handler for diffusers models (Stable Diffusion, SDXL, Flux, etc.).

    This handler is designed for diffusion models from the diffusers library.

    Example usage:
        model = DiffusersModelHandler(
            model_path="runwayml/stable-diffusion-v1-5",
            model_variant="sd",  # optional: sd, sdxl, flux, or auto
        )
    """

    resource_keys: tuple[str, ...] = ("model_path", "adapter_path")
    json_config_keys: tuple[str, ...] = ("model_variant", "load_kwargs")

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
        model_variant: Union[str, DiffusersModelVariant] = DiffusersModelVariant.AUTO,
        load_kwargs: Optional[dict[str, Any]] = None,
        adapter_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[dict[str, Any]] = None,
    ):
        """Initialize DiffusersModelHandler.

        Args:
            model_path: Path to diffusion model (local path or HuggingFace model ID).
            model_variant: Model variant: 'sd15', 'sdxl', 'flux', or 'auto' for auto-detection.
            load_kwargs: Additional kwargs for loading the model (e.g., torch_dtype, variant).
            adapter_path: Path to LoRA adapter weights.
            model_attributes: Additional model attributes.

        """
        if not is_valid_diffusers_model(model_path):
            raise ValueError(f"The provided model_path '{model_path}' is not a valid diffusion model.")

        super().__init__(
            framework=Framework.PYTORCH,
            model_file_format=ModelFileFormat.PYTORCH_ENTIRE_MODEL,
            model_path=model_path,
            model_attributes=model_attributes or {},
        )
        self.add_resources(locals())

        self.model_variant = DiffusersModelVariant(model_variant)
        self.load_kwargs = load_kwargs or {}
        self._pipeline = None

    @property
    def adapter_path(self) -> Optional[str]:
        """Return the path to the LoRA adapter."""
        return self.get_resource("adapter_path")

    @property
    def size_on_disk(self) -> int:
        """Compute size of the model on disk.

        For diffusers models, this typically includes weights from multiple components.
        Returns 0 if unable to compute (e.g., for HuggingFace Hub IDs).
        """
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                # Remote model (HuggingFace Hub ID)
                return 0

            # Sum up all files in the model directory
            total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

            # Add adapter size if present
            if self.adapter_path:
                adapter_path = Path(self.adapter_path)
                if adapter_path.exists():
                    total_size += sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file())

            return total_size
        except Exception as exc:
            logger.warning("Failed to compute model size on disk: %s", exc)
            return 0

    @property
    def detected_model_variant(self) -> DiffusersModelVariant:
        """Detect the diffusion model variant from config files."""
        if self.model_variant != DiffusersModelVariant.AUTO:
            return self.model_variant

        detected = self._detect_variant_from_config()
        if detected is None:
            raise ValueError(
                f"Cannot detect model variant for '{self.model_path}'. "
                "Please specify model_variant explicitly: 'sd', 'sdxl', 'sd3', 'flux', or 'sana'."
            )

        self.model_variant = detected
        return detected

    def _detect_variant_from_config(self) -> Optional[DiffusersModelVariant]:
        """Detect model variant by reading config files.

        This method checks the _class_name field in transformer/config.json or unet/config.json
        to determine the model variant.

        Returns:
            Detected model variant, or None if detection failed.

        """
        try:
            from diffusers import ConfigMixin
        except ImportError as exc:
            logger.debug("Failed to import diffusers.ConfigMixin: %s", exc)
            return None

        # Try transformer config first (for SD3, Flux, Sana)
        try:
            transformer_config = ConfigMixin.load_config(self.model_path, subfolder="transformer")
            class_name = transformer_config.get("_class_name", "")

            if "Sana" in class_name:
                return DiffusersModelVariant.SANA
            if "Flux" in class_name:
                return DiffusersModelVariant.FLUX
            if "SD3" in class_name:
                return DiffusersModelVariant.SD3
        except Exception as exc:
            logger.debug("No transformer config found: %s", exc)

        # Try unet config (for SD, SDXL)
        try:
            unet_config = ConfigMixin.load_config(self.model_path, subfolder="unet")
            # SDXL has cross_attention_dim >= 2048, SD has 768
            if unet_config.get("cross_attention_dim", 768) >= 2048:
                return DiffusersModelVariant.SDXL
            return DiffusersModelVariant.SD
        except Exception as exc:
            logger.debug("No unet config found: %s", exc)

        return None

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

    def get_exportable_components(self) -> list[DC]:
        """Get list of exportable components for the pipeline variant.

        Returns:
            List of component names that should be exported.

        """
        variant = self.detected_model_variant
        variant_components = {
            DiffusersModelVariant.SD: [
                DC.TEXT_ENCODER,
                DC.UNET,
                DC.VAE_ENCODER,
                DC.VAE_DECODER,
            ],
            DiffusersModelVariant.SDXL: [
                DC.TEXT_ENCODER,
                DC.TEXT_ENCODER_2,
                DC.UNET,
                DC.VAE_ENCODER,
                DC.VAE_DECODER,
            ],
            DiffusersModelVariant.SD3: [
                DC.TEXT_ENCODER,
                DC.TEXT_ENCODER_2,
                DC.TEXT_ENCODER_3,
                DC.TRANSFORMER,
                DC.VAE_ENCODER,
                DC.VAE_DECODER,
            ],
            DiffusersModelVariant.FLUX: [
                DC.TEXT_ENCODER,
                DC.TEXT_ENCODER_2,
                DC.TRANSFORMER,
                DC.VAE_ENCODER,
                DC.VAE_DECODER,
            ],
            DiffusersModelVariant.SANA: [
                DC.TEXT_ENCODER,
                DC.TRANSFORMER,
                DC.VAE_ENCODER,
                DC.VAE_DECODER,
            ],
        }
        if variant not in variant_components:
            raise ValueError(f"Unknown model variant: {variant}")
        return variant_components[variant]

    def get_pipeline_type(self) -> DiffusersModelVariant:
        """Get the pipeline type for OnnxConfig lookup.

        Returns:
            Pipeline type (e.g., DiffusersModelVariant.SD).

        """
        return self.detected_model_variant
