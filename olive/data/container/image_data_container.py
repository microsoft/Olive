# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Image Data Container with chained preprocessing support for SD LoRA and DreamBooth."""

import logging
from typing import ClassVar, Optional

from olive.common.pydantic_v1 import BaseModel, Field
from olive.data.constants import DataComponentType, DataContainerType
from olive.data.container.data_container import DataContainer
from olive.data.registry import Registry

logger = logging.getLogger(__name__)


class PreProcessStep(BaseModel):
    """Configuration for a single preprocessing step in the chain."""

    type: str  # registered preprocess function name
    params: dict = Field(default_factory=dict)
    enabled: bool = True  # allow disabling steps without removing them


class PreProcessChain(BaseModel):
    """Configuration for a chain of preprocessing steps."""

    steps: list[PreProcessStep] = Field(default_factory=list)

    def add_step(self, step_type: str, params: Optional[dict] = None, enabled: bool = True) -> "PreProcessChain":
        """Add a preprocessing step to the chain."""
        self.steps.append(PreProcessStep(type=step_type, params=params or {}, enabled=enabled))
        return self

    def insert_step(
        self, index: int, step_type: str, params: Optional[dict] = None, enabled: bool = True
    ) -> "PreProcessChain":
        """Insert a preprocessing step at a specific position."""
        self.steps.insert(index, PreProcessStep(type=step_type, params=params or {}, enabled=enabled))
        return self

    def remove_step(self, step_type: str) -> "PreProcessChain":
        """Remove all steps of a given type."""
        self.steps = [s for s in self.steps if s.type != step_type]
        return self

    def disable_step(self, step_type: str) -> "PreProcessChain":
        """Disable all steps of a given type."""
        for step in self.steps:
            if step.type == step_type:
                step.enabled = False
        return self

    def enable_step(self, step_type: str) -> "PreProcessChain":
        """Enable all steps of a given type."""
        for step in self.steps:
            if step.type == step_type:
                step.enabled = True
        return self

    def get_enabled_steps(self) -> list[PreProcessStep]:
        """Get only enabled steps."""
        return [s for s in self.steps if s.enabled]


# ============================================================================
# Default preprocessing chains
# ============================================================================

def get_lora_default_chain(base_resolution: int = 512, bucket_mode: str = "auto") -> PreProcessChain:
    """Get default preprocessing chain for LoRA training.

    Works for both SD 1.5 and SDXL, just change base_resolution:
    - SD 1.5: base_resolution=512
    - SDXL: base_resolution=1024, bucket_mode="sdxl"

    Default order:
    1. image_filtering - Remove low quality images (disabled)
    2. auto_caption - Generate captions (disabled)
    3. auto_tagging - Generate tags (disabled)
    4. caption_tag_merge - Merge captions and tags (disabled)
    5. image_resizing - Resize images (disabled, bucketing handles this)
    6. aspect_ratio_bucketing - Group by aspect ratio (enabled)
    """
    min_size = base_resolution // 2  # 256 for SD1.5, 512 for SDXL

    chain = PreProcessChain()
    chain.add_step("image_filtering", {"min_width": min_size, "min_height": min_size}, enabled=False)
    chain.add_step("auto_caption", {"model_type": "blip2", "caption_extension": ".caption"}, enabled=False)
    chain.add_step("auto_tagging", {"model_type": "wd14", "tag_extension": ".tags"}, enabled=False)
    chain.add_step("caption_tag_merge", {"caption_extension": ".caption", "tag_extension": ".tags"}, enabled=False)
    chain.add_step("image_resizing", {"target_resolution": base_resolution, "resize_mode": "bucket"}, enabled=False)
    chain.add_step("aspect_ratio_bucketing", {"base_resolution": base_resolution, "bucket_mode": bucket_mode}, enabled=True)
    return chain


def get_dreambooth_default_chain(target_resolution: int = 512) -> PreProcessChain:
    """Get default preprocessing chain for DreamBooth training.

    Default order:
    1. image_filtering - Remove low quality images (disabled)
    2. dreambooth_prompt - Generate fixed prompts with identifier (enabled)
    3. generate_class_images - Generate regularization images (disabled)
    4. image_resizing - Resize to fixed size (enabled)
    """
    chain = PreProcessChain()
    chain.add_step("image_filtering", {"min_width": 256, "min_height": 256}, enabled=False)
    chain.add_step(
        "dreambooth_prompt",
        {
            "prompt_template": "a photo of {identifier} {classname}",
            "identifier_token": "sks",
            "class_token": "dog",
        },
        enabled=True,
    )
    chain.add_step(
        "generate_class_images",
        {"num_class_images": 200, "class_prompt": "a photo of dog"},
        enabled=False,
    )
    chain.add_step(
        "image_resizing",
        {"target_resolution": target_resolution, "resize_mode": "cover", "crop_position": "center"},
        enabled=True,
    )
    return chain


# ============================================================================
# Base mixin for chain execution
# ============================================================================

class ChainPreProcessMixin:
    """Mixin class for containers that support chained preprocessing."""

    # Subclasses should override this
    default_chain_fn: ClassVar = lambda: get_lora_default_chain()

    def pre_process(self, dataset):
        """Run chained preprocessing."""
        params = self.config.pre_process_params

        # Get or build the chain
        chain = self._build_chain(params)

        # Run each enabled step
        for step in chain.get_enabled_steps():
            logger.info("Running preprocess step: %s", step.type)
            try:
                preprocess_fn = Registry.get_pre_process_component(step.type)
                dataset = preprocess_fn(dataset, **step.params)
            except KeyError:
                logger.warning("Preprocess component '%s' not found, skipping", step.type)
            except Exception as e:
                logger.error("Error in preprocess step '%s': %s", step.type, e)
                raise

        return dataset

    def _build_chain(self, params: dict) -> PreProcessChain:
        """Build preprocessing chain from params."""
        # Check for custom steps
        if "steps" in params:
            chain = PreProcessChain(
                steps=[PreProcessStep(**s) for s in params["steps"]]
            )
        else:
            # Use container's default chain
            chain = self.default_chain_fn()

        # Enable specific steps
        enable_steps = params.get("enable_steps") or []
        for step_type in enable_steps:
            chain.enable_step(step_type)

        # Disable specific steps
        disable_steps = params.get("disable_steps") or []
        for step_type in disable_steps:
            chain.disable_step(step_type)

        # Override step params
        step_params = params.get("step_params") or {}
        for step in chain.steps:
            if step.type in step_params:
                step.params.update(step_params[step.type])

        # Apply global params to all steps (e.g., device, overwrite)
        global_params = {k: v for k, v in params.items()
                        if k not in ("steps", "enable_steps", "disable_steps", "step_params")}
        for step in chain.steps:
            for k, v in global_params.items():
                if k not in step.params:
                    step.params[k] = v

        return chain


# ============================================================================
# SD LoRA Data Container (works for both SD 1.5 and SDXL)
# ============================================================================

@Registry.register(DataContainerType.DATA_CONTAINER, name="SDLoRADataContainer")
class SDLoRADataContainer(ChainPreProcessMixin, DataContainer):
    """Data container for Stable Diffusion LoRA training.

    Works for both SD 1.5 and SDXL - just set base_resolution:
    - SD 1.5: base_resolution=512 (default)
    - SDXL: base_resolution=1024, bucket_mode="sdxl"

    Default preprocessing chain:
    1. image_filtering (disabled) - Filter low quality images
    2. auto_caption (disabled) - Generate captions with BLIP-2/Florence-2
    3. auto_tagging (disabled) - Generate tags with WD14 tagger
    4. caption_tag_merge (disabled) - Merge captions and tags
    5. image_resizing (disabled) - Resize images
    6. aspect_ratio_bucketing (enabled) - Group by aspect ratio

    Usage (SD 1.5):
        config = DataConfig(
            name="my_lora",
            type="SDLoRADataContainer",
            load_dataset_config={
                "type": "image_folder_dataset",
                "params": {"data_dir": "/path/to/images"}
            },
            pre_process_data_config={
                "type": "sd_lora_preprocess",
                "params": {
                    "enable_steps": ["auto_caption", "auto_tagging"],
                }
            }
        )

    Usage (SDXL):
        config = DataConfig(
            name="my_sdxl_lora",
            type="SDLoRADataContainer",
            pre_process_data_config={
                "type": "sd_lora_preprocess",
                "params": {
                    "base_resolution": 1024,
                    "bucket_mode": "sdxl",
                    "enable_steps": ["auto_caption"],
                }
            }
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "image_folder_dataset",
        DataComponentType.PRE_PROCESS_DATA.value: "sd_lora_preprocess",
        DataComponentType.POST_PROCESS_DATA.value: "default_post_process_data",
        DataComponentType.DATALOADER.value: "image_bucket_dataloader",
    }

    default_chain_fn: ClassVar = staticmethod(lambda: get_lora_default_chain(512, "auto"))

    def _build_chain(self, params: dict) -> PreProcessChain:
        """Build chain with support for base_resolution and bucket_mode params."""
        if "steps" in params:
            return super()._build_chain(params)

        # Extract resolution params
        base_resolution = params.pop("base_resolution", 512)
        bucket_mode = params.pop("bucket_mode", "auto" if base_resolution <= 512 else "sdxl")

        # Create chain with specified resolution
        chain = get_lora_default_chain(base_resolution, bucket_mode)

        # Apply the rest of the config
        enable_steps = params.get("enable_steps") or []
        for step_type in enable_steps:
            chain.enable_step(step_type)

        disable_steps = params.get("disable_steps") or []
        for step_type in disable_steps:
            chain.disable_step(step_type)

        step_params = params.get("step_params") or {}
        for step in chain.steps:
            if step.type in step_params:
                step.params.update(step_params[step.type])

        global_params = {k: v for k, v in params.items()
                        if k not in ("enable_steps", "disable_steps", "step_params")}
        for step in chain.steps:
            for k, v in global_params.items():
                if k not in step.params:
                    step.params[k] = v

        return chain


# ============================================================================
# DreamBooth Data Container
# ============================================================================

@Registry.register(DataContainerType.DATA_CONTAINER, name="DreamBoothDataContainer")
class DreamBoothDataContainer(ChainPreProcessMixin, DataContainer):
    """Data container for DreamBooth training.

    Default preprocessing chain:
    1. image_filtering (disabled) - Filter low quality images
    2. dreambooth_prompt (enabled) - Generate fixed prompts with identifier
    3. generate_class_images (disabled) - Generate regularization images
    4. image_resizing (enabled) - Resize to fixed size (512, center crop)

    Usage:
        config = DataConfig(
            name="my_dreambooth",
            type="DreamBoothDataContainer",
            load_dataset_config={
                "type": "image_folder_dataset",
                "params": {
                    "data_dir": "/path/to/instance_images",
                    "class_data_dir": "/path/to/class_images",
                }
            },
            pre_process_data_config={
                "type": "dreambooth_preprocess",
                "params": {
                    "step_params": {
                        "dreambooth_prompt": {
                            "identifier_token": "sks",
                            "class_token": "dog",
                        }
                    }
                }
            }
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "image_folder_dataset",
        DataComponentType.PRE_PROCESS_DATA.value: "dreambooth_preprocess",
        DataComponentType.POST_PROCESS_DATA.value: "default_post_process_data",
        DataComponentType.DATALOADER.value: "default_dataloader",  # DreamBooth uses fixed size
    }

    default_chain_fn: ClassVar = staticmethod(lambda: get_dreambooth_default_chain(512))

    def _build_chain(self, params: dict) -> PreProcessChain:
        """Build chain with support for target_resolution param."""
        if "steps" in params:
            return super()._build_chain(params)

        # Extract resolution param
        target_resolution = params.pop("target_resolution", 512)

        # Create chain with specified resolution
        chain = get_dreambooth_default_chain(target_resolution)

        # Apply the rest of the config
        enable_steps = params.get("enable_steps") or []
        for step_type in enable_steps:
            chain.enable_step(step_type)

        disable_steps = params.get("disable_steps") or []
        for step_type in disable_steps:
            chain.disable_step(step_type)

        step_params = params.get("step_params") or {}
        for step in chain.steps:
            if step.type in step_params:
                step.params.update(step_params[step.type])

        global_params = {k: v for k, v in params.items()
                        if k not in ("enable_steps", "disable_steps", "step_params")}
        for step in chain.steps:
            for k, v in global_params.items():
                if k not in step.params:
                    step.params[k] = v

        return chain
