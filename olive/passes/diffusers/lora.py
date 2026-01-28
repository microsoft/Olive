# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from olive.common.utils import StrEnumBase
from olive.constants import DiffusersModelVariant
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import DiffusersModelHandler
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pass_config import BasePassConfig

logger = logging.getLogger(__name__)


class MixedPrecision(StrEnumBase):
    """Mixed precision training mode."""

    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"


class LRSchedulerType(StrEnumBase):
    """Learning rate scheduler type."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class DiffusionTrainingArguments:
    """Training arguments for diffusion model LoRA fine-tuning."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        max_train_steps: int = 1000,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        gradient_checkpointing: bool = True,
        mixed_precision: Union[str, MixedPrecision] = MixedPrecision.BF16,
        lr_scheduler: Union[str, LRSchedulerType] = LRSchedulerType.CONSTANT,
        lr_warmup_steps: int = 0,
        snr_gamma: Optional[float] = None,
        max_grad_norm: float = 1.0,
        checkpointing_steps: int = 500,
        logging_steps: int = 10,
        seed: Optional[int] = None,
        # Flux-specific
        guidance_scale: float = 3.5,
    ):
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.snr_gamma = snr_gamma
        self.max_grad_norm = max_grad_norm
        self.checkpointing_steps = checkpointing_steps
        self.logging_steps = logging_steps
        self.seed = seed
        self.guidance_scale = guidance_scale


class SDLoRA(Pass):
    """Run LoRA fine-tuning on diffusion models.

    Supports:
    - Stable Diffusion 1.5: UNet-based, CLIP text encoder
    - Stable Diffusion XL: UNet-based, dual CLIP text encoders
    - Flux.1: DiT-based (Transformer), CLIP + T5 text encoders

    Trains LoRA adapters on the denoising model (UNet for SD, Transformer for Flux).
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            # Model config
            "model_variant": PassConfigParam(
                type_=DiffusersModelVariant,
                default_value=DiffusersModelVariant.AUTO,
                description="Model variant: 'sd15', 'sdxl', 'flux', or 'auto' to detect automatically.",
            ),
            # LoRA config
            "r": PassConfigParam(
                type_=int,
                default_value=16,
                description="LoRA rank. Flux typically needs higher rank (16-64) than SD (4-16).",
            ),
            "alpha": PassConfigParam(
                type_=float,
                default_value=None,
                description="LoRA alpha for scaling. Defaults to r if not specified.",
            ),
            "lora_dropout": PassConfigParam(
                type_=float,
                default_value=0.0,
                description="Dropout probability for LoRA layers.",
            ),
            "target_modules": PassConfigParam(
                type_=list[str],
                default_value=None,
                description=(
                    "Target modules for LoRA. Defaults depend on model variant:\n"
                    "- SD/SDXL: ['to_k', 'to_q', 'to_v', 'to_out.0']\n"
                    "- Flux: ['to_k', 'to_q', 'to_v', 'to_out.0', 'add_k_proj', 'add_q_proj', 'add_v_proj']"
                ),
            ),
            # Data config
            "train_data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                required=True,
                description="Data config for training dataset.",
            ),
            # Training config
            "training_args": PassConfigParam(
                type_=Union[DiffusionTrainingArguments, dict],
                default_value=None,
                description="Training arguments. See DiffusionTrainingArguments for options.",
            ),
            # Output config
            "merge_lora": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Merge LoRA weights into base model and save merged model. "
                    "If False, only saves LoRA adapter weights."
                ),
            ),
            # DreamBooth config
            "dreambooth": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Enable DreamBooth training for learning specific subjects. "
                    "Use instance_prompt to specify a fixed prompt for all images."
                ),
            ),
            "instance_prompt": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Fixed prompt for all training images in DreamBooth mode. "
                    "Required when dreambooth=True. Example: 'a photo of sks dog'."
                ),
            ),
            "with_prior_preservation": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Enable prior preservation to prevent language drift during DreamBooth training. "
                    "Requires class_prompt to be set."
                ),
            ),
            "class_prompt": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Prompt for generating class images in prior preservation. "
                    "Required when with_prior_preservation=True. Example: 'a photo of a dog'."
                ),
            ),
            "class_data_dir": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Directory containing class images for prior preservation. "
                    "If not provided or has fewer than num_class_images, images will be auto-generated."
                ),
            ),
            "num_class_images": PassConfigParam(
                type_=int,
                default_value=200,
                description="Number of class images for prior preservation.",
            ),
            "prior_loss_weight": PassConfigParam(
                type_=float,
                default_value=1.0,
                description="Weight of the prior preservation loss.",
            ),
        }

    def _run_for_config(
        self, model: DiffusersModelHandler, config: BasePassConfig, output_model_path: str
    ) -> DiffusersModelHandler:
        """Run diffusion model LoRA training."""
        # Validate DreamBooth config
        if config.dreambooth and not config.instance_prompt:
            raise ValueError("instance_prompt is required when dreambooth=True.")
        if config.with_prior_preservation:
            if not config.dreambooth:
                raise ValueError("with_prior_preservation requires dreambooth=True.")
            if not config.class_prompt:
                raise ValueError("class_prompt is required when with_prior_preservation=True.")

        # Initialize training args
        if config.training_args is None:
            training_args = DiffusionTrainingArguments()
        elif isinstance(config.training_args, dict):
            training_args = DiffusionTrainingArguments(**config.training_args)
        else:
            training_args = config.training_args

        # Detect model variant
        model_variant = self._detect_model_variant(model, config)
        logger.info("Detected model variant: %s", model_variant)

        # Route to appropriate training method
        if model_variant == DiffusersModelVariant.FLUX:
            return self._train_flux(model, config, training_args, output_model_path)
        else:
            return self._train_sd(model, config, training_args, model_variant, output_model_path)

    def _train_sd(
        self,
        model: DiffusersModelHandler,
        config: BasePassConfig,
        training_args: DiffusionTrainingArguments,
        model_variant: DiffusersModelVariant,
        output_model_path: str,
    ) -> DiffusersModelHandler:
        """Train LoRA for Stable Diffusion (SD1.5/SDXL)."""
        import torch
        import torch.nn.functional as F
        from accelerate import Accelerator
        from accelerate.utils import ProjectConfiguration, set_seed
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import compute_snr
        from peft import LoraConfig
        from tqdm.auto import tqdm
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        # Setup accelerator
        with tempfile.TemporaryDirectory(prefix="olive_sd_lora_") as temp_dir:
            project_config = ProjectConfiguration(
                project_dir=temp_dir,
                logging_dir=os.path.join(temp_dir, "logs"),
            )
            accelerator = Accelerator(
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                mixed_precision=training_args.mixed_precision,
                project_config=project_config,
            )

            if training_args.seed is not None:
                set_seed(training_args.seed)

            # Load models
            model_path = model.model_path
            logger.info("Loading SD models from %s", model_path)

            # Load text encoders (frozen, for encoding prompts only)
            text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
            text_encoder_2 = None
            if model_variant == DiffusersModelVariant.SDXL:
                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2")

            # Load VAE and UNet
            vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

            # Load noise scheduler
            noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

            # Set weight dtype
            weight_dtype = torch.float32
            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16

            # Freeze all base models and move to device
            vae.requires_grad_(False)
            vae.to(accelerator.device, dtype=weight_dtype)

            text_encoder.requires_grad_(False)
            text_encoder.to(accelerator.device, dtype=weight_dtype)

            if text_encoder_2:
                text_encoder_2.requires_grad_(False)
                text_encoder_2.to(accelerator.device, dtype=weight_dtype)

            unet.requires_grad_(False)
            unet.to(accelerator.device, dtype=weight_dtype)

            # Setup LoRA for UNet only (after moving to device)
            lora_alpha = config.alpha if config.alpha is not None else config.r
            target_modules = config.target_modules or ["to_k", "to_q", "to_v", "to_out.0"]

            unet_lora_config = LoraConfig(
                r=config.r,
                lora_alpha=lora_alpha,
                lora_dropout=config.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            unet.add_adapter(unet_lora_config)

            # LoRA trainable parameters should be fp32 for stable training with mixed precision
            for param in unet.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)

            # Log trainable parameters
            trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in unet.parameters())
            logger.info(
                "UNet trainable parameters: %d / %d (%.2f%%)",
                trainable_params,
                total_params,
                100 * trainable_params / total_params,
            )

            # Enable gradient checkpointing
            if training_args.gradient_checkpointing:
                unet.enable_gradient_checkpointing()

            # Prepare class images for prior preservation
            class_data_dir = None
            if config.with_prior_preservation:
                class_data_dir = self._prepare_class_images(
                    model_path=model_path,
                    class_prompt=config.class_prompt,
                    class_data_dir=config.class_data_dir,
                    num_class_images=config.num_class_images,
                    model_variant=model_variant,
                )

            # Load dataset
            train_dataset = self._load_dataset(config)
            train_dataloader = self._create_dataloader(
                train_dataset,
                training_args,
                model_path,
                model_variant,
                prior_preservation=config.with_prior_preservation,
                class_data_dir=class_data_dir,
                class_prompt=config.class_prompt,
                instance_prompt=config.instance_prompt,
            )

            # Calculate training steps
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
            num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

            # Setup optimizer (UNet only)
            params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-8,
            )

            # Setup LR scheduler
            lr_scheduler = get_scheduler(
                training_args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=training_args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=training_args.max_train_steps * accelerator.num_processes,
            )

            # Prepare with accelerator
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

            # Training loop
            logger.info("***** Running SD LoRA training *****")
            logger.info("  Num examples = %d", len(train_dataset))
            logger.info("  Num epochs = %d", num_train_epochs)
            logger.info("  Batch size = %d", training_args.train_batch_size)
            logger.info("  Gradient accumulation steps = %d", training_args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", training_args.max_train_steps)
            if config.dreambooth:
                logger.info("  Prior preservation enabled with weight = %.2f", config.prior_loss_weight)

            global_step = 0
            progress_bar = tqdm(
                range(training_args.max_train_steps),
                disable=not accelerator.is_local_main_process,
                desc="Training",
            )

            for _ in range(num_train_epochs):
                unet.train()

                for _, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(unet):
                        # Encode images to latents
                        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        # Sample noise and timesteps
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                        ).long()

                        # Add noise
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get text embeddings (frozen)
                        with torch.no_grad():
                            if model_variant == DiffusersModelVariant.SDXL:
                                encoder_hidden_states, pooled = self._encode_prompt_sdxl(
                                    batch, text_encoder, text_encoder_2
                                )
                                add_time_ids = self._compute_time_ids_batch(batch, latents.device, weight_dtype)
                                added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}
                            else:
                                encoder_hidden_states = self._encode_prompt_sd15(batch, text_encoder)
                                added_cond_kwargs = None

                        # UNet forward (with gradients for LoRA training)
                        # Cast to weight_dtype to match UNet's expected input dtype
                        noisy_latents = noisy_latents.to(dtype=weight_dtype)
                        encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)

                        if model_variant == DiffusersModelVariant.SDXL:
                            model_pred = unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]
                        else:
                            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                        # Compute loss target
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                        # Batch is [instance_images, class_images] concatenated
                        if config.with_prior_preservation:
                            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                            target, target_prior = torch.chunk(target, 2, dim=0)

                            # Instance loss
                            if training_args.snr_gamma is not None:
                                # SNR weighting only on instance portion
                                instance_timesteps = timesteps[: len(timesteps) // 2]
                                snr = compute_snr(noise_scheduler, instance_timesteps)
                                mse_loss_weights = torch.stack(
                                    [snr, torch.full_like(snr, training_args.snr_gamma)], dim=1
                                ).min(dim=1)[0]
                                if noise_scheduler.config.prediction_type == "v_prediction":
                                    mse_loss_weights = mse_loss_weights + 1
                                mse_loss_weights = mse_loss_weights / snr
                                instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                                instance_loss = (
                                    instance_loss.mean(dim=list(range(1, len(instance_loss.shape)))) * mse_loss_weights
                                )
                                instance_loss = instance_loss.mean()
                            else:
                                instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                            # Prior loss
                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                            # Combined loss
                            loss = instance_loss + config.prior_loss_weight * prior_loss
                        else:
                            # Standard LoRA training without prior preservation
                            if training_args.snr_gamma is not None:
                                snr = compute_snr(noise_scheduler, timesteps)
                                mse_loss_weights = torch.stack(
                                    [snr, torch.full_like(snr, training_args.snr_gamma)], dim=1
                                ).min(dim=1)[0]
                                if noise_scheduler.config.prediction_type == "v_prediction":
                                    mse_loss_weights = mse_loss_weights + 1
                                mse_loss_weights = mse_loss_weights / snr
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                                loss = loss.mean()
                            else:
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(params_to_optimize, training_args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        if accelerator.is_main_process:
                            if global_step % training_args.logging_steps == 0:
                                logger.info(
                                    "Step %d: loss=%.4f, lr=%.6f",
                                    global_step,
                                    loss.detach().item(),
                                    lr_scheduler.get_last_lr()[0],
                                )

                            if global_step % training_args.checkpointing_steps == 0:
                                save_path = os.path.join(temp_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)

                    if global_step >= training_args.max_train_steps:
                        break

            # Save final model
            accelerator.wait_for_everyone()

            # Save adapter weights
            adapter_path = Path(output_model_path) / "adapter"

            if accelerator.is_main_process:
                adapter_path.mkdir(parents=True, exist_ok=True)

                if config.merge_lora:
                    # Merge LoRA and save full UNet
                    unet_unwrapped = accelerator.unwrap_model(unet)
                    unet_merged = unet_unwrapped.merge_and_unload()
                    unet_merged.save_pretrained(adapter_path)
                    logger.info("Saved merged UNet to %s", adapter_path)
                else:
                    # Save LoRA adapter in diffusers-compatible format
                    unet_unwrapped = accelerator.unwrap_model(unet)
                    unet_unwrapped = unet_unwrapped.to(torch.float32)

                    from peft import get_peft_model_state_dict

                    unet_lora_state_dict = get_peft_model_state_dict(unet_unwrapped)

                    # Use appropriate pipeline class based on model variant
                    if model_variant == DiffusersModelVariant.SDXL:
                        from diffusers import StableDiffusionXLPipeline

                        StableDiffusionXLPipeline.save_lora_weights(
                            save_directory=str(adapter_path),
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )
                    else:
                        from diffusers import StableDiffusionPipeline

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=str(adapter_path),
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )
                    logger.info("Saved UNet LoRA to %s", adapter_path)

            accelerator.end_training()

        # Clean up
        del unet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return model handler
        output_model = deepcopy(model)
        output_model.set_resource("adapter_path", str(adapter_path))
        return output_model

    def _train_flux(
        self,
        model: DiffusersModelHandler,
        config: BasePassConfig,
        training_args: DiffusionTrainingArguments,
        output_model_path: str,
    ) -> DiffusersModelHandler:
        """Train LoRA for Flux models."""
        import torch
        import torch.nn.functional as F
        from accelerate import Accelerator
        from accelerate.utils import ProjectConfiguration, set_seed
        from diffusers import AutoencoderKL, FluxTransformer2DModel
        from diffusers.optimization import get_scheduler
        from peft import LoraConfig
        from tqdm.auto import tqdm
        from transformers import CLIPTextModel, T5EncoderModel

        # Flux requires bfloat16
        if training_args.mixed_precision == "fp16":
            logger.warning("Flux requires bfloat16, switching from fp16")
            training_args.mixed_precision = "bf16"

        with tempfile.TemporaryDirectory(prefix="olive_flux_lora_") as temp_dir:
            project_config = ProjectConfiguration(
                project_dir=temp_dir,
                logging_dir=os.path.join(temp_dir, "logs"),
            )
            accelerator = Accelerator(
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                mixed_precision=training_args.mixed_precision,
                project_config=project_config,
            )

            if training_args.seed is not None:
                set_seed(training_args.seed)

            model_path = model.model_path
            logger.info("Loading Flux models from %s", model_path)

            # Set weight dtype (Flux needs bfloat16)
            weight_dtype = torch.bfloat16

            # Load text encoders (frozen, for encoding prompts only)
            text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
            text_encoder_2 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2")

            # Load VAE and Transformer
            vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer")

            # Freeze all base models and move to device
            vae.requires_grad_(False)
            vae.to(accelerator.device, dtype=weight_dtype)

            text_encoder.requires_grad_(False)
            text_encoder.to(accelerator.device, dtype=weight_dtype)

            text_encoder_2.requires_grad_(False)
            text_encoder_2.to(accelerator.device, dtype=weight_dtype)

            transformer.requires_grad_(False)
            transformer.to(accelerator.device, dtype=weight_dtype)

            # Setup LoRA for transformer only (after moving to device)
            lora_alpha = config.alpha if config.alpha is not None else config.r
            target_modules = config.target_modules or [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
            ]

            transformer_lora_config = LoraConfig(
                r=config.r,
                lora_alpha=lora_alpha,
                lora_dropout=config.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            transformer.add_adapter(transformer_lora_config)

            # LoRA trainable parameters should be fp32 for stable training with mixed precision
            for param in transformer.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)

            # Log trainable parameters
            trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in transformer.parameters())
            logger.info(
                "Flux Transformer trainable parameters: %d / %d (%.2f%%)",
                trainable_params,
                total_params,
                100 * trainable_params / total_params,
            )

            # Enable gradient checkpointing
            if training_args.gradient_checkpointing:
                transformer.enable_gradient_checkpointing()

            # Prepare class images for prior preservation
            class_data_dir = None
            if config.with_prior_preservation:
                class_data_dir = self._prepare_class_images(
                    model_path=model_path,
                    class_prompt=config.class_prompt,
                    class_data_dir=config.class_data_dir,
                    num_class_images=config.num_class_images,
                    model_variant=DiffusersModelVariant.FLUX,
                )

            # Load dataset
            train_dataset = self._load_dataset(config)
            train_dataloader = self._create_dataloader(
                train_dataset,
                training_args,
                model_path,
                DiffusersModelVariant.FLUX,
                prior_preservation=config.with_prior_preservation,
                class_data_dir=class_data_dir,
                class_prompt=config.class_prompt,
                instance_prompt=config.instance_prompt,
            )

            # Calculate training steps
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
            num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

            # Setup optimizer (transformer only)
            params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-8,
            )

            # Setup LR scheduler
            lr_scheduler = get_scheduler(
                training_args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=training_args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=training_args.max_train_steps * accelerator.num_processes,
            )

            # Prepare with accelerator
            transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                transformer, optimizer, train_dataloader, lr_scheduler
            )

            # Training loop
            logger.info("***** Running Flux LoRA training *****")
            logger.info("  Num examples = %d", len(train_dataset))
            logger.info("  Num epochs = %d", num_train_epochs)
            logger.info("  Batch size = %d", training_args.train_batch_size)
            logger.info("  Gradient accumulation steps = %d", training_args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", training_args.max_train_steps)
            if config.dreambooth:
                logger.info("  Prior preservation enabled with weight = %.2f", config.prior_loss_weight)

            global_step = 0
            progress_bar = tqdm(
                range(training_args.max_train_steps),
                disable=not accelerator.is_local_main_process,
                desc="Training Flux",
            )

            for _ in range(num_train_epochs):
                transformer.train()

                for _, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(transformer):
                        # Encode images to latents
                        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                        # Save latent dimensions before packing (needed for image IDs)
                        batch_size, _channels, latent_height, latent_width = latents.shape

                        # Pack latents for Flux
                        latents = self._pack_latents(latents)

                        # Sample noise and timesteps (flow matching)
                        noise = torch.randn_like(latents)

                        # Flux uses continuous timesteps in [0, 1]
                        u = torch.rand(batch_size, device=latents.device, dtype=weight_dtype)
                        timesteps = (u * 1000).to(latents.device)

                        # Flow matching: sigmas = t
                        sigmas = self._get_sigmas(timesteps, latents.ndim, latents.dtype, latents.device)
                        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents

                        # Get text embeddings (frozen)
                        with torch.no_grad():
                            prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt_flux(
                                batch, text_encoder, text_encoder_2
                            )
                            # Get latent image IDs
                            latent_image_ids = self._prepare_latent_image_ids(
                                batch_size, latent_height // 2, latent_width // 2, latents.device, weight_dtype
                            )

                        # Transformer forward (with gradients for LoRA training)
                        model_pred = transformer(
                            hidden_states=noisy_latents,
                            timestep=timesteps / 1000,
                            guidance=torch.full(
                                (batch_size,), training_args.guidance_scale, device=latents.device, dtype=weight_dtype
                            ),
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

                        # Flow matching target: velocity = noise - data
                        target = noise - latents

                        # Prior preservation: split predictions and targets (official HuggingFace approach)
                        if config.with_prior_preservation:
                            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                            target, target_prior = torch.chunk(target, 2, dim=0)

                            # Instance loss
                            instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                            # Prior loss
                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                            # Combined loss
                            loss = instance_loss + config.prior_loss_weight * prior_loss
                        else:
                            # Standard LoRA training
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(params_to_optimize, training_args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        if accelerator.is_main_process:
                            if global_step % training_args.logging_steps == 0:
                                logger.info(
                                    "Step %d: loss=%.4f, lr=%.6f",
                                    global_step,
                                    loss.detach().item(),
                                    lr_scheduler.get_last_lr()[0],
                                )

                            if global_step % training_args.checkpointing_steps == 0:
                                save_path = os.path.join(temp_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)

                    if global_step >= training_args.max_train_steps:
                        break

            # Save final model
            accelerator.wait_for_everyone()
            output_path = Path(output_model_path)

            # Save adapter weights
            adapter_path = output_path / "adapter"

            if accelerator.is_main_process:
                adapter_path.mkdir(parents=True, exist_ok=True)

                if config.merge_lora:
                    # Merge LoRA and save full transformer
                    transformer_unwrapped = accelerator.unwrap_model(transformer)
                    transformer_merged = transformer_unwrapped.merge_and_unload()
                    transformer_merged.save_pretrained(adapter_path)
                    logger.info("Saved merged Transformer to %s", adapter_path)
                else:
                    # Save LoRA adapter in diffusers-compatible format
                    transformer_unwrapped = accelerator.unwrap_model(transformer)
                    transformer_unwrapped = transformer_unwrapped.to(torch.float32)

                    from diffusers import FluxPipeline
                    from peft import get_peft_model_state_dict

                    transformer_lora_state_dict = get_peft_model_state_dict(transformer_unwrapped)

                    FluxPipeline.save_lora_weights(
                        save_directory=str(adapter_path),
                        transformer_lora_layers=transformer_lora_state_dict,
                        safe_serialization=True,
                    )
                    logger.info("Saved Flux Transformer LoRA to %s", adapter_path)

            accelerator.end_training()

        # Clean up
        del transformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return model handler
        output_model = deepcopy(model)
        output_model.set_resource("adapter_path", str(adapter_path))
        return output_model

    def _detect_model_variant(self, model: DiffusersModelHandler, config: BasePassConfig) -> DiffusersModelVariant:
        """Detect the model variant."""
        if config.model_variant != DiffusersModelVariant.AUTO:
            return config.model_variant

        return model.detected_model_variant

    def _prepare_class_images(
        self,
        model_path: str,
        class_prompt: str,
        class_data_dir: Optional[str],
        num_class_images: int,
        model_variant: "DiffusersModelVariant",
    ) -> Path:
        """Generate class images for prior preservation if needed.

        Args:
            model_path: Path to the pretrained diffusion model.
            class_prompt: Prompt for generating class images.
            class_data_dir: Directory to store class images. If None, uses Olive cache.
            num_class_images: Target number of class images.
            model_variant: Type of diffusion model.

        Returns:
            Path to directory containing class images.

        """
        import hashlib

        import torch

        from olive.cache import OliveCache

        # Determine output directory
        if class_data_dir:
            output_dir = Path(class_data_dir)
        else:
            cache = OliveCache.from_cache_env()
            prompt_hash = hashlib.md5(class_prompt.encode()).hexdigest()[:8]
            output_dir = cache.get_cache_dir() / "class_images" / prompt_hash

        output_dir.mkdir(parents=True, exist_ok=True)

        # Count existing images
        existing_images = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
        num_existing = len(existing_images)

        if num_existing >= num_class_images:
            logger.info("Found %d existing class images in %s", num_existing, output_dir)
            return output_dir

        # Generate missing images
        num_to_generate = num_class_images - num_existing
        logger.info("Generating %d class images with prompt: '%s'", num_to_generate, class_prompt)

        # Load appropriate pipeline
        if model_variant == DiffusersModelVariant.FLUX:
            from diffusers import FluxPipeline

            pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        elif model_variant == DiffusersModelVariant.SDXL:
            from diffusers import StableDiffusionXLPipeline

            pipeline = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            from diffusers import StableDiffusionPipeline

            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )

        pipeline = pipeline.to("cuda")

        # Generate images in batches
        batch_size = 4
        for i in range(0, num_to_generate, batch_size):
            current_batch = min(batch_size, num_to_generate - i)
            images = pipeline(
                [class_prompt] * current_batch,
                num_inference_steps=50,
            ).images

            for j, img in enumerate(images):
                img_path = output_dir / f"class_{num_existing + i + j:04d}.png"
                img.save(img_path)

            logger.info("Generated %d/%d class images", min(i + batch_size, num_to_generate), num_to_generate)

        # Clean up pipeline to free memory
        del pipeline
        torch.cuda.empty_cache()

        return output_dir

    def _load_dataset(self, config):
        """Load training dataset.

        Returns the dataset with bucket_assignments preserved for SDXL time embeddings.
        """
        data_config = config.train_data_config
        if isinstance(data_config, dict):
            data_config = DataConfig(**data_config)

        # Load and preprocess dataset through container
        data_container = data_config.to_data_container()
        return data_container.pre_process(data_container.load_dataset())

    def _create_dataloader(
        self,
        dataset,
        training_args,
        model_path,
        model_variant,
        prior_preservation=False,
        class_data_dir=None,
        class_prompt=None,
        instance_prompt=None,
    ):
        """Create training dataloader with image loading and tokenization.

        Args:
            dataset: Dataset with image_path, caption, and bucket_assignments.
            training_args: Training arguments.
            model_path: Path to the diffusion model for loading tokenizers.
            model_variant: Type of diffusion model (SD, SDXL, FLUX).
            prior_preservation: Whether to include class images for prior preservation.
            class_data_dir: Directory containing class images (for prior preservation).
            class_prompt: Prompt for class images (for prior preservation).
            instance_prompt: Fixed prompt for all instance images (for DreamBooth).

        Raises:
            ValueError: If dataset has not been preprocessed with aspect_ratio_bucketing or image_resizing.

        """
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoTokenizer

        # Validate that dataset has been preprocessed
        bucket_assignments = getattr(dataset, "bucket_assignments", None)
        if not bucket_assignments:
            raise ValueError(
                "Dataset must be preprocessed with 'aspect_ratio_bucketing' or 'image_resizing'. "
                "Please configure train_data_config with appropriate preprocessing steps. "
                "Example: {'pre_process_config': {'name': 'aspect_ratio_bucketing', 'params': {'base_resolution': 1024}}}"
            )

        # Load tokenizers based on model variant
        tokenizers = {}
        if model_variant == DiffusersModelVariant.FLUX:
            tokenizers["clip"] = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            tokenizers["t5"] = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        elif model_variant == DiffusersModelVariant.SDXL:
            tokenizers["one"] = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            tokenizers["two"] = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        else:  # SD
            tokenizers["main"] = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        def process_image(image_path):
            """Load and process a single image.

            Raises:
                ValueError: If image_path is not found in bucket_assignments.

            """
            if image_path not in bucket_assignments:
                raise ValueError(
                    f"Image '{image_path}' not found in bucket_assignments. "
                    "All images (including class images) must be preprocessed."
                )

            assignment = bucket_assignments[image_path]
            original_size = assignment["original_size"]
            bucket_size = assignment["bucket"]
            crops_coords = assignment["crops_coords_top_left"]

            # Load image - should already be resized during preprocessing
            img = Image.open(image_path).convert("RGB")

            # Verify image size matches bucket assignment
            bucket_w, bucket_h = bucket_size
            if img.size != (bucket_w, bucket_h):
                raise ValueError(
                    f"Image '{image_path}' size {img.size} does not match bucket assignment {bucket_size}. "
                    "Please ensure images are resized during preprocessing (set resize_images=True)."
                )

            img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0
            pixel_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))

            return pixel_tensor, original_size, bucket_size, crops_coords

        def tokenize_captions(captions_list):
            """Tokenize a list of captions."""
            result = {}
            for name, tok in tokenizers.items():
                max_len = 512 if name == "t5" else 77
                tokens = tok(
                    captions_list, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
                )
                key_map = {
                    "main": "input_ids",
                    "one": "input_ids_one",
                    "two": "input_ids_two",
                    "clip": "input_ids_one",
                    "t5": "input_ids_t5",
                }
                result[key_map[name]] = tokens.input_ids
            return result

        # Load class images list if prior preservation is enabled
        class_images_list = []
        if prior_preservation and class_data_dir:
            class_data_path = Path(class_data_dir)
            class_images_list = sorted(list(class_data_path.glob("*.png")) + list(class_data_path.glob("*.jpg")))
            if not class_images_list:
                raise ValueError(f"No class images found in {class_data_dir}")

        class_image_idx = [0]  # Use list to allow mutation in closure

        def collate_fn(examples):
            # Instance images
            pixel_values = []
            captions = []
            original_sizes = []
            target_sizes = []
            crops_coords = []

            for ex in examples:
                # Process instance image
                image_path = ex.get("image_path", "")
                pixel_tensor, orig_size, tgt_size, crop_coords = process_image(image_path)
                pixel_values.append(pixel_tensor)
                original_sizes.append(orig_size)
                target_sizes.append(tgt_size)
                crops_coords.append(crop_coords)
                # For DreamBooth: use instance_prompt for all images
                # For standard LoRA: use dataset captions
                if instance_prompt:
                    captions.append(instance_prompt)
                else:
                    captions.append(ex.get("caption", ""))

            # For DreamBooth with prior preservation: concatenate class images after instance images
            # This follows the official HuggingFace approach - single forward pass, then torch.chunk
            if prior_preservation and class_images_list:
                for _ in examples:
                    # Get class image (cycle through available images)
                    class_img_path = class_images_list[class_image_idx[0] % len(class_images_list)]
                    class_image_idx[0] += 1

                    # Load and process class image to match instance image size
                    img = Image.open(class_img_path).convert("RGB")
                    # Resize to match the first instance image's target size
                    target_w, target_h = target_sizes[0]
                    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

                    img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0
                    pixel_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))

                    pixel_values.append(pixel_tensor)
                    original_sizes.append((img.width, img.height))
                    target_sizes.append((target_w, target_h))
                    crops_coords.append((0, 0))
                    captions.append(class_prompt or "")

            result = {"pixel_values": torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()}

            # Tokenize all captions (instance + class)
            tokens = tokenize_captions(captions)
            result.update(tokens)

            # Add size info for SDXL
            if model_variant == DiffusersModelVariant.SDXL:
                result["original_sizes"] = original_sizes
                result["target_sizes"] = target_sizes
                result["crops_coords_top_left"] = crops_coords

            return result

        # Use bucket batch sampler to ensure images in each batch have the same dimensions
        from olive.data.component.sd_lora.dataloader import BucketBatchSampler

        batch_sampler = BucketBatchSampler(
            dataset,
            batch_size=training_args.train_batch_size,
            drop_last=False,
            shuffle=True,
            seed=training_args.seed,
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    def _encode_prompt_sd15(self, batch, text_encoder):
        """Encode prompts for SD 1.5 (frozen text encoder)."""
        input_ids = batch["input_ids"].to(text_encoder.device)
        return text_encoder(input_ids, return_dict=False)[0]

    def _encode_prompt_sdxl(self, batch, text_encoder, text_encoder_2):
        """Encode prompts for SDXL (frozen text encoders)."""
        import torch

        input_ids_one = batch["input_ids_one"].to(text_encoder.device)
        input_ids_two = batch["input_ids_two"].to(text_encoder_2.device)

        encoder_hidden_states = text_encoder(input_ids_one, output_hidden_states=True).hidden_states[-2]
        encoder_output_2 = text_encoder_2(input_ids_two, output_hidden_states=True)
        pooled = encoder_output_2[0]
        encoder_hidden_states_2 = encoder_output_2.hidden_states[-2]

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)
        return encoder_hidden_states, pooled

    def _encode_prompt_flux(self, batch, text_encoder, text_encoder_2):
        """Encode prompts for Flux (frozen text encoders)."""
        import torch

        input_ids_clip = batch["input_ids_one"].to(text_encoder.device)
        input_ids_t5 = batch.get("input_ids_t5", batch.get("input_ids_two")).to(text_encoder_2.device)

        # CLIP encoder for pooled embeddings
        clip_output = text_encoder(input_ids_clip, output_hidden_states=True)
        pooled_prompt_embeds = clip_output.pooler_output

        # T5 encoder for main text embeddings
        t5_output = text_encoder_2(input_ids_t5)
        prompt_embeds = t5_output[0]

        # Create text IDs
        text_ids = torch.zeros(prompt_embeds.shape[0], prompt_embeds.shape[1], 3, device=prompt_embeds.device)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _compute_time_ids_batch(self, batch, device, dtype):
        """Compute SDXL time IDs for a batch."""
        import torch

        add_time_ids_list = []
        for i in range(len(batch["original_sizes"])):
            original_size = batch["original_sizes"][i]
            target_size = batch["target_sizes"][i]
            crops_coords = batch["crops_coords_top_left"][i]
            add_time_ids = list(original_size + crops_coords + target_size)
            add_time_ids_list.append(add_time_ids)

        return torch.tensor(add_time_ids_list, dtype=dtype, device=device)

    def _pack_latents(self, latents):
        """Pack latents for Flux (reshape to sequence format)."""
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)

    def _get_sigmas(self, timesteps, n_dim, dtype, device):
        """Get sigmas for flow matching."""
        sigmas = timesteps / 1000.0
        while len(sigmas.shape) < n_dim:
            sigmas = sigmas.unsqueeze(-1)
        return sigmas.to(dtype=dtype, device=device)

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        """Prepare latent image IDs for Flux."""
        import torch

        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids.unsqueeze(0).expand(batch_size, -1, -1)
