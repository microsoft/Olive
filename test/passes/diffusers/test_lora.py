# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import torch

from olive.constants import DiffusersModelVariant
from olive.passes.diffusers.lora import SDLoRA
from olive.passes.olive_pass import create_pass_from_dict

# Constants
SD15_SCALING_FACTOR = 0.18215
SDXL_SCALING_FACTOR = 0.13025
DEFAULT_TRAINING_ARGS = {"max_train_steps": 1, "train_batch_size": 1}


def get_pass_config(data_dir, **kwargs):
    return {
        "train_data_config": {
            "name": "test_data",
            "type": "ImageDataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": data_dir},
            },
        },
        **kwargs,
    }


def setup_tokenizer_mock(mock_auto_tokenizer):
    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 77
    mock_token_output = MagicMock()
    mock_token_output.input_ids = torch.ones(1, 77, dtype=torch.long)
    mock_tokenizer.return_value = mock_token_output
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    return mock_tokenizer


def setup_sd_mocks(
    mock_unet,
    mock_vae,
    mock_clip,
    mock_scheduler,
    mock_get_peft_model,
    mock_torch_model,
    scaling_factor=SD15_SCALING_FACTOR,
):
    # UNet
    mock_unet.from_pretrained.return_value = mock_torch_model
    mock_get_peft_model.return_value = mock_torch_model

    # VAE
    mock_vae_model = MagicMock()
    mock_vae_model.config = MagicMock()
    mock_vae_model.config.scaling_factor = scaling_factor
    mock_vae_model.requires_grad_ = MagicMock(return_value=mock_vae_model)
    mock_vae_model.to = MagicMock(return_value=mock_vae_model)
    mock_latent_dist = MagicMock()
    mock_latent_dist.sample.return_value = torch.randn(1, 4, 8, 8)
    mock_vae_model.encode.return_value = mock_latent_dist
    mock_vae.from_pretrained.return_value = mock_vae_model

    # CLIP
    mock_clip_model = MagicMock()
    mock_clip_model.return_value = (torch.randn(1, 77, 768),)
    mock_clip_model.requires_grad_ = MagicMock(return_value=mock_clip_model)
    mock_clip_model.to = MagicMock(return_value=mock_clip_model)
    mock_clip.from_pretrained.return_value = mock_clip_model

    # Noise scheduler
    mock_noise_scheduler = MagicMock()
    mock_noise_scheduler.config.num_train_timesteps = 1000
    mock_noise_scheduler.config.prediction_type = "epsilon"
    mock_noise_scheduler.add_noise.return_value = torch.randn(1, 4, 8, 8)
    mock_scheduler.from_pretrained.return_value = mock_noise_scheduler

    return mock_vae_model, mock_clip_model


@patch("diffusers.StableDiffusionPipeline.save_lora_weights")
@patch("peft.get_peft_model")
@patch("peft.LoraConfig")
@patch("diffusers.DDPMScheduler")
@patch("diffusers.UNet2DConditionModel")
@patch("diffusers.AutoencoderKL")
@patch("transformers.CLIPTextModel")
@patch("transformers.AutoTokenizer")
@patch("accelerate.Accelerator")
@patch("diffusers.optimization.get_scheduler")
def test_sd_lora_train_sd15(
    mock_get_scheduler,
    mock_accelerator_cls,
    mock_auto_tokenizer,
    mock_clip,
    mock_vae,
    mock_unet,
    mock_scheduler,
    mock_lora_config,
    mock_get_peft_model,
    mock_save_lora,
    test_image_folder,
    output_folder,
    mock_accelerator,
    mock_torch_model,
    mock_input_model_sd15,
):
    mock_accelerator_cls.return_value = mock_accelerator
    mock_get_scheduler.return_value = MagicMock()
    setup_tokenizer_mock(mock_auto_tokenizer)
    setup_sd_mocks(mock_unet, mock_vae, mock_clip, mock_scheduler, mock_get_peft_model, mock_torch_model)

    config = get_pass_config(
        test_image_folder,
        model_variant=DiffusersModelVariant.SD,
        training_args=DEFAULT_TRAINING_ARGS,
    )
    p = create_pass_from_dict(SDLoRA, config, disable_search=True)
    result = p.run(mock_input_model_sd15, output_folder)

    assert result is not None
    mock_unet.from_pretrained.assert_called_once()
    mock_vae.from_pretrained.assert_called_once()
    mock_clip.from_pretrained.assert_called_once()


@patch("peft.get_peft_model")
@patch("peft.LoraConfig")
@patch("diffusers.DDPMScheduler")
@patch("diffusers.UNet2DConditionModel")
@patch("diffusers.AutoencoderKL")
@patch("transformers.CLIPTextModel")
@patch("transformers.AutoTokenizer")
@patch("accelerate.Accelerator")
@patch("diffusers.optimization.get_scheduler")
def test_sd_lora_merge_lora(
    mock_get_scheduler,
    mock_accelerator_cls,
    mock_auto_tokenizer,
    mock_clip,
    mock_vae,
    mock_unet,
    mock_scheduler,
    mock_lora_config,
    mock_get_peft_model,
    test_image_folder,
    output_folder,
    mock_accelerator,
    mock_torch_model,
    mock_input_model_sd15,
):
    mock_accelerator_cls.return_value = mock_accelerator
    mock_get_scheduler.return_value = MagicMock()
    setup_tokenizer_mock(mock_auto_tokenizer)

    # Add merge-specific mocks
    mock_torch_model.merge_and_unload = MagicMock(return_value=mock_torch_model)
    mock_torch_model.save_pretrained = MagicMock()

    setup_sd_mocks(mock_unet, mock_vae, mock_clip, mock_scheduler, mock_get_peft_model, mock_torch_model)

    config = get_pass_config(
        test_image_folder,
        model_variant=DiffusersModelVariant.SD,
        merge_lora=True,
        training_args=DEFAULT_TRAINING_ARGS,
    )
    p = create_pass_from_dict(SDLoRA, config, disable_search=True)
    result = p.run(mock_input_model_sd15, output_folder)

    assert result is not None
    mock_torch_model.merge_and_unload.assert_called_once()
    mock_torch_model.save_pretrained.assert_called_once()


@patch("diffusers.StableDiffusionXLPipeline.save_lora_weights")
@patch("peft.get_peft_model")
@patch("peft.LoraConfig")
@patch("diffusers.DDPMScheduler")
@patch("diffusers.UNet2DConditionModel")
@patch("diffusers.AutoencoderKL")
@patch("transformers.CLIPTextModel")
@patch("transformers.CLIPTextModelWithProjection")
@patch("transformers.AutoTokenizer")
@patch("accelerate.Accelerator")
@patch("diffusers.optimization.get_scheduler")
def test_sd_lora_train_sdxl(
    mock_get_scheduler,
    mock_accelerator_cls,
    mock_auto_tokenizer,
    mock_clip_proj,
    mock_clip,
    mock_vae,
    mock_unet,
    mock_scheduler,
    mock_lora_config,
    mock_get_peft_model,
    mock_save_lora,
    test_image_folder,
    output_folder,
    mock_accelerator,
    mock_torch_model,
    mock_input_model_sdxl,
):
    mock_accelerator_cls.return_value = mock_accelerator
    mock_get_scheduler.return_value = MagicMock()
    setup_tokenizer_mock(mock_auto_tokenizer)
    setup_sd_mocks(
        mock_unet, mock_vae, mock_clip, mock_scheduler, mock_get_peft_model, mock_torch_model, SDXL_SCALING_FACTOR
    )

    # SDXL second text encoder
    mock_clip_proj_model = MagicMock()
    mock_clip_proj_model.return_value = (torch.randn(1, 77, 1280),)
    mock_clip_proj_model.requires_grad_ = MagicMock(return_value=mock_clip_proj_model)
    mock_clip_proj_model.to = MagicMock(return_value=mock_clip_proj_model)
    mock_clip_proj.from_pretrained.return_value = mock_clip_proj_model

    config = get_pass_config(
        test_image_folder,
        model_variant=DiffusersModelVariant.SDXL,
        training_args=DEFAULT_TRAINING_ARGS,
    )
    p = create_pass_from_dict(SDLoRA, config, disable_search=True)
    result = p.run(mock_input_model_sdxl, output_folder)

    assert result is not None
    mock_unet.from_pretrained.assert_called_once()
    mock_clip.from_pretrained.assert_called_once()
    mock_clip_proj.from_pretrained.assert_called_once()


@patch("diffusers.FluxPipeline.save_lora_weights")
@patch("peft.LoraConfig")
@patch("diffusers.FluxTransformer2DModel")
@patch("diffusers.AutoencoderKL")
@patch("transformers.CLIPTextModel")
@patch("transformers.T5EncoderModel")
@patch("transformers.AutoTokenizer")
@patch("accelerate.Accelerator")
@patch("diffusers.optimization.get_scheduler")
def test_sd_lora_train_flux(
    mock_get_scheduler,
    mock_accelerator_cls,
    mock_auto_tokenizer,
    mock_t5,
    mock_clip,
    mock_vae,
    mock_transformer,
    mock_lora_config,
    mock_save_lora,
    test_image_folder,
    output_folder,
    mock_accelerator,
    mock_torch_model,
    mock_input_model_flux,
):
    mock_accelerator_cls.return_value = mock_accelerator
    mock_get_scheduler.return_value = MagicMock()

    # Tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 77
    mock_token_output = MagicMock()
    mock_token_output.input_ids = torch.ones(1, 77, dtype=torch.long)
    mock_tokenizer.return_value = mock_token_output
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    # Transformer
    mock_torch_model.add_adapter = MagicMock()
    mock_torch_model.enable_gradient_checkpointing = MagicMock()
    mock_transformer.from_pretrained.return_value = mock_torch_model

    # VAE for Flux
    mock_vae_model = MagicMock()
    mock_vae_model.config = MagicMock()
    mock_vae_model.config.scaling_factor = SDXL_SCALING_FACTOR
    mock_vae_model.requires_grad_ = MagicMock(return_value=mock_vae_model)
    mock_vae_model.to = MagicMock(return_value=mock_vae_model)
    mock_latent_dist = MagicMock()
    mock_latent_dist.latent_dist.sample.return_value = torch.randn(1, 16, 8, 8)
    mock_vae_model.encode.return_value = mock_latent_dist
    mock_vae.from_pretrained.return_value = mock_vae_model

    # CLIP
    mock_clip_model = MagicMock()
    mock_clip_model.return_value = MagicMock(pooler_output=torch.randn(1, 768))
    mock_clip_model.requires_grad_ = MagicMock(return_value=mock_clip_model)
    mock_clip_model.to = MagicMock(return_value=mock_clip_model)
    mock_clip.from_pretrained.return_value = mock_clip_model

    # T5
    mock_t5_model = MagicMock()
    mock_t5_model.return_value = (torch.randn(1, 512, 4096),)
    mock_t5_model.requires_grad_ = MagicMock(return_value=mock_t5_model)
    mock_t5_model.to = MagicMock(return_value=mock_t5_model)
    mock_t5.from_pretrained.return_value = mock_t5_model

    config = get_pass_config(
        test_image_folder,
        model_variant=DiffusersModelVariant.FLUX,
        training_args=DEFAULT_TRAINING_ARGS,
    )
    p = create_pass_from_dict(SDLoRA, config, disable_search=True)
    result = p.run(mock_input_model_flux, output_folder)

    assert result is not None
    mock_transformer.from_pretrained.assert_called_once()
    mock_clip.from_pretrained.assert_called_once()
    mock_t5.from_pretrained.assert_called_once()


@patch("diffusers.StableDiffusionPipeline.save_lora_weights")
@patch("peft.get_peft_model")
@patch("peft.LoraConfig")
@patch("diffusers.DDPMScheduler")
@patch("diffusers.UNet2DConditionModel")
@patch("diffusers.AutoencoderKL")
@patch("transformers.CLIPTextModel")
@patch("transformers.AutoTokenizer")
@patch("accelerate.Accelerator")
@patch("diffusers.optimization.get_scheduler")
def test_sd_lora_dreambooth_sd15(
    mock_get_scheduler,
    mock_accelerator_cls,
    mock_auto_tokenizer,
    mock_clip,
    mock_vae,
    mock_unet,
    mock_scheduler,
    mock_lora_config,
    mock_get_peft_model,
    mock_save_lora,
    test_image_folder,
    output_folder,
    mock_accelerator,
    mock_torch_model,
    mock_input_model_sd15,
):
    mock_accelerator_cls.return_value = mock_accelerator
    mock_get_scheduler.return_value = MagicMock()
    setup_tokenizer_mock(mock_auto_tokenizer)

    # DreamBooth needs batch size 2 tensors
    mock_torch_model.return_value = (torch.randn(2, 4, 8, 8),)
    mock_unet.from_pretrained.return_value = mock_torch_model
    mock_get_peft_model.return_value = mock_torch_model

    # VAE
    mock_vae_model = MagicMock()
    mock_vae_model.config = MagicMock()
    mock_vae_model.config.scaling_factor = SD15_SCALING_FACTOR
    mock_vae_model.requires_grad_ = MagicMock(return_value=mock_vae_model)
    mock_vae_model.to = MagicMock(return_value=mock_vae_model)
    mock_latent_dist = MagicMock()
    mock_latent_dist.sample.return_value = torch.randn(2, 4, 8, 8)
    mock_vae_model.encode.return_value = mock_latent_dist
    mock_vae.from_pretrained.return_value = mock_vae_model

    # CLIP
    mock_clip_model = MagicMock()
    mock_clip_model.return_value = (torch.randn(2, 77, 768),)
    mock_clip_model.requires_grad_ = MagicMock(return_value=mock_clip_model)
    mock_clip_model.to = MagicMock(return_value=mock_clip_model)
    mock_clip.from_pretrained.return_value = mock_clip_model

    # Noise scheduler
    mock_noise_scheduler = MagicMock()
    mock_noise_scheduler.config.num_train_timesteps = 1000
    mock_noise_scheduler.config.prediction_type = "epsilon"
    mock_noise_scheduler.add_noise.return_value = torch.randn(2, 4, 8, 8)
    mock_scheduler.from_pretrained.return_value = mock_noise_scheduler

    config = get_pass_config(
        test_image_folder,
        model_variant=DiffusersModelVariant.SD,
        dreambooth=True,
        instance_prompt="a photo of sks dog",
        prior_loss_weight=1.0,
        training_args=DEFAULT_TRAINING_ARGS,
    )
    p = create_pass_from_dict(SDLoRA, config, disable_search=True)
    result = p.run(mock_input_model_sd15, output_folder)

    assert result is not None
