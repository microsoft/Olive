# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import inspect
import os
from typing import Callable, Optional, Union

import numpy as np
import onnxruntime as ort
import sd_utils
import sd_utils.config
import torch
from diffusers import OnnxStableDiffusionPipeline
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

# ruff: noqa: T201


def update_qdq_config(config: dict, provider: str, submodel_name: str):
    used_passes = {}
    if sd_utils.config.only_conversion:
        used_passes = {"convert"}
        config["evaluator"] = None
    elif submodel_name == "text_encoder":
        used_passes = {"convert", "dynamic_shape_to_fixed", "surgery", "optimize_qdq", "quantization"}
    elif submodel_name == "unet":
        used_passes = {"convert", "dynamic_shape_to_fixed", "optimize_qdq", "quantization"}
    else:
        used_passes = {"convert", "dynamic_shape_to_fixed", "quantization"}

    for pass_name in set(config["passes"].keys()):
        if pass_name not in used_passes:
            config["passes"].pop(pass_name, None)

    if provider == "cuda":
        config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = ["CUDAExecutionProvider"]
    elif provider == "qnn":
        config["systems"]["local_system"]["accelerators"][0]["device"] = "npu"
        config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = ["QNNExecutionProvider"]
    else:
        config["systems"]["local_system"]["accelerators"][0]["device"] = "cpu"
        config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = ["CPUExecutionProvider"]
        # not meaningful to evaluate QDQ latency on CPU
        config["evaluator"] = None
    return config


class OnnxStableDiffusionPipelineWithSave(OnnxStableDiffusionPipeline):
    def __call__(
        self,
        prompt: Union[str, list[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, list[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        # check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.save_data_dir:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np",
            ).input_ids.astype(np.int32)
            np.savez(self.save_data_dir / "text_inputs.npz", input_ids=text_inputs)

            uncond_input = self.tokenizer(
                negative_prompt if negative_prompt else "",
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np",
            ).input_ids.astype(np.int32)
            np.savez(self.save_data_dir / "uncond_input.npz", input_ids=uncond_input)

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (unet_input.type for unet_input in self.unet.model.get_inputs() if unet_input.name == "timestep"),
            "tensor(float)",
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        if do_classifier_free_guidance:
            splits = np.split(prompt_embeds, 2)
            neg_embeds, text_embeds = splits[0], splits[1]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)

            if do_classifier_free_guidance:
                # Note that in QDQ, we need to use static dimensions (batch is fixed to 1), so we need to split
                unet_input = {"sample": latent_model_input, "timestep": timestep, "encoder_hidden_states": neg_embeds}
                if self.save_data_dir:
                    np.savez(self.save_data_dir / f"{i}_unet_input_neg.npz", **unet_input)
                noise_pred_uncond = self.unet(**unet_input)
                noise_pred_uncond = noise_pred_uncond[0]

                unet_input = {"sample": latent_model_input, "timestep": timestep, "encoder_hidden_states": text_embeds}
                if self.save_data_dir:
                    np.savez(self.save_data_dir / f"{i}_unet_input.npz", **unet_input)
                noise_pred_text = self.unet(**unet_input)
                noise_pred_text = noise_pred_text[0]

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                unet_input = {
                    "sample": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                }
                if self.save_data_dir:
                    np.savez(self.save_data_dir / f"{i}_unet_input.npz", **unet_input)
                noise_pred = self.unet(**unet_input)
                noise_pred = noise_pred[0]

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        if self.save_data_dir:
            np.savez(self.save_data_dir / "latent.npz", latent_sample=latents[0:1])
        image = np.concatenate([self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])])
        if self.save_data_dir:
            np.savez(self.save_data_dir / "output_img.npz", sample=image)

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="np"
            ).pixel_values.astype(image.dtype)

            images, has_nsfw_concept = [], []
            for i in range(image.shape[0]):
                image_i, has_nsfw_concept_i = self.safety_checker(
                    clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                )
                images.append(image_i)
                has_nsfw_concept.append(has_nsfw_concept_i[0])
            image = np.concatenate(images)
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def get_qdq_pipeline(model_dir, common_args, qdq_args, script_dir):
    ort.set_default_logger_severity(3)

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()
    provider_options = [{}]

    provider = common_args.provider

    provider_map = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "qnn": "QNNExecutionProvider",
    }
    assert provider in provider_map, f"Unsupported provider: {provider}"

    if provider == "qnn":
        provider_options[0]["backend_path"] = "QnnHtp.dll"

    pipeline = OnnxStableDiffusionPipelineWithSave.from_pretrained(
        model_dir, provider=provider_map[provider], sess_options=sess_options, provider_options=provider_options
    )
    if qdq_args.save_data:
        pipeline.save_data_dir = script_dir / qdq_args.data_dir / common_args.prompt
        os.makedirs(pipeline.save_data_dir, exist_ok=True)
    else:
        pipeline.save_data_dir = None
    return pipeline
