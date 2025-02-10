# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import inspect
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from diffusers import OnnxStableDiffusionPipeline
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


def update_qnn_config(config: Dict, submodel_name: str):
    # TODO onnx or onnxruntime needs to fix this
    if submodel_name == "unet":
        config["input_model"]["io_config"]["dynamic_axes"] = None
        config["pass_flows"] = [["convert", "qnn_preprocess", "quantization"]]
    else:
        config["pass_flows"] = [["convert", "dynamic_shape_to_fixed", "qnn_preprocess", "quantization"]]
    config["systems"]["local_system"]["accelerators"][0]["device"] = "npu"
    config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = ["QNNExecutionProvider"]
    config["evaluator"] = None
    return config


class QnnStableDiffusionPipeline(OnnxStableDiffusionPipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
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
            text_inputs.tofile(self.save_data_dir / "text_inputs.raw")

            uncond_input = self.tokenizer(
                negative_prompt if negative_prompt else "",
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np",
            ).input_ids.astype(np.int32)
            uncond_input.tofile(self.save_data_dir / "uncond_input.raw")

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
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        if do_classifier_free_guidance:
            neg_embeds, text_embeds = np.split(prompt_embeds, 2)
            if self.save_data_dir:
                neg_embeds.tofile(self.save_data_dir / "neg_embeds.raw")
                text_embeds.tofile(self.save_data_dir / "text_embeds.raw")
        elif self.save_data_dir:
            prompt_embeds.tofile(self.save_data_dir / "text_embeds.raw")

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)

            if self.save_data_dir:
                latent_model_input.tofile(self.save_data_dir / f"{i}_latent.raw")
                timestep.tofile(self.save_data_dir / f"{i}_timestep.raw")

            if do_classifier_free_guidance:
                # Note that in QNN, we need to use static dimensions (batch is fixed to 1), so we need to split
                noise_pred_uncond = self.unet(
                    sample=latent_model_input, timestep=timestep, encoder_hidden_states=neg_embeds
                )
                noise_pred_uncond = noise_pred_uncond[0]
                noise_pred_text = self.unet(
                    sample=latent_model_input, timestep=timestep, encoder_hidden_states=text_embeds
                )
                noise_pred_text = noise_pred_text[0]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.unet(
                    sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds
                )
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
            latents[0:1].tofile(self.save_data_dir / "latent.raw")
        image = np.concatenate([self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])])
        if self.save_data_dir:
            image.tofile(self.save_data_dir / "output_img.raw")

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


def get_qnn_pipeline(model_dir, common_args, qnn_args, script_dir):
    ort.set_default_logger_severity(3)

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()

    # TODO diffusers needs to support new parameter for QNN
    # See https://github.com/huggingface/diffusers/issues/10658
    pipeline = QnnStableDiffusionPipeline.from_pretrained(
        model_dir, provider="CPUExecutionProvider", sess_options=sess_options
    )
    if qnn_args.save_data:
        pipeline.save_data_dir = script_dir / qnn_args.data_dir / common_args.prompt
        os.makedirs(pipeline.save_data_dir, exist_ok=True)
    else:
        pipeline.save_data_dir = None
    return pipeline
