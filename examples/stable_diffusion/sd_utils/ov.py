# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Example modified from: https://docs.openvino.ai/2023.3/notebooks/225-stable-diffusion-text-to-image-with-output.html
# --------------------------------------------------------------------------
import inspect
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import openvino as ov
import PIL
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from openvino.runtime import Model
from transformers import CLIPTokenizer

OV_OPTIMIZED_MODEL_INFO = "ov_optimized_model_info.json"

# ruff: noqa: T201


def scale_fit_to_window(dst_width: int, dst_height: int, image_width: int, image_height: int):
    """Preprocessing helper function for calculating image size.

    Calculating image size for resize with peserving original aspect ratio and fitting image to specific window size.

    Args:
      dst_width (int): destination window width
      dst_height (int): destination window height
      image_width (int): source image width
      image_height (int): source image height

    Returns:
      result_width (int): calculated width for resize
      result_height (int): calculated height for resize

    """
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)


def preprocess(image: PIL.Image.Image):
    """Image preprocessing function.

    Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
    then converts it to np.ndarray and adds padding with zeros on right or
    bottom side of image (depends from aspect ratio), after that
    converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally,
    converts data layout from planar NHWC to NCHW.
    The function returns preprocessed input tensor and padding size, which can be used in postprocessing.

    Args:
      image (PIL.Image.Image): input image
    Returns:
       image (np.ndarray): preprocessed image tensor
       meta (Dict): dictionary with preprocessing metadata info

    """
    src_width, src_height = image.size
    dst_width, dst_height = scale_fit_to_window(512, 512, src_width, src_height)
    image = np.array(image.resize((dst_width, dst_height), resample=PIL.Image.Resampling.LANCZOS))[None, :]
    pad_width = 512 - dst_width
    pad_height = 512 - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    image = image.astype(np.float32) / 255.0
    image = 2.0 * image - 1.0
    image = image.transpose(0, 3, 1, 2)
    return image, {"padding": pad, "src_width": src_width, "src_height": src_height}


@dataclass
class OvStableDiffusionPipelineOutput:
    """Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.

    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]] = None


class OVStableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae_decoder: Model,
        text_encoder: Model,
        tokenizer: CLIPTokenizer,
        unet: Model,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        vae_encoder: Model = None,
    ):
        """Pipeline for text-to-image generation using Stable Diffusion.

        Args:
            vae_decoder (Model):
                Variational Auto-Encoder (VAE) Model to decode images to and from latent representations.
            text_encoder (Model):
                Frozen text-encoder. Stable Diffusion uses the text portion of
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
                the clip-vit-large-patch14(https://huggingface.co/openai/clip-vit-large-patch14) variant.
            tokenizer (CLIPTokenizer):
                Tokenizer of class CLIPTokenizer
                (https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            unet (Model): Conditional U-Net architecture to denoise the encoded image latents.
            scheduler (SchedulerMixin):
                A scheduler to be used in combination with unet to denoise the encoded image latents. Can be one of
                DDIMScheduler, LMSDiscreteScheduler, or PNDMScheduler.
            vae_encoder (Model):
                Variational Auto-Encoder (VAE) Model to encode images to and from latent representations.

        """
        super().__init__()
        self.scheduler = scheduler
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.text_encoder = text_encoder
        self.unet = unet
        self._text_encoder_output = text_encoder.output(0)
        self._unet_output = unet.output(0)
        self._vae_d_output = vae_decoder.output(0)
        self._vae_e_output = vae_encoder.output(0) if vae_encoder is not None else None
        self.height = 512
        self.width = 512
        self.tokenizer = tokenizer

    def __call__(
        self,
        prompts: Union[str, List[str]],
        image: PIL.Image.Image = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        negative_prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        output_type: Optional[str] = "pil",
        seed: Optional[int] = None,
        strength: float = 0.5,
        gif: Optional[bool] = False,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ):
        """Invoke when calling the pipeline for generation.

        Args:
            prompts (str or List[str]):
                The prompts to guide the image generation.
            image (PIL.Image.Image, *optional*, None):
                 Intinal image for generation.
            height (int, *optional*, defaults to 512):
                Height of the generated image.
            width (int, *optional*, defaults to 512):
                Width of the generated image.
            num_inference_steps (int, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            negative_prompt (str or List[str]):
                The negative prompt or prompts to guide the image generation.
            num_images_per_prompt (int, *optional*, defaults to 1):
                Number of images to generate per prompt.
            guidance_scale (float, *optional*, defaults to 7.5):
                Guidance scale as defined in Classifier-Free Diffusion Guidance(https://arxiv.org/abs/2207.12598).
                guidance_scale is defined as `w` of equation 2.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt,
                usually at the expense of lower image quality.
            eta (float, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [DDIMScheduler], will be ignored for others.
            output_type (`str`, *optional*, defaults to "pil"):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): PIL.Image.Image or np.array.
            seed (int, *optional*, None):
                Seed for random generator state initialization.
            strength (float):
                Controls the amount of noise that is added to the input image.
            gif (bool, *optional*, False):
                Flag for storing all steps results or not.
            callback (Callable[[int, int, np.ndarray], None], *optional*, None):
                Callback function for each step of generation.
            callback_steps (int, *optional*, 1):
                Number of steps between callback calls.
            kwargs: Additional keyword arguments.

        Returns:
            Dictionary with keys:
                sample - the last generated image PIL.Image.Image or np.array
                iterations - *optional* (if gif=True) images for all diffusion steps,
                List of PIL.Image.Image or np.array.

        """
        if seed is not None:
            np.random.seed(seed)

        batch_size = len(prompts) if isinstance(prompts, list) else 1

        print(f"Start inference with prompt: {prompts}")
        img_buffer = []
        do_classifier_free_guidance = guidance_scale > 1.0

        # get prompt text embeddings
        text_embeddings = self._encode_prompt(
            prompts,
            num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = timesteps[:1]

        # get the initial random noise unless the user supplied it
        latents, meta = self.prepare_latents(batch_size, num_images_per_prompt, height, width, image, latent_timestep)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if you are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet([latent_model_input, t, text_embeddings])[self._unet_output]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )["prev_sample"].numpy()
            if gif:
                image = self.vae_decoder(latents * (1 / 0.18215))[self._vae_d_output]
                image = self.postprocess_image(image, meta, output_type)
                img_buffer.extend(image)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        # scale and decode the image latents with vae
        image = self.vae_decoder(latents * (1 / 0.18215))[self._vae_d_output]

        image = self.postprocess_image(image, meta, output_type)

        return OvStableDiffusionPipelineOutput(images=image)

    def _encode_prompt(
        self,
        prompt: str,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
    ):
        """Encode the prompt into text encoder hidden states.

        Args:
            prompt (str): prompt to be encoded
            num_images_per_prompt (int): number of images to generate per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states

        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(text_input_ids)[self._text_encoder_output]

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(
        self,
        batch_size,
        num_images_per_prompt,
        height,
        width,
        image: PIL.Image.Image = None,
        latent_timestep: torch.Tensor = None,
    ):
        """Get initial latents for starting generation.

        Args:
            batch_size (int):
                Batch size for generation
            num_images_per_prompt (int):
                Number of images to generate per prompt
            height (int):
                Height of generated image
            width (int):
                Width of generated image
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space

        """
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None and isinstance(self.scheduler, LMSDiscreteScheduler):
            # if you use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
            noise = noise * self.scheduler.sigmas[0].numpy()
            return noise, {}
        input_image, meta = preprocess(image)
        latents = self.vae_encoder(input_image)[self._vae_e_output] * 0.18215
        latents = self.scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        return latents, meta

    def postprocess_image(self, image: np.ndarray, meta: Dict, output_type: str = "pil"):
        """Postprocessing for decoded image.

        Takes generated image decoded by VAE decoder, unpad it to initila image size (if required),
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format

        Args:
            image (np.ndarray):
                Generated image
            meta (Dict):
                Metadata obtained on latents preparing step, can be empty
            output_type (str, *optional*, pil):
                Output format for result, can be pil or numpy
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images

        """
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [img.resize((orig_width, orig_height), PIL.Image.Resampling.LANCZOS) for img in image]
        else:
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [cv2.resize(img, (orig_width, orig_width)) for img in image]
        return image

    def get_timesteps(self, num_inference_steps: int, strength: float):
        """Get scheduler timesteps for generation.

        In case of image-to-image generation, it updates number of steps according to strength

        Args:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image.
               Values that approach 1.0 enable lots of variations
               but will also produce images that are not semantically consistent with the input.

        """
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start


def update_ov_config(config: Dict):
    config["pass_flows"] = [["ov_convert"]]
    config["search_strategy"] = False
    config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = ["CPUExecutionProvider"]
    del config["evaluators"]
    del config["evaluator"]
    return config


def save_optimized_ov_submodel(run_res, submodel, optimized_model_dir, optimized_model_path_map):
    res = next(iter(run_res.values()))

    output_model_dir = res.get_output_model_path()
    optimized_model_path = optimized_model_dir / submodel
    shutil.copytree(output_model_dir, optimized_model_path)
    model_path = (optimized_model_path / submodel).with_suffix(".xml")
    optimized_model_path_map[submodel] = str(model_path)


def get_ov_pipeline(common_args, ov_args, optimized_model_dir):
    if common_args.test_unoptimized:
        return StableDiffusionPipeline.from_pretrained(common_args.model_id)

    with (optimized_model_dir / OV_OPTIMIZED_MODEL_INFO).open("r") as model_info_file:
        optimized_model_path_map = json.load(model_info_file)

    device = ov_args.device

    core = ov.Core()
    text_enc = core.compile_model(optimized_model_path_map["text_encoder"], device)
    unet_model = core.compile_model(optimized_model_path_map["unet"], device)

    ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if device != "CPU" else {}

    vae_decoder = core.compile_model(optimized_model_path_map["vae_decoder"], device, ov_config)
    vae_encoder = core.compile_model(optimized_model_path_map["vae_encoder"], device, ov_config)

    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    return OVStableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_enc,
        unet=unet_model,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        scheduler=lms,
    )


def run_ov_image_inference(
    pipe, image_path, prompt, strength, guidance_scale, image_size, num_inference_steps, common_args, generator=None
):
    image = None
    if image_path:
        img_path = Path(image_path)
        print(f"Image path is {img_path}")
        if not img_path.exists():
            print("Image doesn't exist.")
            sys.exit(1)
        image = PIL.Image.open(img_path)

    return pipe(
        prompts=[prompt] * common_args.batch_size,
        image=image,
        num_inference_steps=num_inference_steps,
        height=image_size,
        width=image_size,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator,
    )


def run_ov_img_to_img_example(pipe, guidance_scale, common_args):
    prompt = "amazing watercolor painting"
    strength = 0.5
    image_path = Path("./assets/dog.png")
    image_size = 512
    num_inference_steps = 10

    return run_ov_image_inference(
        pipe, image_path, prompt, strength, guidance_scale, image_size, num_inference_steps, common_args
    )


def save_ov_model_info(model_info, optimized_model_dir):
    model_info_path = optimized_model_dir / OV_OPTIMIZED_MODEL_INFO
    with model_info_path.open("w") as model_info_file:
        json.dump(model_info, model_info_file, indent=4)
