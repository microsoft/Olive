# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Example modified from: https://docs.openvino.ai/2023.2/notebooks/225-stable-diffusion-text-to-image-with-output.html
# --------------------------------------------------------------------------
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import openvino as ov
import PIL
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from openvino.runtime import Model
from transformers import CLIPTokenizer

from olive.workflows import run as olive_run


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
                Tokenizer of class CLIPTokenizer(https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
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
        prompts: List[str],
        image: PIL.Image.Image = None,
        num_inference_steps: Optional[int] = 50,
        negative_prompt: Union[str, List[str]] = None,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        output_type: Optional[str] = "pil",
        seed: Optional[int] = None,
        strength: float = 0.5,
        gif: Optional[bool] = False,
        **kwargs,
    ):
        """Invoke when calling the pipeline for generation.

        Args:
            prompts (List[str]):
                The prompts to guide the image generation.
            image (PIL.Image.Image, *optional*, None):
                 Intinal image for generation.
            num_inference_steps (int, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            negative_prompt (str or List[str]):
                The negative prompt or prompts to guide the image generation.
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
            kwargs: Additional keyword arguments.

        Returns:
            Dictionary with keys:
                sample - the last generated image PIL.Image.Image or np.array
                iterations - *optional* (if gif=True) images for all diffusion steps,
                List of PIL.Image.Image or np.array.
        """
        if seed is not None:
            np.random.seed(seed)

        return [
            self._infer(
                prompt, negative_prompt, num_inference_steps, guidance_scale, eta, output_type, gif, image, strength
            )
            for prompt in prompts
        ]

    def _infer(
        self, prompt: str, negative_prompt, num_inference_steps, guidance_scale, eta, output_type, gif, image, strength
    ):
        print(f"Start inference with prompt: {prompt}")
        img_buffer = []
        do_classifier_free_guidance = guidance_scale > 1.0

        # get prompt text embeddings
        text_embeddings = self._encode_prompt(
            prompt, do_classifier_free_guidance=do_classifier_free_guidance, negative_prompt=negative_prompt
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
        latents, meta = self.prepare_latents(image, latent_timestep)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for t in self.progress_bar(timesteps):
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

        # scale and decode the image latents with vae
        image = self.vae_decoder(latents * (1 / 0.18215))[self._vae_d_output]

        image = self.postprocess_image(image, meta, output_type)
        return {"sample": image, "iterations": img_buffer}

    def _encode_prompt(
        self,
        prompt: str,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
    ):
        """Encode the prompt into text encoder hidden states.

        Args:
            prompt (str): prompt to be encoded
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

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(self, image: PIL.Image.Image = None, latent_timestep: torch.Tensor = None):
        """Get initial latents for starting generation.

        Args:
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)
        noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None:
            # if you use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
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


def optimize(model_id: str, optimized_model_dir: Path):
    optimized_model_path_map = {}
    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "sd_openvino", ignore_errors=True)
    shutil.rmtree(optimized_model_dir, ignore_errors=True)

    submodel_names = ["text_encoder", "vae_encoder", "vae_decoder", "unet"]

    for submodel in submodel_names:
        print(f"Start optimizing {submodel}")
        olive_config = None
        with (script_dir / f"config_{submodel}.json").open() as f:
            olive_config = json.load(f)

        olive_config["input_model"]["config"]["model_path"] = model_id

        # TODO(xiaoyuz): simplify output api
        res = next(iter(olive_run(olive_config).values()))

        output_model_dir = res.get_output_model_path()
        optimized_model_path = optimized_model_dir / submodel
        shutil.copytree(output_model_dir, optimized_model_path)
        model_path = (optimized_model_path / submodel).with_suffix(".xml")
        optimized_model_path_map[submodel] = model_path
    return optimized_model_path_map


def inference(
    device: str,
    optimized_model_path_map: dict,
    prompt: List[str],
    guidance_scale: float = 7.5,
    num_steps: int = 20,
    seed: int = 42,
    strength: float = 1.0,
    image=None,
):
    core = ov.Core()
    text_enc = core.compile_model(optimized_model_path_map["text_encoder"], device)
    unet_model = core.compile_model(optimized_model_path_map["unet"], device)

    ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if device != "CPU" else {}

    vae_decoder = core.compile_model(optimized_model_path_map["vae_decoder"], device, ov_config)
    vae_encoder = core.compile_model(optimized_model_path_map["vae_encoder"], device, ov_config)

    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    ov_pipe = OVStableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_enc,
        unet=unet_model,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        scheduler=lms,
    )

    return ov_pipe(
        prompt, image, num_inference_steps=num_steps, guidance_scale=guidance_scale, seed=seed, strength=strength
    )


def save_image(image, output_dir, index):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    image_path = output_dir / f"prompt_{index}.png"
    image.save(image_path)
    print(f"Image saved to {image_path}")


def evaluate(model_id: str, device: str, optimized_model_path_map: dict):
    """Evaluate the inference latency of stable diffusion pipeline from huggingface model hub and OpenVINO model.

    Total 5 prompts are used for evaluation.
    """
    prompt_list = [
        "a dog in the park",
        "a cat in the park",
        "a bird in the park",
        "a man in the park",
        "a woman in the park",
    ]
    num_steps = 20
    guidance_scale = 7.5
    evaluate_path = Path("./evaluate")

    print("Start evaluating stable diffusion pipeline from huggingface model hub...")
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    if device == "GPU":
        pipe.to("cuda")
    latencies = []
    for prompt in prompt_list:
        start = time.perf_counter()
        pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]
        latencies.append(time.perf_counter() - start)
    sd_lat_avg = sum(latencies) / len(latencies)

    print("Start evaluating stable diffusion pipeline from OpenVINO model...")
    latencies = []
    for prompt in prompt_list:
        start = time.perf_counter()
        inference(device, optimized_model_path_map, prompt=[prompt], num_steps=num_steps, guidance_scale=guidance_scale)
        latencies.append(time.perf_counter() - start)
    ov_lat_avg = sum(latencies) / len(latencies)
    print(f"Average latency for stable diffusion Huggingface pipeline is {sd_lat_avg} seconds")
    print(f"Average latency for stable diffusion OpenVINO pipeline is {ov_lat_avg} seconds")
    print(f"compare to huggingface model hub, OpenVINO model is {sd_lat_avg / ov_lat_avg} times faster")
    print("Clean up the outputs...")
    shutil.rmtree(evaluate_path, ignore_errors=True)
    print("Evaluation finished.")


def main(raw_args=None):
    print("Start running stable diffusion OpenVINO script...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str, help="Directory to save the generated images")
    parser.add_argument("--device", choices=["CPU", "GPU", "VPU"], default="CPU", type=str)
    parser.add_argument("--image_path", default=None, type=str)
    parser.add_argument("--evaluate", action="store_true", help="Runs the evaluation")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--img_to_img_example", action="store_true", help="Runs the image to image example")
    parser.add_argument("--inference", action="store_true", help="Runs the inference step")
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=(
            [
                "castle surrounded by water and nature, village, volumetric lighting, photorealistic, "
                "detailed and intricate, fantasy, epic cinematic shot, mountains, 8k ultra hd"
            ]
        ),
    )
    parser.add_argument(
        "--num_images_per_prompt ", default=1, type=int, help="The number of images to generate per prompt."
    )
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance",
    )
    parser.add_argument(
        "--strength",
        default=1.0,
        type=float,
        help="Value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. "
        "Values that approach 1.0 enable lots of variations but will also produce images "
        "that are not semantically consistent with the input.",
    )
    parser.add_argument("--num_steps", default=20, type=int, help="Number of denoising steps")
    parser.add_argument("--seed", default=42, type=int, help="Seed for random generator state initialization")

    args = parser.parse_args(raw_args)
    model_id = args.model_id
    device = args.device

    script_dir = Path(__file__).resolve().parent
    optimized_dir_name = f"optimized_{device}"
    optimized_model_dir = script_dir / "models" / optimized_dir_name / model_id

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    optimized_model_path_map = {}
    if args.optimize:
        optimized_model_path_map = optimize(model_id, optimized_model_dir)

    if args.inference:
        image = None
        prompt = args.prompt
        num_steps = args.num_steps
        strength = args.strength

        if args.image_path:
            img_path = Path(args.image_path)
            print(f"Image path is {img_path}")
            if not img_path.exists():
                print("Image doesn't exist.")
                return
            image = PIL.Image.open(img_path)

        if args.img_to_img_example:
            prompt = ["amazing watercolor painting"]
            num_steps = 10
            strength = 0.5
            image = PIL.Image.open(Path("./assets/dog.png"))

        results = inference(
            device,
            optimized_model_path_map,
            prompt=prompt,
            image=image,
            guidance_scale=args.guidance_scale,
            num_steps=num_steps,
            seed=args.seed,
            strength=strength,
        )

        for index, result in enumerate(results):
            final_image = result["sample"][0]
            if result["iterations"]:
                all_frames = result["iterations"]
                img = next(iter(all_frames))
                img.save(
                    fp="result.gif",
                    format="GIF",
                    append_images=iter(all_frames),
                    save_all=True,
                    duration=len(all_frames) * 5,
                    loop=0,
                )
            save_image(final_image, args.output_dir, index)

    if args.evaluate:
        if not optimized_model_path_map:
            optimized_model_path_map = optimize(model_id, optimized_model_dir)
        evaluate(model_id, device, optimized_model_path_map)


if __name__ == "__main__":
    main()
