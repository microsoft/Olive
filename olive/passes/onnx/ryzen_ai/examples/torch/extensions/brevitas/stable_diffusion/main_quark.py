"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import torch
import os
import torch.nn as nn
import argparse
import numpy as np
import json
import time
import pandas as pd

from torch.utils.data import Dataset
from datetime import datetime
from dependencies import value
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from brevitas_examples.stable_diffusion.sd_quant.constants import (
    SD_2_1_EMBEDDINGS_SHAPE,
)
from brevitas_examples.stable_diffusion.sd_quant.constants import SD_XL_EMBEDDINGS_SHAPE
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_latents
from brevitas_examples.stable_diffusion.sd_quant.utils import (
    generate_unet_21_rand_inputs,
)
from brevitas_examples.stable_diffusion.sd_quant.utils import (
    generate_unet_xl_rand_inputs,
)
from brevitas_examples.stable_diffusion.sd_quant.utils import unet_input_shape

from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.common.parse_utils import quant_format_validator

from brevitas.nn.quant_activation import QuantIdentity
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.stable_diffusion.sd_quant.export import (
    export_onnx as brevitas_export_onnx,
)
from brevitas_examples.stable_diffusion.sd_quant.export import export_quant_params
import adapter as quark_brevitas

from typing import List, Optional

TEST_SEED = 123456
torch.manual_seed(TEST_SEED)

NEGATIVE_PROMPTS = [
    "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
]

CALIBRATION_PROMPTS = [
    "A man in a space suit playing a guitar, inspired by Cyril Rolando, highly detailed illustration, full color illustration, very detailed illustration, dan mumford and alex grey style",
    "a living room, bright modern Scandinavian style house, large windows, magazine photoshoot, 8k, studio lighting",
    "cute rabbit in a spacesuit",
    "minimalistic plolygon geometric car in brutalism warehouse, Rick Owens",
]

TESTING_PROMPTS = [
    "batman, cute modern disney style, Pixar 3d portrait, ultra detailed, gorgeous, 3d zbrush, trending on dribbble, 8k render",
    "A beautiful stack of rocks sitting on top of a beach, a picture, red black white golden colors, chakras, packshot, stock photo",
    "A painting of a fish on a black background, a digital painting, by Jason Benjamin, colorful vector illustration, mixed media style illustration, epic full color illustration, mascot illustration",
    "close up photo of a rabbit, forest in spring, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot",
]


def load_calib_prompts(calib_data_path, sep="\t"):
    df = pd.read_csv(calib_data_path, sep=sep)
    lst = df["caption"].tolist()
    return lst


def export_onnx(
    pipe, output_dir, trace_inputs, weight_quant_granularity, export_weight_q_node
):
    export_manager = StdQCDQONNXManager
    export_manager.change_weight_export(export_weight_q_node=export_weight_q_node)
    brevitas_export_onnx(pipe, trace_inputs, output_dir, export_manager)


def run_test_inference(
    pipe,
    resolution,
    prompts,
    seeds,
    output_path,
    device,
    dtype,
    use_negative_prompts,
    guidance_scale,
    name_prefix="",
):
    images = dict()
    with torch.no_grad():
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        test_latents = generate_latents(
            seeds, device, dtype, unet_input_shape(resolution)
        )
        neg_prompts = NEGATIVE_PROMPTS * len(seeds) if use_negative_prompts else []
        for prompt in prompts:
            prompt_images = pipe(
                [prompt] * len(seeds),
                latents=test_latents,
                negative_prompt=neg_prompts,
                guidance_scale=guidance_scale,
            ).images
            images[prompt] = prompt_images

        i = 0
        for prompt, prompt_images in images.items():
            for image in prompt_images:
                file_path = os.path.join(output_path, f"{name_prefix}{i}.png")
                print(f"Saving to {file_path}")
                image.save(file_path)
                i += 1
    return images


def run_val_inference(
    pipe,
    resolution,
    prompts,
    seeds,
    device,
    dtype,
    use_negative_prompts,
    guidance_scale,
    total_steps,
    test_latents=None,
):
    with torch.no_grad():

        if test_latents is None:
            test_latents = generate_latents(
                seeds[0], device, dtype, unet_input_shape(resolution)
            )

        neg_prompts = NEGATIVE_PROMPTS if use_negative_prompts else []
        for prompt in tqdm(prompts):
            # We don't want to generate any image, so we return only the latent encoding pre VAE
            pipe(
                prompt,
                negative_prompt=neg_prompts[0],
                latents=test_latents,
                output_type="latent",
                guidance_scale=guidance_scale,
                num_inference_steps=total_steps,
            )


class SDXLPipeCalibrationDataset(Dataset):
    def __init__(
        self,
        prompts: List[str],
        seeds: List[int],
        resolution,
        device,
        dtype,
        guidance_scale: float,
        total_steps: int,
        test_latents=None,
        negative_prompts: Optional[List[str]] = None,
    ):
        self.prompts = prompts
        if test_latents is None:
            test_latents = generate_latents(
                seeds[0], device, dtype, unet_input_shape(resolution)
            )
        self.test_latents = test_latents
        self.negative_prompts = negative_prompts or NEGATIVE_PROMPTS
        self.total_steps = total_steps
        self.guidance_scale = guidance_scale

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return (self.prompts[index],), {
            "negative_prompt": self.negative_prompts[0],
            "latents": self.test_latents,
            "output_type": "latent",
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.total_steps,
        }


def main(args):

    dtype = getattr(torch, args.dtype)

    calibration_prompts = CALIBRATION_PROMPTS
    if args.calibration_prompt_path is not None:
        calibration_prompts = load_calib_prompts(args.calibration_prompt_path)
    print(args.calibration_prompt, len(calibration_prompts))
    assert args.calibration_prompt <= len(
        calibration_prompts
    ), f"Only {len(calibration_prompts)} prompts are available"
    calibration_prompts = calibration_prompts[: args.calibration_prompt]

    latents = None
    if args.path_to_latents is not None:
        latents = torch.load(args.path_to_latents).to(torch.float16)

    # Create output dir. Move to tmp if None
    ts = datetime.fromtimestamp(time.time())
    str_ts = ts.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f"{str_ts}")
    os.mkdir(output_dir)

    # Dump args to json
    with open(os.path.join(output_dir, "args.json"), "w") as fp:
        json.dump(vars(args), fp)

    # Extend seeds based on batch_size
    test_seeds = [TEST_SEED] + [TEST_SEED + i for i in range(1, args.batch_size)]

    # Load model from float checkpoint
    print(f"Loading model from {args.model}...")
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    print(f"Model loaded from {args.model}.")

    # Move model to target device
    print(f"Moving model to {args.device}...")
    pipe = pipe.to(args.device)

    if args.prompt > 0:
        print("Running inference with prompt ...")
        testing_prompts = TESTING_PROMPTS[: args.prompt]
        float_images = run_test_inference(
            pipe,
            args.resolution,
            testing_prompts,
            test_seeds,
            output_dir,
            args.device,
            dtype,
            guidance_scale=args.guidance_scale,
            use_negative_prompts=args.use_negative_prompts,
            name_prefix="float_",
        )

    # Detect Stable Diffusion XL pipeline
    is_sd_xl = isinstance(pipe, StableDiffusionXLPipeline)

    # Enable attention slicing
    if args.attention_slicing:
        pipe.enable_attention_slicing()

    # Extract list of layers to avoid
    blacklist = []
    for name, _ in pipe.unet.named_modules():
        if "time_emb" in name:
            blacklist.append(name.split(".")[-1])
    print(f"Blacklisted layers: {blacklist}")

    # Make sure there all LoRA layers are fused first, otherwise raise an error
    for m in pipe.unet.modules():
        if hasattr(m, "lora_layer") and m.lora_layer is not None:
            raise RuntimeError(
                "LoRA layers should be fused in before calling into quantization."
            )

    pipe.set_progress_bar_config(disable=True)

    # Quantize model
    if args.quantize:

        @value
        def weight_bit_width(module):
            if isinstance(module, nn.Linear):
                return args.linear_weight_bit_width
            elif isinstance(module, nn.Conv2d):
                return args.conv_weight_bit_width
            else:
                raise RuntimeError(f"Module {module} not supported.")

        @value
        def input_bit_width(module):
            if isinstance(module, nn.Linear):
                return args.linear_input_bit_width
            elif isinstance(module, nn.Conv2d):
                return args.conv_input_bit_width
            elif isinstance(module, QuantIdentity):
                return args.quant_identity_bit_width
            else:
                raise RuntimeError(f"Module {module} not supported.")

        qconfig = quark_brevitas.BrevitasSDXLQuantizationConfig(
            dtype=dtype,
            device=args.device,
            weight_bit_width=weight_bit_width,
            weight_quant_format=args.weight_quant_format,
            weight_quant_type=args.weight_quant_type,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            quantize_input_zero_point=args.quantize_input_zero_point,
            input_bit_width=input_bit_width,
            input_quant_format=args.input_quant_format,
            input_scale_type=args.input_scale_type,
            input_scale_precision=args.input_scale_precision,
            input_param_method=args.input_param_method,
            input_quant_type=args.input_quant_type,
            input_quant_granularity=args.input_quant_granularity,
            input_scale_stats_op=args.input_scale_stats_op,
            input_zp_stats_op=args.input_zp_stats_op,
            use_ocp=args.use_ocp,
            use_fnuz=args.use_fnuz,
            linear_weight_bit_width=args.linear_weight_bit_width,
            linear_input_bit_width=args.linear_input_bit_width,
            conv_weight_bit_width=args.conv_weight_bit_width,
            conv_input_bit_width=args.conv_input_bit_width,
            blacklist=blacklist,
            quantize_sdp_1=args.quantize_sdp_1,
            quantize_sdp_2=args.quantize_sdp_2,
            activation_equalization=args.activation_equalization,
            activation_equalization_alpha=args.act_eq_alpha,
            activation_equalization_exclude_blacklist=args.exclude_blacklist_act_eq,
            gptq=args.gptq,
            bias_correction=args.bias_correction,
        )

        quantizer = quark_brevitas.BrevitasModelQuantizer(qconfig)
        # NB: Temporary solution for unet vs pipe as the callable during calibration.

        def _calibration_callable(data):
            args, kwargs = data
            return pipe(*args, **kwargs)

        pipe.unet.calibration_callable = _calibration_callable
        dataloader = SDXLPipeCalibrationDataset(
            calibration_prompts,
            test_seeds,
            args.resolution,
            args.device,
            dtype,
            args.guidance_scale,
            args.calibration_steps,
        )

        quantizer.quantize_model(pipe.unet, dataloader)

        pipe.set_progress_bar_config(disable=True)

    if args.checkpoint_name is not None:
        torch.save(
            pipe.unet.state_dict(), os.path.join(output_dir, args.checkpoint_name)
        )

    # Perform inference
    if args.prompt > 0:
        print("Computing accuracy on default prompt")
        testing_prompts = TESTING_PROMPTS[: args.prompt]
        assert args.prompt <= len(
            TESTING_PROMPTS
        ), f"Only {len(TESTING_PROMPTS)} prompts are available"

        quant_images = run_test_inference(
            pipe,
            args.resolution,
            testing_prompts,
            test_seeds,
            output_dir,
            args.device,
            dtype,
            use_negative_prompts=args.use_negative_prompts,
            guidance_scale=args.guidance_scale,
            name_prefix="quant_",
        )

        float_images_values = float_images.values()
        float_images_values = [x for x_nested in float_images_values for x in x_nested]
        float_images_values = torch.tensor(
            [np.array(image) for image in float_images_values]
        )
        float_images_values = float_images_values.permute(0, 3, 1, 2)

        quant_images_values = quant_images.values()
        quant_images_values = [x for x_nested in quant_images_values for x in x_nested]
        quant_images_values = torch.tensor(
            [np.array(image) for image in quant_images_values]
        )
        quant_images_values = quant_images_values.permute(0, 3, 1, 2)

        fid = FrechetInceptionDistance(normalize=False)
        fid.update(float_images_values, real=True)
        fid.update(quant_images_values, real=False)
        print(f"FID: {float(fid.compute())}")

    if args.export_target:
        # Move to cpu and to float32 to enable CPU export
        if args.export_cpu_float32:
            pipe.unet.to("cpu").to(torch.float32)
        pipe.unet.eval()
        device = next(iter(pipe.unet.parameters())).device
        dtype = next(iter(pipe.unet.parameters())).dtype

        # Define tracing input
        if is_sd_xl:
            generate_fn = generate_unet_xl_rand_inputs
            shape = SD_XL_EMBEDDINGS_SHAPE
        else:
            generate_fn = generate_unet_21_rand_inputs
            shape = SD_2_1_EMBEDDINGS_SHAPE
        trace_inputs = generate_fn(
            embedding_shape=shape,
            unet_input_shape=unet_input_shape(args.resolution),
            device=device,
            dtype=dtype,
        )

        if args.export_target == "onnx":
            export_onnx(
                pipe,
                output_dir,
                trace_inputs,
                args.weight_quant_granularity,
                args.export_weight_q_node,
            )
        if args.export_target == "params_only":
            export_quant_params(pipe, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion quantization")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/scratch/hf_models/stable-diffusion-2-1-base",
        help="Path or name of the model.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="Target device for quantized model.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=2,
        help="How many seeds to use for each image during validation. Default: 2",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=4,
        help="Number of prompt to use for testing. Default: 4",
    )
    parser.add_argument(
        "--calibration-prompt",
        type=int,
        default=2,
        help="Number of prompt to use for calibration. Default: 2",
    )
    parser.add_argument(
        "--calibration-prompt-path",
        type=str,
        default=None,
        help="Path to calibration prompt",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Name to use to store the checkpoint in the output dir. If not provided, no checkpoint is saved.",
    )
    parser.add_argument(
        "--path-to-latents",
        type=str,
        default=None,
        help="Load pre-defined latents. If not provided, they are generated based on an internal seed.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution along height and width dimension. Default: 512.",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="Guidance scale."
    )
    parser.add_argument(
        "--calibration-steps",
        type=float,
        default=8,
        help="Steps used during calibration",
    )
    add_bool_arg(
        parser,
        "output-path",
        str_true=True,
        default=".",
        help="Path where to generate output folder.",
    )
    add_bool_arg(
        parser, "quantize", default=True, help="Toggle quantization. Default: Enabled"
    )
    add_bool_arg(
        parser,
        "activation-equalization",
        default=False,
        help="Toggle Activation Equalization. Default: Disabled",
    )
    add_bool_arg(parser, "gptq", default=False, help="Toggle gptq. Default: Disabled")
    add_bool_arg(
        parser,
        "bias-correction",
        default=True,
        help="Toggle bias-correction. Default: Enabled",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Model Dtype, choices are float32, float16, bfloat16. Default: float16",
    )
    add_bool_arg(
        parser,
        "attention-slicing",
        default=False,
        help="Enable attention slicing. Default: Disabled",
    )
    parser.add_argument(
        "--export-target",
        type=str,
        default="",
        choices=["", "onnx", "params_only"],
        help="Target export flow.",
    )
    add_bool_arg(
        parser,
        "export-weight-q-node",
        default=False,
        help="Enable export of floating point weights + QDQ rather than integer weights + DQ. Default: Disabled",
    )
    parser.add_argument(
        "--conv-weight-bit-width",
        type=int,
        default=8,
        help="Weight bit width. Default: 8.",
    )
    parser.add_argument(
        "--linear-weight-bit-width",
        type=int,
        default=8,
        help="Weight bit width. Default: 8.",
    )
    parser.add_argument(
        "--conv-input-bit-width",
        type=int,
        default=0,
        help="Input bit width. Default: 0 (not quantized)",
    )
    parser.add_argument(
        "--act-eq-alpha",
        type=float,
        default=0.9,
        help="Alpha for activation equalization. Default: 0.9",
    )
    parser.add_argument(
        "--linear-input-bit-width",
        type=int,
        default=0,
        help="Input bit width. Default: 0 (not quantized).",
    )
    parser.add_argument(
        "--weight-param-method",
        type=str,
        default="stats",
        choices=["stats", "mse"],
        help="How scales/zero-point are determined. Default: stats.",
    )
    parser.add_argument(
        "--input-param-method",
        type=str,
        default="stats",
        choices=["stats", "mse"],
        help="How scales/zero-point are determined. Default: stats.",
    )
    parser.add_argument(
        "--input-scale-stats-op",
        type=str,
        default="minmax",
        choices=["minmax", "percentile"],
        help="Define what statics op to use for input scale. Default: minmax.",
    )
    parser.add_argument(
        "--input-zp-stats-op",
        type=str,
        default="minmax",
        choices=["minmax", "percentile"],
        help="Define what statics op to use for input zero point. Default: minmax.",
    )
    parser.add_argument(
        "--weight-scale-precision",
        type=str,
        default="float_scale",
        choices=["float_scale", "po2_scale"],
        help="Whether scale is a float value or a po2. Default: float_scale.",
    )
    parser.add_argument(
        "--input-scale-precision",
        type=str,
        default="float_scale",
        choices=["float_scale", "po2_scale"],
        help="Whether scale is a float value or a po2. Default: float_scale.",
    )
    parser.add_argument(
        "--weight-quant-type",
        type=str,
        default="asym",
        choices=["sym", "asym"],
        help="Weight quantization type. Default: asym.",
    )
    parser.add_argument(
        "--input-quant-type",
        type=str,
        default="asym",
        choices=["sym", "asym"],
        help="Input quantization type. Default: asym.",
    )
    parser.add_argument(
        "--weight-quant-format",
        type=quant_format_validator,
        default="int",
        help="Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. Default: int.",
    )
    parser.add_argument(
        "--input-quant-format",
        type=quant_format_validator,
        default="int",
        help="Input quantization type. Either int or eXmY, with X+Y==input_bit_width-1. Default: int.",
    )
    parser.add_argument(
        "--weight-quant-granularity",
        type=str,
        default="per_channel",
        choices=["per_channel", "per_tensor", "per_group"],
        help="Granularity for scales/zero-point of weights. Default: per_channel.",
    )
    parser.add_argument(
        "--input-quant-granularity",
        type=str,
        default="per_tensor",
        choices=["per_tensor"],
        help="Granularity for scales/zero-point of inputs. Default: per_tensor.",
    )
    parser.add_argument(
        "--input-scale-type",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="Whether to do static or dynamic input quantization. Default: static.",
    )
    parser.add_argument(
        "--weight-group-size",
        type=int,
        default=16,
        help="Group size for per_group weight quantization. Default: 16.",
    )
    add_bool_arg(
        parser,
        "quantize-weight-zero-point",
        default=True,
        help="Quantize weight zero-point. Default: Enabled",
    )
    add_bool_arg(
        parser,
        "exclude-blacklist-act-eq",
        default=False,
        help="Exclude unquantized layers from activation equalization. Default: Disabled",
    )
    add_bool_arg(
        parser,
        "quantize-input-zero-point",
        default=False,
        help="Quantize input zero-point. Default: Enabled",
    )
    add_bool_arg(
        parser,
        "export-cpu-float32",
        default=False,
        help="Export FP32 on CPU. Default: Disabled",
    )
    add_bool_arg(
        parser,
        "use-ocp",
        default=False,
        help="Use OCP format for float quantization. Default: True",
    )
    add_bool_arg(
        parser,
        "use-fnuz",
        default=True,
        help="Use FNUZ format for float quantization. Default: True",
    )
    add_bool_arg(
        parser,
        "use-negative-prompts",
        default=True,
        help="Use negative prompts during generation/calibration. Default: Enabled",
    )
    add_bool_arg(
        parser, "quantize-sdp-1", default=False, help="Quantize SDP. Default: Disabled"
    )
    add_bool_arg(
        parser, "quantize-sdp-2", default=False, help="Quantize SDP. Default: Disabled"
    )
    args = parser.parse_args()
    print("Args: " + str(vars(args)))
    main(args)
