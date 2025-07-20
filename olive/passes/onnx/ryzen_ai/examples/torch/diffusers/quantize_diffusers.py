# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import cv2
import glob
import json
import types
import torch
import torch.nn as nn
import random
import argparse
import fnmatch
import numpy as np
import smoothquant
from PIL import Image
from dataclasses import replace
from torch.utils.data import Dataset
from diffusers import DiffusionPipeline, ControlNetModel, \
    StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, \
    StableDiffusion3Pipeline, AutoencoderKL, UniPCMultistepScheduler, \
    FluxPipeline
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.utils import load_image
from quark.torch import ModelQuantizer, ModelExporter, load_params
from quark.torch.quantization import Bfloat16Spec, \
    FP8E4M3PerTensorSpec, Int8PerTensorSpec, Int8PerChannelSpec, \
    Int4PerTensorSpec, Int4PerChannelSpec
from quark.torch.quantization.config.config import Config, QuantizationConfig
from quark.torch.export import ExporterConfig, OnnxExporterConfig
from tqdm import tqdm

FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max",
                                           is_dynamic=False).to_quantization_spec()

INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(observer_method="min_max",
                                         symmetric=True,
                                         scale_type="float",
                                         round_method="half_even",
                                         is_dynamic=False).to_quantization_spec()

INT8_PER_CHANNEL_SPEC = Int8PerChannelSpec(symmetric=True,
                                           scale_type="float",
                                           round_method="half_even",
                                           ch_axis=0,
                                           is_dynamic=False).to_quantization_spec()

INT4_PER_TENSOR_SPEC = Int4PerTensorSpec(observer_method="min_max",
                                         symmetric=True,
                                         scale_type="float",
                                         round_method="half_even",
                                         is_dynamic=False).to_quantization_spec()

INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(symmetric=True,
                                           scale_type="float",
                                           round_method="half_even",
                                           ch_axis=0,
                                           is_dynamic=False).to_quantization_spec()

BFLOAT16_SPEC = Bfloat16Spec().to_quantization_spec()

ADDTIONAL_ARGS = {
    "flux-dev": {
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
    },
    "sdxl_shape": {
        'batch_size': 2,
        'sequence_length': 16,
        'num_choices': 4,
        'width': 64,
        'height': 64,
        'num_channels': 3,
        'point_batch_size': 3,
        'nb_points_per_image': 2,
        'feature_size': 80,
        'nb_max_frames': 3000,
        'audio_sequence_length': 16000
    }
}

def lora_forward(self, x, scale=None):
    return self._torch_forward(x)

def replace_lora_layers(module, exclude_layers, parent_name=""):
    def filter_by_name(test_module_name):
        for name_pattern in exclude_layers:
            if fnmatch.fnmatch(test_module_name, name_pattern):
                return True
        return False

    for name, child in module.named_children():
        full_name = parent_name + "." + name
        if filter_by_name(full_name):
            continue
        if isinstance(child, LoRACompatibleConv):
            in_channels = child.in_channels
            out_channels = child.out_channels
            kernel_size = child.kernel_size
            stride = child.stride
            padding = child.padding
            dilation = child.dilation
            groups = child.groups
            bias = child.bias

            new_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias is not None,
            )
            new_conv.weight.data = child.weight.data.clone().to(child.weight.data.device)
            if bias is not None:
                new_conv.bias.data = child.bias.data.clone().to(child.bias.data.device)
            setattr(module, name, new_conv)
            new_conv._torch_forward = new_conv.forward
            new_conv.forward = types.MethodType(lora_forward, new_conv)

        elif isinstance(child, LoRACompatibleLinear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias
            new_linear = nn.Linear(in_features, out_features, bias=bias is not None)
            new_linear.weight.data = child.weight.data.clone().to(child.weight.data.device)
            if bias is not None:
                new_linear.bias.data = child.bias.data.clone().to(child.bias.data.device)
            setattr(module, name, new_linear)
            new_linear._torch_forward = new_linear.forward
            new_linear.forward = types.MethodType(lora_forward, new_linear)

        replace_lora_layers(child, exclude_layers, full_name)

    for name, child in module._modules.items():
        if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
            for sub_name, sub_child in child.named_children():
                full_name = parent_name + "." + name + "." + sub_name
                if filter_by_name(full_name):
                    continue
                if isinstance(sub_child, LoRACompatibleConv):
                    in_channels = sub_child.in_channels
                    out_channels = sub_child.out_channels
                    kernel_size = sub_child.kernel_size
                    stride = sub_child.stride
                    padding = sub_child.padding
                    dilation = sub_child.dilation
                    groups = sub_child.groups
                    bias = sub_child.bias

                    new_conv = nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias is not None,
                    )
                    new_conv.weight.data = sub_child.weight.data.clone().to(sub_child.weight.data.device)
                    if bias is not None:
                        new_conv.bias.data = sub_child.bias.data.clone().to(sub_child.bias.data.device)
                    setattr(child, sub_name, new_conv)
                    new_conv._torch_forward = new_conv.forward
                    new_conv.forward = types.MethodType(lora_forward, new_conv)

                elif isinstance(sub_child, LoRACompatibleLinear):
                    in_features = sub_child.in_features
                    out_features = sub_child.out_features
                    bias = sub_child.bias

                    new_linear = nn.Linear(in_features, out_features, bias=bias is not None)

                    new_linear.weight.data = sub_child.weight.data.clone().to(sub_child.weight.data.device)
                    if bias is not None:
                        new_linear.bias.data = sub_child.bias.data.clone().to(sub_child.bias.data.device)
                    setattr(child, sub_name, new_linear)
                    new_linear._torch_forward = new_linear.forward
                    new_linear.forward = types.MethodType(lora_forward, new_linear)

                replace_lora_layers(sub_child, exclude_layers, full_name)


def get_export_config() -> ExporterConfig:
    export_config = ExporterConfig(json_export_config=None, onnx_export_config=OnnxExporterConfig())
    # export_config.json_export_config.min_kv_scale = 1.0
    return export_config


class CustomizedExit(Exception):
    pass


class WrappingModelForDumpData(torch.nn.Module):
    def __init__(self, model, register_name):
        super(WrappingModelForDumpData, self).__init__()
        self.call_count = 0
        self.inner_model = model
        self.register_name = register_name
        self.dump_data_folder = g_args.dump_data_folder
        os.makedirs(self.dump_data_folder, exist_ok=True)

    def forward(self, *args, **kwargs):
        os.makedirs(os.path.join(self.dump_data_folder, self.register_name), exist_ok=True)
        # dump module input
        torch.save({
            "args" : args,
            "kwargs" : kwargs
            },
            os.path.join(self.dump_data_folder, self.register_name, self.register_name + "_" + str(self.call_count) + ".pt")
        )
        if self.register_name == 'unet':
            self.call_count = self.call_count + 1
            if self.call_count % g_args.n_steps == 0 and self.call_count != 0:
                print("save input number:", self.call_count)
                raise CustomizedExit
            else:
                return self.inner_model(*args, **kwargs)
        elif self.register_name == 'vae.decoder':
            self.call_count = self.call_count + 1
            return self.inner_model(*args, **kwargs)
        else:
            self.call_count = self.call_count + 1
            raise CustomizedExit

    def __getattr__(self, name):
        if name in self.__dict__.keys():
            return self.__dict__[name]
        if name in self._modules.keys():
            return self._modules[name]
        return getattr(self.inner_model, name)


class WrappingModelForExtendInterface(torch.nn.Module):
    def __init__(self, model):
        super(WrappingModelForExtendInterface, self).__init__()
        self.inner_model = model

    def forward(self, data):
        return self.inner_model(*data[0][0], **data[0][1])

    def __getattr__(self, name):
        if name in self.__dict__.keys():
            return self.__dict__[name]
        if name in self._modules.keys():
            return self._modules[name]
        return getattr(self.inner_model, name)


class DumpDatasetFrom(Dataset):
    def __init__(self, module_name):
        self.is_fp32 = True if module_name == 'vae.decoder' else False
        self.data_list = get_files_with_prefix(g_args.dump_data_folder, module_name)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        device_count = torch.cuda.device_count()
        map_location = torch.device(f"cuda:{device_count - 1}") if device_count > 0 else torch.device("cpu")
        dump_data = torch.load(self.data_list[index], map_location=map_location)
        if self.is_fp32:
            dump_data['args'] = list(dump_data['args'])
            dump_data['args'][0] = dump_data['args'][0].to(torch.float32)
        return ((dump_data['args'], dump_data['kwargs']),)


def get_files_with_prefix(directory, prefix):
    files_with_prefix = []
    try:
        for filename in os.listdir(os.path.join(directory, prefix)):
            if filename.startswith(prefix):
                files_with_prefix.append(os.path.join(directory, prefix, filename))
    except FileNotFoundError:
        print(f"The directory {directory} does not exist")
    return files_with_prefix

def dump_input_data(module_name, dump_pipe, prompts_for_calib, latents):
    org_module = dump_pipe.__dict__[module_name]
    if module_name == 'vae':
        org_decoder = dump_pipe.vae.decoder
        dump_pipe.vae.decoder = WrappingModelForDumpData(dump_pipe.vae.decoder, 'vae.decoder')
    else:
        dump_pipe.__dict__[module_name] = WrappingModelForDumpData(dump_pipe.__dict__[module_name], module_name)

    for i in tqdm(range(len(prompts_for_calib)), desc=f"Dumping input data of {module_name}"):
        prompt = prompts_for_calib[i]
        try:
            get_image_by_prompt(prompt, dump_pipe, latents)
        except CustomizedExit:
            pass

    dump_pipe.__dict__[module_name] = org_module
    if module_name == 'vae':
        dump_pipe.vae.decoder = org_decoder

def get_dataset_prompts(cocos_file_path):
    # Read prompts from a file and return as a list of dictionaries
    prompts = []
    with open(cocos_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split("\t")
            prompts.append(arr[2])
    prompts = prompts[1:]
    return prompts

def setup_seed():
    seed = g_args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def quant_module(pipe, module_name, quant_config, exclude_layers, algo_configs, device):
    if module_name == 'vae':
        dataloader = DumpDatasetFrom('vae.decoder')
        model = WrappingModelForExtendInterface(pipe.vae.decoder)
        quant_config = replace(quant_config, exclude=exclude_layers)
        # Apply smooth quant
        for algo_config in algo_configs:
            if algo_config["name"] == "smooth":
                cache_act_scales = None
                cache_act_scales = smoothquant.calibrate_smoothquant(model.to(torch.float32), dataloader)
                smoothquant.apply_smoothquant(model, cache_act_scales, exclude_layers=exclude_layers, alpha=algo_config["alpha"])

        quantizer = ModelQuantizer(quant_config)
        dtype = next(model.parameters()).dtype
        quant_model = quantizer.quantize_model(model.to(torch.float32), dataloader)
        pipe.vae.decoder = quant_model.inner_model.to(dtype)
    else:
        dataloader = DumpDatasetFrom(module_name)
        model = WrappingModelForExtendInterface(pipe.__dict__[module_name])
        if module_name == 'unet':
            replace_lora_layers(model, exclude_layers)
        quant_config = replace(quant_config, exclude=exclude_layers)
        # Apply smooth quant
        for algo_config in algo_configs:
            if algo_config["name"] == "smooth":
                cache_act_scales = None
                cache_act_scales = smoothquant.calibrate_smoothquant(model, dataloader)
                smoothquant.apply_smoothquant(model, cache_act_scales, exclude_layers=exclude_layers, alpha=algo_config["alpha"])

        quantizer = ModelQuantizer(quant_config)
        quant_model = quantizer.quantize_model(model, dataloader)
        pipe.__dict__[module_name] = quant_model.inner_model


@torch.no_grad()
def test_coco2014_dataset(pipe, latents):
    if not os.path.exists(g_args.save_images_dir):
        os.makedirs(g_args.save_images_dir)

    test_prompts = get_dataset_prompts(g_args.test_prompts)
    test_prompts = test_prompts[:g_args.test_size]
    for idx in tqdm(range(len(test_prompts)), desc="Generating Images"):
        prompt = test_prompts[idx]
        image = get_image_by_prompt(prompt, pipe, latents=latents)
        image.save(os.path.join(g_args.save_images_dir, str(idx) + ".png"))
        with open(os.path.join(g_args.save_images_dir, str(idx) + ".txt"), 'w') as f:
            f.write(prompt)


def find_image_files(directory):
    image_files = []
    pattern_png = os.path.join(directory, '**', '*.png')
    png_files = glob.glob(pattern_png, recursive=True)
    image_files.extend(png_files)
    return image_files


def evaluating_coco_result():
    img_files = find_image_files(g_args.save_images_dir)
    image_numpy = []
    caption_list = []
    for i in tqdm(range(len(img_files)), desc="Getting Caption List"):
        img = img_files[i]
        image = Image.open(img)
        image_numpy.append(np.array(image, dtype=np.uint8))
        with open(img[:-3] + 'txt', 'r') as f:
            data = f.read()
            caption_list.append(data)
    from tools.clip.clip_encoder import CLIPEncoder
    from tools.fid.fid_score import compute_fid

    # clip score
    clip_scores = []
    clip = CLIPEncoder(device=torch.device('cuda'))
    for k in tqdm(range(len(caption_list)), desc="Computing Clip Scores"):
        caption = caption_list[k]
        generated = Image.fromarray(image_numpy[k])
        clip_scores.append(
            100 * clip.get_clip_score(caption, generated).item()
        )

    clip_score = np.mean(clip_scores)
    print("clip_score:", clip_score)

    # fid score
    print("Computing FID:")
    statistics_path = "./inference/text_to_image/tools/val2014.npz"
    fid = compute_fid(image_numpy, statistics_path, torch.device('cuda'))
    print("fid:", fid)


def get_image_by_prompt(prompt, pipe, latents=None):
    prompts = [prompt]
    negative_prompt = ["normal quality, low quality, worst quality, low res, blurry, nsfw, nude."]
    if g_args.controlnet_id is not None:
        if "lllyasviel/control_v11p_sd15_canny" in g_args.controlnet_id:
            # load image: input str or PIL.Image.Image
            image = load_image(g_args.input_image)
            # get canny image: the process can be modified
            image = np.array(image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image)
            image.save(os.path.join(g_args.export_path, "sd15_canny_control.png"))
        elif "diffusers/controlnet-canny-sdxl-1.0" in g_args.controlnet_id:
            # load image: input str or PIL.Image.Image
            image = load_image(g_args.input_image)
            # get canny image: the process can be modified
            image = np.array(image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image)
            image.save(os.path.join(g_args.export_path, "sdxl_canny_control.png"))

        if "stable-diffusion-3" in g_args.model_id.lower() or "flux" in g_args.model_id.lower():
            latents = None

        # generate image
        image = pipe(
            prompt=prompts,
            negative_prompt=negative_prompt * len(prompts),
            image=image,
            num_inference_steps=g_args.n_steps,
            controlnet_conditioning_scale=g_args.controlnet_conditioning_scale,
            latents=latents,
            guidance_scale=8.0
        ).images[0]

    # flux
    elif "flux.1-dev" in g_args.model_id.lower():
        image = pipe(
            prompt=prompts,
            num_inference_steps=g_args.n_steps,
            **ADDTIONAL_ARGS['flux-dev']
        ).images[0]

    # diffusion
    else:
        image = pipe(
            prompt=prompts,
            num_inference_steps=g_args.n_steps,
            negative_prompt=negative_prompt * len(prompts),
            latents=latents,
            guidance_scale=8.0
        ).images[0]

    return image


def main(g_args: argparse.Namespace):
    setup_seed()
    latents_path = "inference/text_to_image/tools/latents.pt"
    latents = torch.load(latents_path).to(torch.float16)
    # sdxl controlnet
    if g_args.controlnet_id is not None:
        if 'lllyasviel/control_v11p_sd15_canny' in g_args.controlnet_id:
            controlnet = ControlNetModel.from_pretrained(
                g_args.controlnet_id,
                torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                g_args.model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
        elif 'diffusers/controlnet-canny-sdxl-1.0' in g_args.controlnet_id:
            controlnet = ControlNetModel.from_pretrained(
                g_args.controlnet_id,
                torch_dtype=torch.float16
            )
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                g_args.model_id,
                controlnet=controlnet,
                vae=vae,
                torch_dtype=torch.float16,
            )
        else:
            raise ValueError(f"Controlnet {g_args.controlnet_id} is not supported. Only support diffusers/controlnet-canny-sdxl-1.0")

        pipe.to(g_args.device)

    # flux
    elif "flux.1" in g_args.model_id.lower():
        pipe = FluxPipeline.from_pretrained(
                g_args.model_id,
                torch_dtype=torch.bfloat16,
                device_map="balanced",
                )
        latents = None

    # stable diffusion 3.5
    elif "stable-diffusion-3.5" in g_args.model_id.lower():
        pipe = StableDiffusion3Pipeline.from_pretrained(
                g_args.model_id,
                torch_dtype=torch.bfloat16
                )
        pipe.to(g_args.device)
        latents = None

    # stable diffusion 3
    elif "stable-diffusion-3" in g_args.model_id.lower():
        pipe = StableDiffusion3Pipeline.from_pretrained(
                g_args.model_id,
                torch_dtype=torch.float16
                )
        pipe.to(g_args.device)
        latents = None

    # stable diffusion
    else:
        pipe = DiffusionPipeline.from_pretrained(
            g_args.model_id,
            torch_dtype=torch.float16,
            variant="fp16" ,
            use_safetensors=True,
            device_map="balanced",
        )

    # Skip quant
    if g_args.skip_quantization:
        prompt = "A cute cat run in city."
        image = get_image_by_prompt(prompt, pipe, latents=latents)
        image.save("orgin_sdxl_example.png")

    # Load unet
    elif g_args.load:
        wrapping_model = WrappingModelForExtendInterface(pipe.unet)
        json_path = os.path.join(g_args.export_path, g_args.model_name + ".json")
        safetensors_path = os.path.join(g_args.export_path, g_args.model_name + ".safetensors")
        wrapping_model = load_params(wrapping_model, json_path=json_path, safetensors_path=safetensors_path)
        pipe.unet = wrapping_model.inner_model

    # quantize
    else:
        # 1. Load pipeline quant config
        with open(g_args.quant_config_file_path) as file:
            pipeline_config = json.load(file)

        # 2. Determine module names
        if g_args.controlnet_id is None:
            module_names = pipeline_config["module_names"]
        else:
            module_names = ["unet"]

        # 3. (Optional) Collect dump data
        if not g_args.using_cache_input_data:
            prompts_for_calib = get_dataset_prompts(g_args.calib_prompts)
            prompts_for_calib = prompts_for_calib[:g_args.calib_size]
            for module_name in module_names:
                print(f"\n[INFO]: Dumping input data of {module_name} ...")
                dump_input_data(module_name, pipe, prompts_for_calib, latents)

        # 4. Quantization
        quant_configs = pipeline_config["quant_configs"]
        for module_name in module_names:
            if module_name not in quant_configs:
                continue
            print(f"\n[INFO]: Quantizing {module_name} ...")

            # 4-1. Set quant scheme
            module_quant_config = quant_configs[module_name]
            quant_scheme = module_quant_config["quant_scheme"]
            layer_type_quant_config = {}
            if quant_scheme == 'w_fp8_a_fp8':
                config = QuantizationConfig(weight=FP8_PER_TENSOR_SPEC, input_tensors=FP8_PER_TENSOR_SPEC)
            elif quant_scheme == 'w_int4_per_channel_sym':
                config = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)
            elif quant_scheme == 'w_int8_per_tensor_sym':
                config = QuantizationConfig(weight=INT8_PER_TENSOR_SPEC)
            elif quant_scheme == 'w_int8_a_int8':
                ConvConfig = QuantizationConfig(weight=INT8_PER_CHANNEL_SPEC, input_tensors=INT8_PER_TENSOR_SPEC)
                config = QuantizationConfig(weight=INT8_PER_CHANNEL_SPEC, input_tensors=INT8_PER_TENSOR_SPEC)
                layer_type_quant_config = {torch.nn.Conv2d: ConvConfig}
            elif quant_scheme == 'bf16':
                config = QuantizationConfig(weight=BFLOAT16_SPEC, input_tensors=BFLOAT16_SPEC)
            else:
                config = None
            quant_config = Config(global_quant_config=config, layer_type_quant_config=layer_type_quant_config)

            # 4-2. Set exclude layers
            exclude_layers = module_quant_config["exclude_layers"]

            # 4-3. Set quant algorithm
            algo_configs = module_quant_config["algo_configs"]

            # 4-4. Quantize module
            quant_module(pipe, module_name, quant_config, exclude_layers, algo_configs, g_args.device)

        if g_args.print_debug_info:
            model_name = g_args.model_id.replace("/", "_")
            with open(f"{model_name}.txt", "w") as f:
                f.write(str(pipe.__dict__[module_name]))

        # (Optional) Export
        if g_args.export == "onnx":
            # SDXL: export all modules by optimum
            if g_args.controlnet_id is None and ("stable-diffusion-xl-base-1.0" in g_args.model_id or "sdxl-turbo" in g_args.model_id or "stable-diffusion-v1-5" in g_args.model_id):
                from optimum.exporters.onnx.convert import onnx_export_from_model
                print("\n[INFO]: Exporting diffusion model to ONNX ...")
                dtype = pipe.dtype
                onnx_export_from_model(
                    model=pipe.to(torch.float32),
                    output=g_args.export_path,
                    monolith=False,
                    no_post_process=False,
                    do_validation=True,
                    _variant='default',
                    legacy=False,
                    preprocessors=[],
                    device='cuda',
                    no_dynamic_axes=False,
                    task='text-to-image',
                    use_subprocess=True,
                    do_constant_folding=False,
                    **ADDTIONAL_ARGS['sdxl_shape']
                )
                pipe.to(dtype)

            # Export quantized modules one by one
            else:
                for module_name in module_names:
                    if module_name not in quant_configs:
                        continue
                    print(f"\n[INFO]: Exporting onnx graph of {module_name} ...")
                    with torch.inference_mode():
                        dataloader = DumpDatasetFrom(module_name)
                        input_data = dataloader.__getitem__(0)
                        if quant_scheme in ["w_int4_per_channel_sym", "w_uint4_per_group_asym", "w_int4_per_group_sym", "w_uint4_a_bfloat16_per_group_asym"]:
                            uint4_int4_flag = True
                        else:
                            uint4_int4_flag = False
                        export_dir = g_args.export_path + f"/{module_name}"
                        export_config = get_export_config()
                        exporter = ModelExporter(config=export_config, export_dir=export_dir)
                        exporter.export_onnx_model(pipe.__dict__[module_name], input_data, uint4_int4_flag=uint4_int4_flag)

        elif g_args.export == "safetensor":
            from quark.torch import save_params
            for module_name in module_names:
                if module_name not in quant_configs:
                    continue
                print(f"\n[INFO]: Exporting {module_name} to safetensors ...")
                dataloader = DumpDatasetFrom("vae.decoder") if module_name == "vae" else DumpDatasetFrom(module_name)
                input_data = dataloader.__getitem__(0)
                save_params(pipe.__dict__[module_name], model_type=module_name, export_dir=g_args.export_path)

    if len(g_args.prompt.strip()) != 0:
        image = get_image_by_prompt(g_args.prompt, pipe, latents)
        save_name = g_args.prompt.replace(" ", "_").replace(".", "_")
        image.save(os.path.join(g_args.export_path, save_name + ".png"))

    if g_args.test:
        print("\n[INFO]: Evaluating model ...")
        test_coco2014_dataset(pipe, latents)
        evaluating_coco_result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Model quantization and evaluation script.")
    # Argument for model
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="the model id from the diffusers")
    parser.add_argument("--device", help="Device for running the quantizer", default="cuda", choices=["cuda"])
    parser.add_argument("--n_steps", type=int, default=20, help="Number of steps for dumping input data during calibration.")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed for reproducibility.")
    parser.add_argument("--prompt", type=str, default="", help="a prompt string to gennerate a image")

    # Argument for controlnet
    parser.add_argument("--controlnet_id", type=str, default=None, help="The controlnet from the diffusers.")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.5, help="The outputs of the ControlNet are multiplied by the scale  \
                        before they are added to the residual in the original unet. Recommend 0.5 for good generalization.")
    parser.add_argument("--input_image", type=str,
                        default="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png",
                        help="The image of constraint to guide the diffusion process in controlnet.")

    # Argument for calibration dataset
    parser.add_argument("--using_cache_input_data", action="store_true", help="Use the previous cached dump input data.")
    parser.add_argument("--dump_data_folder", default="coco2014_calib_data", type=str, help="Folder path to store dumped input data.")
    parser.add_argument("--calib_prompts", type=str, default="", help="The file path for calibration data ")
    parser.add_argument("--calib_size", type=int, default=500, help="Number of steps for dumping input data during calibration.")

    # Argument for sdxl pipeline quantization
    parser.add_argument("--skip_quantization", action="store_true", help="Perform inference using floating-point precision without quantization.")
    parser.add_argument("--quant_config_file_path", type=str, default=None)

    # Argument for save model
    parser.add_argument("--load", action="store_true", help="Load a pre-exported model for inference.")
    parser.add_argument("--model_name", type=str, default="sdxl", help="Name of the model to be loaded or exported.")
    parser.add_argument("--export", default=None, type=str, choices=["safetensor", "onnx", None])
    parser.add_argument("--export_path", default="./quantized_models", type=str, help="Folder path to store quantized models.")

    # Argument for evaluation
    parser.add_argument("--test", action="store_true", help="Run coco2014 test dataset.")
    parser.add_argument("--test_prompts", type=str, default="", help="The file path for test data ")
    parser.add_argument("--test_size", type=int, default=5000, help="Number of steps for dumping input data during calibration.")
    parser.add_argument("--save_images_dir", type=str, default="test_coco2014_result", help="the dir for save images")

    # Argument for debug
    parser.add_argument("--print_debug_info", action="store_true", help="print some debug information.")

    g_args = parser.parse_args()

    main(g_args)
