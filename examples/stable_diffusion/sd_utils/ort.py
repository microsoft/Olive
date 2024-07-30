# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import onnxruntime as ort
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline
from onnxruntime import __version__ as OrtVersion
from packaging import version

from olive.model import ONNXModelHandler

# ruff: noqa: TID252, T201


def update_cuda_config(config: Dict):
    if version.parse(OrtVersion) < version.parse("1.17.0"):
        # disable skip_group_norm fusion since there is a shape inference bug which leads to invalid models
        config["passes"]["optimize_cuda"]["optimization_options"] = {"enable_skip_group_norm": False}
    config["pass_flows"] = [["convert", "optimize_cuda"]]
    config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = ["CUDAExecutionProvider"]
    return config


def validate_args(args, provider):
    ort.set_default_logger_severity(4)
    if args.static_dims:
        print(
            "WARNING: the --static_dims option is deprecated, and static shape optimization is enabled by default. "
            "Use --dynamic_dims to disable static shape optimization."
        )

    validate_ort_version(provider)


def validate_ort_version(provider: str):
    if provider == "dml" and version.parse(OrtVersion) < version.parse("1.16.0"):
        print("This script requires onnxruntime-directml 1.16.0 or newer")
        sys.exit(1)
    elif provider == "cuda" and version.parse(OrtVersion) < version.parse("1.17.0"):
        if version.parse(OrtVersion) < version.parse("1.16.2"):
            print("This script requires onnxruntime-gpu 1.16.2 or newer")
            sys.exit(1)
        print(
            f"WARNING: onnxruntime {OrtVersion} has known issues with shape inference for SkipGroupNorm. Will disable"
            " skip_group_norm fusion. onnxruntime-gpu 1.17.0 or newer is strongly recommended!"
        )


def save_optimized_onnx_submodel(submodel_name, provider, model_info):
    footprints_file_path = (
        Path(__file__).resolve().parents[1] / "footprints" / f"{submodel_name}_gpu-{provider}_footprints.json"
    )
    with footprints_file_path.open("r") as footprint_file:
        footprints = json.load(footprint_file)

        conversion_footprint = None
        optimizer_footprint = None
        for footprint in footprints.values():
            if footprint["from_pass"] == "OnnxConversion":
                conversion_footprint = footprint
            elif footprint["from_pass"] == "OrtTransformersOptimization":
                optimizer_footprint = footprint

        assert conversion_footprint
        assert optimizer_footprint

        unoptimized_olive_model = ONNXModelHandler(**conversion_footprint["model_config"]["config"])
        optimized_olive_model = ONNXModelHandler(**optimizer_footprint["model_config"]["config"])

        model_info[submodel_name] = {
            "unoptimized": {
                "path": Path(unoptimized_olive_model.model_path),
            },
            "optimized": {
                "path": Path(optimized_olive_model.model_path),
            },
        }

        print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")
        print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")


def save_onnx_pipeline(
    has_safety_checker, model_info, optimized_model_dir, unoptimized_model_dir, pipeline, submodel_names
):
    # Save the unoptimized models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    print("\nCreating ONNX pipeline...")

    if has_safety_checker:
        safety_checker = OnnxRuntimeModel.from_pretrained(model_info["safety_checker"]["unoptimized"]["path"].parent)
    else:
        safety_checker = None

    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(model_info["vae_encoder"]["unoptimized"]["path"].parent),
        vae_decoder=OnnxRuntimeModel.from_pretrained(model_info["vae_decoder"]["unoptimized"]["path"].parent),
        text_encoder=OnnxRuntimeModel.from_pretrained(model_info["text_encoder"]["unoptimized"]["path"].parent),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(model_info["unet"]["unoptimized"]["path"].parent),
        scheduler=pipeline.scheduler,
        safety_checker=safety_checker,
        feature_extractor=pipeline.feature_extractor,
        requires_safety_checker=True,
    )

    print("Saving unoptimized models...")
    onnx_pipeline.save_pretrained(unoptimized_model_dir)

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    print("Copying optimized models...")
    shutil.copytree(unoptimized_model_dir, optimized_model_dir, ignore=shutil.ignore_patterns("weights.pb"))
    for submodel_name in submodel_names:
        src_path = model_info[submodel_name]["optimized"]["path"]
        dst_path = optimized_model_dir / submodel_name / "model.onnx"
        shutil.copyfile(src_path, dst_path)

    print(f"The optimized pipeline is located here: {optimized_model_dir}")


def get_ort_pipeline(model_dir, common_args, ort_args, guidance_scale):
    ort.set_default_logger_severity(3)

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False

    static_dims = not ort_args.dynamic_dims
    batch_size = common_args.batch_size
    image_size = common_args.image_size
    provider = common_args.provider

    if static_dims:
        hidden_batch_size = batch_size if (guidance_scale == 0.0) else batch_size * 2
        # Not necessary, but helps DML EP further optimize runtime performance.
        # batch_size is doubled for sample & hidden state because of classifier free guidance:
        # https://github.com/huggingface/diffusers/blob/46c52f9b9607e6ecb29c782c052aea313e6487b7/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L672
        sess_options.add_free_dimension_override_by_name("unet_sample_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        sess_options.add_free_dimension_override_by_name("unet_sample_height", image_size // 8)
        sess_options.add_free_dimension_override_by_name("unet_sample_width", image_size // 8)
        sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        sess_options.add_free_dimension_override_by_name("unet_hidden_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

    provider_map = {
        "dml": "DmlExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }
    assert provider in provider_map, f"Unsupported provider: {provider}"
    return OnnxStableDiffusionPipeline.from_pretrained(
        model_dir, provider=provider_map[provider], sess_options=sess_options
    )
