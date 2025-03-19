# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import glob
import json
import os
import readline
import shutil
from pathlib import Path

import onnx

from olive.common.utils import run_subprocess
from olive.model import ModelConfig
from olive.passes.onnx.common import model_proto_to_file
from olive.workflows import run as olive_run
from olive.workflows.run.config import RunConfig

# flake8: noqa: T201
# phi3-vision only supports CPU and CUDA targets for now
TARGETS = ["cpu", "cuda"]
config_path = Path(__file__).parent / "vision" / "config_templates"


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="phi3 optimization")

    parser.add_argument(
        "--target",
        type=str,
        default="cpu",
        required=False,
        choices=TARGETS,
        help=f"Choose from {TARGETS}",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="int4",
        choices=["int4", "fp16"],
        help=(
            (
                "Precision of optimized model. "
                "int4: run quantization on the model, which is able to run on CPU and CUDA."
                "fp16: no quantization, only run on CUDA."
            ),
        ),
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference with optimized model",
    )
    parser.add_argument(
        "--optimized_model_path",
        type=str,
        help="Run inference with optimized model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/phi3-vision-128k-instruct",
        required=False,
        help="Path to folder to store ONNX model and additional files (e.g. GenAI config, external data files, etc.)",
    )
    parser.add_argument(
        "--cache_dir",
        required=False,
        default="cache",
        help="Path to cache directory",
    )

    return parser.parse_args(raw_args)


def is_model_ready(input_model_path):
    if not input_model_path.exists():
        return False
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        config = AutoConfig.from_pretrained(input_model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(input_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(input_model_path, trust_remote_code=True)
        del model, processor, config
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    return True


def resave_onnx_model(model_path, target_output_path, pass_config):
    model_proto_to_file(
        onnx.load(model_path),
        target_output_path,
        save_as_external_data=pass_config.get("save_as_external_data", False),
        all_tensors_to_one_file=pass_config.get("all_tensors_to_one_file", True),
        external_data_name=pass_config.get("external_data_name", None),
        size_threshold=pass_config.get("size_threshold", 1024),
        convert_attribute=pass_config.get("convert_attribute", False),
    )


def run_and_save(config, output_dir, suffix):
    run_output = olive_run(config)
    output_node = next(iter(run_output.values())).get_top_ranked_nodes(1)[0]
    # "model_path" resource can be folder for model with external data
    model_path = ModelConfig.parse_file_or_obj(output_node.model_config).create_model().model_path
    pass_config = output_node.pass_run_config
    resave_onnx_model(model_path, output_dir / f"phi-3-v-128k-instruct-{suffix}.onnx", pass_config)


def main(raw_args=None):
    args = get_args(raw_args)

    if args.precision == "fp16" and args.target == "cpu":
        raise ValueError("fp16 precision is only supported on CUDA target, try --precision fp16 --target cuda instead")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.inference and args.optimized_model_path:
        generate(args.optimized_model_path)
        return

    input_model_path = output_dir / "pytorch"
    if not is_model_ready(input_model_path):
        print(f"Model not found from {input_model_path}, preparing the model...")
        # prepare the input model
        run_subprocess(
            [
                "bash",
                "vision/scripts/prepare_phi3_vision_for_olive.sh",
                str(output_dir),
            ],
            env=os.environ,
        )
    # if device is gpu, rewrite the genai_config.json
    genai_config_path = output_dir / "genai_config.json"
    with genai_config_path.open("r") as f:
        genai_config = json.load(f)
    if args.target == "cuda":
        genai_config["model"]["decoder"]["session_options"]["provider_options"] = [{"cuda": {}}]
    else:
        genai_config["model"]["decoder"]["session_options"]["provider_options"] = []
    with genai_config_path.open("w") as f:
        json.dump(genai_config, f, indent=4)

    text_embedding_config = generate_text_embedding_config(args, input_model_path)
    run_and_save(text_embedding_config, output_dir, "text-embedding")

    # Generate Olive configuration file for specific target
    print("\nGenerating Olive configuration file...")
    vision_config = generate_vision_config(args, input_model_path)
    run_and_save(vision_config, output_dir, "vision")

    text_config = generate_text_config(args, input_model_path)
    try:
        run_and_save(text_config, output_dir, "text")
    except Exception:
        config_link = "https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/tree/main/cpu-int4-rtn-block-32-acc-level-4."
        print(
            "Ignore the error during Olive run.\nThis script will copy the "
            "genai_config.json and processor_config.json from ",
            config_link,
        )
        # even if the Olive run fails, partially generated model
        # are still useful. Copy the model file to the output directory
        text_config = RunConfig.parse_obj(text_config)
        cache_dir = text_config.engine.cache_dir
        pass_config = text_config.passes["builder"].config
        # find the model file in the cache directory
        model_path = list(cache_dir.rglob("*.onnx"))[-1]
        resave_onnx_model(model_path, output_dir / "phi-3-v-128k-instruct-text.onnx", pass_config)
    print("Model generation completed, output saved to ", output_dir)
    # clean up the output directory in olive.workflows
    to_remove_folders = [
        Path(args.output_dir).resolve() / "vision",
        Path(args.output_dir).resolve() / "text",
        Path(args.output_dir).resolve() / "text_embedding",
    ]
    for folder in to_remove_folders:
        shutil.rmtree(folder, ignore_errors=True)

    if args.inference:
        generate(output_dir)


def generate_vision_config(args, input_model_path):
    config = json.load((config_path / "vision_config.json").open())
    config["input_model"]["model_path"] = input_model_path
    config["input_model"]["model_script"] = config_path.parent / "scripts" / "user_script.py"

    if args.precision == "fp16" or (args.precision == "int4" and args.target == "cuda"):
        config["passes"]["convert"]["torch_dtype"] = "float16"
    else:
        config["passes"]["convert"]["torch_dtype"] = "float32"

    if args.target == "cpu":
        config["passes"]["matmul_4bits"]["accuracy_level"] = 4
    else:
        config["passes"]["convert"]["device"] = "cuda"
        config["systems"]["local_system"]["accelerators"] = [
            {"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}
        ]
    if args.precision != "int4":
        del config["passes"]["matmul_4bits"]

    config["engine"] = {
        "cache_dir": Path(args.cache_dir).resolve() / "vision",
        "output_dir": Path(args.output_dir).resolve() / "vision",
    }
    return config


def generate_text_config(args, input_model_path):
    config = json.load((config_path / "text_config.json").open())
    config["input_model"]["model_path"] = input_model_path

    config["passes"]["builder"]["precision"] = args.precision

    if args.target == "cpu" and args.precision == "int4":
        config["passes"]["builder"]["int4_accuracy_level"] = 4
    elif args.target == "cuda":
        config["systems"]["local_system"]["accelerators"] = [
            {"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}
        ]
    config["engine"] = {
        "cache_dir": Path(args.cache_dir).resolve() / "text",
        "output_dir": Path(args.output_dir).resolve() / "text",
    }
    return config


def generate_text_embedding_config(args, input_model_path):
    config = json.load((config_path / "text_embedding_config.json").open())
    config["input_model"]["model_path"] = input_model_path

    if args.precision == "fp16" or (args.precision == "int4" and args.target == "cuda"):
        config["passes"]["convert"]["torch_dtype"] = "float16"
    else:
        config["passes"]["convert"]["torch_dtype"] = "float32"

    if args.target == "cuda":
        config["passes"]["convert"]["device"] = "cuda"
        config["systems"]["local_system"]["accelerators"] = [
            {"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}
        ]
    config["engine"] = {
        "cache_dir": Path(args.cache_dir).resolve() / "text_embedding",
        "output_dir": Path(args.output_dir).resolve() / "text_embedding",
    }
    return config


def _complete(text, state):
    return [*glob.glob(text + "*"), None][state]  # noqa: PTH207


def generate(model_path):
    import onnxruntime_genai as og

    print("Loading model...")
    model = og.Model(str(model_path))
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    while True:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_complete)
        image_path = input("Image Path (leave empty if no image, only local image supported): ")

        image = None
        prompt = "<|user|>\n"
        if len(image_path) == 0:
            print("No image provided")
        else:
            print("Loading image...")

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = og.Images.open(image_path)
            prompt += "<|image_1|>\n"

        text = input("Prompt: ")
        prompt += f"{text}<|end|>\n<|assistant|>\n"
        print("Processing image and prompt...")
        inputs = processor(prompt, images=image)

        print("Generating response...")
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=3072)

        generator = og.Generator(model, params)

        while not generator.is_done():
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)

        for _ in range(3):
            print()

        # Delete the generator to free the captured graph before creating another one
        del generator


if __name__ == "__main__":
    main()
