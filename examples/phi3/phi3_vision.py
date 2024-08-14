# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import glob
import json
import os
import readline
from pathlib import Path

from olive.common.utils import run_subprocess
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
        choices=["int4"],
        help=("Precision of quantized model. Currently only int4 is supported. "),
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
        default="cache/phi3-vision-128k-instruct",
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


def build_text_embedding(args, input_model_path):
    #########################################
    # Functions/variables from model builder
    #########################################
    import numpy as np
    import torch
    from onnx import TensorProto, external_data_helper, helper, numpy_helper, save_model
    from transformers import AutoConfig, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(input_model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(input_model_path, trust_remote_code=True)

    # User inputs
    io_dtype = TensorProto.FLOAT16 if args.precision == torch.float16 else TensorProto.FLOAT
    os.makedirs(args.output_dir, exist_ok=True)

    # Map TensorProto dtypes
    to_torch_dtype = {
        TensorProto.FLOAT16: torch.float16,
        TensorProto.FLOAT: torch.float32,
    }
    to_numpy_dtype = {
        TensorProto.FLOAT16: np.float16,
        TensorProto.FLOAT: np.float32,
    }

    def make_external_tensor(np_data, name, **kwargs):
        tensor = numpy_helper.from_array(np_data)
        tensor.name = name

        filename = f"{name}.bin"
        external_data_helper.set_external_data(tensor, location=filename)
        with open(os.path.join(args.output_dir, filename), "wb") as f:
            f.write(tensor.raw_data)
        tensor.ClearField("raw_data")
        tensor.data_location = TensorProto.EXTERNAL

        return tensor

    # Make model
    embedding = model.model.embed_tokens.weight.to(to_torch_dtype[io_dtype]).detach().cpu().numpy()
    weight_name = "model.embed_tokens.weight"
    embed_weight = make_external_tensor(embedding.astype(to_numpy_dtype[io_dtype]), weight_name)
    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid("", 14), helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=7,
        producer_name="onnxruntime-genai-olive",
        producer_version="0.0.0",
        graph=helper.make_graph(
            name="main_graph",
            inputs=[
                helper.make_tensor_value_info("input_ids", TensorProto.INT64, shape=["batch_size", "sequence_length"])
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "inputs_embeds", io_dtype, shape=["batch_size", "sequence_length", config.hidden_size]
                )
            ],
            initializer=[embed_weight],
            value_info=[],
            nodes=[
                helper.make_node(
                    "Gather",
                    inputs=[weight_name, "input_ids"],
                    outputs=["inputs_embeds"],
                    name="/model/embed_tokens/Gather",
                )
            ],
        ),
    )

    external_data_helper.load_external_data_for_model(model, args.output_dir)

    # Delete external data files on disk before re-saving
    for path in os.listdir(args.output_dir):
        if path.endswith(".bin"):
            (Path(args.output_dir) / path).unlink()

    # Save ONNX model with only one external data file and delete any existing duplicate copies
    filename = "phi-3-v-128k-instruct-text-embedding.onnx"
    output_path = Path(args.output_dir) / filename
    save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{filename}.data",
        size_threshold=0,
        convert_attribute=False,
    )


def resave_onnx_model(model_path, target_output_path, pass_config):
    import onnx

    from olive.passes.onnx.common import model_proto_to_file

    model_proto_to_file(
        onnx.load(model_path),
        target_output_path,
        save_as_external_data=pass_config.get("save_as_external_data", False),
        all_tensors_to_one_file=pass_config.get("all_tensors_to_one_file", True),
        external_data_name=pass_config.get("external_data_name", None),
        size_threshold=pass_config.get("size_threshold", 1024),
        convert_attribute=pass_config.get("convert_attribute", False),
    )


def main(raw_args=None):
    args = get_args(raw_args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.inference and args.optimized_model_path:
        generate(args.optimized_model_path)
        return

    input_model_path = output_dir / "phi3-vision-128k-instruct" / "pytorch"
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

    build_text_embedding(args, input_model_path)

    # Generate Olive configuration file for specific target
    print("\nGenerating Olive configuration file...")
    vision_config = generate_vision_config(args, input_model_path)
    vision_output = olive_run(vision_config)
    output_node = next(iter(vision_output.values())).get_top_ranked_nodes(1)[0]
    model_path = output_node.model_config["config"]["model_path"]
    pass_config = output_node.pass_run_config
    resave_onnx_model(model_path, output_dir / "phi-3-v-128k-instruct-vision.onnx", pass_config)

    text_config = generate_text_config(args, input_model_path)
    try:
        text_output = olive_run(text_config)
        output_node = next(iter(text_output.values())).get_top_ranked_nodes(1)[0]
        model_path = output_node.model_config["config"]["model_path"]
        pass_config = output_node.pass_run_config
        resave_onnx_model(model_path, output_dir / "phi-3-v-128k-instruct-text.onnx", pass_config)
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

    if args.inference:
        generate(output_dir)


def generate_vision_config(args, input_model_path):
    config = json.load((config_path / "vision_config.json").open())
    config["input_model"]["model_path"] = input_model_path
    config["input_model"]["model_script"] = config_path.parent / "scripts" / "user_script.py"

    if args.target == "cpu":
        config["passes"]["convert"]["torch_dtype"] = "float32"
        config["passes"]["matmul_4bits"]["accuracy_level"] = 4
    else:
        config["passes"]["convert"]["torch_dtype"] = "float16"
        config["passes"]["convert"]["device"] = "cuda"
        config["systems"]["local_system"]["accelerators"] = [
            {"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}
        ]

    config["engine"] = {
        "cache_dir": Path(args.cache_dir).resolve() / "vision",
        "output_dir": Path(args.output_dir).resolve() / "vision",
    }
    return config


def generate_text_config(args, input_model_path):
    config = json.load((config_path / "text_config.json").open())
    config["input_model"]["model_path"] = input_model_path

    if args.target == "cpu":
        config["passes"]["builder"]["int4_accuracy_level"] = 4
    else:
        config["systems"]["local_system"]["accelerators"] = [
            {"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}
        ]
    config["engine"] = {
        "cache_dir": Path(args.cache_dir).resolve() / "text",
        "output_dir": Path(args.output_dir).resolve() / "text",
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
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)

        for _ in range(3):
            print()

        # Delete the generator to free the captured graph before creating another one
        del generator


if __name__ == "__main__":
    main()
