# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path

import onnxruntime_genai as og

logger = logging.getLogger(__name__)


def _find_dir_contains_sub_dir(current_dir: Path, target_dir_name):
    curr_path = Path(current_dir).absolute()
    target_dir = glob.glob(target_dir_name, root_dir=curr_path)
    if target_dir:
        return Path(curr_path / target_dir[0]).absolute()
    else:
        if curr_path.parent == curr_path:
            # Root dir
            return None
        return _find_dir_contains_sub_dir(curr_path / "..", target_dir_name)


def _complete(text, state):
    return (glob.glob(text + "*") + [None])[state]


def run(args: argparse.Namespace):
    logger.info("Loading model...")
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            logger.info(f"Setting model to {args.execution_provider}...")
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    logger.info("Model loaded")

    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()
    stream = processor.create_stream()

    interactive = not args.non_interactive

    while True:
        if interactive:
            try:
                import readline

                readline.set_completer_delims(" \t\n;")
                readline.parse_and_bind("tab: complete")
                readline.set_completer(_complete)
            except ImportError:
                # Not available on some platforms. Ignore it.
                pass
            image_paths = [
                image_path.strip()
                for image_path in input("Image Path (comma separated; leave empty if no image): ").split(",")
            ]
        else:
            if args.image_paths:
                image_paths = args.image_paths
            else:
                image_paths = [str(Path(__file__).parent / "images" / "dog.jpg")]

        image_paths = [image_path for image_path in image_paths if image_path]

        images = None
        if len(image_paths) == 0:
            logger.info("No image provided")
        else:
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                logger.info(f"Using image: {image_path}")

            images = og.Images.open(*image_paths)

        if interactive:
            text = input("Prompt: ")
        else:
            if args.prompt:
                text = args.prompt
            else:
                text = "What is shown in this image?"

        # Construct the "messages" argument passed to apply_chat_template
        messages = []
        if model.type == "phi3v":
            # Combine all image tags and text into one user message
            content = "".join([f"<|image_{i + 1}|>\n" for i in range(len(image_paths))]) + text
            messages.append({"role": "user", "content": content})
        else:
            # Gemma3-style multimodal: structured content
            content_list = [{"type": "image"} for _ in image_paths]
            content_list.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content_list})

        # Apply the chat template using the tokenizer
        message_json = json.dumps(messages)
        prompt = tokenizer.apply_chat_template(message_json, add_generation_prompt=True)

        logger.info("Processing images and prompt...")
        inputs = processor(prompt, images=images)

        logger.info("Generating response...")
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=1024)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        start_time = time.time()

        while not generator.is_done():
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            logger.info(stream.decode(new_token), end="", flush=True)

        total_run_time = time.time() - start_time
        logger.info(f"Total Time : {total_run_time:.2f}")

        # Delete the generator to free the captured graph before creating another one
        del generator

        if not interactive:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, default="", required=True, help="Path to the folder containing the model"
    )
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        required=False,
        default="follow_config",
        choices=["cpu", "cuda", "dml", "follow_config"],
        help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.",
    )
    parser.add_argument(
        "--image_paths", nargs="*", type=str, required=False, help="Path to the images, mainly for CI usage"
    )
    parser.add_argument(
        "-pr", "--prompt", required=False, help="Input prompts to generate tokens from, mainly for CI usage"
    )
    parser.add_argument(
        "--non-interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        required=False,
        help="Non-interactive mode, mainly for CI usage",
    )
    args = parser.parse_args()
    run(args)
