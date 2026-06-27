#!/usr/bin/env python
"""VLM (Phi-4-multimodal) inference using ORT GenAI with exported ONNX models.

Usage:
    # Text-only
    python vlm_inference.py --prompt "The capital of France is"

    # With image
    python vlm_inference.py --prompt "Describe this image." --image photo.jpg

    # Custom model directory
    python vlm_inference.py --model_dir exported_vlm_pkg --prompt "What is 2+2?"
"""

import argparse
import os

import onnxruntime_genai as og


def generate_text(model_dir: str, prompt: str, max_new_tokens: int = 128) -> str:
    """Run text-only generation."""
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)

    input_ids = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + max_new_tokens)

    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)

    tokenizer_stream = tokenizer.create_stream()
    generated = []
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        generated.append(token)
        print(tokenizer_stream.decode(token), end="", flush=True)
        if len(generated) >= max_new_tokens:
            break

    print()
    del generator
    return tokenizer.decode(generated)


def generate_with_image(model_dir: str, prompt: str, image_path: str, max_new_tokens: int = 128) -> str:
    """Run multimodal generation with image input."""
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()

    images = og.Images.open(image_path)
    inputs = processor(prompt, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    tokenizer_stream = tokenizer.create_stream()
    generated = []
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        generated.append(token)
        print(tokenizer_stream.decode(token), end="", flush=True)
        if len(generated) >= max_new_tokens:
            break

    print()
    del generator
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="VLM inference with ORT GenAI")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--image", default=None, help="Path to an image file for vision input")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--model_dir", default="exported_vlm_pkg")
    args = parser.parse_args()

    genai_config = os.path.join(args.model_dir, "genai_config.json")
    if not os.path.exists(genai_config):
        print(f"Error: genai_config.json not found in {args.model_dir}")
        print("Run export first:")
        print(
            "  olive capture-onnx-graph --model_name_or_path microsoft/Phi-4-multimodal-instruct "
            "--use_mobius_builder --trust_remote_code --output_path exported_vlm_pkg"
        )
        print("Then create genai_config.json and save tokenizer (see README.md).")
        return

    print(f"Model: {args.model_dir}")
    print(f"Prompt: {args.prompt}")
    if args.image:
        print(f"Image: {args.image}")
    print("-" * 50)

    if args.image:
        output = generate_with_image(args.model_dir, args.prompt, args.image, args.max_new_tokens)
    else:
        output = generate_text(args.model_dir, args.prompt, args.max_new_tokens)

    print("-" * 50)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
