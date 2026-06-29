#!/usr/bin/env python
"""SD3 end-to-end inference using all ONNX components (text encoders + transformer + VAE).

Usage:
    python sd3_inference.py --prompt "A photo of a cat sitting on a windowsill"
    python sd3_inference.py --prompt "A futuristic city" --steps 50 --output city.png
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import CLIPTokenizer, T5TokenizerFast

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
ONNX_DIR = "exported_sd3_full2"


def encode_text(prompt: str, onnx_dir: str, model_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode prompt using ONNX CLIP-L, CLIP-G, and T5-XXL text encoders.

    Returns:
        encoder_hidden_states: [1, 410, 4096]
        pooled_projections: [1, 2048]

    """
    # Load tokenizers (lightweight, no model weights)
    tokenizer_l = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_g = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3")

    # Load ONNX sessions
    sess_l = ort.InferenceSession(os.path.join(onnx_dir, "text_encoder", "model.onnx"))
    sess_g = ort.InferenceSession(os.path.join(onnx_dir, "text_encoder_2", "model.onnx"))
    sess_t5 = ort.InferenceSession(os.path.join(onnx_dir, "text_encoder_3", "model.onnx"))

    # CLIP-L
    tokens_l = tokenizer_l(prompt, padding="max_length", max_length=77, return_tensors="np", truncation=True)
    out_l = sess_l.run(
        None,
        {
            "input_ids": tokens_l["input_ids"].astype(np.int64),
            "attention_mask": tokens_l["attention_mask"].astype(np.int64),
        },
    )
    clip_l_hidden = out_l[0]  # last_hidden_state [1, 77, 768]
    clip_l_pooled = out_l[1]  # text_embeds [1, 768]

    # CLIP-G
    tokens_g = tokenizer_g(prompt, padding="max_length", max_length=77, return_tensors="np", truncation=True)
    out_g = sess_g.run(
        None,
        {
            "input_ids": tokens_g["input_ids"].astype(np.int64),
            "attention_mask": tokens_g["attention_mask"].astype(np.int64),
        },
    )
    clip_g_hidden = out_g[0]  # last_hidden_state [1, 77, 1280]
    clip_g_pooled = out_g[1]  # text_embeds [1, 1280]

    # T5-XXL
    tokens_t5 = tokenizer_t5(prompt, padding="max_length", max_length=256, return_tensors="np", truncation=True)
    out_t5 = sess_t5.run(None, {"input_ids": tokens_t5["input_ids"].astype(np.int64)})
    t5_hidden = out_t5[0]  # last_hidden_state [1, 256, 4096]

    # Pad CLIP outputs to 4096 and concatenate
    clip_l_padded = np.pad(clip_l_hidden, ((0, 0), (0, 0), (0, 4096 - 768)))  # [1, 77, 4096]
    clip_g_padded = np.pad(clip_g_hidden, ((0, 0), (0, 0), (0, 4096 - 1280)))  # [1, 77, 4096]
    encoder_hidden_states = np.concatenate([clip_l_padded, clip_g_padded, t5_hidden], axis=1)  # [1, 410, 4096]
    pooled_projections = np.concatenate([clip_l_pooled, clip_g_pooled], axis=-1)  # [1, 2048]

    return encoder_hidden_states.astype(np.float32), pooled_projections.astype(np.float32)


def denoise(
    onnx_dir: str,
    encoder_hidden_states: np.ndarray,
    pooled_projections: np.ndarray,
    scheduler: FlowMatchEulerDiscreteScheduler,
    latent_shape: tuple = (1, 16, 64, 64),
    seed: int = 42,
) -> torch.Tensor:
    """Run the denoising loop using the ONNX transformer."""
    sess = ort.InferenceSession(os.path.join(onnx_dir, "transformer", "model.onnx"))

    torch.manual_seed(seed)
    latents = torch.randn(latent_shape)

    for i, t in enumerate(scheduler.timesteps):
        noise_pred = sess.run(
            None,
            {
                "sample": latents.numpy(),
                "timestep": np.array([t.item()], dtype=np.int64),
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_projections": pooled_projections,
            },
        )[0]
        latents = scheduler.step(torch.from_numpy(noise_pred), t, latents, return_dict=False)[0]
        if i % 7 == 0:
            print(f"  Step {i}/{len(scheduler.timesteps)}, t={t.item():.1f}")

    return latents


def decode_latents(latents: torch.Tensor, onnx_dir: str) -> np.ndarray:
    """Decode latents to image using the ONNX VAE decoder."""
    sess = ort.InferenceSession(os.path.join(onnx_dir, "vae_decoder", "model.onnx"))

    # SD3 VAE scaling: latents / scaling_factor + shift_factor
    # SD3 defaults: scaling_factor=1.5305, shift_factor=0.0609
    scaling_factor = 1.5305
    shift_factor = 0.0609
    latents_scaled = latents / scaling_factor + shift_factor

    output = sess.run(None, {"latent_sample": latents_scaled.numpy()})[0]
    # output: [1, 3, H, W] in [-1, 1]
    image = (output / 2 + 0.5).clip(0, 1)
    image = np.transpose(image[0], (1, 2, 0))  # [H, W, 3]
    return (image * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="SD3 all-ONNX inference")
    parser.add_argument("--prompt", default="A photo of a cat sitting on a windowsill")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="sd3_output.png")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--onnx_dir", default=ONNX_DIR)
    args = parser.parse_args()

    # Verify exported model exists
    transformer_path = os.path.join(args.onnx_dir, "transformer", "model.onnx")
    if not os.path.exists(transformer_path):
        print(f"Error: ONNX model not found at {args.onnx_dir}/")
        print(
            "Run: olive capture-onnx-graph --model_name_or_path "
            "stabilityai/stable-diffusion-3-medium-diffusers "
            "--use_mobius_builder --output_path exported_sd3_full2"
        )
        return

    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, Seed: {args.seed}")
    print(f"ONNX dir: {args.onnx_dir}")

    print("\n1. Encoding text (ONNX CLIP-L + CLIP-G + T5-XXL)...")
    encoder_hidden_states, pooled_projections = encode_text(args.prompt, args.onnx_dir, args.model_id)
    print(f"   encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"   pooled_projections: {pooled_projections.shape}")

    print("\n2. Denoising (ONNX SD3 transformer)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    scheduler.set_timesteps(args.steps)
    latents = denoise(args.onnx_dir, encoder_hidden_states, pooled_projections, scheduler, seed=args.seed)

    print("\n3. Decoding latents (ONNX VAE decoder)...")
    image = decode_latents(latents, args.onnx_dir)
    Image.fromarray(image).save(args.output)
    print(f"\nSaved: {args.output} ({image.shape[1]}x{image.shape[0]})")


if __name__ == "__main__":
    main()
