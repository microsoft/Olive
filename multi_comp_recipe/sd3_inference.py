#!/usr/bin/env python
"""SD3 end-to-end inference using ONNX transformer + PyTorch text encoders and VAE.

Usage:
    python sd3_inference.py --prompt "A photo of a cat sitting on a windowsill"
    python sd3_inference.py --prompt "A futuristic city" --steps 50 --output city.png
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
ONNX_DIR = "out/transformer"


def encode_text(prompt: str, model_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode prompt using CLIP-L, CLIP-G, and T5-XXL text encoders.

    Returns:
        encoder_hidden_states: [1, 410, 4096]
        pooled_projections: [1, 2048]
    """
    tokenizer_l = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder_l = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32)

    tokenizer_g = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_g = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=torch.float32)

    tokenizer_t5 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3")
    text_encoder_t5 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3", torch_dtype=torch.float32)

    with torch.no_grad():
        # CLIP-L
        tokens_l = tokenizer_l(prompt, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        out_l = text_encoder_l(**tokens_l, output_hidden_states=True)
        clip_l_hidden = out_l.hidden_states[-2]  # [1, 77, 768]
        clip_l_pooled = out_l.pooler_output  # [1, 768]

        # CLIP-G
        tokens_g = tokenizer_g(prompt, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        out_g = text_encoder_g(**tokens_g, output_hidden_states=True)
        clip_g_hidden = out_g.hidden_states[-2]  # [1, 77, 1280]
        clip_g_pooled = out_g.pooler_output  # [1, 1280]

        # T5-XXL
        tokens_t5 = tokenizer_t5(prompt, padding="max_length", max_length=256, return_tensors="pt", truncation=True)
        t5_hidden = text_encoder_t5(**tokens_t5).last_hidden_state  # [1, 256, 4096]

    # Pad CLIP outputs to 4096 and concatenate
    clip_l_padded = torch.nn.functional.pad(clip_l_hidden, (0, 4096 - 768))  # [1, 77, 4096]
    clip_g_padded = torch.nn.functional.pad(clip_g_hidden, (0, 4096 - 1280))  # [1, 77, 4096]
    encoder_hidden_states = torch.cat([clip_l_padded, clip_g_padded, t5_hidden], dim=1)  # [1, 410, 4096]
    pooled_projections = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)  # [1, 2048]

    return encoder_hidden_states.numpy(), pooled_projections.numpy()


def denoise(
    onnx_path: str,
    encoder_hidden_states: np.ndarray,
    pooled_projections: np.ndarray,
    scheduler: FlowMatchEulerDiscreteScheduler,
    latent_shape: tuple = (1, 16, 64, 64),
    seed: int = 42,
) -> torch.Tensor:
    """Run the denoising loop using the ONNX transformer."""
    sess = ort.InferenceSession(onnx_path)

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


def decode_latents(latents: torch.Tensor, model_id: str) -> np.ndarray:
    """Decode latents to image using the VAE decoder."""
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    with torch.no_grad():
        latents_scaled = latents / vae.config.scaling_factor + vae.config.shift_factor
        image = vae.decode(latents_scaled, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).numpy()[0]
    return (image * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="SD3 inference with ONNX transformer")
    parser.add_argument("--prompt", default="A photo of a cat sitting on a windowsill")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="sd3_output.png")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--onnx_dir", default=ONNX_DIR)
    args = parser.parse_args()

    onnx_path = os.path.join(args.onnx_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        print("Run: olive capture-onnx-graph --model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers "
              "--use_mobius_builder --output_path out")
        return

    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, Seed: {args.seed}")

    print("\n1. Encoding text...")
    encoder_hidden_states, pooled_projections = encode_text(args.prompt, args.model_id)
    print(f"   encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"   pooled_projections: {pooled_projections.shape}")

    print("\n2. Denoising...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    scheduler.set_timesteps(args.steps)
    latents = denoise(onnx_path, encoder_hidden_states, pooled_projections, scheduler, seed=args.seed)

    print("\n3. Decoding latents...")
    image = decode_latents(latents, args.model_id)
    Image.fromarray(image).save(args.output)
    print(f"\nSaved: {args.output} ({image.shape[1]}x{image.shape[0]})")


if __name__ == "__main__":
    main()
