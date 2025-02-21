import torch
from diffusers import StableDiffusionPipeline, OnnxStableDiffusionPipeline
import numpy as np

model_id = "stabilityai/stable-diffusion-2-1"
prompt = "folding electric bike"
prompt = "Road improvements coming along in west Gulfport"
np_generator = np.random.RandomState(seed=0)
np_latents = np_generator.randn(1, 4, 64, 64).astype(np.float32)

if not False:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    generator = torch.Generator().manual_seed(0)
    latents = torch.from_numpy(np_latents)
else:
    pipe = OnnxStableDiffusionPipeline.from_pretrained("models/unoptimized/" + model_id)
    generator = np_generator
    latents = np_latents
image = pipe(prompt,num_inference_steps=10,height=512,width=512,guidance_scale=0,generator=generator,latents=latents).images[0]
image.save(f"{prompt}.png")