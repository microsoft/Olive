import numpy as np
from diffusers import DiffusionPipeline

image = np.zeros([1, 1024, 1024, 3])
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9")
prompt = "An astronaut riding a green horse"
images = pipe(prompt=prompt, image=image).images
