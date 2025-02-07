# Generate base model

python stable_diffusion.py --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 --provider qnn --optimize --use_random_data --data_num 1

# Generate image

python stable_diffusion.py --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 --provider qnn --num_inference_steps 5 --guidance_scale 7.5 --prompt "cat swims in the river" --seed 0
