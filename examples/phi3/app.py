import onnxruntime_genai as og
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

MODEL_PATH = args.model
ADAPTER_PATH = os.path.join(MODEL_PATH, "adapter_weights.npz")

phrase = "Calculate the sum of a list of integers."
# prompt = f"<|user|>\n{phrase}<|end|>\n<|assistant|>\n"
prompt = f"### Question: {phrase} \n### Answer: "

print("Loading model...")
model=og.Model(MODEL_PATH)
print("Model loaded.")

tokenizer = og.Tokenizer(model)
tokens = tokenizer.encode(prompt)

## load adapter data
tiny_codes_weights = np.load(ADAPTER_PATH)

# create zero weights for the base model
base_zero_weights = {key: np.zeros_like(value) for key, value in tiny_codes_weights.items()}
random_weights = {key: np.random.rand(*value.shape).astype(value.dtype) for key, value in tiny_codes_weights.items()}

params=og.GeneratorParams(model)
params.set_search_options(max_length=200)
params.input_ids = tokens

for key in tiny_codes_weights.keys():
   params.add_extra_input(key, tiny_codes_weights[key])
print("Tiny Codes adapter weights loaded.")

generator=og.Generator(model, params)
tokenizer_stream=tokenizer.create_stream()

print("Adapter results..")
while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()
    print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end='', flush=True)

print("================================================\n")
print("Base model results for comparison..")

## ======Base model======
params=og.GeneratorParams(model)
params.set_search_options(max_length=200)
params.input_ids = tokens


for key in base_zero_weights.keys():
   params.add_extra_input(key, base_zero_weights[key])
print("Base model zero weights adapter loaded.")

generator=og.Generator(model, params)
tokenizer_stream=tokenizer.create_stream()

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()
    print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end='', flush=True)
