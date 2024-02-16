import torch
from transformers.models.phi import configuration_phi, modeling_phi
from transformers import AutoTokenizer

config = configuration_phi.PhiConfig()
config.hidden_size = 2560
config.intermediate_size = 10240
config.num_hidden_layers = 32
config.resid_pdrop = 0.0
config.partial_rotary_factor = 0.4
config.torch_dtype = "float16"

torch.set_default_device("cuda")

torch.set_printoptions(profile="full")

model = modeling_phi.PhiForCausalLM(config).half()

model.eval()

checkpoint = torch.load("C:\\Users\\xianz\\work\\Olive\\examples\\directml\\phi\\hf_converted\\phi-2_pytorch_model.bin")

model.load_state_dict(checkpoint, strict=False)

torch.set_printoptions(profile="full", sci_mode=False)

import time

model_dir = "C:\\Users\\xianz\\work\\Olive\examples\\directml\\llm\\models\\optimized\\microsoft_phi-2"
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)
max_length = 200
before_time = time.perf_counter()
outputs = model.generate(**inputs, max_length=max_length)
print (outputs.shape)
text = tokenizer.batch_decode(outputs)[0]

after_time = time.perf_counter()
duration = after_time - before_time
tokens_per_second = (max_length-23) / duration

print(f"Execution took {duration:0.4f} seconds (generated {tokens_per_second:0.2f} tokens per second)")

print(text)
