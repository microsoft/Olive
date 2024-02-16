
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


max_length = 590 + 23
inputs = tokenizer('''write a extremely long story starting with once upon a time''', return_tensors="pt", return_attention_mask=False)

before_time = time.perf_counter()

streamer = TextStreamer(tokenizer, skip_prompt=True)
outputs =  model.generate(**inputs, max_length=max_length, streamer=streamer)

after_time = time.perf_counter()
duration = after_time - before_time
tokens_per_second = (outputs.shape[1]-23) / duration

# print (outputs.shape)
print(f"Generated {outputs.shape[1]-23:0.0f} tokens in: {duration:0.4f} seconds (generated {tokens_per_second:0.2f} tokens per second)")
# # print (outputs)
# text = tokenizer.batch_decode(outputs)[0]
#print(text)
