from transformers import LlamaForCausalLM, LlamaTokenizer

prompt = "What data were you trained on?"

tokenizer = LlamaTokenizer.from_pretrained(r"C:\Users\pavignol\projects\transformers\llama_weights")
tokens = tokenizer(prompt, return_tensors="pt").input_ids

decoder_model = LlamaForCausalLM.from_pretrained(r"C:\Users\pavignol\projects\transformers\llama_weights").to("cuda")

generation_output = decoder_model.generate(input_ids=tokens.to("cuda"), max_new_tokens=32)
print(tokenizer.decode(generation_output[0]))
