from transformers import AutoTokenizer
import torch
import config
from user_script import get_or_create_decoder_model
from llm import set_config_parameters
import transformers

# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

device = "cuda"
data_type = torch.float16

# pipeline = transformers.pipeline(
#     "text-generation", model="meta-llama/Llama-2-7b-chat-hf", tokenizer=tokenizer, torch_dtype=torch.float32, device="cpu"
# )

# config.num_layers = 32
# config.vocab_size = 65024
# config.hidden_size = 4544
# config.intermediate_size = config.hidden_size * 4
# config.num_heads = 71
# config.num_key_value_heads = 1
# config.normalization_type = "layer_norm"
# config.epsilon = 1e-05
# config.state_dict = pipeline.model.state_dict()
# config.strict_weights_loading = True

torch.set_default_dtype(data_type)

set_config_parameters("", config.num_layers)

torch.set_default_device(device)
# torch.set_printoptions(profile="full", sci_mode=False)

model = get_or_create_decoder_model()


# ---------------------------
import os
# Initialize the series of tokens to the prompt
# prompt = "Once upon a time, there was a "
model_dir = "C:\\Users\\xianz\\work\\Olive\examples\\directml\\llm\\models\\optimized\\microsoft_phi-2"
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
x = tokenizer('''def print_prime(n):
    """
    Print all primes between 1 and n
    """''', return_tensors="pt", return_attention_mask=False)["input_ids"]

print (x)
max_seq_len = 50

seq_len = x.shape[1]

head_dim = int(config.hidden_size / config.num_heads)

cache = [None] * config.num_layers
for layer_idx in range(config.num_layers):
    cache[layer_idx] = {
        "key": torch.zeros((1, config.num_key_value_heads, 0, head_dim)),
        "value": torch.zeros((1, config.num_key_value_heads, 0, head_dim)),
    }

for idx in range(max_seq_len):
    attn_mask = torch.ones(1, seq_len)

    if idx == 0:
        position_ids = torch.arange(seq_len, dtype=torch.int64).reshape((1, seq_len))
        model.set_use_cache(False)
    else:
        position_ids = torch.tensor(seq_len - 1, dtype=torch.int64).reshape((1, 1))
        model.set_use_cache(True)

    outputs = model(x, position_ids, attn_mask, cache)

    logits = outputs[0][:, -1, :]
    
    import numpy as np
    # print (logits.cpu().detach().numpy())
    for layer_idx in range(config.num_layers):
        cache[layer_idx]["key"] = outputs[1 + layer_idx * 2]
        cache[layer_idx]["value"] = outputs[2 + layer_idx * 2]

        # print (f"cache.{layer_idx}.key")
        # print (outputs[1 + layer_idx * 2].shape)
        # print (outputs[1 + layer_idx * 2][:, :, -1, :].cpu().detach().numpy())
        # print (f"cache.{layer_idx}.value")
        # print (outputs[2 + layer_idx * 2].shape)
        # print (outputs[2 + layer_idx * 2][:, :, -1, :].cpu().detach().numpy())
    next_token_id = logits.argmax(-1)
    # print (next_token_id)
    if (next_token_id.item() == tokenizer.eos_token_id):
        break
    
    # print (next_token_id)
    t = tokenizer.decode(next_token_id.to('cpu'), skip_special_tokens=True)
    print(f'{t}', end='')

    # update the cache.
    seq_len += 1
    x = next_token_id.unsqueeze(0)
