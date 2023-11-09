import torch
from transformers import AutoConfig

model_id = "01-ai/Yi-6B-200K"
config = AutoConfig.from_pretrained(model_id)

def dummy_inputs(model):
    attention_mask_sequence_length = 19
    sequence_length = 19
    inputs = {
        "input_ids": torch.randint(10, (1, sequence_length), dtype=torch.int64),
        "attention_mask": torch.randint(10, (1, attention_mask_sequence_length), dtype=torch.int64),
    }
    return inputs