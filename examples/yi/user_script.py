import torch


def dummy_inputs(model):
    attention_mask_sequence_length = 19
    sequence_length = 19
    return {
        "input_ids": torch.randint(10, (1, sequence_length), dtype=torch.int64),
        "attention_mask": torch.randint(10, (1, attention_mask_sequence_length), dtype=torch.int64),
    }
