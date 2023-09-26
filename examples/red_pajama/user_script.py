import torch
import transformers

MIN_TRANSFORMERS_VERSION = "4.30.2"

# check transformers version
assert (
    transformers.__version__ >= MIN_TRANSFORMERS_VERSION
), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."


def _get_inputs(batch_size, torch_dtype, device):
    attention_mask_sequence_length = 1
    sequence_length = 2

    return {
        "input_ids": torch.randint(10000, (batch_size, sequence_length), dtype=torch.int64, device=device),
        "attention_mask": torch.ones((batch_size, attention_mask_sequence_length), dtype=torch.int64, device=device),
    }


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, batch_size, torch_dtype, torch_device):
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.torch_device = torch_device

    def __getitem__(self, idx):
        return (
            _get_inputs(self.batch_size, self.torch_dtype, self.torch_device),
            None,
        )


def create_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(batch_size, torch.float16, "cuda")
