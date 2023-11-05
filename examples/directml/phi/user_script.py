import torch
from transformers import AutoModelForCausalLM


def load_pytorch_origin_model(model_path):
    return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)


class DataLoader:
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __getitem__(self, idx):
        sequence_length = 20
        inputs = {
            "input_ids": torch.randint(10, (self.batchsize, sequence_length), dtype=torch.int64),
        }
        return inputs, None


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    return DataLoader(batchsize)


def dummy_inputs(model):
    return torch.zeros((1, 77), dtype=torch.int64)
