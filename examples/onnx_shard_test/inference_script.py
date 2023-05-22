import torch
from torch.utils.data import DataLoader, Dataset

class BloomDataset(Dataset):
    def __init__(self, size, input_shape):
        self.size = size
        self._input_shape = input_shape

    def __getitem__(self, idx):
        return {'input.1' : torch.ones(self._input_shape, dtype=torch.int64),
                'onnx::Cast_1' : torch.ones(self._input_shape, dtype=torch.float32)}, torch.rand(10)

    def __len__(self):
        return self.size

def create_bloom_dataloader(datadir, batchsize):
    dataloader = BloomDataset(5, (1, 128))
    return dataloader