import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from olive.constants import Framework

ort.set_default_logger_severity(3)


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    return tokenizer(examples["text"])


class Dataloader:
    def __init__(self, pad_max=196, batch_size=1):
        self.pad_max = pad_max
        self.batch_size = batch_size
        dataset = load_dataset("lambada", split="validation")
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

    def collate_batch(self, batch):
        input_ids_padded = []
        attention_mask_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids) + 1)
            attention_mask[0] = 0
            input_ids = pad(input_ids, (0, pad_len), value=1)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)

        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)), torch.tensor(last_ind)

    def __iter__(self):
        try:
            for (input_ids, _attention_mask), last_ind in self.dataloader:
                yield input_ids, last_ind
        except StopIteration:
            return

    def __len__(self):
        return len(self.dataloader)


class OnnxDataloader(Dataloader):
    def __init__(self, pad_max=196, batch_size=1):
        super().__init__(pad_max, batch_size)

    def __iter__(self):
        try:
            for (input_ids, attention_mask), last_ind in self.dataloader:
                data = [input_ids.detach().cpu().numpy().astype("int64")]
                for _ in range(28):
                    data.append(np.zeros((input_ids.shape[0], 16, 1, 256), dtype="float32"))
                    data.append(np.zeros((input_ids.shape[0], 16, 1, 256), dtype="float32"))
                data.append(attention_mask.detach().cpu().numpy().astype("int64"))
                yield data, last_ind.detach().cpu().numpy()
        except StopIteration:
            return


def create_pt_dataloader(data_dir, batch_size, *args, **kwargs):
    return Dataloader(batch_size=batch_size)


def create_onnx_dataloader(data_dir, batch_size=1, *args, **kwargs):
    return OnnxDataloader(batch_size=batch_size)


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    model_framework = kwargs.pop("model_framework")
    if model_framework == Framework.ONNX:
        dataloader = create_onnx_dataloader(data_dir, batch_size)
    elif model_framework == Framework.PYTORCH:
        dataloader = create_pt_dataloader(data_dir, batch_size)
    return dataloader
