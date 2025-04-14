import torch
from torch.utils.data import Dataset

from olive.data.registry import Registry


class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sample = self.text[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.int64)
        token_type_ids = torch.tensor(sample["token_type_ids"], dtype=torch.int64)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.int64)

        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}, idx


@Registry.register_pre_process()
def dataset_pre_process(dataset, **kwargs):
    from transformers import AutoTokenizer

    max_samples = kwargs.get("max_samples", 128)
    model_name = kwargs.get("model_name")
    texts = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_dict = tokenizer(sample["text"], max_length=8192, padding=True, truncation=True)
        texts.append(
            {
                "input_ids": batch_dict["input_ids"],
                "token_type_ids": batch_dict["token_type_ids"],
                "attention_mask": batch_dict["attention_mask"],
            }
        )
    return TextDataset(texts)
