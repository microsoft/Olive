from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertModel
from transformers.modeling_outputs import ModelOutput as _ModelOutput

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class ModelOutput(_ModelOutput):
    """Wrapper for ModelOutput class from transformers.modeling_outputs.

    Always returns None for missing keys when accessed with __getitem__
    or __getattr__.
    """

    def __init__(self, lhs, po, **kwargs):
        """Initialize the constructor.

        It is just a placeholder for future init
        """
        super().__init__(last_hidden_state=lhs, pooler_output=po, **kwargs)

    def __getitem__(self, k):
        if isinstance(k, str) and k not in self.keys():
            return None
        return super().__getitem__(k)

    def __getattr__(self, k):
        if k in self.keys():
            return self[k]
        return None


class SimpleBert(torch.nn.Module):
    def __init__(self, model: BertModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embeddings = model.embeddings
        self.encoder = model.encoder
        self.pooler = model.pooler

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        **kwargs,
    ) -> ModelOutput:
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
        )[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return ModelOutput(
            lhs=sequence_output,
            po=pooled_output,
        )


def create_4d_mask(
    mask: torch.Tensor,
    input_shape: Union[torch.Size, tuple[int, int]],
    masked_value: float = -50.0,
) -> torch.Tensor:
    # (batch_size, num_heads, seq_len, head_dim)
    batch_sz, seq_len = input_shape
    expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.float()
    return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)


def tokenize_hfdataset(
    dataset: "Dataset",
    tokenizer: "Union[PreTrainedTokenizer, PreTrainedTokenizerFast]",
    input_cols: list[str],
    label_col: Optional[str] = None,
    seq_length: int = 512,
    max_samples: Optional[int] = None,
    batch_size: int = 1024,
):
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    all_items = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for i, sample in tqdm(enumerate(loader)):
        encoded_input = tokenizer(
            *[sample[input_col] for input_col in input_cols],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        batch_sz = encoded_input.input_ids.shape[0]
        input_ids = encoded_input.input_ids
        attention_mask = create_4d_mask(
            encoded_input.attention_mask,
            (batch_sz, seq_length),
        )
        token_type_ids = (
            encoded_input.token_type_ids
            if "token_type_ids" in encoded_input
            else torch.zeros(seq_length).expand(batch_sz, -1)
        )

        batch = {
            "input_ids": input_ids.int(),
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids.int(),
            **(
                {
                    label_col: sample.get(
                        label_col,
                        range(i * batch_size, i * batch_size + batch_sz),
                    )
                }
                if label_col is not None
                else {}
            ),
        }

        items = [dict(zip(batch.keys(), sample)) for sample in zip(*batch.values())]
        all_items.extend(items)

    return all_items
