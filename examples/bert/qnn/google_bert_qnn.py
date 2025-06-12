from collections import OrderedDict
from typing import Optional

import torch
from bert_common import SimpleBert, tokenize_hfdataset
from transformers import AutoModelForNextSentencePrediction as AutoModelNSP
from transformers import AutoTokenizer
from wikitext import create_nsp_dataset

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


def load_bert_nsp_model(model_name: str):
    model = AutoModelNSP.from_pretrained(model_name).eval()
    model.bert = SimpleBert(model.bert)
    return model


@Registry.register_dataset()
def dataset_to_nsp_dataset(
    data_path: str,
    data_name: str,
    data_split: str,
    input_cols: list[str],
    label_col: str,
    max_samples: Optional[int],
):
    return create_nsp_dataset(
        dataset=data_path,
        name=data_name,
        split=data_split,
        sent_cols=input_cols,
        label_col=label_col,
        max_samples=max_samples,
    )


@Registry.register_pre_process()
def tokenize_dataset(
    dataset,
    model_name: str,
    input_cols: list[str],
    label_col: str,
    max_samples: Optional[int],
    seq_length=512,
    **kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = tokenize_hfdataset(
        dataset,
        tokenizer,
        input_cols,
        label_col=label_col,
        seq_length=seq_length,
        max_samples=max_samples,
    )
    return BaseDataset(dataset, label_col)


@Registry.register_post_process()
def bert_scl_post_process(outputs) -> torch.Tensor:
    """Post-processing for Sequence Classification tasks."""
    if isinstance(outputs, torch.Tensor):
        return outputs.argmax(dim=-1)
    if isinstance(outputs, (OrderedDict, dict)):
        if "logits" in outputs:
            return outputs["logits"].argmax(dim=-1)
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
    raise ValueError(f"Unsupported output type: {type(outputs)}")
