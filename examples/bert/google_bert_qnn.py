from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from bert_common import (
    SimpleBert,
    npz_to_hfdataset,
    tokenize_hfdataset2,
)
from datasets import load_dataset
from transformers import (
    AutoModelForNextSentencePrediction as AutoModelNSP,
)
from transformers import (
    AutoModelForSequenceClassification as AutoModelSCL,
)
from transformers import (
    AutoTokenizer,
    BertModel,
)
from wikitext import create_nsp_dataset

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


def load_bert_model(model_name: str):
    model = BertModel.from_pretrained(model_name).eval()
    return SimpleBert(model)


def load_bert_scl_model(model_name: str):
    model = AutoModelSCL.from_pretrained(model_name).eval()
    model.bert = SimpleBert(model.bert)
    return model


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


@Registry.register_dataset()
def load_csv_dataset(
    data_file: str,
    split: str = "train",
):
    return load_dataset(path="csv", data_files=data_file, split=split)


@Registry.register_dataset()
def load_npz_dataset(
    data_path: Path,
    max_samples: int,
):
    return npz_to_hfdataset(data_path, max_samples)


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
    dataset = tokenize_hfdataset2(
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


@Registry.register_post_process()
def bert_tcl_post_process(outputs) -> torch.Tensor:
    """Post-processing for Token Classification tasks."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (OrderedDict, dict)):
        if "logits" in outputs:
            return outputs["logits"]
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
    raise ValueError(f"Unsupported output type: {type(outputs)}")


@Registry.register_post_process()
def bert_qa_post_process(outputs) -> torch.Tensor:
    """Post-processing for Question Answering tasks."""
    if isinstance(outputs, (OrderedDict, dict)) and ("start_logits" in outputs and "end_logits" in outputs):
        logits = [outputs["start_logits"], outputs["end_logits"]]
        return torch.stack(logits, dim=1)
    raise ValueError(f"Unsupported output type: {type(outputs)}")
