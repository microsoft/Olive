import torch
from transformers import RobertaForSequenceClassification

from olive.data.template import huggingface_data_config_template

BERT_MODEL_CONFIG = {
    "model_name": "roberta-large-mnli",
    "task": "text-classification",
    "dataset": {
        "data_name": "glue",
        "subset": "mnli_matched",
        "split": "validation",
        "input_cols": ["premise", "hypothesis"],
        "label_cols": ["label"],
        "batch_size": 1,
        "max_samples": 100,
        "component_kwargs": {"pre_process_data": {"align_labels": True}},
    },
}


def get_data_container(model_name, task, **kwargs):
    return huggingface_data_config_template(model_name, task, **kwargs).to_data_container()


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    return get_data_container(
        BERT_MODEL_CONFIG["model_name"], BERT_MODEL_CONFIG["task"], **BERT_MODEL_CONFIG["dataset"]
    ).create_dataloader(data_dir)


def torch_complied_model(model_path):
    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
    return torch.compile(model)


def post_process(result):
    return get_data_container(
        BERT_MODEL_CONFIG["model_name"], BERT_MODEL_CONFIG["task"], **BERT_MODEL_CONFIG["dataset"]
    ).post_process(result)
