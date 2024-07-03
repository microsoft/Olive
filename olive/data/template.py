# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.data.config import DataComponentConfig, DataConfig


def dummy_data_config_template(input_shapes, input_names=None, input_types=None) -> DataConfig:
    """Convert the dummy data config to the data container.

    input_names: list
        The input names of the model.
    input_shapes: list
        The input shapes of the model.
    input_types: list
        The input types of the model.
    """
    return DataConfig(
        name="dummy_data_config_template",
        type="DummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "input_shapes": input_shapes,
                "input_names": input_names,
                "input_types": input_types,
            }
        ),
    )


def huggingface_data_config_template(model_name, task, **kwargs) -> DataConfig:
    """Convert the huggingface data config to the data container.

    model_name: str
        The model name of huggingface.
    task: str
        The task type of huggingface.
    **kwargs: dict
        Additional arguments:
        - olive.data.component.load_dataset_config.huggingface_dataset
            - `data_name`: str, data name in huggingface dataset, e.g.: "glue", "squad"
            - `subset`: str, subset of data, e.g.: "train", "validation", "test"
            - `split`: str, split of data, e.g.: "train", "validation", "test"
            - `data_files`: str | list | dict, path to source data file(s).
            e.g.
                load_dataset_config={
                    "params": {
                        "data_name": "glue",
                        "subset": "train",
                        "split": "train",
                        "data_files": "whatever.pt"
                    }
                }

        - olive.data.component.pre_process_data_config.huggingface_pre_process
            - `align_labels`: true | false, whether to align the dataset labels with huggingface model config
            more details in https://huggingface.co/docs/datasets/nlp_process#align
            - `model_config_path`: str, path to the model config file
            - `input_cols`: list, input columns of data
            - `label_cols`: list, label columns of data
            - `max_samples`: int, maximum number of samples in the dataset
            - others is used for huggingface tokenizer

            e.g.:
                pre_process_data_config={
                    "params": {
                        "align_labels": true
                        "max_samples": 1024
                        "input_cols": [0],
                        "label_cols": ["text"],
                    }
                }
        - olive.data.component.dataloader_config
            - `batch_size`: int, batch size of data

            e.g.:
                dataloader_config={
                    "batch_size": 1
                }
    """
    for component_config_name in ["pre_process_data_config", "post_process_data_config"]:
        component_config = kwargs.get(component_config_name, None) or {}
        if isinstance(component_config, DataComponentConfig):
            component_config = component_config.dict()
        component_config["params"] = component_config.get("params") or {}
        component_config["params"].update({"model_name": model_name, "task": task})
        kwargs[component_config_name] = component_config
    return DataConfig(
        name="huggingface_data_config",
        type="HuggingfaceContainer",
        **kwargs,
    )


def raw_data_config_template(
    data_dir,
    input_names,
    input_shapes,
    input_types=None,
    input_dirs=None,
    input_suffix=None,
    input_order_file=None,
    annotations_file=None,
) -> DataConfig:
    """Convert the raw data config to the data container.

    Refer to olive.data.component.dataset.RawDataset for more details.
    """
    return DataConfig(
        name="raw_data_config_template",
        type="RawDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "data_dir": data_dir,
                "input_names": input_names,
                "input_shapes": input_shapes,
                "input_types": input_types,
                "input_dirs": input_dirs,
                "input_suffix": input_suffix,
                "input_order_file": input_order_file,
                "annotations_file": annotations_file,
            }
        ),
    )


def transformers_dummy_data_config_template(
    model_name,
    batch_size=1,
    seq_len=128,
    past_seq_len=128,
    max_seq_len=1024,
    model_framework="onnx",
    use_fp16=False,
    shared_kv=False,
    generative=False,
    ort_past_key_name="past_key_values.<id>.key",
    ort_past_value_name="past_key_values.<id>.value",
) -> DataConfig:
    """Convert the transformer dummy data config to the data container.

    Refer to olive.data.component.dataset.TransformersDummyDataset for more details.
    """
    return DataConfig(
        name="transformer_dummy_data_config_template",
        type="TransformersDummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "model_name": model_name,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "past_seq_len": past_seq_len,
                "max_seq_len": max_seq_len,
                "model_framework": model_framework,
                "use_fp16": use_fp16,
                "shared_kv": shared_kv,
                "generative": generative,
                "ort_past_key_name": ort_past_key_name,
                "ort_past_value_name": ort_past_value_name,
            }
        ),
    )


def transformers_prompt_dummy_data_config_template(
    model_name,
    batch_size=2,
    seq_len=8,
    past_seq_len=0,
    max_seq_len=1024,
    model_framework="onnx",
    use_fp16=False,
    shared_kv=False,
    generative=False,
    ort_past_key_name="past_key_values.<id>.key",
    ort_past_value_name="past_key_values.<id>.value",
) -> DataConfig:
    """Convert the transformer dummy data config to the data container.

    Refer to olive.data.component.dataset.TransformersDummyDataset for more details.
    """
    return DataConfig(
        name="transformer_prompt_dummy_data_config_template",
        type="TransformersPromptDummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "model_name": model_name,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "past_seq_len": past_seq_len,
                "max_seq_len": max_seq_len,
                "model_framework": model_framework,
                "use_fp16": use_fp16,
                "shared_kv": shared_kv,
                "generative": generative,
                "ort_past_key_name": ort_past_key_name,
                "ort_past_value_name": ort_past_value_name,
            }
        ),
    )


def transformers_token_dummy_data_config_template(
    model_name,
    batch_size=2,
    seq_len=1,
    past_seq_len=8,
    max_seq_len=1024,
    model_framework="onnx",
    use_fp16=False,
    shared_kv=False,
    generative=False,
    ort_past_key_name="past_key_values.<id>.key",
    ort_past_value_name="past_key_values.<id>.value",
) -> DataConfig:
    """Convert the transformer dummy data config to the data container.

    Refer to olive.data.component.dataset.TransformersDummyDataset for more details.
    """
    return DataConfig(
        name="transformer_token_dummy_data_config_template",
        type="TransformersTokenDummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "model_name": model_name,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "past_seq_len": past_seq_len,
                "max_seq_len": max_seq_len,
                "model_framework": model_framework,
                "use_fp16": use_fp16,
                "shared_kv": shared_kv,
                "generative": generative,
                "ort_past_key_name": ort_past_key_name,
                "ort_past_value_name": ort_past_value_name,
            }
        ),
    )
