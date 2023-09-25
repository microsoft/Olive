# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.data.config import DataConfig


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
        params_config={
            "input_shapes": input_shapes,
            "input_names": input_names,
            "input_types": input_types,
        },
    )


def huggingface_data_config_template(model_name, task, **kwargs) -> DataConfig:
    """Convert the huggingface data config to the data container.

    model_name: str
        The model name of huggingface.
    task: str
        The task type of huggingface.
    **kwargs: dict
        The additional arguments. Will be passed as `params_config` to the data container.
        - `data_name`: str, data name in huggingface dataset, e.g.: "glue", "squad"
        - `subset`: str, subset of data, e.g.: "train", "validation", "test"
        - `split`: str, split of data, e.g.: "train", "validation", "test"
        - `data_files`: str | list | dict, path to source data file(s).
        - `input_cols`: list, input columns of data
        - `label_cols`: list, label columns of data
        - `batch_size`: int, batch size of data
        - `max_samples`: int, maximum number of samples in the dataset
        and other arguments in
            - olive.data.component.load_dataset.huggingface_dataset
            - olive.data.component.pre_process_data.huggingface_pre_process
                - `align_labels`: true | false, whether to align the dataset labels with huggingface model config
                more details in https://huggingface.co/docs/datasets/nlp_process#align
                - `model_config_path`: str, path to the model config file
                - others is used for huggingface tokenizer
                e.g.:
                "component_kwargs": {
                    "pre_process_data": {
                        "align_labels": true
                    }
                }
    """
    return DataConfig(
        name="huggingface_data_config_template",
        type="HuggingfaceContainer",
        params_config={
            "model_name": model_name,
            "task": task,
            **kwargs,
        },
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
        params_config={
            "data_dir": data_dir,
            "input_names": input_names,
            "input_shapes": input_shapes,
            "input_types": input_types,
            "input_dirs": input_dirs,
            "input_suffix": input_suffix,
            "input_order_file": input_order_file,
            "annotations_file": annotations_file,
        },
    )
