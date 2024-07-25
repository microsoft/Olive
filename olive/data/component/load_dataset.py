# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Mapping, Optional, Sequence, Union

from olive.constants import Framework
from olive.data.component.dataset import DummyDataset, RawDataset, TransformersDummyDataset
from olive.data.registry import Registry


@Registry.register_dataset()
@Registry.register_default_dataset()
@Registry.register_dataset("simple_dataset")
def local_dataset(**kwargs):
    """Create a simple local dataset from input data."""
    # TODO(olivedevs): Implement this feature if and when needed.
    return


@Registry.register_dataset()
def huggingface_dataset(
    data_name: str,
    subset: Optional[str] = None,
    data_dir: Optional[str] = None,
    split: Optional[str] = "validation",
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    col_filters: Optional[Mapping[str, Union[str, int, float]]] = None,
    **kwargs
):
    """Create a dataset from huggingface datasets."""
    from datasets.utils.logging import disable_progress_bar, set_verbosity_error

    disable_progress_bar()
    set_verbosity_error()
    from datasets import load_dataset

    assert data_name is not None, "Please specify the data name"
    dataset = load_dataset(path=data_name, name=subset, data_dir=data_dir, split=split, data_files=data_files, **kwargs)
    if col_filters:
        dataset = dataset.filter(lambda x: _filter_dataset(x, col_filters))
    return dataset


def _filter_dataset(example: dict, col_filters: dict):
    """Filter the dataset based on the column filters."""
    return all(example[col] == value for col, value in col_filters.items())


@Registry.register_dataset()
def dummy_dataset(input_shapes, input_names=None, input_types=None, max_samples=32, **kwargs):
    return DummyDataset(input_shapes, input_names, input_types, max_samples, **kwargs)


@Registry.register_dataset()
def raw_dataset(
    data_dir,
    input_names,
    input_shapes,
    input_types=None,
    input_dirs=None,
    input_suffix=None,
    input_order_file=None,
    annotations_file=None,
):
    return RawDataset(
        data_dir=data_dir,
        input_names=input_names,
        input_shapes=input_shapes,
        input_types=input_types,
        input_dirs=input_dirs,
        input_suffix=input_suffix,
        input_order_file=input_order_file,
        annotations_file=annotations_file,
    )


@Registry.register_dataset()
def transformers_dummy_dataset(
    data_dir,
    model_name,
    seq_len: int = 128,
    past_seq_len: int = 128,
    max_seq_len: int = 1024,
    model_framework: str = Framework.ONNX,
    use_fp16: bool = False,
    shared_kv: bool = False,
    generative: bool = False,
    ort_past_key_name: str = "past_key_values.<id>.key",
    ort_past_value_name: str = "past_key_values.<id>.value",
    trust_remote_code=None,
    max_samples=32,
    use_step=False,
    ignore_input_fields=None,
):
    return TransformersDummyDataset(
        model_name=model_name,
        seq_len=seq_len,
        past_seq_len=past_seq_len,
        max_seq_len=max_seq_len,
        model_framework=model_framework,
        use_fp16=use_fp16,
        shared_kv=shared_kv,
        generative=generative,
        ort_past_key_name=ort_past_key_name,
        ort_past_value_name=ort_past_value_name,
        trust_remote_code=trust_remote_code,
        max_samples=max_samples,
        use_step=use_step,
        ignore_input_fields=ignore_input_fields,
    )


@Registry.register_dataset()
def transformers_prompt_dummy_dataset(
    data_dir,
    model_name,
    seq_len: int = 8,
    past_seq_len: int = 0,
    max_seq_len: int = 2048,
    model_framework: str = Framework.ONNX,
    use_fp16: bool = False,
    shared_kv: bool = False,
    generative: bool = False,
    ort_past_key_name: str = "past_key_values.<id>.key",
    ort_past_value_name: str = "past_key_values.<id>.value",
    trust_remote_code=None,
    max_samples=32,
    use_step=False,
    ignore_input_fields=None,
):
    return TransformersDummyDataset(
        model_name=model_name,
        seq_len=seq_len,
        past_seq_len=past_seq_len,
        max_seq_len=max_seq_len,
        model_framework=model_framework,
        use_fp16=use_fp16,
        shared_kv=shared_kv,
        generative=generative,
        ort_past_key_name=ort_past_key_name,
        ort_past_value_name=ort_past_value_name,
        trust_remote_code=trust_remote_code,
        max_samples=max_samples,
        use_step=use_step,
        ignore_input_fields=ignore_input_fields,
    )


@Registry.register_dataset()
def transformers_token_dummy_dataset(
    data_dir,
    model_name,
    seq_len: int = 1,
    past_seq_len: int = 8,
    max_seq_len: int = 2048,
    model_framework: str = Framework.ONNX,
    use_fp16: bool = False,
    shared_kv: bool = False,
    generative: bool = False,
    ort_past_key_name: str = "past_key_values.<id>.key",
    ort_past_value_name: str = "past_key_values.<id>.value",
    trust_remote_code=None,
    max_samples=32,
    use_step=False,
    ignore_input_fields=None,
):
    return TransformersDummyDataset(
        model_name=model_name,
        seq_len=seq_len,
        past_seq_len=past_seq_len,
        max_seq_len=max_seq_len,
        model_framework=model_framework,
        use_fp16=use_fp16,
        shared_kv=shared_kv,
        generative=generative,
        ort_past_key_name=ort_past_key_name,
        ort_past_value_name=ort_past_value_name,
        trust_remote_code=trust_remote_code,
        max_samples=max_samples,
        use_step=use_step,
        ignore_input_fields=ignore_input_fields,
    )
