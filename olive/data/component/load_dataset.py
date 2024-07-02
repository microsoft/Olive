# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.constants import Framework
from olive.data.component.dataset import DummyDataset, RawDataset, TransformersDummyDataset
from olive.data.registry import Registry


@Registry.register_default_dataset()
def local_dataset(data_dir, label_cols=None, **kwargs):
    pass


@Registry.register_dataset()
def simple_dataset(data_dir, input_data, label_cols=None, **kwargs):
    """Create a simple dataset from input data.

    The input data can be:
    1. a text
    2. a tensor
    """


@Registry.register_dataset()
def huggingface_dataset(data_dir, data_name=None, subset=None, split="validation", data_files=None, **kwargs):
    """Create a dataset from huggingface datasets."""
    from datasets.utils.logging import disable_progress_bar, set_verbosity_error

    disable_progress_bar()
    set_verbosity_error()
    from datasets import load_dataset

    assert data_name is not None, "Please specify the data name"
    return load_dataset(path=data_name, name=subset, data_dir=data_dir, split=split, data_files=data_files, **kwargs)


@Registry.register_dataset()
def dummy_dataset(data_dir, input_shapes, input_names=None, input_types=None):
    return DummyDataset(input_shapes, input_names, input_types)


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
    batch_size: int = 1,
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
):
    return TransformersDummyDataset(
        model_name=model_name,
        batch_size=batch_size,
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
    )
