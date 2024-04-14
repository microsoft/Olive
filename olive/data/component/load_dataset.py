# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.data.component.dataset import DummyDataset, RawDataset
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
