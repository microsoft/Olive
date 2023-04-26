# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from torch.utils.data import Dataset

from olive.data_container.constants import DataComponentType, DefaultDataComponent
from olive.data_container.registry import Registry


class BaseDataset(Dataset):
    """
    This class is used to define the Olive dataset which should return the data with following format:
    1. [data, label] for supervised learning
    2. [data] for unsupervised learning
    The data should be a list or dict of numpy arrays or torch tensors
    """

    def to_numpy(self):
        """
        This function is used to convert the dataset to numpy array
        """
        pass

    def to_torch_tensor(self):
        """
        This function is used to convert the dataset to torch tensor
        """
        pass

    def to_snap_dataset(self):
        """
        This function is used to convert the dataset to snap dataset
        """
        pass


@Registry.register(DataComponentType.DATASET, name=DefaultDataComponent.DATASET.value)
def local_dataset(data_dir=None, label_name=None, **kwargs):
    pass


@Registry.register(DataComponentType.DATASET)
def simple_dataset(input_data, label_name=None, **kwargs):
    """
    This function is used to create a simple dataset from input data which can be:
    1. a text
    2. a tensors
    """
    pass


@Registry.register(DataComponentType.DATASET)
def huggingface_dataset(data_name, **kwargs):
    """
    This function is used to create a dataset from huggingface datasets
    """
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()
    from datasets import load_dataset

    return load_dataset(data_name, **kwargs)
