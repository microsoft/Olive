# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from torch.utils.data import Dataset

from olive.data_container.registry import Registry


class BaseDataset(Dataset):
    """
    This class is used to define the Olive dataset which should return the data with following format:
    1. [data, label] for supervised learning
    2. [data] for unsupervised learning
    The data should be a list or dict of numpy arrays or torch tensors
    """

    def __init__(self, data, label_cols=None, **kwargs):
        """
        This function is used to initialize the dataset
        """
        self.data = data
        self.label_cols = label_cols or []

    def __len__(self):
        """
        This function is used to return the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items() if k not in self.label_cols}
        label = self.data[index][self.label_cols[0]]
        return data, label

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


@Registry.register_default_dataset()
def local_dataset(data_dir=None, label_cols=None, **kwargs):
    pass


@Registry.register_dataset()
def simple_dataset(input_data, label_cols=None, **kwargs):
    """
    This function is used to create a simple dataset from input data which can be:
    1. a text
    2. a tensor
    """
    pass


@Registry.register_dataset()
def huggingface_dataset(data_name=None, subset=None, split="validation", **kwargs):
    """
    This function is used to create a dataset from huggingface datasets
    """
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()
    from datasets import load_dataset

    assert data_name is not None, "Please specify the data name"
    return load_dataset(path=data_name, name=subset, split=split, **kwargs)
