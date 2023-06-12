# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset


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


class DummyDataset(BaseDataset):
    def __init__(self, input_names, input_shapes, input_types):
        """
        This function is used to initialize the dummy dataset
        if input_names is None, the dummy dataset will return a tuple of tensors
        else the dummy dataset will return a dict of tensors
        """
        self.input_names = input_names
        self.input_shapes = input_shapes
        self.input_types = input_types

    def __len__(self):
        return 256

    def __getitem__(self, index):
        str_to_type = {
            "float32": torch.float32,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
            "int8": torch.int8,
            "bool": torch.bool,
        }
        input_types = self.input_types or ["float32"] * len(self.input_shapes)
        input_types = [str_to_type[type_] for type_ in input_types]

        if not self.input_names:
            dummy_inputs = []
            for shape, dtype in zip(self.input_shapes, input_types):
                dummy_inputs.append(torch.ones(shape, dtype=dtype))
            dummy_inputs = tuple(dummy_inputs) if len(dummy_inputs) > 1 else dummy_inputs[0]
        else:
            dummy_inputs = {}
            for input_name, input_shape, input_type in zip(self.input_names, self.input_shapes, input_types):
                dummy_inputs.update({input_name: torch.ones(input_shape, dtype=input_type)})
            dummy_inputs = dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[self.input_names[0]]
        label = 0
        return dummy_inputs, label
