# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
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

    def to_snpe_dataset(self):
        """
        This function is used to convert the dataset to snpe dataset
        """
        pass


class DummyDataset(BaseDataset):
    def __init__(self, input_shapes, input_names=None, input_types=None):
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


class RawDataset(BaseDataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        input_names: List[str],
        input_shapes: List[List[int]],
        input_types: Optional[List[str]] = None,
        input_dirs: Optional[List[str]] = None,
        input_suffix: Optional[str] = None,
        input_order_file: Optional[str] = None,
        annotations_file: Optional[str] = None,
    ):
        """
        Initialize the raw dataset.

        :param data_dir:  Directory containing the raw data files. This contains the input files, annotations file and
        input order file. Each input file is assumed to be a binary file containing a numpy array.
        :param input_names: List of input names.
        :param input_shapes: List of input shapes. Each element is a list of integers. Length of this list should be
        equal to the number of inputs. The batch dimension, if any, is assumed to be included in the input shapes.
        :param input_types: List of input types. Each element is a string. Length of this list should be equal to the
        number of inputs. Default is None, in which case all inputs are assumed to be of type float32.
        :param input_dirs: List of input directories. Each element is a string and corresponds to the sub-directory in
        the data_dir where the files for the corresponding input are located. Length of this list should be equal to the
        number of inputs. Default is None, in which case the input dirs are assumed to be the same as the input names.
        :param input_suffix: Suffix of the input files. It is used to filter out the files in the input directories.
        Default is None, in which case all files in the input directories are considered.
        :param input_order_file: Name of the file containing the input order. This file should be present in the
        data_dir. It is assumed to be a text file with a file name at each line. All input sub-directories should
        contain the same set of files. Default is None, in which case the input order is assumed to be the ascending
        order of the input files in the first input sub-directory.
        :param annotations_file: Name of the file containing the annotations. This file should be present in the
        data_dir. It is assumed to be a .npy file containing a numpy array. Default is None.
        """
        self.data_dir = Path(data_dir).resolve()
        self.input_names = input_names
        assert len(input_names) == len(input_shapes), "Number of input shapes should be equal to number of input names."

        input_types = input_types or ["float32"] * len(input_names)
        assert len(input_names) == len(input_types), "Number of input types should be equal to number of input names."

        input_dirs = input_dirs or input_names
        assert len(input_names) == len(input_dirs), "Number of input dirs should be equal to number of input names."

        # store input information in a dictionary
        self.input_specs = {}
        for input_name, input_shape, input_type, input_dir in zip(input_names, input_shapes, input_types, input_dirs):
            self.input_specs[input_name] = {"shape": input_shape, "type": input_type, "dir": input_dir}

        # get input order
        if input_order_file is None:
            input_dir = self.data_dir / self.input_specs[self.input_names[0]]["dir"]
            glob_pattern = "*" if input_suffix is None else f"*{input_suffix}"
            input_files = sorted(
                filter(
                    lambda x: x not in [input_order_file, annotations_file],
                    map(lambda x: x.name, input_dir.glob(glob_pattern)),
                )
            )
            self.input_files = input_files
        else:
            with open(self.data_dir / input_order_file, "r") as f:
                self.input_files = [line.strip() for line in f.readlines()]

        # get annotations
        self.annotations = None
        if annotations_file is not None:
            self.annotations = np.load(self.data_dir / annotations_file)
            assert len(self.annotations) == len(
                self.input_files
            ), "Number of annotations should be equal to number of input files."

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index: int):
        input_file = self.input_files[index]
        data = {}
        for input_name, input_spec in self.input_specs.items():
            input_path = self.data_dir / input_spec["dir"] / input_file
            data[input_name] = np.fromfile(input_path, dtype=input_spec["type"]).reshape(input_spec["shape"])
        label = 0 if self.annotations is None else self.annotations[index]
        return data, label
