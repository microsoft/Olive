# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from olive.common.utils import find_first_matched_value, resolve_torch_dtype
from olive.constants import Framework


class BaseDataset(TorchDataset):
    """Define the Olive dataset which should return the data with following format.

    1. [data, label] for supervised learning
    2. [data] for unsupervised learning
    The data should be a list or dict of numpy arrays or torch tensors
    """

    def __init__(self, data, label_cols=None, max_samples=None, **kwargs):
        """Initialize the dataset."""
        self.data = data
        self.label_cols = label_cols or []
        self.max_samples = max_samples

    def __len__(self):
        """Return the length of the dataset."""
        num_samples = len(self.data)
        if self.max_samples is not None:
            # if max_samples is not None, return the min of num_samples and max_samples
            num_samples = min(num_samples, self.max_samples)
        return num_samples

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items() if k not in self.label_cols}
        label = self.data[index][self.label_cols[0]]
        return data, label

    def to_numpy(self):
        """Convert the dataset to numpy array."""

    def to_torch_tensor(self):
        """Convert the dataset to torch tensor."""

    def to_snpe_dataset(self):
        """Convert the dataset to snpe dataset."""

    def to_hf_dataset(self, label_name="label"):
        """Convert the dataset to huggingface dataset.

        :param label_name: The name of the label column in the new dataset. Default is "label".
        """
        from datasets import Dataset

        if hasattr(self, "data") and isinstance(self.data, Dataset):
            # some children classes may not have data attribute
            # this part assumes the class follows the format of BaseDataset and has data and label_cols attributes
            # deepcopy the dataset since we might modify it
            hf_dataset = deepcopy(self.data)
            for col_name in self.label_cols[1:]:
                # label_cols is a list but we only use the first element for now
                # remove the other label columns
                hf_dataset = hf_dataset.remove_columns(col_name)
            # rename the label column
            if self.label_cols[0] != label_name:
                if label_name in hf_dataset.column_names:
                    raise ValueError(f"Cannot rename label column to {label_name} since it already exists")
                hf_dataset = hf_dataset.rename_column(self.label_cols[0], label_name)
            # truncate the dataset to len (happen when max_samples is not None)
            # this is not costly since the dataset is sliced when selected with range
            hf_dataset = hf_dataset.select(range(len(self)))
        else:
            first_input, _ = self[0]
            if not isinstance(first_input, dict):
                raise ValueError("Cannot convert to huggingface dataset since the input is not a dict")
            # convert the dataset to dict of lists
            data_dict = {k: [] for k in first_input}
            data_dict[label_name] = []
            # loop over the dataset
            for _, d in enumerate(self):
                data, label = deepcopy(d)
                for k, v in data.items():
                    data_dict[k].append(v)
                data_dict[label_name].append(label)
            # convert the dict of lists to huggingface dataset
            hf_dataset = Dataset.from_dict(data_dict)
            hf_dataset.set_format("torch", output_all_columns=True)
        return hf_dataset


class DummyDataset(BaseDataset):
    def __init__(self, input_shapes, input_names: Optional[List] = None, input_types: Optional[List] = None):
        """Initialize the dummy dataset.

        if input_names is None, the dummy dataset will return a tuple of tensors
        else the dummy dataset will return a dict of tensors
        """
        # pylint: disable=super-init-not-called
        self.input_shapes = input_shapes
        self.input_names = input_names
        self.input_types = input_types or ["float32"] * len(input_shapes)

    def __len__(self):
        return 256

    def __getitem__(self, index):
        # From https://docs.python.org/3/reference/datamodel.html#object.__getitem__,
        # __getitem__ should raise IndexError when index is out of range
        # Otherwise, the enumerate function will enter infinite loop
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        input_types = [resolve_torch_dtype(dtype_str) for dtype_str in self.input_types]

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
        """Initialize the raw dataset.

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
        # pylint: disable=super-init-not-called
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
                    (x.name for x in input_dir.glob(glob_pattern)),
                )
            )
            self.input_files = input_files
        else:
            with open(self.data_dir / input_order_file) as f:
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


class TransformersDummyDataset(DummyDataset):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        seq_len: int,
        past_seq_len: int,
        max_seq_len: int,
        model_framework: str = Framework.ONNX,
        use_fp16: bool = False,
        shared_kv: bool = False,
        generative: bool = False,
        ort_past_key_name: str = "past_key_values.<id>.key",
        ort_past_value_name: str = "past_key_values.<id>.value",
        trust_remote_code: Optional[bool] = None,
    ):
        # pylint: disable=super-init-not-called
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.past_seq_len = past_seq_len
        self.max_seq_len = max_seq_len
        self.model_framework = model_framework
        if shared_kv and (model_framework != Framework.ONNX or not use_fp16):
            raise ValueError("shared_kv is only supported for ONNX model with FP16")
        self.use_fp16 = use_fp16
        self.shared_kv = shared_kv
        self.generative = generative
        self.ort_past_key_name = ort_past_key_name
        self.ort_past_value_name = ort_past_value_name
        self.trust_remote_code = trust_remote_code

    def __getitem__(self, idx):
        input_ids, attention_mask, position_ids, past_kv = self.get_merged_sample_with_past_kv_inputs(
            model_name=self.model_name,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            past_seq_len=self.past_seq_len,
            use_fp16=self.use_fp16,
            trust_remote_code=self.trust_remote_code,
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not self.generative:
            inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_kv,
                }
            )

        if self.model_framework == Framework.ONNX:
            inputs.update(self.flatten_past_kv_inputs(past_kv))
            inputs.pop("past_key_values", None)

            if self.shared_kv:
                inputs = self.enable_past_present_share_buffer(inputs, self.past_seq_len, self.max_seq_len)

        return (inputs, None)

    def enable_past_present_share_buffer(self, ort_inputs: dict, past_seq_len: int, max_seq_len: int):
        """Enable past-present share buffer for GQA. For ONNX model + FP16 + GQA only."""
        for k, v in ort_inputs.items():
            # Allocate new buffers with max_seq_len for GQA
            if "past_key_values" in k:
                # Copy v (BxSxPxH) into new_v (BxSxMxH)
                batch_size, num_heads, _, head_size = v.shape
                new_v = torch.zeros((batch_size, num_heads, max_seq_len, head_size), dtype=v.dtype)
                new_v[:batch_size, :num_heads, :past_seq_len, :head_size] = v
                ort_inputs[k] = new_v
        return ort_inputs

    def get_merged_sample_with_past_kv_inputs(
        self,
        model_name: str,
        batch_size: int,
        seq_len: int,
        past_seq_len: int,
        use_fp16: bool = False,
        trust_remote_code=None,
    ):
        """Get inputs for forward pass with past_key_values.

        This is for the "merged" model which can be used for both prompt generation and token generation.
        For prompt generation, past_seq_len = 0 and seq_len >= 1. past_kv is a list of tuples of empty tensors.
        For token generation, past_seq_len >= 1 and seq_len = 1.

        Shape of outputs:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, past_seq_len + seq_len)
            position_ids: (batch_size, seq_len)
            past_key: (batch_size, num_heads, past_seq_len, head_size)
            past_value: (batch_size, num_heads, past_seq_len, head_size)
        """
        # Note: No need for separate function for legacy prompt and token generation
        # prompt generation (get_sample_inputs):
        #   past_seq_len = 0, seq_len >= 1, use_gqa = False, use_fp16 = False
        #   and remove past_kv from the output
        # token generation (get_sample_with_past_kv_inputs):
        #   past_seq_len >= 1, seq_len = 1, use_gqa = False, use_fp16 = False
        # By using a single function with no default values, we can avoid confusion and are deliberate about the sizes
        # can instead write dummy input functions like 'get_merged_decoder_with_past_dummy_inputs' if needed

        # Using Namespace class to access dict items like class attributes
        from transformers import AutoConfig

        model_attributes = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code).__dict__
        world_size = model_attributes.get("world_size", 1)
        vocab_size = model_attributes.get("vocab_size", 50256)
        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
        attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.int64)
        position_ids = self.get_position_ids(attention_mask, past_seq_len=past_seq_len)
        position_ids = position_ids.to(torch.int64)
        past_kv = self.get_past_kv_inputs(
            model_attributes, batch_size, past_seq_len, use_fp16=use_fp16, world_size=world_size
        )

        return (input_ids, attention_mask, position_ids, past_kv)

    def get_position_ids(self, attention_mask: torch.Tensor, past_seq_len: int):
        """Get position_ids from attention_mask."""
        # this is generic but in practice we only expect to see two scenarios for (past_seq_len, seq_len)
        # prompt generation: (0, seq_len) -> position_ids = (batch_size, seq_len)
        # token generation: (past_seq_len, 1) -> position_ids = (batch_size, 1)
        # Note: The merged model only works in these two scenarios
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids[:, past_seq_len:]

    def get_past_kv_inputs(
        self, model_attributes, batch_size: int, past_seq_len: int, use_fp16: bool, world_size: int = 1
    ):
        """Get past_key_values for all layers.

        Shape of past_key_values is (batch_size, num_heads, past_seq_len, head_size).
        """
        from olive.model.utils.hf_mappings import HIDDEN_SIZE_NAMES, NUM_HEADS_NAMES, NUM_HIDDEN_LAYER_NAMES

        num_attention_heads = find_first_matched_value(model_attributes, NUM_HEADS_NAMES)
        head_size = find_first_matched_value(model_attributes, HIDDEN_SIZE_NAMES) // num_attention_heads
        num_hidden_layers = find_first_matched_value(model_attributes, NUM_HIDDEN_LAYER_NAMES)
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        return [
            (
                torch.rand(batch_size, num_attention_heads, past_seq_len, head_size, dtype=torch_dtype),
                torch.rand(batch_size, num_attention_heads, past_seq_len, head_size, dtype=torch_dtype),
            )
            for _ in range(num_hidden_layers)
        ]

    def flatten_past_kv_inputs(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Flatten past_key_values to a dict of past_key and past_value. For ONNX model only."""
        past_kv = {}
        # Convert list of past_kv to dict of past_key and past_value
        for i, (past_k, past_v) in enumerate(past_key_values):
            past_kv[self.ort_past_key_name.replace("<id>", str(i))] = past_k
            past_kv[self.ort_past_value_name.replace("<id>", str(i))] = past_v
        return past_kv
