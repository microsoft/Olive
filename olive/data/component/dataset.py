# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from olive.common.hf.utils import get_model_config
from olive.common.utils import resolve_torch_dtype
from olive.constants import Framework


class BaseDataset(TorchDataset):
    """Define the Olive dataset which should return the data with following format.

    1. [data, label] for supervised learning
    2. [data] for unsupervised learning
    The data should be a list or dict of numpy arrays or torch tensors
    """

    def __init__(self, data, label_col, max_samples=None, **kwargs):
        """Initialize the dataset."""
        self.data = data
        self.label_col = label_col
        self.max_samples = max_samples

    def __len__(self):
        """Return the length of the dataset."""
        num_samples = len(self.data)
        if self.max_samples is not None:
            # if max_samples is not None, return the min of num_samples and max_samples
            num_samples = min(num_samples, self.max_samples)
        return num_samples

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items() if k != self.label_col}
        label = self.data[index][self.label_col]
        return data, label


class DummyDataset(BaseDataset):
    def __init__(
        self,
        input_shapes,
        input_names: Optional[List] = None,
        input_types: Optional[List] = None,
        max_samples: Optional[int] = 32,
        **kwargs,
    ):
        """Initialize the dataset with dummy data.

        if input_names is not provided, the dataset will return a tuple of tensors
        else the dataset will return a dict of tensors
        """
        # pylint: disable=super-init-not-called
        if not input_types:
            input_types = ["float32"] * len(input_shapes)
        input_types = [resolve_torch_dtype(dtype_str) for dtype_str in input_types]

        if input_names:
            dummy_data = {}
            for input_name, input_shape, input_type in zip(input_names, input_shapes, input_types):
                dummy_data.update({input_name: torch.ones(input_shape, dtype=input_type)})
            dummy_data = dummy_data if len(dummy_data) > 1 else dummy_data[input_names[0]]
        else:
            dummy_data = []
            for shape, dtype in zip(input_shapes, input_types):
                dummy_data.append(torch.ones(shape, dtype=dtype))
            dummy_data = tuple(dummy_data) if len(dummy_data) > 1 else dummy_data[0]

        self.max_samples = max_samples
        self.dummy_data = dummy_data, torch.tensor([0])

    def __len__(self):
        return self.max_samples

    def __getitem__(self, index):
        # From https://docs.python.org/3/reference/datamodel.html#object.__getitem__,
        # __getitem__ should raise IndexError when index is out of range
        # Otherwise, the enumerate function will enter infinite loop
        if index < 0 or index >= self.max_samples:
            raise IndexError("Index out of range")

        return self.dummy_data


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


class TransformersDummyDataset(BaseDataset):
    def __init__(
        self,
        model_name: str,
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
        max_samples: Optional[int] = 32,
        use_step: bool = False,
        ignore_input_fields: Optional[List[str]] = None,
    ):
        # pylint: disable=super-init-not-called
        self.model_name = model_name
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
        self.max_samples = max_samples
        self.use_step = use_step
        self.ignore_input_fields = ignore_input_fields

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        input_ids, attention_mask, position_ids, past_kv = self.get_merged_sample_with_past_kv_inputs(
            model_name=self.model_name,
            seq_len=self.seq_len,
            past_seq_len=self.past_seq_len,
            use_fp16=self.use_fp16,
            trust_remote_code=self.trust_remote_code,
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.use_step and self.model_framework == Framework.ONNX:
            inputs["step"] = torch.tensor(0, dtype=torch.int64)

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

        # filter the ignore_input_fields
        if self.ignore_input_fields:
            inputs = {k: v for k, v in inputs.items() if k not in self.ignore_input_fields}
        return (inputs, 1)

    def enable_past_present_share_buffer(self, ort_inputs: dict, past_seq_len: int, max_seq_len: int):
        """Enable past-present share buffer for GQA. For ONNX model + FP16 + GQA only."""
        for k, v in ort_inputs.items():
            # Allocate new buffers with max_seq_len for GQA
            if "past_key_values" in k:
                # Copy v (BxSxPxH) into new_v (BxSxMxH)
                num_heads, _, head_size = v.shape
                new_v = torch.zeros((num_heads, max_seq_len, head_size), dtype=v.dtype)
                new_v[:num_heads, :past_seq_len, :head_size] = v
                ort_inputs[k] = new_v
        return ort_inputs

    def get_merged_sample_with_past_kv_inputs(
        self,
        model_name: str,
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
            input_ids: (seq_len)
            attention_mask: (past_seq_len + seq_len)
            position_ids: (seq_len)
            past_key: (num_heads, past_seq_len, head_size)
            past_value: (num_heads, past_seq_len, head_size)
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
        model_attributes = get_model_config(model_name, trust_remote_code=trust_remote_code).to_dict()
        world_size = model_attributes.get("world_size", 1)
        vocab_size = model_attributes.get("vocab_size", 50256)
        input_ids = torch.randint(low=0, high=vocab_size, size=(seq_len,), dtype=torch.int64)
        attention_mask = torch.ones(past_seq_len + seq_len, dtype=torch.int64)
        position_ids = self.get_position_ids(attention_mask, past_seq_len=past_seq_len)
        position_ids = position_ids.to(torch.int64)
        past_kv = self.get_past_kv_inputs(model_attributes, past_seq_len, use_fp16=use_fp16, world_size=world_size)

        return (input_ids, attention_mask, position_ids, past_kv)

    def get_position_ids(self, attention_mask: torch.Tensor, past_seq_len: int):
        """Get position_ids from attention_mask."""
        # this is generic but in practice we only expect to see two scenarios for (past_seq_len, seq_len)
        # prompt generation: (0, seq_len) -> position_ids = (seq_len)
        # token generation: (past_seq_len, 1) -> position_ids = (1)
        # Note: The merged model only works in these two scenarios
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids[past_seq_len:]

    def get_past_kv_inputs(self, model_attributes, past_seq_len: int, use_fp16: bool, world_size: int = 1):
        """Get past_key_values for all layers.

        Shape of past_key_values is (num_heads, past_seq_len, head_size).
        """
        from olive.common.hf.wrapper import ModelWrapper

        model_wrapper = ModelWrapper(model_attributes)
        num_key_value_heads = model_wrapper.num_key_value_heads
        num_attention_heads = model_wrapper.num_attention_heads
        hidden_size = model_wrapper.hidden_size
        if num_attention_heads is None or hidden_size is None:
            raise ValueError("Cannot find num_attention_heads or hidden_size in model attributes")
        num_attention_heads = num_attention_heads // world_size

        head_size = hidden_size // num_attention_heads
        if num_key_value_heads is not None:
            num_key_value_heads = num_key_value_heads // world_size
            # adjust num_attention_heads to num_key_value_heads for MoE models to get the right shape
            num_attention_heads = num_key_value_heads

        num_hidden_layers = model_wrapper.num_hidden_layers
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        return [
            (
                torch.rand(num_attention_heads, past_seq_len, head_size, dtype=torch_dtype),
                torch.rand(num_attention_heads, past_seq_len, head_size, dtype=torch_dtype),
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
