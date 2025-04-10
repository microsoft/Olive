# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np
from onnxruntime import OrtValue

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from onnxruntime import IOBinding
    from torch import Tensor


class Cache(ABC):
    """Abstract class for KV cache management."""

    def __init__(
        self,
        past_names: List[str],
        present_names: List[str],
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str = "float32",
    ):
        """Initialize the cache.

        :param past_names: List of past key and value names.
        :param present_names: List of present key and value names.
            Names must be in the same order as past_names.
        :param batch_size: Batch size.
        :param num_kv_heads: Number of key-value heads.
        :param head_dim: Dimension of each key-value head.
        :param dtype: Data type of the key-value tensors.
        """
        self.past_names = past_names
        self.present_names = present_names
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

    @abstractmethod
    def update(self, present_kvs: List["NDArray"]):
        """Update the cache with the present key-value tensors.

        :param present_kvs: List of present key-value tensors. This must be past key-value tensors
            concatenated with the key-value tensors from the current step.
        """
        raise NotImplementedError

    @abstractmethod
    def get_kv_inputs(self) -> Dict[str, "NDArray"]:
        """Get the key-value tensors to be used as inputs for the next step."""
        raise NotImplementedError


class DynamicCache(Cache):
    """Dynamic cache that doesn't have a fixed size for the cache tensors.

    It stores the most recent key-value tensors and uses them as inputs for the next step.
    """

    def __init__(
        self,
        past_names: List[str],
        present_names: List[str],
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str = "float32",
    ):
        """Initialize the cache.

        :param past_names: List of past key and value names.
        :param present_names: List of present key and value names.
            Names must be in the same order as past_names.
        :param batch_size: Batch size.
        :param num_kv_heads: Number of key-value heads.
        :param head_dim: Dimension of each key-value head.
        :param dtype: Data type of the key-value tensors.
        """
        super().__init__(past_names, present_names, batch_size, num_kv_heads, head_dim, dtype)
        # cache before prompt processing is empty tensor
        self.cache = {
            k: np.zeros((self.batch_size, self.num_kv_heads, 0, self.head_dim), dtype=self.dtype)
            for k in self.past_names
        }

    def update(self, present_kvs: List["NDArray"]):
        """Update the cache with the present key-value tensors.

        :param present_kvs: List of present key-value tensors.
        """
        for k, v in zip(self.past_names, present_kvs):
            self.cache[k] = v

    def get_kv_inputs(self) -> Dict[str, "NDArray"]:
        """Get the key-value tensors to be used as inputs for the next step.

        :return: Dictionary of key-value tensors.
        """
        return self.cache


class StaticCache(Cache):
    """Static cache that has a fixed size for the cache tensors.

    During prompt processing, it stores the present key-value tensors at the beginning of the cache.
    During token generation, it stores the new key-value tensors at the first empty slot in the cache.
    """

    def __init__(
        self,
        past_names: List[str],
        present_names: List[str],
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str = "float32",
        max_cache_len: int = 2048,
    ):
        """Initialize the cache.

        :param past_names: List of past key and value names.
        :param present_names: List of present key and value names.
            Names must be in the same order as past_names.
        :param batch_size: Batch size.
        :param num_kv_heads: Number of key-value heads.
        :param head_dim: Dimension of each key-value head.
        :param dtype: Data type of the key-value tensors.
        :param max_cache_len: Maximum length of the cache.
        """
        super().__init__(past_names, present_names, batch_size, num_kv_heads, head_dim, dtype)
        self.max_cache_len = max_cache_len
        # allocate cache with zeros
        self.cache = {
            k: np.zeros((self.batch_size, self.num_kv_heads, self.max_cache_len, self.head_dim), dtype=self.dtype)
            for k in self.past_names
        }
        # keep track of the length of the cache
        self.seen_len = 0

    def update(self, present_kvs: List["NDArray"]):
        """Update the cache with the present key-value tensors.

        At the prompt processing step, i.e., when the cache is empty, the present key-value tensors can have any length
        smaller than or equal to max_cache_len.
        In token generation step, i.e., with one new token at each step, the present key-value tensors must have length
        equal to max_cache_len + 1. The last key-value tensor at the end of present_kvs is inserted into the first empty
        slot in the cache.

        :param present_kvs: List of present key-value tensors.
        """
        present_len = present_kvs[0].shape[2]
        assert present_len > 0, "present_kvs cannot be empty"

        if self.seen_len == 0:
            assert present_len <= self.max_cache_len, (
                "present_kvs is longer than max_cache_len during prompt processing"
            )
            # prompt processing
            for k, v in zip(self.past_names, present_kvs):
                self.cache[k][:, :, :present_len] = v
            self.seen_len = present_len
            return

        assert present_len == self.max_cache_len + 1, (
            "present_kvs must be one step longer than max_cache_len in token generation"
        )
        for k, v in zip(self.past_names, present_kvs):
            self.cache[k][:, :, self.seen_len] = v[:, :, -1]
        self.seen_len += 1

    def get_kv_inputs(self) -> Dict[str, "NDArray"]:
        """Get the key-value tensors to be used as inputs for the next step.

        During prompt processing, this returns an kev-value tensors with 0 length.
        During token generation, this returns the kev-value tensors with length max_cache_len.
            Appropriate attention mask should be applied to the key-value tensors to mask out
            padded key-value tensors from previous steps and the unused slots.

        :return: Dictionary of key-value tensors.
        """
        if self.seen_len == 0:
            return {
                k: np.zeros((self.batch_size, self.num_kv_heads, 0, self.head_dim), dtype=self.dtype)
                for k in self.past_names
            }
        return self.cache


class IOBoundCache(ABC):
    """Abstract class for KV cache management with IO binding support."""

    def __init__(
        self,
        past_names: List[str],
        present_names: List[str],
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str = "float32",
        device: str = "cpu",
        device_id: int = 0,
        backend: str = "ort",
    ):
        """Initialize the cache.

        :param past_names: List of past key and value names.
        :param present_names: List of present key and value names.
            Names must be in the same order as past_names.
        :param batch_size: Batch size.
        :param num_kv_heads: Number of key-value heads.
        :param head_dim: Dimension of each key-value head.
        :param dtype: Data type of the key-value tensors.
        :param device: Device type for the cache tensors.
        :param device_id: Device ID for the cache tensors.
        :param backend: Backend for the cache tensors. Options: "ort" or "torch".
        """
        self.past_names = past_names
        self.present_names = present_names
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.device_id = device_id
        self.backend = backend

        # torch backend specific
        self.torch_device = None
        self.torch_dtype = None
        if self.backend == "torch":
            assert self.device in {"cpu", "cuda"}, f"device {self.device} is not supported with `torch` backend"

            import torch

            self.torch_device = "cpu" if self.device == "cpu" else torch.device("cuda", self.device_id)
            self.torch_dtype = getattr(torch, self.dtype)

    def get_empty_ortvalue(self, *shape) -> OrtValue:
        """Get an empty OrtValue with the given shape.

        :param shape: Shape of the OrtValue.
        :return: OrtValue with the given shape with zeros.
        """
        return OrtValue.ortvalue_from_shape_and_type(shape, self.dtype, self.device, self.device_id)

    def get_empty_buffer(self, *shape) -> Union["Tensor", OrtValue]:
        """Get an empty buffer with the given shape.

        :param shape: Shape of the buffer.
        :return: Buffer with the given shape with zeros. If backend is "torch", returns a torch.Tensor.
            If backend is "ort", returns an OrtValue.
        """
        if self.backend == "torch":
            import torch

            return torch.zeros(shape, dtype=self.torch_dtype, device=self.torch_device)
        return self.get_empty_ortvalue(*shape)

    @abstractmethod
    def update(self, present_kvs: List[OrtValue]):
        """Update the cache with the present key-value tensors.

        :param present_kvs: List of present key-value tensors as OrtValue.
        """
        raise NotImplementedError

    @abstractmethod
    def bind_kv_io(self, io_binding: "IOBinding"):
        """Bind the cache tensors to the IO binding object.

        Past key-value tensors are bound as inputs and present key-value tensors are bound as outputs.

        :param io_binding: IOBinding object to bind the cache tensors.
        """
        raise NotImplementedError


class DynamicIOBoundCache(IOBoundCache):
    """Dynamic IO bound cache that doesn't have a fixed size for the cache tensors."""

    def __init__(
        self,
        past_names: List[str],
        present_names: List[str],
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str = "float32",
        device: str = "cpu",
        device_id: int = 0,
        backend: str = "ort",
    ):
        """Initialize the cache.

        :param past_names: List of past key and value names.
        :param present_names: List of present key and value names.
            Names must be in the same order as past_names.
        :param batch_size: Batch size.
        :param num_kv_heads: Number of key-value heads.
        :param head_dim: Dimension of each key-value head.
        :param dtype: Data type of the key-value tensors.
        :param device: Device type for the cache tensors.
        :param device_id: Device ID for the cache tensors.
        :param backend: Backend for the cache tensors. Options: "ort" or "torch".
            There is no implemetation difference for this class between "ort" and "torch".
        """
        super().__init__(
            past_names, present_names, batch_size, num_kv_heads, head_dim, dtype, device, device_id, backend
        )
        # will just use ortvalue since we don't need to access the cache
        self.cache = {
            k: self.get_empty_ortvalue(self.batch_size, self.num_kv_heads, 0, self.head_dim) for k in self.past_names
        }
        # won't pre-allocate output cache since shape is dynamic and we can just bind the output

    def update(self, present_kvs: List[OrtValue]):
        """Update the cache with the present key-value tensors.

        :param present_kvs: List of present key-value tensors as OrtValue.
        """
        for k, ort_value in zip(self.past_names, present_kvs):
            self.cache[k] = ort_value

    def bind_kv_io(self, io_binding: "IOBinding"):
        """Bind the cache tensors to the IO binding object.

        Past key-value tensors are bound as inputs and present key-value tensors are bound as outputs.
        Outputs are not pre-allocated and will be allocated dynamically by the session.

        :param io_binding: IOBinding object to bind the cache tensors.
        """
        for k, ort_value in self.cache.items():
            io_binding.bind_ortvalue_input(k, ort_value)
        for k in self.present_names:
            # let the output cache be allocated dynamically
            io_binding.bind_output(k, self.device, self.device_id)


class StaticIOBoundCache(IOBoundCache):
    """Static IO bound cache that has a fixed size for the cache tensors.

    During prompt processing, it stores the present key-value tensors at the beginning of the cache.
    During token generation, it stores the new key-value tensors at the first empty slot in the cache.
    Torch backend is faster than ort backend.
    """

    def __init__(
        self,
        past_names: List[str],
        present_names: List[str],
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str = "float32",
        device: str = "cpu",
        device_id: int = 0,
        backend: str = "ort",
        max_cache_len: int = 2048,
    ):
        """Initialize the cache.

        :param past_names: List of past key and value names.
        :param present_names: List of present key and value names.
            Names must be in the same order as past_names.
        :param batch_size: Batch size.
        :param num_kv_heads: Number of key-value heads.
        :param head_dim: Dimension of each key-value head.
        :param dtype: Data type of the key-value tensors.
        :param device: Device type for the cache tensors.
        :param device_id: Device ID for the cache tensors.
        :param backend: Backend for the cache tensors. Options: "ort" or "torch".
            If backed is ort, the cache tensors are stored as OrtValue.
            If backend is torch, the cache tensors are stored as torch.Tensor. This is faster than ort backend.
        :param max_cache_len: Maximum length of the cache.
        """
        super().__init__(
            past_names, present_names, batch_size, num_kv_heads, head_dim, dtype, device, device_id, backend
        )
        self.max_cache_len = max_cache_len
        # allocate cache with zeros
        self.cache = {
            k: self.get_empty_buffer(self.batch_size, self.num_kv_heads, self.max_cache_len, self.head_dim)
            for k in self.past_names
        }
        # pre-allocate output cache for token generation
        # might be helpful when using cuda graph since the buffer is same across iterations
        self.output_cache = {
            k: self.get_empty_buffer(self.batch_size, self.num_kv_heads, self.max_cache_len + 1, self.head_dim)
            for k in self.present_names
        }
        # keep track of the length of the cache
        self.seen_len = 0

    def update(self, present_kvs: List[OrtValue]):
        """Update the cache with the present key-value tensors.

        At the prompt processing step, i.e., when the cache is empty, the present key-value tensors can have any length
        smaller than or equal to max_cache_len.
        In token generation step, i.e. with one new token at each step, the present key-value tensors must have length
        equal to max_cache_len + 1. The last key-value tensor at the end of present_kvs is inserted into the first empty
        slot in the cache.

        :param present_kvs: List of present key-value tensors as OrtValue.
        """
        if self.seen_len == 0:
            # prompt processing
            present_len = present_kvs[0].shape()[2]
            assert present_len > 0, "present_kvs cannot be empty"
            assert present_len <= self.max_cache_len, (
                "present_kvs is longer than max_cache_len during prompt processing"
            )
            for k, v in zip(self.past_names, present_kvs):
                if self.backend == "torch":
                    import torch

                    self.cache[k][:, :, :present_len] = torch.tensor(v.numpy(), device=self.torch_device)
                else:
                    np_value = self.cache[k].numpy()
                    np_value[:, :, :present_len] = v.numpy()
                    self.cache[k].update_inplace(np_value)
            self.seen_len = present_len
            return

        # token generation
        for past_k, present_k, v in zip(self.past_names, self.present_names, present_kvs):
            assert self.output_cache[present_k].data_ptr() == v.data_ptr(), (
                "out cache ortvalue should be same as present ortvalue"
            )
            if self.backend == "torch":
                self.cache[past_k][:, :, self.seen_len] = self.output_cache[present_k][:, :, -1]
            else:
                np_value = self.cache[past_k].numpy()
                np_value[:, :, self.seen_len] = v.numpy()[:, :, -1]
                self.cache[past_k].update_inplace(np_value)
        self.seen_len += 1

    def bind_kv_io(self, io_binding: "IOBinding"):
        """Bind the cache tensors to the IO binding object.

        Past key-value tensors are bound as inputs and present key-value tensors are bound as outputs.
        During prompt processing, the past key-value tensors have length 0 and the present key-value tensors
            will be allocated dynamically by the session.
        During token generation, the past key-value tensors have length max_cache_len and the present key-value tensors
            have length max_cache_len + 1. Appropriate attention mask should be applied to the key-value tensors to mask
            out padded key-value tensors from previous steps and the unused slots.

        :param io_binding: IOBinding object to bind the cache tensors.
        """
        if self.seen_len == 0:
            # just use ort value since we don't need to access this after prompt processing
            for k in self.past_names:
                io_binding.bind_ortvalue_input(
                    k, self.get_empty_ortvalue(self.batch_size, self.num_kv_heads, 0, self.head_dim)
                )
            for k in self.present_names:
                io_binding.bind_output(k, self.device, self.device_id)
            return

        for k, v in self.cache.items():
            if self.backend == "torch":
                io_binding.bind_input(
                    name=k,
                    device_type=self.device,
                    device_id=self.device_id,
                    element_type=self.dtype,
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )
            else:
                io_binding.bind_ortvalue_input(k, v)
        for k, v in self.output_cache.items():
            if self.backend == "torch":
                io_binding.bind_output(
                    name=k,
                    device_type=self.device,
                    device_id=self.device_id,
                    element_type=self.dtype,
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )
            else:
                io_binding.bind_ortvalue_output(k, v)


class GQASharedCache(IOBoundCache):
    """GroupQueryAttention (GQA) contrib op compatible cache.

    It uses the same fixed length tensor buffers for both past and present key-value tensors.
    """

    def __init__(
        self,
        past_names: List[str],
        present_names: List[str],
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int = 2048,
        dtype: str = "float32",
        device: str = "gpu",
        device_id: int = 0,
        backend: str = "ort",
    ):
        """Initialize the cache.

        :param past_names: List of past key and value names.
        :param present_names: List of present key and value names.
            Names must be in the same order as past_names.
        :param batch_size: Batch size.
        :param num_kv_heads: Number of key-value heads.
        :param head_dim: Dimension of each key-value head.
        :param max_cache_len: Maximum length of the cache.
        :param dtype: Data type of the key-value tensors.
        :param device: Device type for the cache tensors.
        :param device_id: Device ID for the cache tensors.
        :param backend: Backend for the cache tensors. Options: "ort" or "torch".
            There is no implemetation difference for this class between "ort" and "torch".
        """
        super().__init__(
            past_names, present_names, batch_size, num_kv_heads, head_dim, dtype, device, device_id, backend
        )
        if device == "dml" and max_cache_len % 4 == 0:
            # there is an overflow bug in DML for max_cache_len % 4 == 0
            max_cache_len += 1
        self.max_cache_len = max_cache_len
        # will just use ortvalue since we don't need to access the cache
        # allocate cache with zeros
        # both past and present will use the same cache
        self.cache = {
            k: self.get_empty_ortvalue(self.batch_size, self.num_kv_heads, self.max_cache_len, self.head_dim)
            for k in self.past_names
        }

    def update(self, present_kvs: List[OrtValue]):
        """Update the cache with the present key-value tensors.

        Expects the present key-value tensors to be the same as the cache tensors.
        There is no padding in between the key-value tensors. For each element in the batch, the key-value tensors
        are inserted right after the previous key-value tensors. In case of uneven sequence lengths, the key-value
        tensors at each step will be ragged with right padding.

        :param present_kvs: List of present key-value tensors as OrtValue.
        """
        for k, v in zip(self.past_names, present_kvs):
            assert self.cache[k].data_ptr() == v.data_ptr(), "cache ortvalue should be same as present ortvalue"

    def bind_kv_io(self, io_binding: "IOBinding"):
        """Bind the cache tensors to the IO binding object.

        Both past and present key-value tensors are bound to the same cache with length max_cache_len.
        Appropriate attention mask should be used to inform the GroupQueryAttention contrib op about the
            valid length of the key-value tensors and input tokens.

        :param io_binding: IOBinding object to bind the cache tensors.
        """
        for past_k, present_k in zip(self.past_names, self.present_names):
            io_binding.bind_ortvalue_input(past_k, self.cache[past_k])
            io_binding.bind_ortvalue_output(present_k, self.cache[past_k])
