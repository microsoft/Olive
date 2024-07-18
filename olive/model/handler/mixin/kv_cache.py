# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

from olive.common.utils import exclude_keys
from olive.model.config import IoConfig, KVCacheConfig
from olive.model.config.io_config import is_kv_cache_required

logger = logging.getLogger(__name__)


class PytorchKvCacheMixin:
    def merge_kv_cache_hook(self, dummy_inputs, past_kv_names: str = "past_key_values"):
        if not isinstance(dummy_inputs, dict):
            return dummy_inputs
        dummy_inputs = self.past_key_values_input_filter_hook(dummy_inputs)
        io_config = IoConfig.parse_obj(self.io_config)
        if isinstance(io_config.kv_cache, dict):
            kv_cache_config = KVCacheConfig.parse_obj(io_config.kv_cache)
            unused_keys = set()
            if kv_cache_config and not dummy_inputs.get(past_kv_names):
                torch_past_key_values = []
                kv_inputs = kv_cache_config.get_ort_past_kv_names()
                for k_input, v_input in zip(kv_inputs[::2], kv_inputs[1::2]):
                    if k_input not in dummy_inputs or v_input not in dummy_inputs:
                        raise ValueError(
                            f"Cannot find past key-value pair for {k_input} and {v_input} in dummy inputs."
                        )
                    torch_past_key_values.append((dummy_inputs[k_input], dummy_inputs[v_input]))
                    unused_keys.add(k_input)
                    unused_keys.add(v_input)
                dummy_inputs[past_kv_names] = torch_past_key_values
            if unused_keys:
                logger.debug("Merged kv inputs: %s to list for torch model inference", unused_keys)
                dummy_inputs = exclude_keys(dummy_inputs, unused_keys)
        return dummy_inputs

    def merge_kv_cache_to_tuple_hook(self, dummy_inputs, past_kv_names: str = "past_key_values"):
        """Merge the key-value cache inputs to a tuple for torch model inference.

        Torch legacy exporter only accept tuple inputs where
        all but the last element of the tuple will be passed as non-keyword arguments,
        and named arguments will be set from the last element. If a named argument is
        not present in the dictionary, it is assigned the default value, or None if a
        default value is not provided.
        Model details from: https://pytorch.org/docs/stable/onnx_torchscript.html#module-torch.onnx
        """
        return (self.merge_kv_cache_hook(dummy_inputs, past_kv_names),)

    # TODO(jambayk): consider removing this since we don't use hf dataset for dummy inputs anymore
    def past_key_values_input_filter_hook(self, dummy_inputs, past_kv_names: str = "past_key_values"):
        if not isinstance(dummy_inputs, dict):
            return dummy_inputs
        # this can happen when we are using an hf dataset to generate dummy inputs
        # only handle dict for now since we cannot get the name of the input from a list/tuple
        io_config = self.io_config
        dummy_input_keys = set(dummy_inputs.keys())

        # after the expansion, user should provide the correct input names for inference
        for name, dm_input in dummy_inputs.items():
            # the `past_key_values` is the argument name from huggingface model class
            # which is independent of the kv-related variables in input list provided by users
            # if user provided the kv-related variables, we should not remove
            # the `past_key_values` from dummy inputs. But if not, we should remove it.
            if (
                name == past_kv_names
                and isinstance(dm_input, list)
                and is_kv_cache_required(dm_input, IoConfig.parse_obj(io_config))
            ):
                dummy_input_keys.discard(name)

        unused_keys = dummy_input_keys - set(io_config.get("input_names"))

        if unused_keys:
            logger.debug("Removing unused dummy inputs: %s", unused_keys)
            dummy_inputs = exclude_keys(dummy_inputs, unused_keys)
        return dummy_inputs
