# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import re
from typing import List, Union

import onnx
import torch
from tqdm import tqdm

from olive.common.hf.model_io import replace_with_extended_mask
from olive.common.utils import format_data
from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_dataloader()
@Registry.register_default_dataloader()
def default_dataloader(dataset, batch_size=1, **kwargs):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, **kwargs)


@Registry.register_dataloader()
def dataloader_with_ignored_batch_fields(
    dataset, batch_size: int = 1, fields_no_batch: Union[str, List] = "step", **kwargs
):
    from torch.utils.data import DataLoader, default_collate

    ignore_fields = [fields_no_batch] if isinstance(fields_no_batch, str) else fields_no_batch

    def ignore_batch_collate_fn(batch):
        nonlocal ignore_fields
        # ignore to batch the data fields for given inputs
        input_data, label = default_collate(batch)
        if isinstance(input_data, dict):
            for k, v in input_data.items():
                if k in ignore_fields:
                    input_data[k] = v[0].item()
        return [input_data, label]

    return DataLoader(dataset, batch_size=batch_size, collate_fn=ignore_batch_collate_fn, **kwargs)


@Registry.register_dataloader()
def no_auto_batch_dataloader(dataset, **kwargs):
    from torch.utils.data import DataLoader

    # torch dataloader will automatically batch if batch_size is not None
    # this dataloader will not batch. Assumes that the dataset already returns a batch
    kwargs.pop("batch_size", None)
    return DataLoader(dataset, batch_size=None, **kwargs)


def default_calibration_dataloader(dataloader, model_path=None, io_config=None, **kwargs):
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationDataReader(CalibrationDataReader):
        def __init__(self, dataloader, io_config=None, **kwargs):
            self.dataloader = dataloader
            self.io_config = io_config
            self.kwargs = kwargs
            self.data_iter = iter(self.dataloader)

        def get_next(self):
            if self.data_iter is None:
                self.data_iter = iter(self.dataloader)
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return None
            if isinstance(batch, (list, tuple)):
                # [input, label] or (input, label)
                batch = batch[0]
            assert isinstance(batch, dict), "Only support dict type batch data"

            if self.io_config:
                batch = format_data(batch, self.io_config)
            else:
                batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
            return batch

        def rewind(self):
            self.data_iter = None

    if model_path and io_config:
        # there is no overhead for non-llm models
        dataloader = LLMAugmentedDataLoader(dataloader, model_path, io_config)

    return _CalibrationDataReader(dataloader, io_config, **kwargs)


class LLMAugmentedDataLoader:
    # TODO(jambayk): provide flags to enable/disable prefill+decode inputs
    def __init__(self, dataloader, model_path, io_config):
        self.dataloader = dataloader
        self.model_path = model_path
        self.io_config = io_config
        # self.io_config["input_shapes"] = dict(zip(self.io_config["input_names"], self.io_config["input_shapes"]))

        self.position_ids = "position_ids" in self.io_config["input_names"]
        self.extended_attention_mask = (
            "attention_mask" in self.io_config["input_names"]
            and len(self.io_config["input_shapes"][self.io_config["input_names"].index("attention_mask")]) == 4
        )

        self.kv_info = self.get_kv_info(self.io_config)
        self.has_gqa = False
        for node in onnx.load(self.model_path, load_external_data=False).graph.node:
            if node.op_type == "GroupQueryAttention":
                self.has_gqa = True
                break
        logger.debug("Model has GroupQueryAttention: %s", self.has_gqa)

        self._session = None

    def __len__(self):
        return len(self.dataloader) * (
            2
            if not self.has_gqa and self.kv_info and self.kv_info["past_names"][0] not in next(iter(self.dataloader))[0]
            else 1
        )

    def __iter__(self):
        progress_bar = tqdm(total=len(self), desc="Onnx Dataloader")
        for batch, label in self.dataloader:
            if self.extended_attention_mask and batch["attention_mask"].ndim == 2:
                # model expects 4d attention mask
                replace_with_extended_mask(batch, "causal", -5000)
            if self.position_ids and "position_ids" not in batch:
                # create position ids from attention mask
                batch["position_ids"] = self.get_position_ids(batch["input_ids"], batch["attention_mask"])

            if self.kv_info and self.kv_info["past_names"][0] not in batch:
                prefill_slice = slice(0, None) if self.has_gqa else slice(None, -1)
                attention_mask = batch["attention_mask"]
                if attention_mask.ndim == 4:
                    attention_mask = (attention_mask[:, 0, -1] == 0).to(batch["input_ids"].dtype)
                # prepend mask for past keys and values
                if not self.has_gqa:
                    attention_mask = torch.cat([torch.zeros_like(attention_mask[:, :1]), attention_mask], dim=-1)

                # prefill: all - 1 tokens, no past keys/values (= 0 past seq len)
                prefill_batch = {
                    "input_ids": batch["input_ids"][:, prefill_slice],
                    "attention_mask": attention_mask[:, prefill_slice],
                    **self.get_empty_kv_cache(batch["input_ids"].shape[0], self.kv_info, self.has_gqa),
                }
                if self.position_ids:
                    prefill_batch["position_ids"] = batch["position_ids"][:, prefill_slice]
                if self.extended_attention_mask:
                    replace_with_extended_mask(prefill_batch, "causal", -5000)

                prefill_label = (
                    label[:, prefill_slice]
                    if (isinstance(label, torch.Tensor) and label.shape == batch["input_ids"].shape)
                    else label
                )

                yield prefill_batch, prefill_label
                progress_bar.update(1)

                if self.has_gqa:
                    continue

                # decode: last token, all - 1 past keys/values generated from prefill
                session_outputs = self.session.run(None, format_data(prefill_batch, self.io_config))
                session_outputs = dict(zip(self.io_config["output_names"], session_outputs))

                decode_batch = {
                    "input_ids": batch["input_ids"][:, -1:],
                    "attention_mask": attention_mask,
                    **{v: session_outputs[k] for k, v in self.kv_info["present_to_past"].items()},
                }
                if self.position_ids:
                    decode_batch["position_ids"] = batch["position_ids"][:, -1:]
                if self.extended_attention_mask:
                    replace_with_extended_mask(decode_batch, "causal", -5000)

                decode_label = (
                    label[:, -1:]
                    if (isinstance(label, torch.Tensor) and label.shape == batch["input_ids"].shape)
                    else label
                )

                yield decode_batch, decode_label
                progress_bar.update(1)
            else:
                yield batch, label
                progress_bar.update(1)

    @property
    def session(self):
        from onnxruntime import InferenceSession

        if self._session is not None:
            return self._session
        try:
            # will try to use CUDAExecutionProvider if possible
            self._session = InferenceSession(self.model_path, providers=["CUDAExecutionProvider"])
        except Exception:
            logger.debug("Failed to use CUDAExecutionProvider for generating past keys/values, falling back to CPU")
            self._session = InferenceSession(self.model_path)
        return self._session

    @staticmethod
    def get_position_ids(input_ids, attention_mask):
        sequence_length = input_ids.shape[-1]
        if attention_mask.ndim == 2:
            return (attention_mask.cumsum(-1) - 1)[:, -sequence_length:]
        elif attention_mask.ndim == 4:
            return ((attention_mask[:, 0, -1] == 0).cumsum(-1) - 1)[:, -sequence_length:]
        else:
            raise ValueError("Invalid attention mask shape")

    @staticmethod
    def get_kv_info(io_config):
        # assuming batch_size, num_kv_heads, past_seq_len, head_size
        kv_options = {
            r"past_key_values.(\d+).key": {
                "past_key": "past_key_values.%d.key",
                "past_value": "past_key_values.%d.value",
                "present_key": "present.%d.key",
                "present_value": "present.%d.value",
            },
            r"past_key_(\d+)": {
                "past_key": "past_key_%d",
                "past_value": "past_value_%d",
                "present_key": "present_key_%d",
                "present_value": "present_value_%d",
            },
        }

        # Find the format of the past keys and values
        # only accept dynamic shapes for now
        kv_format = None
        for idx, i_name in enumerate(io_config["input_names"]):
            for pattern in kv_options:
                if re.match(pattern, i_name) and not isinstance(io_config["input_shapes"][idx][2], int):
                    kv_format = pattern
                    break
            if kv_format:
                break

        if kv_format is None:
            return None

        num_layers = 0
        for i_name in io_config["input_names"]:
            num_layers += int(re.match(kv_format, i_name) is not None)
        logger.debug("Found %d layers with past keys/values", num_layers)

        past_names = []
        present_to_past = {}
        for k in ["key", "value"]:
            past_names.extend([kv_options[kv_format][f"past_{k}"] % i for i in range(num_layers)])
            present_to_past.update(
                {
                    kv_options[kv_format][f"present_{k}"] % i: kv_options[kv_format][f"past_{k}"] % i
                    for i in range(num_layers)
                }
            )

        past_shape = io_config["input_shapes"][io_config["input_names"].index(past_names[0])]

        return {
            "past_names": past_names,
            "present_to_past": present_to_past,
            "num_kv_heads": past_shape[1],
            "head_size": past_shape[3],
        }

    @staticmethod
    def get_empty_kv_cache(batch_size, kv_info, has_gqa):
        # TODO(jambayk): should we have atleast seq len 1?
        # if so, would need to prepend to 2d attention mask
        return {
            k: torch.zeros(
                (batch_size, kv_info["num_kv_heads"], 0 if has_gqa else 1, kv_info["head_size"]), dtype=torch.float32
            )
            for k in kv_info["past_names"]
        }
