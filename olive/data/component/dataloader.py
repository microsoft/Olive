# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from typing import Dict, List, Optional, Union

import torch

from olive.common.hf.model_io import get_kv_info
from olive.common.utils import format_data
from olive.data.registry import Registry
from olive.logging import get_verbosity

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


@Registry.register_dataloader()
def default_calibration_dataloader(
    dataloader,
    model_path: Optional[str] = None,
    io_config: Optional[Dict] = None,
    **kwargs,
):
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationDataReader(CalibrationDataReader):
        # pylint: disable=W0223
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


# TODO(jambayk): generalize this and make this available for dataloaders
class LLMAugmentedDataLoader:
    def __init__(self, dataloader, model_path: str, io_config: Dict):
        import onnx

        self.dataloader = dataloader
        self.model_path = model_path
        self.io_config = io_config

        self.position_ids = "position_ids" in self.io_config["input_names"]
        self.past_seq_len = "past_seq_len" in self.io_config["input_names"]
        self.kv_info = get_kv_info(self.io_config)
        self.has_gqa = False
        for node in onnx.load(self.model_path, load_external_data=False).graph.node:
            if node.op_type == "GroupQueryAttention":
                self.has_gqa = True
                logger.debug("Model has GroupQueryAttention: %s", self.has_gqa)
                break

    def __len__(self):
        try:
            return len(self.dataloader)
        except (TypeError, NotImplementedError):
            # len() not implemented for dataloader
            return 0

    def __iter__(self):
        # progress bar
        progress_bar = None
        if get_verbosity() == logging.DEBUG and len(self) > 0:
            from tqdm import tqdm

            progress_bar = tqdm(total=len(self), desc="LLMAugmentedDataLoader")

        for data in self.dataloader:
            if isinstance(data, (list, tuple)):
                batch, label = data
            else:
                batch = data
                label = None
            assert isinstance(batch, dict), "Only support dict type batch data"

            if self.kv_info and self.kv_info["past_names"][0] not in batch:
                attention_mask = batch["attention_mask"]
                # prepend mask for past keys and values
                if not self.has_gqa:
                    # GQA: no past. Quantizer needs to exclude GQA from calibration
                    # No GQA: one unattended past token so that the calibrator data collection works
                    attention_mask = torch.cat([torch.zeros_like(attention_mask[:, :1]), attention_mask], dim=-1)

                batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": attention_mask,
                    **self.get_empty_kv_cache(batch["input_ids"].shape[0], self.kv_info, self.has_gqa),
                }

            yield batch, label
            if progress_bar:
                progress_bar.update(1)

    def add_extra_inputs(self, batch: Dict[str, torch.Tensor]):
        """Add extra inputs to the batch.

        :param batch: A dictionary containing the batch data.
        :return: The updated batch with extra inputs added.
        """
        if "attention_mask" not in batch:
            # no attention mask
            return

        attention_mask = batch["attention_mask"]
        if self.position_ids and "position_ids" not in batch:
            # position ids: current tokens
            # attention mask: past + current tokens
            batch["position_ids"] = (attention_mask.cumsum(-1) - 1)[:, -batch["input_ids"].shape[-1] :]
        if self.past_seq_len and "past_seq_len" not in batch:
            # past seq len: past tokens
            # attention mask: past + current tokens
            batch["past_seq_len"] = attention_mask.sum(-1, keepdim=True) - 1
            batch["total_seq_len"] = torch.tensor(attention_mask.shape[-1])
            del batch["attention_mask"]

    @staticmethod
    def get_empty_kv_cache(batch_size: int, kv_info: Dict, has_gqa: bool):
        """Return an empty cache for past keys and values.

        :param batch_size: The batch size.
        :param kv_info: A dictionary containing information about the keys and values.
        :param has_gqa: A boolean indicating whether the model has GQA.
        :return: A dictionary with empty tensors for past keys and values.
            If the model has GQA, the past keys and values will have a length of 0, else they will have a length of 1.
        """
        return {
            k: torch.zeros(
                (batch_size, kv_info["num_kv_heads"], 0 if has_gqa else 1, kv_info["head_size"]), dtype=torch.float32
            )
            for k in kv_info["past_names"]
        }
