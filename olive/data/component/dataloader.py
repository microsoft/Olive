# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import List, Union

from olive.data.registry import Registry


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
def default_calibration_dataloader(dataloader, **kwargs):
    # TODO(trajep): consider other quantization calibration interface in SNPE/INC/OpenVINO/Torch and etc.
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationDataReader(CalibrationDataReader):
        def __init__(self, dataloader, **kwargs):
            self.dataloader = dataloader
            self.kwargs = kwargs
            self.data_iter = iter(self.dataloader)

        def get_next(self):
            if self.data_iter is None:
                self.data_iter = iter(self.dataloader)
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return None
            if isinstance(batch, list):
                batch = batch[0]
            if isinstance(batch, dict):
                batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
            return batch

        def rewind(self):
            self.data_iter = None

    return _CalibrationDataReader(dataloader, **kwargs)
