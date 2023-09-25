# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from torch.utils.data import DataLoader

from olive.data.registry import Registry


@Registry.register_dataloader()
def skip_dataloader(_dataset):
    return _dataset


@Registry.register_default_dataloader()
def default_dataloader(_dataset, batch_size=1, **kwargs):
    return DataLoader(_dataset, batch_size=batch_size, **kwargs)


@Registry.register_dataloader()
def no_auto_batch_dataloader(_dataset, **kwargs):
    # torch dataloader will automatically batch if batch_size is not None
    # this dataloader will not batch. Assumes that the dataset already returns a batch
    return DataLoader(_dataset, batch_size=None, **kwargs)


@Registry.register_dataloader()
def default_calibration_dataloader(_dataloader, **kwargs):
    # TODO(trajep): consider other quantization calibration interface in SNPE/INC/OpenVINO/Torch and etc.
    from onnxruntime.quantization import CalibrationDataReader

    dataloader = _dataloader

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
