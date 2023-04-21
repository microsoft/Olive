# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from torch.utils.data import DataLoader

from olive.data_container.constants import DataComponentType, DefaultDataComponent
from olive.data_container.registry import Registry


@Registry.register(DataComponentType.DATALOADER, name=DefaultDataComponent.DATALOADER)
def default_dataloader(dataset, batch_size=1, **kwargs):
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


@Registry.register(DataComponentType.DATALOADER)
def default_calibration_dataloader(dataset, batch_size=1, **kwargs):
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationDataReader(CalibrationDataReader):
        def __init__(self, dataset, batch_size: int = 1, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size
            self.kwargs = kwargs
            self.data_loader = DataLoader(dataset, batch_size=batch_size, **kwargs)
            self.data_iter = iter(self.data_loader)

        def get_next(self):
            if self.data_iter is None:
                self.data_iter = iter(self.data_loader)
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return None
            if isinstance(batch, (list, tuple)):
                batch = [v.numpy() for v in batch]
            elif isinstance(batch, dict):
                batch = {k: v.numpy() for k, v in batch.items()}
            return batch

        def rewind(self):
            self.data_iter = None

    return _CalibrationDataReader(dataset, batch_size, **kwargs)
