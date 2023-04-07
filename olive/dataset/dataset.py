from abc import ABC, abstractmethod

from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset


class OliveInputDataset(Dataset):
    """
    Load the dataset with the format of [data, label, meta_data(optional)]
    """

    def __init__(self, input):
        self.input = input

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        if len(self.input[-1]) == 2:
            return self.input[idx][0], self.input[idx][1]
        else:
            return self.input[idx][0], self.input[idx][1], self.input[idx][2]


class OliveCalibrationDataset(CalibrationDataReader):
    def __init__(self, iter_data, post_func=lambda x: x[0]):
        self.iter_data = iter_data
        self.iter = iter(self.iter_data)
        self.post_func = post_func

    def get_next(self):
        if self.iter is None:
            self.iter = iter(self.iter_data)
        try:
            return self.post_func(next(self.iter))
        except StopIteration:
            return None

    def rewind(self):
        self.iter = None


class DatasetPipeline(ABC):
    def __init__(
        self,
        batch_size: int = 1,
        inputs: list = None,
        **kwargs,
    ):
        """
        batch_size: batch size
        inputs: user directly give the input samplers with the format of ['data', 'label', 'meta_data'(optional)]
        kwargs: other parameters
        """
        self.batch_size = batch_size
        self.inputs = inputs
        self.kwargs = kwargs

        if self.inputs is not None:
            self.dataset = OliveInputDataset()
        else:
            self.dataset = self.load_dataset(kwargs["load_dataset"]).pre_process()

    @abstractmethod
    def load_dataset(self, **kwargs):
        pass

    @abstractmethod
    def pre_process(self):
        raise NotImplementedError

    @abstractmethod
    def post_process(self):
        raise NotImplementedError

    @abstractmethod
    def calibration_post_func(self):
        raise NotImplementedError

    def dataloader(self, **kwargs):
        assert self.dataset is not None and len(self.dataset) > 0, "Dataset is empty"
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            **kwargs,
        )

    def calibration_dataset(self):
        assert self.dataset is not None and len(self.dataset) > 0, "Dataset is empty"
        return OliveCalibrationDataset(self.dataset, self.calibration_post_func)
