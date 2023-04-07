Olive Dataset Design
====================

### Overview

Currently, we have a few different dataset implementations in Olive. We should unify them under standardized Olive.Dataset/Olive.Dataloader interface which will be introduced in this design document.

### User scenarios

There are two main user scenarios for Olive dataset:
- Evaluate the intermediate models on the dataset.
    In this scenario, user will provide the function to load dataset and dataloader in user script which is passed into olive_pass with config string like following:
    ```json
    "evaluators": {
        "common_evaluator": {
            "metrics":[
                {
                    "name": "accuracy",
                    "type": "accuracy",
                    "sub_type": "accuracy_score",
                    "user_config":{
                        "post_processing_func": "post_process",
                        "user_script": "user_script.py",
                        "dataloader_func": "create_dataloader",
                        "batch_size": 1
                    }
                },
                {
                    "name": "latency",
                    "type": "latency",
                    "sub_type": "avg",
                    "user_config":{
                        "user_script": "user_script.py",
                        "dataloader_func": "create_dataloader",
                        "batch_size": 1
                    }
                }
            ],
            "target": "local_system"
        }
    },
    ```

    When the control flow reaches the evaluator, Olive will **load the user script** and **call the dataloader function** to get the dataloader.

    Then Olive will **iterate the dataloader** and call the model to get the prediction. Finally, Olive will call the post **processing function** to get the final prediction and calculate the metrics.

- Run Olive passes:
    - Conversion: requires user to provide the input samplers or input name and shape for Olive to generate the faked data.
    ```json
    "conversion": {
        "type": "OnnxConversion",
        "config": {
            "input_names": ["input_ids", "attention_mask", "token_type_ids"],
            "input_shapes": [[1, 128], [1, 128], [1, 128]],
            "input_types": ["int64", "int64", "int64"],
            "output_names": ["output"],
            "target_opset": 13
        }
    }
    ```
    - PerfTuning: same with Conversion.
    ```json
    "perf_tuning": {
        "type": "OrtPerfTuning",
        "config": {
            "user_script": "user_script.py",
            "dataloader_func": "create_dataloader",
            "batch_size": 1
        }
    }
    ```
    - Quantization: requires a set of calibration dataset, for different pass, there might be different calibration datasets.
    ```json
    "quantization": {
        "type": "OnnxQuantization",
        "config": {
            "user_script": "user_script.py",
            "dataloader_func": "glue_calibration_reader"
        }
    },
    ```
    - QAT: requires a whole dataset with annotations which can hold the same format with evaluation dataset.
    ```json
    "quantization_aware_training":{
        "type": "QuantizationAwareTraining",
        "config":{
            "input_shapes": [[1, 128], [1, 128], [1, 128]],
            "input_types": ["int64", "int64", "int64"],
            "user_script": "user_script.py",
            "training_loop_func": "training_loop_func"
        }
    },
    ```
    - Distillation: same with QAT

In above cases, user must to provide complete Dataset, Dataloader and Calibration Dataset.
We should provide a unified interface for Olive to load the dataset and try best to simplify the user experience.

We will tried to optimize Olive dataset related api from following aspects:
- [ ] **Unify dataset interface**: Provide a complete Olive Dataset interface which can be used in all above scenarios. The interface should be compatible with
    1. Torch dataset
    2. Huggingface dataset
    3. SNAP dataset (olive/snap/data_loader.py)
    4. OpenVINO dataset
    5. To add more...

- [ ] **Generate dummy input** For those pass or metric evaluation like perf-tuning and latency measurement
which can just take simple dummy tensors input or data shape/type as input,
we should provide a simple interface to generate the dummy input.
    **Requirement from AML integration, to make Olive v2 support dummy input for Perf-tuning pass*

- [ ] **Implement Dataset examples** Provide a series of Olive Dataset examples on popular datasets.
    - [ ] GLUE
    - [ ] SQUAD
    - [ ] MNIST
    - [ ] CIFAR10
    - [ ] To add more...
    The examples can be used as dataset input or as reference for user to implement their own dataset.

- [ ] **TBD user_config parameters sharing**: We noticed the different passes and evaluations will
try to load data respectively. That is helpful when the host and target system are not the same one.
As different pass run and evaluation are hold on different devices.
But when it comes to local run, the same dataset are load repeatedly.
We should try to optimize this with low priority?


### Design

#### Unify dataset interface
Before the introduction on Olive `DatasetPipeline` interface, we will first give the general workflow for
dataset load and processing in machine learning.
![figure](../images/dataset-flow.png)

Then based on this figure, we expect:

1. User to prepare the dataset with given format before call Olive: `[data, annotation(optional), meta_data(optional)]`.
    - `data`: the input data for model to process, it can be a single tensor/text/files... or a list of tensors/text/files.
    - `annotation`: the label for the data, it can be a single tensor/text/files or a list of tensors/text/files.
    - `meta_data`: the meta data for the data, some information to tell the structure/parse way... of current data
2. After the data preparation, user can implement their DataPipeline(interface is shown below) to load the dataset and get the dataloader.
The steps would be like:
    - User to inherit the DataPipeline class to implement data pipeline:
        - `load_dataset` to load the dataset.
        - `pre_process` to format the dataset and make it ready for model to consume. Could be optional if the dataset is ready for model inference.
        - `post_precess` to format the model output to the format which can be used for evaluation. Could be optional.
        - `calibration_post_func` to format the dataset and make it ready for quantization. Could be optional if there is no quantization.
    After these, the DataPipeline will generate the dataloader with given batch_size/sample_ratio which can be used for model inference. User can also overwrite it with their own version.

3. After the dataloader is ready, user can call Olive to run the pass or evaluation.

Besides that, Olive can also follow the `DataPipeline` to implement the interface for huggingface dataset, might be called `HuggingfaceDataPipeline`.

```python
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod
from onnxruntime.quantization import CalibrationDataReader


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
        raise NotImplementedError

    def pre_process(self):
        pass

    def post_process(self):
        pass

    def calibration_post_func(self):
        pass

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
```


#### Generate dummy input
For those pass or metric evaluation like perf-tuning where the dummy data is OK, Olive can generate the dummy input with settings like:
1. input_name: `[intput1, input2, input3, ...]`
2. dtype: `[int8, float16, float32, ...]`
3. shape, `[[1, 128], [1, 128], [1, 256], ...]`

This improvement would be like some pass-oriented API/config to generate the dummy input for the model.
For auto optimization which does not require user to provide passes, we may need to extract to common config with appropriate default key/value.

#### Implement Dataset examples
When dataset interface is unified, we can tried to conduct more examples.

### Reference
- [Torch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#datasets-dataloaders)
- [HuggingFace Dataset](https://huggingface.co/docs/datasets/index)
- [Arch Dataset: Fast HF Dataset](https://microsoft.github.io/archai/getting_started/notebooks/nlp/fast_hf_dataset_provider.html#)
- [OpenVINO dataloader](https://docs.openvino.ai/latest/pot_compression_api_README.html#dataloader)
