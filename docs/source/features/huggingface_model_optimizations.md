# Huggingface Model Optimization


## Introduction
This document describes how to configure a  workflow to optimize Huggingface models using Olive. The user can simply specify the model name, task, dataset and metric to optimize a model.
1. Model name: is in Huggingface model hub, e.g. `bert-base-uncased`.
2. Task name: introduces the task specific head for the model, e.g. `text-classification`. More task names can be found [here](https://huggingface.co/tasks). Task name is used to:
    - Load model in which user can just provide the model name and task name. Olive will automatically load the model from Huggingface model hub for specific task.
    - Load specific tokenizer and data processor. Currently, we simplify the user experience only for `text-classification` task, which will be demonstrated in the following examples. *For other tasks, it is under development actively.*
3. The task specific dataset: is hosted in [Huggingface datasets](https://huggingface.co/datasets), e.g. `glue` dataset for text classification task.
4. Metric name: is supported by [Huggingface evaluate](https://huggingface.co/docs/evaluate/index). User can refer to [huggingface metrics](https://huggingface.co/metrics) for full metrics list.

## Example

### Model config and dataset config
Take `bert-base-uncased` as an example, user can specify task name as `text-classification` and dataset as `glue` to form the Huggingface config `hf_config` as follows:
```json
"hf_config": {
    "model_name": "bert-base-uncased",
    "task": "text-classification",
    "dataset": {
        "data_name":"glue",
        "subset": "mrpc",
        "split": "validation",
        "input_cols": ["sentence1", "sentence2"],
        "label_cols": ["label"],
        "batch_size": 1
    }
}
```
Please refer to [hf_config](../overview/options.md#hf_config) for more details.

### Metric config
```json
{
    "name": "accuracy",
    "type": "accuracy",
    "backend": "huggingface_metrics",
    "sub_types": [
        {"name": "accuracy", "priority": -1},
        {"name": "f1"}
    ]
}
```
Please refer to [metrics](../overview/options.md#metrics) for more details.

### Custom components config
You can use your own custom components functions for your model. You will need to define the details of your components in your script as functions.
```json
{
    "input_model": {
        "type": "PyTorchModel",
        "config": {
            "model_script": "code/user_script.py",
            "script_dir": "code",
            "hf_config": {
                "model_class": "WhisperForConditionalGeneration",
                "model_name": "openai/whisper-medium",
                "components": [
                    {
                        "name": "encoder_decoder_init",
                        "io_config": "get_encdec_io_config",
                        "component_func": "get_encoder_decoder_init",
                        "dummy_inputs_func": "encoder_decoder_init_dummy_inputs"
                    },
                    {
                        "name": "decoder",
                        "io_config": "get_dec_io_config",
                        "component_func": "get_decoder",
                        "dummy_inputs_func": "decoder_dummy_inputs"
                    }
                ]
            }
        }
    },
}
```
#### Script example
```
# my_script.py
def get_dec_io_config(model_name: str):
    # return your io dict
    ...

def get_decoder(model_name: str):
    # your component implementation
    ...

def dummy_inputs_func():
    # return the dummy input for your component
    ...
```

### E2E example
For the complete example, please refer to [Bert Optimization with PTQ on CPU](https://github.com/microsoft/Olive/tree/main/examples/bert#bert-optimization-with-ptq-on-cpu).
