(Quick-tour)=
# Quick Tour

Below is a quick guide to get the packages installed to use Olive for model optimization. We will start with a
PyTorch model and then convert and quantize it to ONNX. If you are new to Olive and model optimization, we recommend
checking the [Design](Design) and Tutorials sections for more in-depth explanations.

## Install Olive and dependencies
Before you begin, install Olive and the necessary packages.
```bash
pip install olive-ai
```

You will also need to install your preferred build of onnxruntime. Let's choose the default CPU package for this tour.
```bash
pip install onnxruntime
```

Refer to the [Installation](Installation) section for more details.

## Model Optimization Workflow
Olive model optimization workflows are defined using config JSON files. You can use the Olive CLI to run the pipeline:

```bash
python -m olive.workflows.run --config user_provided_info.json
```

or in python code:

```python
from olive.workflows import run as olive_run
olive_run("user_provided_info.json")
```

```{note}
`olive.workflows.run` in python code also accepts python dictionary equivalent of the config JSON object.
```

Now, let's take a look at the information you can provide to Olive to optimize your model.

### Input Model

You provide input model location and type. PyTorchModel, ONNXModel, OpenVINOModel and SNPEModel are supported model types.
```json
"input_model":{
    "type": "PyTorchModel",
    "config": {
        "model_path": "resnet.pt",
        "is_file": true
    }
}
```

### Host and Target Systems
An optimization technique, which we call a Pass, can be run on a variety of **host** systems and the resulting model evaluated
on desired **target** systems. More details for the available systems can be found at [OliveSystems api reference](systems).

In this guide, you will use your local system as both the hosts for passes and target for evaluation.

```json
"systems": {
    "local_system": {"type": "LocalSystem"}
}
```

### Evaluator
In order to chose the set of Pass configuration parameters that lead to the "best" model, Olive requires an evaluator that
returns metrics values for each output model.

```json
"evaluators": {
    "common_evaluator":{
        "metrics":[
            {
                "name": "latency",
                "type": "latency",
                "sub_type": "avg",
                "user_config":{
                    "user_script": "user_script.py",
                    "data_dir": "data",
                    "dataloader_func": "create_dataloader",
                    "batch_size": 16
                }
            }
        ],
        "target": "local_system"
    }
}
```

`latency_metric` requires you to provide a function as value for `dataloader_func` that returns a dataloader object when called on `data_dir` and `batch_size`. You can provide the function object directly but here, let's give it a function name `"create_dataloader"` that can be imported from `user_script`.

[This file](https://github.com/microsoft/Olive/blob/main/examples/resnet_ptq_cpu/user_script.py)
has an example of how to write user scripts.
Refer to [](user_script) for more details on user scripts.

You can provide more than one metric to the evaluator `metrics` list.

### Engine
The engine which handles the auto-tuning process. You can select search strategy here.

```json
"engine": {
    "cache_dir": ".cache"
    "search_strategy": {
        "execution_order": "joint",
        "search_algorithm": "exhaustive",
    }
}
```

### Passes
You list the Passes that you want to apply on the input model. In this example,
let us first convert the pytorch model to ONNX and quantize it.

```json
"onnx_conversion": {
    "type": "OnnxConversion",
    "config": {
        "input_names": ["input"],
        "input_shapes": [[1, 3, 32, 32]],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {"0": "batch_size"},
            "output": {"0": "batch_size"}
        },
        "target_opset": 13
    },
    "host": {"type": "LocalSystem"}
}
```

```json
"onnx_quantization": {
    "type": "OnnxDynamicQuantization",
    "config": {
        "user_script": "user_script.py",
        "data_dir": "data",
        "dataloader_func": "resnet_calibration_reader",
        "weight_type" : "QUInt8"
    }
}
```

### Example JSON
Here is the complete json configuration file as we discussed above which you use to optimizer your input model using following command

```bash
python -m olive.workflows.run --config config.json
```

```json
{
    "verbose": true,
    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "model_path": "resnet.pt",
            "is_file": true
        }
    },
    "systems": {
        "local_system": {"type": "LocalSystem"}
    },
    "evaluators": {
        "common_evaluator":{
            "metrics":[
                {
                    "name": "latency",
                    "type": "latency",
                    "sub_type": "avg",
                    "user_config":{
                        "user_script": "user_script.py",
                        "data_dir": "data",
                        "dataloader_func": "create_dataloader",
                        "batch_size": 16
                    }
                }
            ],
            "target": "local_system"
        }
    },
    "passes": {
        "onnx_conversion": {
            "type": "OnnxConversion",
            "config": {
                "input_names": ["input"],
                "input_shapes": [[1, 3, 32, 32]],
                "output_names": ["output"],
                "dynamic_axes": {
                    "input": {"0": "batch_size"},
                    "output": {"0": "batch_size"}
                },
                "target_opset": 13
            },
            "host": {"type": "LocalSystem"}
        },
        "onnx_quantization": {
            "type": "OnnxDynamicQuantization",
            "config": {
                "user_script": "user_script.py",
                "data_dir": "data",
                "dataloader_func": "resnet_calibration_reader",
                "weight_type" : "QUInt8"
            }
        }
    },
    "engine": {
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "exhaustive"
        },
        "evaluator": "common_evaluator",
        "host": {"type": "LocalSystem"},
    }
}
```

### Olive Footprint
<!-- TODO -->
