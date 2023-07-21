# Quick Tour

Here is a quick guide on using Olive for model optimization. We will focus on accelerating a PyTorch model on the CPU. It is a simple three step process.

1. **Install Olive and necessary packages.**

You can install olive using pip install.

```bash
pip install olive-ai
```

2. **Describe your model and your needs in a json configuration file. This will be the input to the Olive.**

Olive needs information about your model. For example, how to load the model, name and shape of input tensors. You can also select the target hardware and list of optimizations you want to perform on the model. You can provide this information in a json file as an input to the Olive.

3. **Accelerate the model using Olive.**

The last step is the simplest one. You just need to run following simple command.
```
python -m olive.workflows.run --config my_model_acceleration_description.json
```
or in python code:
```python
from olive.workflows import run as olive_run
olive_run("my_model_acceleration_description.json")
```

```{note}
`olive.workflows.run` in python code also accepts python dictionary equivalent of the config JSON object.

You can use setup mode `python -m olive.workflows.run --config my_model_acceleration_description.json --setup` to identify list of additional packages you may need to install for your workflow.
```

## Information needed to accelerate a model

Typically, you need to have input model information such as model type, input names and shapes, where the model is stored. You would also know your desired performance requirements in terms of Latency, Accuracy etc. Along with this information you need to provide Olive list of model transformations and optimizations you want to apply. Optionally you can also select target hardware and select additional Olive features. Now, let's take a look at how you can include this information in the json configuration file you will use as an Olive input.

### Input Model

Let's use a PyTorch resnet model as an example which you can describe in the json file as follows. You can use any PyTorch model.

```json
    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "model_path": "resnet.pt",
            "io_config": {
                "input_names": ["input"],
                "input_shapes": [[1, 3, 32, 32]],
                "output_names": ["output"],
            }
        }
    }
```

It is possible to provide additional information such as dataset you want to use. You could also directly select HuggingFace model and task. See [Input Model configuration](../overview/options.md/#input-model-information) for more information.

### Performance Requirement

Let's assume you want to optimize for latency and provide the following information to the Olive evaluator, which is responsible to measure the performance metric.

```json
    "evaluators": {
        "my_evaluator":{
            "metrics":[
                {
                    "name": "my_latency_metric",
                    "type": "latency",
                    "sub_types": [{"name": "avg"}]
                }
            ]
        }
    },
```

You can also specify accuracy requirements. See [Evaluator](../overview/options.md/#evaluators-information) for more information.

### Passes to apply

Olive can apply various transformations and optimizations, also known as passes, on the input model to produce the accelerated output model. See [Passes](../overview/options.md/#passes-information) for the full list of passes supported by Olive.

Let's apply ONNX conversation and use dynamic quantization technique to quantize the model by specifying following in the json file.

```json
    "passes": {
        "onnx_conversion": {
            "type": "OnnxConversion",
            "config": {
                "target_opset": 13
            }
        },
        "quantization": {
            "type": "OnnxDynamicQuantization"
        },
    }
```

### Olive Engine

Finally, you can select Olive features such as which performance metrics you want to use in this run, verbosity level etc.

```json
    "engine": {
        "log_severity_level": 0,
        "evaluator": "my_evaluator"
    }
```

Now you have a complete json file that you can use to accelerate the resnet model. For more detailed information about all the features supported by Olive, please refer to the [Olive Options](../overview/options.md) and Tutorials.

------

```json
{
    "description" : "Complete my_model_acceleration_description.json used in this quick tour",
    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "model_path": "resnet.pt",
            "io_config": {
                "input_names": ["input"],
                "input_shapes": [[1, 3, 32, 32]],
                "output_names": ["output"],
            }
        }
    },
    "evaluators": {
        "my_evaluator":{
            "metrics":[
                {
                    "name": "my_latency_metric",
                    "type": "latency",
                    "sub_types": [{"name": "avg"}]
                }
            ]
        }
    },
    "passes": {
        "onnx_conversion": {
            "type": "OnnxConversion",
            "config": {
                "target_opset": 13
            }
        },
        "quantization": {
            "type": "OnnxDynamicQuantization"
        },
    },
    "engine": {
        "log_severity_level": 0,
        "evaluator": "common_evaluator"
    }
}
```
