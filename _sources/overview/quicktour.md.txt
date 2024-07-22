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
```bash
olive run --config my_model_acceleration_description.json
```
**Note:** If `olive` cannot be found in your path, you can use `python -m olive` instead.

or in python code:
```python
from olive.workflows import run as olive_run
olive_run("my_model_acceleration_description.json")
```

```{note}
`olive.workflows.run` in python code also accepts python dictionary equivalent of the config JSON object.

You can use setup mode `olive run --config my_model_acceleration_description.json --setup` to identify list of additional packages you may need to install for your workflow.

To include user implemented (or proprietary, or private) passes as part of workflow, clone olive_config.json and update it.
Provide the path to the cloned _olive_config.json_ file at launch using the '--package-config' command line option.

You can also change the default directory for temporary files and directories using `--tempdir` option.
Set this to a local directory if you want to avoid using the default tempdir for reasons such as disk space and permissions.

If you want to use different device ids specially for cuda device, please set `CUDA_VISIBLE_DEVICES` to the desired device ids, like:

    # linux
    CUDA_VISIBLE_DEVICES=2,3 olive run --config my_model_acceleration_description.json

    # windows
    set CUDA_VISIBLE_DEVICES=2,3 & olive run --config my_model_acceleration_description.json

    # python
    import os
    from olive.workflows import run as olive_run
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    olive_run("my_model_acceleration_description.json")

```

## Information needed to accelerate a model

Typically, you need to have input model information such as model type, input names and shapes, where the model is stored. You would also know your desired performance requirements in terms of Latency, Accuracy etc. Along with this information you need to provide Olive list of model transformations and optimizations you want to apply. Optionally you can also select target hardware and select additional Olive features. Now, let's take a look at how you can include this information in the json configuration file you will use as an Olive input.

### Input Model

Olive can accept ONNX, Torch, OpenVINO and SNPE models as of now. The config for each of these models is slightly different. You can find more information about each of these models in [Input Model configuration](https://microsoft.github.io/Olive/api/models.html).

Let's use a PyTorch resnet model as an example which you can describe in the json file as follows. You can use any PyTorch model.

```json
    "input_model":{
        "type": "PyTorchModel",
        "model_path": "resnet.pt",
        "io_config": {
            "input_names": ["input"],
            "input_shapes": [[1, 3, 32, 32]],
            "output_names": ["output"],
        }
    }
```

It is possible to provide additional information such as dataset you want to use. You could also directly select HuggingFace model and task. See [Input Model configuration](../overview/options.md/#input-model-information) for more information.

### Passes to apply

Olive can apply various transformations and optimizations, also known as passes, on the input model to produce the accelerated output model. See [Passes](../overview/options.md/#passes-information) for the full list of passes supported by Olive.

Let's apply ONNX conversation and use dynamic quantization technique to quantize the model by specifying following in the json file.

```json
    "passes": {
        "onnx_conversion": {
            "type": "OnnxConversion",
            "target_opset": 13
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
        "log_severity_level": 0
    }
```

Now you have a complete json file that you can use to accelerate the resnet model. For more detailed information about all the features supported by Olive, please refer to the [Olive Options](../overview/options.md) and Tutorials.

------

```json
{
    "description" : "Complete my_model_acceleration_description.json used in this quick tour",
    "input_model":{
        "type": "PyTorchModel",
        "model_path": "resnet.pt",
        "io_config": {
            "input_names": ["input"],
            "input_shapes": [[1, 3, 32, 32]],
            "output_names": ["output"],
        }
    },
    "passes": {
        "onnx_conversion": {
            "type": "OnnxConversion",
            "target_opset": 13
        },
        "quantization": {
            "type": "OnnxDynamicQuantization"
        },
    },
    "engine": {
        "log_severity_level": 0
    }
}
```
