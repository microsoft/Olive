# Advanced User Tour

Olive provides simple  Python and command line interface to optimize the input model. See [Quick Tour](../overview/quicktour.md) for more information.
```bash
olive run --config user_provided_info.json
```

```python
from olive.workflows import run as olive_run
olive_run(user_provided_info_json_file)
```

Olive provides Python interface to advanced user to instantiate, register and run individual optimization techniques. These
approach may not take advantage of all the features supported by standard Olive interface.

Now, let's take a look at how you can use advance Python interface.

## Input Model
Start by creating an instance of an OliveModelHandler to represent the model to be optimized. Depending on the model framework, the
model can be loaded from file or using a model loader function. For a complete of available models and their initialization options, refer to [OliveModels api reference](models).

```python
from olive.models import Modelconfig

config = {
    "type": "PyTorchModel",
    "model_path": "resnet.pt",
    "io_config": {
        "input_names": ["input"],
        "input_shapes": [[1, 3, 32, 32]],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    }
}
input_model = ModelConfig.parse_obj(config)
```

## Host and Target Systems
An optimization technique, which we call a Pass, can be run on a variety of **host** systems and the resulting model evaluated
on desired **target** systems. More details for the available systems can be found at [OliveSystems api reference](systems).

In this guide, you will use your local system as both the hosts for passes and target for evaluation.

```python
from olive.systems.local import LocalSystem

local_system = LocalSystem()
```

## Evaluator
In order to chose the set of Pass configuration parameters that lead to the "best" model, Olive requires an evaluator that
returns metrics values for each output model.

```python
from olive.evaluator.metric import LatencySubType, Metric, MetricType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig

# create latency metric instance
latency_metric = Metric(
    name="latency",
    type=MetricType.LATENCY,
        sub_types=[{
        "name": LatencySubType.AVG,
        "priority": 1,
        "metric_config": {"warmup_num": 0, "repeat_test_num": 5, "sleep_num": 2},
    }],
    user_config={
        "user_script": "user_script.py",
        "data_dir": "data",
        "dataloader_func": "create_dataloader",
        "batch_size": 16,
    }
)

# create evaluator configuration
evaluator_config =  OliveEvaluatorConfig(metrics=[latency_metric])
```

`latency_metric` requires you to provide a function as value for `dataloader_func` that returns a dataloader object when called on `data_dir`, `batch_size`, optional positional argument list and keyword argument dictionary. You can provide the function object directly but here, let's give it a function name `"create_dataloader"` that can be imported from `user_script`.

[This file](https://github.com/microsoft/Olive/blob/main/examples/resnet/user_script.py) for
has an example of how to write user scripts.
<!-- Refer to [User Scripts and Script Dir]() for more details on how Olive handles user scripts. -->

You can provide more than one metric to the evaluator `metrics` list.

## Engine
You are now ready create the engine which handles the auto-tuning process.

```python
from olive.engine import Engine

# configuration options for engine
engine_config = {
    "cache_dir": ".cache"
    "search_strategy": {
        "execution_order": "joint",
        "search_algorithm": "exhaustive",
    }
}

engine = Engine(**engine_config, evaluator=evaluator_config)
```

## Register Passes
The engine has now been created. You need to register the Passes that you want to apply on the input model. In this example,
let us first convert the pytorch model to ONNX and quantize it. More information about the
Passes available in Olive can be found at ...

```python
from olive.passes import OnnxConversion, OnnxQuantization

# Onnx conversion pass
onnx_conversion_config = {
    "target_opset": 13,
}
# override the default host with pass specific host
engine.register(OnnxConversion, onnx_conversion_config, False, host=LocalSystem())

# onnx quantization pass
quantization_config = {
    "user_script": "user_script.py",
    "data_dir": "data",
    "dataloader_func": "resnet_calibration_reader",
    "weight_type" : "QUInt8"
}
# search over the values for the other config parameters
engine.register(OnnxQuantization, quantization_config, False)
```

## Run the engine
Finally, run the engine on your input model. The output will be the best set of parameters for the passes and the output
model. Note: the engine run result will be updated soon.

```python
best_execution = engine.run(input_model, [DEFAULT_CPU_ACCELERATOR])
```
