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

First, install required packages according to passes.
```
python -m olive.workflows.run --config user_provided_info.json --setup
```
Then, optimize the model
```
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

Here is the complete json configuration file to optimize your input model using following command

```bash
python -m olive.workflows.run --config config.json
```

```json
{
    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "model_path": "resnet.pt",
            "model_storage_kind": "file",
            "io_config": {
                "input_names": ["input"],
                "input_shapes": [[1, 3, 32, 32]],
                "output_names": ["output"],
                "dynamic_axes": {
                    "input": {"0": "batch_size"},
                    "output": {"0": "batch_size"}
                }
            }
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
        "target": {"type": "LocalSystem"}
    }
}
```
It is composed with 5 parts:
### [Input Model](../tutorials/options.md/#input-model-information)
You provide input model location and type. PyTorchModel, ONNXModel, OpenVINOModel and SNPEModel are supported model types.

### [Host and Target Systems](../tutorials/options.md/#systems-information)
An optimization technique, which we call a Pass, can be run on a variety of host systems and the resulting model evaluated on desired target systems. More details for the available systems can be found at at [OliveSystems api reference](systems).

### [Evaluator](../tutorials/options.md/#evaluators-information)
In order to chose the set of Pass configuration parameters that lead to the “best” model, Olive requires an evaluator that returns metrics values for each output model.

### [Passes](../tutorials/options.md/#passes-information)
You list the Passes that you want to apply on the input model. In this example, let us first convert the pytorch model to ONNX and quantize it.

### [Engine](../tutorials/options.md/#engine-information)
The engine is used to handle the auto-tuning process.

## Olive Optimization Result
### Olive Footprint
When the optimization process is completed, Olive will generate a report(json) under the `output_dir` if you specified already in `engine.run`. The report contains the:
- `footprints.json`: A dictionary of all the footprints generated during the optimization process. The structure of footprints value is:
```python
class FootprintNode(ConfigBase):
    # None for no parent which means current model is the input model
    parent_model_id: str = None
    model_id: str
    model_config: Dict = None
    from_pass: str = None
    pass_run_config: Dict = None
    is_pareto_frontier: bool = False
    metrics: FootprintNodeMetric = FootprintNodeMetric()
    date_time: float = datetime.now().timestamp()

class FootprintNodeMetric(ConfigBase):
    """
    value: {"metric_name": metrics_value, ...}
    cmp_direction: will be auto suggested. The format will be like: {"metric_name": 1, ...},
        1: higher is better, -1: lower is better
    is_goals_met: if the goals set by users is met
    """
    value: Dict = None
    cmp_direction: Dict = None
    is_goals_met: bool = False
```
- `pareto_frontier_footprints.json`: A dictionary of the footprints that are on the Pareto frontier based on the metrics goal you set in config of `evaluators.metrics`.

Here is an example of that:
```json
{
    "24_OrtTransformersOptimization-23-28b039f9e50b7a04f9ab69bcfe75b9a2": {
        "parent_model_id": "23_OnnxConversion-9d98a0131bcdfd593432adfa2190016b-fa609d8c8586ea9b21b129a124e3fdb0",
        "model_id": "24_OrtTransformersOptimization-23-28b039f9e50b7a04f9ab69bcfe75b9a2",
        "model_config": {
            "type": "ONNXModel",
            "config": {
                "model_path": "path",
                "name": null,
                "model_storage_kind": "file",
                "version": null,
                "inference_settings": null
            }
        },
        "from_pass": "OrtTransformersOptimization",
        "pass_run_config": {
            "model_type": "bert",
            "num_heads": 0,
            "hidden_size": 0,
            "optimization_options": null,
            "opt_level": null,
            "use_gpu": false,
            "only_onnxruntime": false,
            "float16": false,
            "input_int32": false,
            "use_external_data_format": false
        },
        "is_pareto_frontier": true,
        "metrics": {
            "value": {
                "accuracy": 0.8602941036224365,
                "latency": 87.4454
            },
            "cmp_direction": {
                "accuracy": 1,
                "latency": -1
            },
            "is_goals_met": true
        },
        "date_time": 1681211541.682187
    }
}
```

You can also call the following methods to plot the Pareto frontier footprints. Also please make sure you installed `plotly` successfully.
```python
from olive.engine.footprint import Footprint
footprint = Footprint.from_file("footprints.json")
footprint.plot_pareto_frontier()
```
### Olive Packaging
Olive also can package output artifacts when user adds `PackagingConfig` to Engine configurations. Please refer to [Packaing Olive artifacts](../tutorials/packaging_output_models.md) for more details.
