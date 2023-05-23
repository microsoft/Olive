(Quick-tour)=
# Quick Tour

Here is a quick guide on using Olive for model optimization with two steps. We will focus on accelerating a PyTorch model on the CPU leveraging ONNX conversion and ONNX quantization.

## Step1 Install Olive
Before you begin, install Olive and the necessary packages.
```bash
pip install olive-ai
```
Refer to the [Installation](Installation) section for more details.

## Step 2 Run Olive with Model Optimization Workflow
Olive model optimization workflows are defined using config JSON files. You can use the Olive CLI to run the pipeline.

First, install required packages according to your config file.
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

Now, let's take a look at the json configuration file you need to provide to optimize your model.

```json
{
    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "model_path": "resnet.pt",
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
        "local_system": {
            "type": "LocalSystem",
            "config": {
                "device": "cpu"
            }
        }
    },

    "evaluators": {
        "common_evaluator":{
            "metrics":[
                {
                    "name": "latency",
                    "type": "latency",
                    "sub_types": [
                        {"name": "avg", "goal": {"type": "percent-min-improvement", "value": 20}},
                        {"name": "max"},
                        {"name": "min"}
                    ]
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
        "quantization": {
            "type": "OnnxDynamicQuantization"
        },
    },

    "engine": {
        "evaluator": "common_evaluator",
        "packaging_config": {}
    }
}
```
It is composed with 5 parts:
### [Input Model](../overview/options.md/#input-model-information)
You provide input model location and type. PyTorchModel, ONNXModel, OpenVINOModel and SNPEModel are supported model types.

### [Systems](../overview/options.md/#systems-information)(Optional)
You can define hardware target in this section for both Olive host system and target system used in the engine section below. Olive host system is for running optimizations and target system for evaluating the optimized model. The default value is local system with CPU. More details for the available systems can be found at [OliveSystems api reference](systems).

### [Evaluator](../overview/options.md/#evaluators-information)
You specify your performance requirements in evaluator, such as accuracy and latency, which the optimized candidate models should meet. Olive utilizes the information to tune the optimal set of optimization parameters for the "best" model.

### [Passes](../overview/options.md/#passes-information)
An optimization technique is called as a Pass in Olive. You list optimizations that you want to apply on the input model. In this example, we first convert the pytorch model to ONNX then quantize it.

### [Engine](../overview/options.md/#engine-information)
The engine tunes optimization passes to produces optimized model(s) based on evaluation criteria. It has default auto-tuning algorithm, also allow you to set your own. You can also define host system and target system in this section if you have to run the optimizations and evaluate the model on different hardware target. Please check [Engine api reference](engine) for more details.

Note: In addition to these five core sectors, Olive provides a rich selection of optional configurations to suit diverse scenarios. For detailed information on these options, please refer to the [options.md](../overview/options.md/) file.

## Olive Optimization Result
### Olive Footprint
When the optimization process is completed, Olive will generate a report(json) under the `output_dir`. The report contains the:
- `footprints.json`: A dictionary of all the footprints generated during the optimization process.
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
Olive also can package output artifacts when user adds `PackagingConfig` to Engine configurations.
```
"engine": {
    ...
    "packaging_config": {
        "type": "Zipfile",
        "name": "OutputModels"
    },
    ...
}
```
Olive packaging will generate a ZIP file which includes 3 folders: `CandidateModels`, `SampleCode` and `ONNXRuntimePackages`:
* `CandidateModels`: top ranked output model set
    * Model file
    * Olive Pass run history configurations for candidate model
    * Inference settings (`onnx` model only)
* `SampleCode`: code sample for ONNX model
    * C++
    * C#
    * Python
* `ONNXRuntimePackages`: ONNXRuntime package files with the same version that were used by Olive Engine in this workflow run.

Please refer to [Packaing Olive artifacts](../tutorials/packaging_output_models.md) for more details.

## Azure ML Client

If you will use Azure ML resources and assets, you need to provide your Azure ML client configurations. For example:
* You have AzureML system for targets or hosts.
* You have Azure ML model as input model.

AzureML authentication credentials is needed. Refer to
[this](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?tabs=sdk)  for
more details.

You can either add configurations to the Olive json config file:
```json
"azureml_client": {
    "subscription_id": "<subscription_id>",
    "resource_group": "<resource_group>",
    "workspace_name": "<workspace_name>"
},
```
or you can also have your config file in a seprate json file in the following format:
```json
{
    "subscription_id": "<subscription_id>",
    "resource_group": "<resource_group>",
    "workspace_name": "<workspace_name>"
}
```
and specify your config file path to `azureml_client`:
```json
"azureml_client": {
    "aml_config_path": "<path to your config file>"
},
```

For more detailed information about Olive, please refer to the [Design](Design) and Tutorials sections, where you can find in-depth explanations.
