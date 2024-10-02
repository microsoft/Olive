# Packaging Olive artifacts

## What is Olive Packaging

Olive will output multiple candidate models based on metrics priorities. It also can package output artifacts when the user requires. Olive packaging can be used in different scenarios. There are 4 packaging types: `Zipfile`, `AzureMLModels`, `AzureMLData`, `AzureMLDeployment` and `Dockerfile`.

### Zipfile

Zipfile packaging will generate a ZIP file which includes 3 folders: `CandidateModels`, `SampleCode` and `ONNXRuntimePackages`, and a `models_rank.json` file in the `output_dir` folder (from Engine Configuration):

* `CandidateModels`: top ranked output model set
  * Model file
  * Olive Pass run history configurations for candidate model
  * Inference settings (`onnx` model only)
* `ONNXRuntimePackages`: ONNXRuntime package files with the same version that were used by Olive Engine in this workflow run.
* `models_rank.json`: A JSON file containing a list that ranks all output models based on specific metrics across all accelerators.

#### CandidateModels

`CandidateModels` includes k folders where k is the number of ranked output models, with name `BestCandidateModel_1`, `BestCandidateModel_2`, ... and `BestCandidateModel_k`. The order is ranked by metrics priorities, starting from 1. e.g., if you have 3 metrics `metric_1`, `metric_2` and `metric_3` with priority `1`, `2` and `3`. The output models will be sorted firstly by `metric_1`. If the value of `metric_1` of 2 output models are same, they will be sorted by `metric_2`, and followed by next lower priority metric.

Each `BestCandidateModel` folder will include model file/folder. The folder also includes a json file which includes the Olive Pass run history configurations since input model, a json file with performance metrics and a json file for inference settings for the candidate model if the candidate model is an ONNX model.

#### Models rank JSON file

A file that contains a JSON list for ranked model info across all accelerators, e.g.:

```json
[
    {
        "rank": 1,
        "model_config": {
            "type": "ONNXModel",
            "config": {
                "model_path": "path/model.onnx",
                "inference_settings": {
                    "execution_provider": [
                        "CPUExecutionProvider"
                    ],
                    "provider_options": [
                        {}
                    ],
                    "io_bind": false,
                    "session_options": {
                        "execution_mode": 1,
                        "graph_optimization_level": 99,
                        "inter_op_num_threads": 1,
                        "intra_op_num_threads": 14
                    }
                },
                "use_ort_extensions": false,
                "model_attributes": {"<model_attributes_key>": "<model_attributes_value>"},
            }
        },
        "metrics": {
            "accuracy-accuracy": {
                "value": 0.8602941176470589,
                "priority": 1,
                "higher_is_better": true
            },
            "latency-avg": {
                "value": 36.2313,
                "priority": 2,
                "higher_is_better": false
            },
        }
    },
    {"rank": 2, "model_config": "<model_config>", "metrics": "<metrics>"},
    {"rank": 3, "model_config": "<model_config>", "metrics": "<metrics>"}
]
```

### AzureMLModels

AzureMLModels packaging will register the output models to your Azure Machine Learning workspace. The asset name will be set as `<packaging_config_name>_<accelerator_spec>_<model_rank>`. The order is ranked by metrics priorities, starting from 1. For instance, if the output model is ONNX model and the packaging config is:

```json
{
    "type": "AzureMLModels",
    "name": "olive_output_model",
    "version": "1",
    "description": "description"
}
```

and for CPU, the best execution provider is CPUExecutionProvider, so the first ranked model name registered on AML will be `olive_output_model_cpu-cpu_1`.

Olive will also upload model configuration file, inference config file, metrics file and model info file to the Azure ML.

### AzureMLData

AzureMLData packaging will upload the output models to your Azure Machine Learning workspace as Data assets. The asset name will be set as `<packaging_config_name>_<accelerator_spec>_<model_rank>`. The order is ranked by metrics priorities, starting from 1. For instance, if the output model is ONNX model and the packaging config is:

```json
{
    "type": "AzureMLData",
    "name": "olive_output_model",
    "version": "1",
    "description": "description"
}
```

and for CPU, the best execution provider is CPUExecutionProvider, so the first ranked model Data name on AML will be `olive_output_model_cpu-cpu_1`.

Olive will also upload model configuration file, inference config file, metrics file and model info file to the Azure ML.

### AzureMLDeployment

AzureMLDeployment packaging will [package](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-package-models?view=azureml-api-2&tabs=sdk) ranked No. 1 model across all output models to Azure ML workspace, and create an endpoint for it if the endpoint doesn't exist, then deploy the output model to this endpoint.

### Dockerfile

Dockerfile packaging will generate a Dockerfile. You can simple run `docker build` for this Dockerfile to build a docker image which includes `onnxruntime` Python package and first ranked output model.

## How to package Olive artifacts

Olive packaging configuration is configured in `PackagingConfig` in Engine configuration. `PackagingConfig` can be a single packaging configuration. Alternatively, if you want to apply multiple packaging types, you can also define a list of packaging configurations.

If not specified, Olive will not package artifacts.

* `PackagingConfig`
  * `type [PackagingType]`:
    Olive packaging type. Olive will package different artifacts based on `type`.
  * `name [str]`:
    For `PackagingType.Zipfile` type, Olive will generate a ZIP file with `name` prefix: `<name>.zip`.
    For `PackagingType.AzureMLModels` and `PackagingType.AzureMLData`, Olive will use this `name` for Azure ML resource.
    The default value is `OutputModels`.
    For `PackagingType.AzureMLDeployment` and `PackagingType.Dockerfile` type, Olive will ignore this attribute.
  * `config [dict]`:
    The packaging config.
    * `Zipfile`
      * `export_in_mlflow_format [bool]`:
        Export model in mlflow format. This is `false` by default.
    * `AzureMLModels`
      * `export_in_mlflow_format [bool]`:
        Export model in mlflow format. This is `false` by default.
      * `version [int | str]`：
        The version for this model registration. This is `1` by default.
      * `description [str]`
        The description for this model registration. This is `None` by default.
    * `AzureMLData`
      * `export_in_mlflow_format [bool]`:
        Export model in mlflow format. This is `false` by default.
      * `version [int | str]`：
        The version for this data asset. This is `1` by default.
      * `description [str]`
        The description for this data asset. This is `None` by default.
    * `AzureMLDeployment`
      * `model_name [str]`:
        The model name when registering your output model to your Azure ML workspace. `olive-deployment-model` by default
      * `model_version [int | str]`:
        The model version when registering your output model to your Azure ML workspace. Please note if there is already a model with the same name and the same version in your workspace, this will override your existing registered model. `1` by default.
      * `description [str]`
        The description for this model registration. This is `None` by default.
      * `model_package [ModelPackageConfig]`:
        The configurations for model packaging.
        * `target_environment [str]`:
          The environment name for the environment created by Olive. `olive-target-environment` by default.
        * `target_environment_version [str]`
          The environment version for the environment created by Olive. Please note if there is already an environment with the same name and the same version in your workspace, your existing environment version will plus 1. This `target_environment_version` will not be applied for your environment. `None` by default.
        * `inferencing_server [InferenceServerConfig]`
          * `type [str]`
            The targeted inferencing server type. `AzureMLOnline` or `AzureMLBatch`.
          * `code_folder [str]`
            The folder path to your scoring script.
          * `scoring_script [str]`
            The scoring script name.
        * `base_environment_id [str]`
          The base environment id that will be used for Azure ML packaging. The format is `azureml:<base-environment-name>:<base-environment-version>`.
        * `environment_variables [dict]`
          Env vars that are required for the package to run, but not necessarily known at Environment creation time. `None` by default.
      * `deployment_config [DeploymentConfig]`
        The deployment configuration.
        * `endpoint_name [str]`
          The endpoint name for the deployment. If the endpoint doesn't exist, Olive will create one endpoint with this name. `olive-default-endpoint` by default.
        * `deployment_name [str]`
          The name of the deployment. `olive-default-deployment` by default.
        * `instance_type [str]`
          Azure compute sku. ManagedOnlineDeployment only. `None` by default.
        * `compute [str]`
          Compute target for batch inference operation. BatchDeployment only. `None` by default.
        * `instance_count [str]`
          Number of instances the interfering will run on. `1` by default.
        * `mini_batch_size [str]`
          Size of the mini-batch passed to each batch invocation. `10` by default.
        * `extra_config [dict]`
          Extra configurations for deployment. `None` by default.
    * `Dockerfile`
      * `requirements_file [str]`:
        `requirements.txt` file path. The packages will be installed to docker image.
  * `include_runtime_packages [bool]`:
    Whether or not to include runtime packages (like onnxruntime) in zip file. Defaults to True

You can add different types `PackagingConfig` as a list to Engine configurations. e.g.:

```json
"engine": {
    "search_strategy": {
        "execution_order": "joint",
        "search_algorithm": "tpe",
        "search_algorithm_config": {
            "num_samples": 5,
            "seed": 0
        }
    },
    "evaluator": "common_evaluator",
    "host": "local_system",
    "target": "local_system",
    "packaging_config": [
        {
            "type": "Zipfile",
            "name": "OutputModels"
        },
        {
            "type": "AzureMLModels",
            "name": "OutputModels"
        },
        {
            "type": "AzureMLData",
            "name": "OutputModels"
        },
        {
            "type": "AzureMLDeployment",
            "model_package": {
                "inferencing_server": {
                    "type": "AzureMLOnline",
                    "code_folder": "code",
                    "scoring_script": "score.py"
                },
                "base_environment_id": "azureml:olive-aml-packaging:1"
            }
        }
    ]
    "cache_dir": "cache"
}
```

## Packaged files

### Inference config file

The inference config file is a json file including `execution_provider` and `session_options`. e.g.:

```json
{
    "execution_provider": [
        [
            "CPUExecutionProvider",
            {}
        ]
    ],
    "session_options": {
        "execution_mode": 1,
        "graph_optimization_level": 99,
        "extra_session_config": null,
        "inter_op_num_threads": 1,
        "intra_op_num_threads": 64
    }
}
```

### Model configuration file

The model configuration file is a json file including the history of applied Passes history to the output model. e.g.:

```json
{
  "53fc6781998a4624b61959bb064622ce": null,
  "0_OnnxConversion-53fc6781998a4624b61959bb064622ce-7a320d6d630bced3548f242238392730": {
    //...
  },
  "1_OrtTransformersOptimization-0-c499e39e42693aaab050820afd31e0c3-cpu-cpu": {
    //...
  },
  "2_OnnxQuantization-1-1431c563dcfda9c9c3bf26c5d61ef58e": {
    //...
  },
  "3_OrtPerfTuning-2-a843d77ae4964c04e145b83567fb5b05-cpu-cpu": {
    //...
  }
}
```

### Metrics file

The metrics file is a json file including input model metrics and output model metrics.
