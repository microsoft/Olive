# Olive Options

Olive enables users to easily compose and customize their own model optimization pipelines. Olive provides a set of passes that can be
used to compose a pipeline. Olive receives input model, target hardware, performance requirements, and list of optimizations techniques
to apply from user in the form of a json dictionary. In this document, we document the options user can set in this dictionary.

The options are organized into following sections:

- [Azure ML client](#azure-ml-client) `azureml_client`
- [Input Model Information](#input-model-information) `input_model`
- [Systems Information](#systems-information) `systems`
- [Evaluators Information](#evaluators-information) `evaluators`
- [Passes Information](#passes-information) `passes`
- [Engine Information](#engine-information) `engine`

## Azure ML Client

If you will use Azure ML resources and assets, you need to provide your Azure ML client configurations. For example:
* You have AzureML system for targets or hosts.
* You have Azure ML model as input model.

AzureML authentication credentials is needed. Refer to
[this](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?tabs=sdk)  for
more details.

`azureml_client: [Dict]`
- `subscription_id: [str]` Azure account subscription id.
- `resource_group: [str]` Azure account resource group name.
- `workspace_name: [str]` Azure ML workspace name.
- `aml_config_path: [str]` The path to Azure config file, if Azure ML client config is in a separate file.
- `read_timeout: [int]` read timeout in seconds for HTTP requests, user can increase if they find the default value too small. The default value from azureml sdk is 3000 which is too large and cause the evaluations and pass runs to sometimes hang for a long time between retries of job stream and download steps.
- `max_operation_retries: [int]` The maximum number of retries for Azure ML operations like resource creation and download.
The default value is 3. User can increase if there are network issues and the operations fail.
- `operation_retry_interval: [int]` The initial interval in seconds between retries for Azure ML operations like resource creation and download. The interval doubles after each retry. The default value is 5. User can increase if there are network issues and the operations fail.

### Example
```json
"azureml_client": {
    "subscription_id": "<subscription_id>",
    "resource_group": "<resource_group>",
    "workspace_name": "<workspace_name>",
    "read_timeout" : 4000,
    "max_operation_retries" : 4,
    "operation_retry_interval" : 5
},
```
## Input Model Information

`input_model: [Dict]`

User should specify input model type and configuration using `input model` dictionary. It contains following items:

- `type: [str]` Type of the input model. The supported types are `PyTorchModel`, `ONNXModel`, `OpenVINOModel`, and `SNPEModel`. It is
case insensitive.

- `config: [Dict]` The input model config dictionary specifies following items:

    - `model_path: [str | Dict]` The model path can be a string or a dictionary. If it is a string, it is either a string name
    used by the model loader or the path to the model file/directory. If it is a dictionary, it contains information about the model path.
    Please refer to [Configuring Model Path](configuring_model_path) for the more information of the model path dictionary.

    - `model_loader: [str]` The name of the function provided by the user to load the model. The function should take the model path as
    input and return the loaded model.

    - `model_script: [str]` The name of the script provided by the user to assist with model loading.

    - <a name="hf_config"></a> `hf_config: [Dict]` Instead of `model_path` or `model_loader`, the model can be specified using a dictionary describing a huggingface
    model. This dictionary specifies the following items:

        - `model_name: [str]`: This the model name of the huggingface model such as `distilbert-base-uncased` which will be used to load the model with huggingface `from_pretrained` method.

        - `task: [str]`: This is the task type for the model such as `text-classification`. The complete list of supported task can be found
        at [huggingface-tasks](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines#transformers.pipeline.task).

        - `feature: [str]`: The ONNX export features. This is only needed for HuggingFace hub model. Default to `default`. You can find more info at [Export to ONNX](https://huggingface.co/docs/transformers/serialization)

        - `model_class: [str]`: Instead of the `task`, the class of the model can be provided as well. Such as `DistilBertForSequenceClassification`

        - `components: [List[HFComponent]]`: HFComponent list:
            - `HFComponent`:
                - `name: [str]`: Component name. Olive will generate a model class with this str as attribute name.
                - `io_config: [str | Dict]`: The io_config of this component. If `str`, Olive will load `io_config` from `model_script`.
                - `component_func: [str]`: The component function name will be loaded from `model_script`.
                - `dummy_inputs_func: [str]`: The dummy input function name will be loaded from `model_script`.

        - `config: [str]`: The config of the model can be provided as well. Such as `WhisperConfig`. See
        [huggingface configurations](https://huggingface.co/docs/transformers/main_classes/configuration)

        - `dataset: [dict]`: If you want to use the huggingface dataset, you need to provide the dataset config. See [huggingface datasets](https://huggingface.co/docs/datasets/loading). Olive exposes the following configs(which will be extended in the future):
            ```python
            "dataset": {
                "model_name": "distilbert-base-uncased",  # the model name of the huggingface model, if not provided, it will use the model_name in hf_config
                "task": "text-classification",  # the task type for the model, if not provided, it will use the task in hf_config
                "data_name":"glue",  # the name of the dataset
                "subset": "mrpc",  # the subset of the dataset, could be "mrpc", "mnli" and etc. You can find the available subsets in the dataset page.
                "split": "validation",  # the split of the dataset, could be "train", "validation", "test" and etc. You can find the available splits in the dataset page.
                "input_cols": ["sentence1", "sentence2"],  # the input columns of the dataset
                "label_cols": ["label"],  # the label columns of the dataset
                "batch_size": 1  # the batch size of the dataloader
            }
            ```
            For cases where you do not want to use the huggingface model but want to use the huggingface dataset, you can provide `dataset` config only like above.

Please find the detailed config options from following table for each model type:

| Model Type | Description |
|:----------|:-------------|
| [PytorchModel](pytorch_model) | Pytorch model |
| [ONNXModel](onnx_model) | ONNX model |
| [OpenVINOModel](openvino_model) | OpenVINO IR model |
| [SNPEModel](snpe_model) | SNPE DLC model |

### Example
```json
"input_model": {
    "type": "PyTorchModel",
    "config": {
        "model_loader": "load_pytorch_origin_model",
        "model_script": "user_script.py",
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
}
```

## Systems Information
`systems: [Dict]`

This is a dictionary that contains the information of systems that are reference by the engine, passes and evaluators. The key of the
dictionary is the name of the system. The value of the dictionary is another dictionary that contains the information of the system. The
information of the system contains following items:

- `type: [str]` The type of the system. The supported types are `LocalSystem`, `AzureML` and `Docker`.
  There are some built-in system alias which could also be used as type. For exmaple, `AzureNDV2System`. Please refer to [System Alias](system_alias) for the complete list of system alias.

- `config: [Dict]` The system config dictionary that contains the system specific information.

Please refer to [Configuring OliveSystem](configuring_olivesystem) for the more information of the system config dictionary.

### Example
```json
"systems": {
    "local_system": {"type": "LocalSystem"},
    "aml_system": {
        "type": "AzureML",
        "config": {
            "aml_compute": "cpu-cluster",
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                "conda_file_path": "conda.yaml"
            }
        }
    }
}
```
## Evaluators Information
`evaluators: [Dict]`

This is a dictionary that contains the information of evaluators that are reference by the engine and passes. The key of the dictionary
is the name of the evaluator. The value of the dictionary is another dictionary that contains the information of the evaluator. The
information of the evaluator contains following items:

- <a name="metrics"></a> `metrics: [List]` This is a list of metrics that the evaluator will use to evaluate the model. Each metric is a dictionary that
    contains following items:

    - `name: [str]` The name of the metric. This must be a unique name among all metrics in the evaluator.

    - `type: [str]` The type of the metric. The supported types are `accuracy`, `latency` and `custom`.

    - `backend: [str]` The type of metrics' backend. Olive implement `torch_metrics` and `huggingface_metrics` backends. The default value is `torch_metrics`.
        - `torch_metrics` backend uses `torchmetrics` library to compute metrics. It supports `accuracy_score`, `f1_score`, `precision`, `recall` and `auc` metrics.
        - `huggingface_metrics` backend uses huggingface `evaluate` library to compute metrics. The supported metrics can be found at [huggingface metrics](https://huggingface.co/metrics).

    - `subtypes: [List[Dict]]` The subtypes of the metric. Cannot be null or empty. Each subtype is a dictionary that contains following items:

        - `name: str` The name of the subtype. Please refer to [AccuracySubtype](accuracy_sub_type) and [LatencySubtype](latency_sub_type)
        for the supported subtypes. For `custom` type, if the result of the evaluation is a dictionary, the name of the subtype should be the key of the dictionary. Otherwise, the name of the subtype could be any unique string user gives.

        - `metric_config` The parameter config used to measure detailed metrics. Please note that when the `backend` is `huggingface_metrics`, you should see the `metric_config` as dictionary of:
            - `load_params`: The parameters used to load the metric, run as `evaluator = evaluate.load("word_length", **load_params)`.
            - `compute_params` The parameters used to compute the metric, run as `evaluator.compute(predictions=preds, references=target, **compute_params)`.
            - `result_key` The key used to extract the metric result with given format. For example, if the metric result is {'accuracy': {'value': 0.9}}, then the result_key should be 'accuracy.value'."

        - `priority: [int]` The priority of the subtype. The higher priority subtype will be given priority during evaluation. Note that it should be unique among all subtypes in the metric.

        - `higher_is_better: [Boolean]` True if the metric is better when it is higher. It is `true` for `accuracy` type and `false` for `latency` type.

        - `goal: [Dict]` The goal of the metric. It is a dictionary that contains following items:

            - `type: [str]` The type of the goal. The supported types are `threshold`, `min-improvement`, `percent-min-improvement`,
            `max-degradation`, and `percent-max-degradation`.

            - `value: [float]` The value of the goal. It is the threshold value for `threshold` type. It is the minimum improvement value
            for `min-improvement` type. It is the minimum improvement percentage for `percent-min-improvement` type. It is the maximum
            degradation value for `max-degradation` type. It is the maximum degradation percentage for `percent-max-degradation` type.

    - `user_config: [Dict]` The user config dictionary that contains the user specific information for the metric. The
       dictionary contains following items:

        - `user_script: [str]` The name of the script provided by the user to assist with metric evaluation.

        - `script_dir: [str]` The directory that contains dependencies for the user script.

        - `data_dir: [str]` The directory that contains the data for the metric evaluation.

        - `batch_size: [int]` The batch size for the metric evaluation.

        - `dataloader_func: [str]` The name of the function provided by the user to load the data for the metric evaluation. The
        function should take the `data_dir` and `batch_size` as input and return the data loader. Only valid for `accuracy` and `latency`
         type.

        - `inference_settings: [Dict]` Inference settings for the different runtime. Only valid for `accuracy` and `latency` type.

        - `post_processing_func: [str]` The name of the function provided by the user to post process the model output. The function
        should take the model output and return the post processed output. Only valid for `accuracy` type.

        - `evaluate_func: [str]` The name of the function provided by the user to evaluate the model. The function should take the
        model, `data_dir` and `batch_size` as input and return the evaluation result. Only valid for `custom` type.

### Example
```json
"evaluators": {
    "common_evaluator": {
        "metrics":[
            {
                "name": "accuracy",
                "type": "accuracy",
                "sub_types": [
                    {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
                    {"name": "f1_score", "metric_config": {"multiclass": false}},
                    {"name": "auc", "metric_config": {"reorder": true}}
                ],
                "user_config":{
                    "post_processing_func": "post_process",
                    "user_script": "user_script.py",
                    "dataloader_func": "create_dataloader",
                    "batch_size": 1
                }
            },
            {
                "name": "accuracy",
                "type": "accuracy",
                "backend": "huggingface_metrics",
                "sub_types": [
                    {"name": "accuracy", "priority": -1},
                    {"name": "f1"}
                ],
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
                "sub_types": [
                    {"name": "avg", "priority": 2, "goal": {"type": "percent-min-improvement", "value": 20}},
                    {"name": "max"},
                    {"name": "min"}
                ],
                "user_config":{
                    "user_script": "user_script.py",
                    "dataloader_func": "create_dataloader",
                    "batch_size": 1
                }
            }
        ]
    }
}
```


## Passes Information
`passes: [Dict]`

This is a dictionary that contains the information of passes that are executed by the engine. The passes are executed
in order of their definition in this dictionary. The key of the dictionary is the name of the pass. The value of the dictionary is
another dictionary that contains the information of the pass. The information of the pass contains following items:

- `type: [str]` The type of the pass.

- `disable_search: [Boolean]` This decides whether to use the default value (`true`) or the default searchable values,
  if any, (`false`) for the optional parameters. This is `false` by default and can be overridden if `search_strategy` under `engine` is
  specified. Otherwise, it is always `true`.

- `config: [Dict]` The configuration of the pass.

- `host: [str | Dict]` The host of the pass. It can be a string or a dictionary. If it is a string, it is the name of a system in
`systems`. If it is a dictionary, it contains the system information. If not specified, the host of the engine will be used.

- `evaluator: [str | Dict]` The evaluator of the pass. It can be a string or a dictionary. If it is a string, it is the name of an
evaluator in `evaluators`. If it is a dictionary, it contains the evaluator information. If not specified, the evaluator of the engine
will be used.

- `clean_run_cache: [Boolean]` This decides whether to clean the run cache of the pass before running the pass. This is `false` by default.

- `output_name: str` In no-search mode (i.e., `search_strategy` is `null`), if `output_name` is provided, the output model of the pass will be
saved to the engine's `output_dir` with the prefix of `output_name`. For the final pass, the engine's `output_name`, if provided, overrides the
`output_name` of the pass.

Please refer to [Configuring Pass](configuring_pass) for more details on `type`, `disable_search` and `config`.

Please also find the detailed options from following table for each pass:

| Pass Name | Description |
|:----------|:-------------|
| [OnnxConversion](onnx_conversion) | Convert a PyTorch model to ONNX model |
| [OnnxModelOptimizer](onnx_model_optimizer) | Optimize ONNX model by fusing nodes. |
| [OnnxTransformersOptimization](onnx_transformers_optimization) | Optimize transformer based models in scenarios where ONNX Runtime does not apply the optimization at load time. It is based on onnxruntime.transformers.optimizer. |
| [OrtPerfTuning](ort_perf_tuning) | Optimize ONNX Runtime inference settings. |
| [OnnxDynamicQuantization](onnx_dynamic_quantization) | ONNX Dynamic Quantization Pass. |
| [OnnxStaticQuantization](onnx_static_quantization) | ONNX Static Quantization Pass. |
| [OnnxQuantization](onnx_quantization) | Quantize ONNX model with onnxruntime where we can search for best parameters for static/dynamic quantization at same time. |
| [IncDynamicQuantization](inc_dynamic_quantization) |  Intel® Neural Compressor Dynamic Quantization Pass. |
| [IncStaticQuantization](inc_static_quantization) |  Intel® Neural Compressor Static Quantization Pass. |
| [IncQuantization](inc_quantization) | Quantize ONNX model with Intel® Neural Compressor where we can search for best parameters for static/dynamic quantization at same time. |
| [QuantizationAwareTraining](onnx_quantization_aware_training) | Run quantization aware training on PyTorch model. |
| [OpenVINOConversion](openvino_conversion) | Converts PyTorch, ONNX or TensorFlow Model to OpenVino Model. |
| [OpenVINOQuantization](openvino_quantization) | Post-training quantization for OpenVINO model. |
| [SNPEConversion](snpe_conversion) | Convert ONNX or TensorFlow model to SNPE DLC. Uses snpe-tensorflow-to-dlc or snpe-onnx-to-dlc tools from the SNPE SDK. |
| [SNPEQuantization](snpe_quantization) | Quantize SNPE model. Uses snpe-dlc-quantize tool from the SNPE SDK. |
| [SNPEtoONNXConversion](snpe_to_onnx_conversion) | Convert a SNPE DLC to ONNX to use with SNPE Execution Provider. Creates a ONNX graph with the SNPE DLC as a node. |
| [VitisAIQuantization](vitis_ai_quantization) | AMD-Xilinx Vitis-AI Quantization Pass.  |
| [OptimumConversion](optimum_conversion) | Convert huggingface models to ONNX via the Optimum library. |
| [OptimumMerging](optimum_merging) | Merge 2 models together with an `if` node via the Optimum library. |

### Example
```json
"passes": {
    "onnx_conversion": {
        "type": "OnnxConversion",
        "config": {
            "target_opset": 13
        }
    },
    "onnx_quantization": {
        "type": "OnnxQuantization",
        "config": {
            "user_script": "user_script.py",
            "data_dir": "data",
            "dataloader_func": "resnet_calibration_reader",
            "weight_type": "QUInt8"
        }
    }
}
```

## Engine Information
`engine: [Dict]`

This is a dictionary that contains the information of the engine. The information of the engine contains following items:

- `search_strategy: [Dict | Boolean | None]` The search strategy of the engine. It contains the following items:

    - `execution_order: [str]` The execution order of the optimizations of passes. The options are `pass-by-pass` and `joint`.

    - `search_algorithm: [str]` The search algorithm of the engine. The available search algorithms are `exhaustive`, `random` and `tpe`.

    - `search_algorithm_config: [Dict]` The configuration of the search algorithm. The configuration of the search algorithm depends on
    the search algorithm.

    - `output_model_num: [int]` The number of output models from the engine based on metric priority. If not specified, the engine will output all qualified models.

    - `stop_when_goals_met: [Boolean]` This decides whether to stop the search when the metric goals, if any,  are met. This is `false` by
    default.

    - `max_iter: [int]` The maximum number of iterations of the search. Only valid for `joint` execution order. By default, there is no
    maximum number of iterations.

    - `max_time: [int]` The maximum time of the search in seconds. Only valid for `joint` execution order. By default, there is no
    maximum time.

  If `search_strategy` is `null` or `false`, the engine will run the passes in the order they were registered without searching. Thus, the passes must
  have empty search spaces. The output of the final pass will be evaluated if there is a valid evaluator. The output of the engine will be
  the output model of the final pass and its evaluation result.

  If `search_strategy` is `true`, the search strategy will be the default search strategy. The default search strategy is `exhaustive` search
  algorithm with `joint` execution order.

- `evaluation_only: [Boolean]` This decides whether to run the engine in evaluation only mode. In this mode, the engine will evaluate the input
    model using the engine's evaluator and return the results. If the engine has no evaluator, it will raise an error. This is `false` by default.

- `host: [str | Dict]` The host of the engine. It can be a string or a dictionary. If it is a string, it is the name of a system in `systems`.
    If it is a dictionary, it contains the system information. If not specified, it is the local system.

- `target: [str | Dict]` The target to run model evaluations on. It can be a string or a dictionary. If it is a string, it is the name of
    a system in `systems`. If it is a dictionary, it contains the system information. If not specified, it is the local system.

- `evaluator: [str | Dict]` The evaluator of the engine. It can be a string or a dictionary. If it is a string, it is the name of an evaluator
    in `evaluators`. If it is a dictionary, it contains the evaluator information. This evaluator will be used to evaluate the input model if
    needed. It is also used to evaluate the output models of passes that don't have their own evaluators.

- `cache_dir: [str]` The directory to store the cache of the engine. If not specified, the cache will be stored in the `.olive-cache` directory
    under the current working directory.

- `clean_cache: [Boolean]` This decides whether to clean the cache of the engine before running the engine. This is `false` by default.

- `clean_evaluation_cache: [Boolean]` This decides whether to clean the evaluation cache of the engine before running the engine. This is
`false` by default.

- `plot_pareto_frontier` This decides whether to plot the pareto frontier of the search results. This is `false` by default.

- `output_dir: [str]` The directory to store the output of the engine. If not specified, the output will be stored in the current working
    directory. For a run with no search, the output is the output model of the final pass and its evaluation result. For a run with search, the
    output is a json file with the search results.

- `output_name: [str]` The name of the output. This string will be used as the prefix of the output file name. If not specified, there is no
    prefix.

- `packaging_config: [PackagingConfig]` Olive artifacts packaging configurations. If not specified, Olive will not package artifacts.

- `log_severity_level: [int]` The log severity level of Olive. The options are `0` for `VERBOSE`, `1` for
    `INFO`, `2` for `WARNING`, `3` for `ERROR`, `4` for `FATAL`. The default value is `1` for `INFO`.

- `ort_log_severity_level: [int]` The log severity level of ONNX Runtime. The options are `0` for `VERBOSE`, `1` for
    `INFO`, `2` for `WARNING`, `3` for `ERROR`, `4` for `FATAL`. The default value is `3` for `ERROR`.

Please find the detailed config options from following table for each search algorithm:

| Algorithm  | Description |
|:----------|:-------------|
| [exhaustive](exhaustive_search_algorithm) | Iterates over the entire search space |
| [random](random_search_algorithm) | Samples random points from the search space with or without replacement |
| [tpe](tpe_search_algorithm) | Sample using TPE (Tree-structured Parzen Estimator) algorithm. |

### Example
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
    "clean_cache": true,
    "cache_dir": "cache"
}
```
