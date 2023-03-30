# Olive Options

Olive enables users to easily compose and customize their own model optimization pipelines. Olive provides a set of passes that can be
used to compose a pipeline. Olive receives input model, target hardware, performance requirements, and list of optimizations techniques
to apply from user in the form of a json dictionary. In this document, we document the options user can set in this dictionary.

The options are organized into following sections:

- [Verbosity](#verbosity) `verbose`
- [Input Model Information](#input-model-information) `input_model`
- [Systems Information](#systems-information) `systems`
- [Evaluators Information](#evaluators-information) `evaluators`
- [Passes Information](#passes-information) `passes`
- [Engine Information](#engine-information) `engine`

## Verbosity
`verbose: [Boolean]`

If set to `true`, Olive will log verbose information during the optimization process. The default value is `false`.

## Input Model Information

`input_model: [Dict]`

User should specify input model type and configuration using `input model` dictionary. It contains following items:

- `type: [str]` Type of the input model. The supported types are `PyTorchModel`, `ONNXModel`, `OpenVINOModel`, and `SNPEModel`. It is
case insensitive.

- `config: [Dict]` The input model config dictionary specifies following items.

    - `model_path: [str]` The model path.

    - `name: [str]` The name of the model.

    - `is_file: [Boolean]` True if the model path points to a file.

    - `model_loader: [str]` The name of the function provided by the user to load the model. The function should take the model path as
    input and return the loaded model.

    - `model_script: [str]` The name of the script provided by the user to assist with model loading.

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
        "model_path": null,
        "is_file": false,
        "model_loader": "load_pytorch_origin_model",
        "model_script": "user_script.py"
    }
}
```

## Systems Information
`systems: [Dict]`

This is a dictionary that contains the information of systems that are reference by the engine, passes and evaluators. The key of the
dictionary is the name of the system. The value of the dictionary is another dictionary that contains the information of the system. The
information of the system contains following items:

- `type: [str]` The type of the system. The supported types are `LocalSystem`, `AzureML` and `Docker`.

- `config: [Dict]` The system config dictionary that contains the system specific information.

Please refer to [Configuring OliveSystem](configuring_olivesystem) for the more information of the system config dictionary.

### Example
```json
"systems": {
    "local_system": {"type": "LocalSystem"},
    "aml_system": {
        "type": "AzureML",
        "config": {
            "aml_config_path": "olive-workspace-config.json",
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

- `metrics: [List]` This is a list of metrics that the evaluator will use to evaluate the model. Each metric is a dictionary that
    contains following items:

    - `name: [str]` The name of the metric. This must be a unique name among all metrics in the evaluator.

    - `type: [str]` The type of the metric. The supported types are `accuracy`, `latency` and `custom`.

    - `subtype: [str]` The subtype of the metric. Please refer to [AccuracySubtype](accuracy_sub_type) and
    [LatencySubtype](latency_sub_type) for the supported sub-types. It is `null` for `custom` type.

    - `higher_is_better: [Boolean]` True if the metric is better when it is higher. It is `true` for `accuracy` type and `false` for `latency` type.

    - `goal: [Dict]` The goal of the metric. It is a dictionary that contains following items:

        - `type: [str]` The type of the goal. The supoorted types are `threshold`, `min-improvement`, `percent-min-improvement`,
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

        - `infererence_settings: [Dict]` Inference settings for the different runtime. Only valid for `accuracy` and `latency` type.

        - `post_processing_func: [str]` The name of the function provided by the user to post process the model output. The function
        should take the model output and return the post processed output. Only valid for `accuracy` type.

        - `evaluate_func: [str]` The name of the function provided by the user to evaluate the model. The function should take the
        model, `data_dir` and `batch_size` as input and return the evaluation result. Only valid for `custom` type.

- `target: [str | Dict]` The target of the evaluator. It can be a string or a dictionary. If it is a string, it is the name of a system
in `systems`. If it is a dictionary, it contains the system information. If not specified, it is the local system.

### Example
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
}
```


## Passes Information
`passes: [Dict]`

This is a dictionary that contains the information of passes that are executed by the engine. The passes are executed
in order of their definintion in this dictionary. The key of the dictionary is the name of the pass. The value of the dictionary is
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

Please refer to [Configuring Pass](configuring_pass) for more details on `type`, `disable_search` and `config`.

Please also find the detailed options from following table for each pass:

| Pass Name | Description |
|:----------|:-------------|
| [OnnxConversion](onnx_conversion) | Convert a PyTorch model to ONNX model |
| [OnnxModelOptimizer](onnx_model_optimizer) | Optimize ONNX model by fusing nodes. |
| [OnnxTransformersOptimization](onnx_transformers_optimization) | Optimize transformer based models in scenarios where ONNX Runtime does not apply the optimization at load time. It is based on onnxruntime.transformers.optimizer. |
| [OnnxThreadTuning](onnx_thread_tuning) | Optimize ONNX Runtime inference settings. |
| [OnnxDynamicQuantization](onnx_dynamic_quantization) | Convert a PyTorch model to ONNX model. |
| [OnnxStaticQuantization](onnx_static_quantization) | ONNX Static Quantization Pass. |
| [OnnxQuantization](onnx_quantization) | Quantize ONNX model with onnxruntime where we can search for best parameters for static/dynamic quantization at same time. |
| [QuantizationAwareTraining](onnx_quantization_aware_training) | Run quantization aware training on PyTorch model. |
| [OpenVINOConversion](openvino_conversion) | Converts PyTorch, ONNX or TensorFlow Model to OpenVino Model. |
| [OpenVINOQuantization](openvino_quantization) | Post-training quantization for OpenVINO model. |
| [SNPEConversion](snpe_conversion) | Convert ONNX or TensorFlow model to SNPE DLC. Uses snpe-tensorflow-to-dlc or snpe-onnx-to-dlc tools from the SNPE SDK. |
| [SNPEQuantization](snpe_quantization) | Quantize SNPE model. Uses snpe-dlc-quantize tool from the SNPE SDK. |
| [SNPEtoONNXConversion](snpe_to_onnx_conversion) | Convert a SNPE DLC to ONNX to use with SNPE Execution Provider. Creates a ONNX graph with the SNPE DLC as a node. |

### Example
```json
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

    - `stop_when_goals_met: [Boolean]` This decides whether to stop the search when the metric goals, if any,  are met. This is `false` by
    default.

    - `max_iter: [int]` The maximum number of iterations of the search. Only valid for `joint` execution order. By default, there is no
    maximum number of iterations.

    - `max_time: [int]` The maximum time of the search in seconds. Only valid for `joint` execution order. By default, there is no
    maximum time.

  If `search_strategy` is `null` or `false`, the engine will run the passes in the order they were registered without. Thus, the passes must
  have empty search spaces. The output of the final pass will be evaluated if there is a valid evaluator. The output of the engine will be
  the output model of the final pass and its evaluation result.

  If `search_strategy` is `true`, the search strategy will be the default search strategy. The default search strategy is `exhaustive` search
  algorithm with `joint` execution order.

- `evaluation_only: [Boolean]` This decides whether to run the engine in evaluation only mode. In this mode, the engine will evaluate the input
    model using the engine's evaluator and return the results. If the engine has no evaluator, it will raise an error. This is `false` by default.

- `host: [str | Dict]` The host of the engine. It can be a string or a dictionary. If it is a string, it is the name of a system in `systems`.
    If it is a dictionary, it contains the system information. If not specified, it is the local system.

- `evaluator: [str | Dict]` The evaluator of the engine. It can be a string or a dictionary. If it is a string, it is the name of an evaluator
    in `evaluators`. If it is a dictionary, it contains the evaluator information. This evaluator will be used to evaluate the input model if
    needed. It is also used to evaluate the output models of passes that don't have their own evaluators.

- `cache_dir: [str]` The directory to store the cache of the engine. If not specified, the cache will be stored in the `.olive-cache` directory
    under the current working directory.

- `clean_cache: [Boolean]` This decides whether to clean the cache of the engine before running the engine. This is `false` by default.

- `clean_evaluation_cache: [Boolean]` This decides whether to clean the evaluation cache of the engine before running the engine. This is
`false` by default.

- `output_dir: [str]` The directory to store the output of the engine. If not specified, the output will be stored in the current working
    directory. For a run with no search, the output is the output model of the final pass and its evaluation result. For a run with search, the
    output is a json file with the search results.

- `output_name: [str]` The name of the output. This string will be used as the prefix of the output file name. If not specified, there is no
    prefix.

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
    "clean_cache": true,
    "cache_dir": "cache"
}
```
