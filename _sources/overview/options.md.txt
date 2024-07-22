# Olive Options

Olive enables users to easily compose and customize their own model optimization pipelines. Olive provides a set of passes that can be
used to compose a pipeline. Olive receives input model, target hardware, performance requirements, and list of optimizations techniques
to apply from user in the form of a json dictionary. In this document, we document the options user can set in this dictionary.

**Note**:
- The json schema for the config file can be found [here](https://microsoft.github.io/Olive/schema.json). It can be used in IDEs like VSCode to provide intellisense by adding the following line at the top of the config file:
```json
"$schema": "https://microsoft.github.io/Olive/schema.json"
```
- The config file can also be provided as a YAML file with the extension `.yaml` or `.yml`.

The options are organized into following sections:

- [Workflow id](#workflow-id) `workflow_id`
- [Azure ML client](#azure-ml-client) `azureml_client`
- [Input Model Information](#input-model-information) `input_model`
- [Systems Information](#systems-information) `systems`
- [Evaluators Information](#evaluators-information) `evaluators`
- [Passes Information](#passes-information) `passes`
- [Engine Information](#engine-information) `engine`
- [Workflow Host](#workflow-host) `workflow_host`


## Workflow ID

You can name the workflow run by specifying `workflow_id` section in your config file. Olive will save the cache under `<cache_dir>/<workflow_id>` folder, and automatically save the current running config in the cache folder.

## Workflow Host

Workflow host is where the Olive workflow will be run. The default value is `None`. If `None` set for workflow host, Olive will run workflow locally. It suppurts `AzureML` system for now.

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
- `default_auth_params: Dict[str, Any]` Default auth parameters for AzureML client. Please refer to [azure DefaultAzureCredential](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python#parameters) for more details. For example, if you want to exclude managed identity credential, you can set the following:
    ```json
    "azureml_client": {
        // ...
        "default_auth_params": {
            "exclude_managed_identity_credential": true
        }
    }
    ```
- `keyvault_name: [str]` The keyvault name to retrieve secrets.

### Example
#### `azureml_client` with `aml_config_path`:
##### `aml_config.json`:
```json
{
    "subscription_id": "<subscription_id>",
    "resource_group": "<resource_group>",
    "workspace_name": "<workspace_name>",
}
```
##### `azureml_client`:
```json
"azureml_client": {
    "aml_config_path": "aml_config.json",
    "read_timeout" : 4000,
    "max_operation_retries" : 4,
    "operation_retry_interval" : 5
},
```
#### `azureml_client` with azureml config fields:
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

<!-- TODO(anyone): Docs for all model handlers-->
## Input Model Information

`input_model: [Dict]`

User should specify input model type and configuration using `input model` dictionary. It contains following items:

- `type: [str]` Type of the input model which is case insensitive.. The supported types contain `HfModelHandler`, `PyTorchModelHandler`, `ONNXModelHandler`, `OpenVINOModelHandler`,`SNPEModelHandler` and etc. You can
find more details in [Olive Models](https://microsoft.github.io/Olive/api/models.html).

- `config: [Dict]` The configuration of the pass. Its fields can be provided directly to the parent dictionary. For example, for `HfModelHandler`, the input model config dictionary specifies following items:

    - `model_path: [str | Dict]` The model path can be a string or a dictionary. If it is a string, it is a huggingface hub model id or a local directory. If it is a dictionary, it contains information about the model path. Please refer to [Configuring Model Path](../tutorials/configure_model_path.md) for the more information of the model path dictionary.

    - `task: [str]` The task of the model. The default task is `text-generation-with-past` which is equivalent to a causal language model with key-value cache enabled.

    - `io_config: [Dict]`: The inputs and outputs information of the model. If not provided, Olive will try to infer the input and output information from the model. The dictionary contains following items:
        - `input_names: [List[str]]` The input names of the model.
        - `input_types: [List[str]]` The input types of the model.
        - `input_shapes: [List[List[int]]]` The input shapes of the model.
        - `output_names: [List[str]]` The output names of the model.
        - `dynamic_axes: [Dict[str, Dict[str, str]]]` The dynamic axes of the model. The key is the name of the input or output and the value is a dictionary that contains the dynamic axes of the input or output. The key of the value dictionary is the index of the dynamic axis and the value is the name of the dynamic axis. For example, `{"input": {"0": "batch_size"}, "output": {"0": "batch_size"}}` means the first dimension of the input and output is dynamic and the name of the dynamic axis is `batch_size`.
        - `string_to_int_dim_params: List[str]` The list of input names in dynamic axes that need to be converted to int value.
        - `kv_cache: Union[bool, Dict[str, str]]` The key value cache configuration. If not provided, it is assumed to be `True` if the `task` ends with `-with-past`.
          - If it is `False`, Olive will not use key value cache.
          - If it is `True`, Olive will infer the cache configuration from the input_names/input_shapes and input model based on default `kv_cache`.
          - If it is a dictionary, it should contains the key value cache configuration. Here is an default configuration example:
            - `ort_past_key_name`: "past_key_values.<id>.key"
                Template for the past key name. The `<id>` will be replaced by the id of the past key.
            - `ort_past_value_name`: "past_key_values.<id>.value"
                Template for the past value name. The `<id>` will be replaced by the id of the past value.
            - `ort_present_key_name`: "present.<id>.key"
                Template for the present key name. The `<id>` will be replaced by the id of the present key.
            - `ort_present_value_name`: "present.<id>.value"
                Template for the present value name. The `<id>` will be replaced by the id of the present value.
            - `world_size`: 1
                It is only used for distributed models.
            - `num_hidden_layers`: null
                If null, Olive will infer the number of hidden layers from the model.
            - `num_attention_heads`: null
                If null, Olive will infer the number of attention heads from the model.
            - `hidden_size`: null
                If null, Olive will infer the hidden size from the model.
            - `past_sequence_length`: null
                If null, Olive will infer the past sequence length from the model.
            - `batch_size`: 0
                The batch size of the model. If it is 0, Olive will use the batch size from the input_shapes if `input_ids`.
            - `dtype`: "float32"
                The data type of the model.
            - `shared_kv`: false
                Whether to share the key value cache between the past and present key value cache. If it is true, the dynamic axes of the past and present key value cache will be the same.
            - `sequence_length_idx`: 2
                For most of the cases, the input shape for kv_cache is like (batch_size, num_attention_heads/world_size, sequence_length, hidden_size/num_attention_heads). The `sequence_length` is the index of the sequence length in the input shape.
            - `past_kv_dynamic_axis`: null
                The dynamic axis of the past key value cache. If it is null, Olive will infer the dynamic axis.
            - `present_kv_dynamic_axis`: null
                The dynamic axis of the present key value cache. If it is null, Olive will infer the dynamic axis.

    - `load_kwargs: [dict]`: Arguments to pass to the `from_pretrained` method of the model class. Refer to [this documentation](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained).

Please find the detailed config options from following table for each model type:

| Model Type | Description |
|:----------|:-------------|
| [HfModelHandler](hf_model) | Hf model |
| [PytorchModelHandler](pytorch_model) | Pytorch model |
| [ONNXModelHandler](onnx_model) | ONNX model |
| [OpenVINOModelHandler](openvino_model) | OpenVINO IR model |
| [SNPEModelHandler](snpe_model) | SNPE DLC model |

### Example
```json
"input_model": {
    "type": "HfModel",
    "model_path": "meta-llama/Llama-2-7b-hf"
}
```

## Systems Information
`systems: [Dict]`

This is a dictionary that contains the information of systems that are reference by the engine, passes and evaluators. The key of the
dictionary is the name of the system. The value of the dictionary is another dictionary that contains the information of the system. The
information of the system contains following items:

- `type: [str]` The type of the system. The supported types are `LocalSystem`, `AzureML` and `Docker`.
  There are some built-in system alias which could also be used as type. For example, `AzureNDV2System`. Please refer to [Olive System Alias](olive_system_alias) for the complete list of system alias.

- `config: [Dict]` The system config dictionary that contains the system specific information. The fields can be provided directly under the parent dictionary.
 - `accelerators: [List[str]]` The accelerators that will be used for this workflow.
 - `hf_token: [bool]` Whether to use a Huggingface token to access Huggingface resources. If it is set to `True`, For local system, Docker system, and PythonEnvironment system, Olive will retrieve the token from the `HF_TOKEN` environment variable or from the token file located at `~/.huggingface/token`. For AzureML system, Olive will retrieve the token from user keyvault secret. If set to `False`, no token will be utilized during this workflow run. The default value is `False`.


Please refer to [How To Configure System](../tutorials/configure_systems.rst) for the more information of the system config dictionary.

### Example
```json
"systems": {
    "local_system": {"type": "LocalSystem"},
    "aml_system": {
        "type": "AzureML",
        "aml_compute": "cpu-cluster",
        "aml_docker_config": {
            "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            "conda_file_path": "conda.yaml"
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

    - `type: [str]` The type of the metric. The supported types are `accuracy`, `latency`, `throughput` and `custom`.

    - `backend: [str]` The type of metrics' backend. Olive implement `torch_metrics` and `huggingface_metrics` backends. The default value is `torch_metrics`.
        - `torch_metrics` backend uses `torchmetrics`(>=0.1.0) library to compute metrics. It supports `accuracy_score`, `f1_score`, `precision`, `recall` and `auroc` metrics which are used for `binary` task (equal to `metric_config:{"task": "binary"}`) by default. You need alter the `task` if needed. Please refer to [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) for more details.
        - `huggingface_metrics` backend uses huggingface `evaluate` library to compute metrics. The supported metrics can be found at [huggingface metrics](https://huggingface.co/metrics).

    - `subtypes: [List[Dict]]` The subtypes of the metric. Cannot be null or empty. Each subtype is a dictionary that contains following items:

        - `name: str` The name of the subtype. Please refer to [AccuracySubtype](accuracy_sub_type), [LatencySubtype](latency_sub_type) and [ThroughputSubtype](throughput_sub_type)
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

        - `data_dir: [str|ResourcePathConfig]` The directory that contains the data for the metric evaluation.

        - `batch_size: [int]` The batch size for the metric evaluation.

        - `inference_settings: [Dict]` Inference settings for the different runtime.

        - `dataloader_func: [str]` The name of the function provided by the user to load the data for the metric evaluation. The function should take the `data_dir`, `batch_size`, `model_framework` (provided as a keyword argument)
        as input and return the data loader. Not valid for `custom` type when `evaluate_func` is provided.

        - `post_processing_func: [str]` The name of the function provided by the user to post process the model output. The function should take the model output as input and return the post processed
        output. Only valid for `accuracy` type or `custom` type when `evaluate_func` is not provided.

        - `evaluate_func: [str]` The name of the function provided by the user to evaluate the model. The function should take the model, `data_dir`, `batch_size`, `device`, `execution_providers` as input
        and return the evaluation result. Only valid for `custom` type.

        - `metric_func: [str]` The name of the function provided by the user to compute metric from the model output. The function should take the post processed output and target as input and return the
        metric result. Only valid for `custom` type when `evaluate_func` is not provided.

        - `func_kwargs: [Dict[str, Dict[str, Any]]]` Keyword arguments for the functions provided by the user. The key is the name of the function and the value is the keyword arguments for the function. The
        functions must be able to take the keyword arguments either through the function signature as keyword/positional parameters after the required positional parameters or through `**kwargs`.

    Note that for above `data_dir` config which is related to resource path, Olive supports local file, local folder or AML Datastore. Take AML Datastore as an example, Olive can parse the resource type automatically from `config dict`, or `url`. Please refer to our [Resnet](https://github.com/microsoft/Olive/tree/main/examples/resnet#resnet-optimization-with-ptq-on-cpu) example for more details.
    ```json
    "data_dir": {
        "type": "azureml_datastore",
        "azureml_client": "azureml_client",
        "datastore_name": "test",
        "relative_path": "cifar-10-batches-py"
    }
    // provide azureml datastore url
    "data_dir": "azureml://subscriptions/test/resourcegroups/test/workspaces/test/datastores/test/cifar-10-batches-py"
    ```

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
                    {"name": "f1_score"},
                    {"name": "auroc"}
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
                    "batch_size": 1,
                    "inference_settings" : {
                        "onnx": {
                            "session_options": {
                                "enable_profiling": true
                            }
                        }
                    }
                }
            }
        ]
    }
}
```


## Passes Information
`passes: [Dict]`

This is a dictionary that contains the information of passes that are executed by the engine. The passes are executed
in order of their definition in this dictionary if `pass_flows` is not specified. The key of the dictionary is the name
of the pass. The value of the dictionary is another dictionary that contains the information of the pass. The information
of the pass contains following items:

- `type: [str]` The type of the pass.

- `disable_search: [Boolean]` This decides whether to use the default value (`true`) or the default searchable values,
  if any, (`false`) for the optional parameters. This is `false` by default and can be overridden if `search_strategy` under `engine` is
  specified. Otherwise, it is always `true`.

- `config: [Dict]` The configuration of the pass. Its fields can be provided directly to the parent dictionary.

- `host: [str | Dict]` The host of the pass. It can be a string or a dictionary. If it is a string, it is the name of a system in
`systems`. If it is a dictionary, it contains the system information. If not specified, the host of the engine will be used.

- `evaluator: [str | Dict]` The evaluator of the pass. It can be a string or a dictionary. If it is a string, it is the name of an
evaluator in `evaluators`. If it is a dictionary, it contains the evaluator information. If not specified, the evaluator of the engine
will be used.

- `clean_run_cache: [Boolean]` This decides whether to clean the run cache of the pass before running the pass. This is `false` by default.

- `output_name: str` In no-search mode (i.e., `search_strategy` is `null`), if `output_name` is provided, the output model of the pass will be
saved to the engine's `output_dir` with the prefix of `output_name`. For the final pass, if the engine's `output_name` is provided, it will override
the `output_name` of the pass.

Please refer to [Configuring Pass](../tutorials/configure_pass.rst) for more details on `type`, `disable_search` and `config`.

Please also find the detailed options from following table for each pass:

| Pass Name | Description |
|:----------|:-------------|
| [OnnxConversion](onnx_conversion) | Convert a PyTorch model to ONNX model |
| [OnnxOpVersionConversion](onnx_op_version_conversion) | Convert a Onnx model to target op version |
| [ModelBuilder](model_builder) | Convert a generative PyTorch model to ONNX model using [ONNX Runtime Generative AI](https://github.com/microsoft/onnxruntime-genai) module |
| [OnnxModelOptimizer](onnx_model_optimizer) | Optimize ONNX model by fusing nodes. |
| [OnnxTransformersOptimization](onnx_transformers_optimization) | Optimize transformer based models in scenarios where ONNX Runtime does not apply the optimization at load time. It is based on onnxruntime.transformers.optimizer. |
| [OrtPerfTuning](ort_perf_tuning) | Optimize ONNX Runtime inference settings. |
| [OnnxDynamicQuantization](onnx_dynamic_quantization) | ONNX Dynamic Quantization Pass. |
| [OnnxStaticQuantization](onnx_static_quantization) | ONNX Static Quantization Pass. |
| [OnnxQuantization](onnx_quantization) | Quantize ONNX model with onnxruntime where we can search for best parameters for static/dynamic quantization at same time. |
| [IncDynamicQuantization](inc_dynamic_quantization) |  Intel® Neural Compressor Dynamic Quantization Pass. |
| [IncStaticQuantization](inc_static_quantization) |  Intel® Neural Compressor Static Quantization Pass. |
| [IncQuantization](inc_quantization) | Quantize ONNX model with Intel® Neural Compressor where we can search for best parameters for static/dynamic quantization at same time. |
| [DynamicToFixedShape](dynamic_to_fixed_shape) | Convert dynamic shape to fixed shape for ONNX model |
| [ExtractAdapters](extract_adapters) | Extract adapters from ONNX model |
| [QuantizationAwareTraining](onnx_quantization_aware_training) | Run quantization aware training on PyTorch model. |
| [OpenVINOConversion](openvino_conversion) | Converts PyTorch, ONNX or TensorFlow Model to OpenVino Model. |
| [OpenVINOQuantization](openvino_quantization) | Post-training quantization for OpenVINO model. |
| [SNPEConversion](snpe_conversion) | Convert ONNX or TensorFlow model to SNPE DLC. Uses snpe-tensorflow-to-dlc or snpe-onnx-to-dlc tools from the SNPE SDK. |
| [SNPEQuantization](snpe_quantization) | Quantize SNPE model. Uses snpe-dlc-quantize tool from the SNPE SDK. |
| [SNPEtoONNXConversion](snpe_to_onnx_conversion) | Convert a SNPE DLC to ONNX to use with SNPE Execution Provider. Creates a ONNX graph with the SNPE DLC as a node. |
| [VitisAIQuantization](vitis_ai_quantization) | AMD-Xilinx Vitis-AI Quantization Pass. |
| [GptqQuantizer](gptq_quantizer) | GPTQ quantization Pass On Pytorch Model. |
| [AutoAWQQuantizer](awq_quantizer) | AWQ quantization Pass On Pytorch Model. |
| [MergeAdapterWeights](merge_adapter_weights) | Merge adapter weights into the base model and save transformer context files. |
| [OptimumConversion](optimum_conversion) | Convert huggingface models to ONNX via the Optimum library. |
| [OptimumMerging](optimum_merging) | Merge 2 models together with an `if` node via the Optimum library. |
| [MixedPrecisionOverrides](mixed_precision_overrides) | Pre-processes the model for mixed precision quantization with qnn configs. |

### Example
```json
"passes": {
    "onnx_conversion": {
        "type": "OnnxConversion",
        "target_opset": 13
    },
    "onnx_quantization": {
        "type": "OnnxQuantization",
        "user_script": "user_script.py",
        "data_dir": "data",
        "dataloader_func": "resnet_calibration_reader",
        "weight_type": "QUInt8"
    }
}
```

## Pass Flows Information
`pass_flows: List[List[str]]`

This is a list of list of pass names. Each list of pass names is a pass flow which will be executed in order.
When `pass_flows` is not specified, the passes are executed in the order of the `passes` dictionary.


### Example
```json
"passes": {
    "onnx_conversion": {
        "type": "OnnxConversion",
        "target_opset": 13
    },
    "transformers_optimization": {
        "type": "OrtTransformersOptimization",
        "model_type": "bert",
        "num_heads": 12,
        "hidden_size": 768,
        "float16": true
    },
    "onnx_quantization": {
        "type": "OnnxQuantization",
        "user_script": "user_script.py",
        "data_dir": "data",
        "dataloader_func": "resnet_calibration_reader",
        "weight_type": "QUInt8"
    }
},
"pass_flows": [
    ["onnx_conversion", "transformers_optimization"],
    ["onnx_conversion", "transformers_optimization", "onnx_quantization"],
    ["onnx_conversion", "onnx_quantization"],
]
```

## Engine Information
`engine: [Dict]`

This is a dictionary that contains the information of the engine. Its fields can be provided directly to the parent dictionary. The information of the engine contains following items:

- `search_strategy: [Dict | Boolean | None]`, `None` by default. The search strategy of the engine. It contains the following items:

    - `execution_order: [str]` The execution order of the optimizations of passes. The options are `pass-by-pass` and `joint`.

    - `search_algorithm: [str]` The search algorithm of the engine. The available search algorithms are `exhaustive`, `random` and `tpe`.

    - `search_algorithm_config: [Dict]` The configuration of the search algorithm. The configuration of the search algorithm depends on
    the search algorithm. Its fields can be provided directly to the parent dictionary.

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

- `evaluate_input_model: [Boolean]` In this mode, the engine will evaluate the input model using the engine's evaluator and return the results. If the engine has no evaluator, it will skip the evaluation. This is `true` by default.

- `host: [str | Dict | None]`, `None` be default. The host of the engine. It can be a string or a dictionary. If it is a string, it is the name of a system in `systems`.
    If it is a dictionary, it contains the system information. If not specified, it is the local system.

- `target: [str | Dict | None]`, `None` be default. The target to run model evaluations on. It can be a string or a dictionary. If it is a string, it is the name of
    a system in `systems`. If it is a dictionary, it contains the system information. If not specified, it is the local system.

- `evaluator: [str | Dict | None]`, `None` by default. The evaluator of the engine. It can be a string or a dictionary. If it is a string, it is the name of an evaluator
    in `evaluators`. If it is a dictionary, it contains the evaluator information. This evaluator will be used to evaluate the input model if
    needed. It is also used to evaluate the output models of passes that don't have their own evaluators. If it is None, skip the evaluation for input model and any output models.

- `cache_dir: [str]`, `.olive-cache` by default. The directory to store the cache of the engine. If not specified, the cache will be stored in the `.olive-cache` directory
    under the current working directory.

- `clean_cache: [Boolean]`, `false` by default. This decides whether to clean the cache of the engine before running the engine.

- `clean_evaluation_cache: [Boolean]` , `false` by default. This decides whether to clean the evaluation cache of the engine before running the engine.

- `plot_pareto_frontier`, `false` by default. This decides whether to plot the pareto frontier of the search results.

- `output_dir: [str]`, `None` by default. The directory to store the output of the engine. If not specified, the output will be stored in the current working
    directory. For a run with no search, the output is the output model of the final pass and its evaluation result. For a run with search, the
    output is a json file with the search results.

- `output_name: [str]`, `None` by default. The name of the output. This string will be used as the prefix of the output file name. If not specified, there is no
    prefix.

- `packaging_config: [PackagingConfig]`, `None` by default. Olive artifacts packaging configurations. If not specified, Olive will not package artifacts.

- `log_severity_level: [int]`, `1` by default. The log severity level of Olive. The options are `0` for `VERBOSE`, `1` for
    `INFO`, `2` for `WARNING`, `3` for `ERROR`, `4` for `FATAL`.

- `ort_log_severity_level: [int]`, `3` by default. The log severity level of ONNX Runtime C++ logs. The options are `0` for `VERBOSE`, `1` for
    `INFO`, `2` for `WARNING`, `3` for `ERROR`, `4` for `FATAL`.

- `ort_py_log_severity_level: [int]`, `3` by default. The log severity level of ONNX Runtime Python logs. The options are `0` for `VERBOSE`, `1` for
    `INFO`, `2` for `WARNING`, `3` for `ERROR`, `4` for `FATAL`.

- `log_to_file: [Boolean]`, `false` by default. This decides whether to log to file. If `true`, the log will be stored in a olive-<timestamp>.log file
    under the current working directory.

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
        "num_samples": 5,
        "seed": 0
    },
    "evaluator": "common_evaluator",
    "host": "local_system",
    "target": "local_system",
    "clean_cache": true,
    "cache_dir": "cache"
}
```
