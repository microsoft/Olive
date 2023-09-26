# Packaging Olive artifacts

## What is Olive Packaging
Olive will output multiple candidate models based on metrics priorities. It also can package output artifacts when the user required. Olive packaging can be used in different scenarios. There is only one packaging type: `Zipfile`.


### Zipfile
Zipfile packaging will generate a ZIP file which includes 3 folders: `CandidateModels`, `SampleCode` and `ONNXRuntimePackages` in the `output_dir` folder (from Engine Configuration):
* `CandidateModels`: top ranked output model set
    * Model file
    * Olive Pass run history configurations for candidate model
    * Inference settings (`onnx` model only)
* `SampleCode`: code sample for ONNX model
    * C++
    * C#
    * Python
* `ONNXRuntimePackages`: ONNXRuntime package files with the same version that were used by Olive Engine in this workflow run.

#### CandidateModels
`CandidateModels` includes k folders where k is the number of output models, with name `BestCandidateModel_1`, `BestCandidateModel_2`, ... and `BestCandidateModel_k`. The order is ranked by metrics priorities. e.g., if you have 3 metrics `metric_1`, `metric_2` and `metric_3` with priority `1`, `2` and `3`. The output models will be sorted firstly by `metric_1`. If the value of `metric_1` of 2 output models are same, they will be sorted by `metric_2`, and followed by next lower priority metric.

Each `BestCandidateModel` folder will include model file/folder. The folder also includes a json file which includes the Olive Pass run history configurations since input model, a json file with performance metrics and a json file for inference settings for the candidate model if the candidate model is an ONNX model.

##### Inference config file
The inference config file is a json file including `execution_provider` and `session_options`. e.g.:

```
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

#### SampleCode
Olive will only provide sample codes for ONNX model. Sample code supports 3 different programming languages: `C++`, `C#` and `Python`. And a code snippet introducing how to use Olive output artifacts to inference candidate model with recommended inference configurations.


## How to package Olive artifacts
Olive packaging configuration is configured in `PackagingConfig` in Engine configuration. If not specified, Olive will not package artifacts.

* `PackagingConfig`
    * `type [PackagingType]`:
      Olive packaging type. Olive will package different artifacts based on `type`.
    * `name [str]`:
      For `PackagingType.Zipfile` type, Olive will generate a ZIP file with `name` prefix: `<name>.zip`. By default, the output artifacts will be named as `OutputModels.zip`.
    * `export_in_mlflow_format [bool]`:
      Export model in mlflow format. This is `false` by default.

You can add `PackagingConfig` to Engine configurations. e.g.:

```
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
    "packaging_config": {
        "type": "Zipfile",
        "name": "OutputModels"
    },
    "clean_cache": true,
    "cache_dir": "cache"
}
```
