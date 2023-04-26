# Packaing Olive artifacts

## What is Olive Packaging
Olive will output multiple candidate models based on metrics priority ranks. It also can package output artifacts when the user required. Olive packaging can be used in different scenarios. There is only one packaging type: `Zipfile`.


### Zipfile
Zipfile packaging will generate a ZIP file which includes 2 folders: `CandidateModels` and `SampleCode` in the `output_dir` folder (from Engine Configuration):
* `CandidateModels`: top ranked output model set
    * Model file
    * Olive Pass run history configurations for candidate model
    * Inference settings (`onnx` model only)
* `SampleCode`: code sample for ONNX model
    * C++
    * C#
    * Python

#### CandidateModels
`CandidateModels` includes k folders where k is the number of output models, with name `BestCandidateModel_1`, `BestCandidateModel_2`, ... and `BestCandidateModel_k`. The order is ranked by metrics priorities. e.g., if you have 3 metrics `metric_1`, `metric_2` and `metric_3` with priority rank `1`, `2` and `3`. The output models will be sorted firstly by `metric_1`. If the value of `metric_1` of 2 output models are same, they will be sorted by `metric_2`, and followed by next lower priority metric.

Each `BestCandidateModel` folder will include model file/folder. A json file which includes the Olive Pass run history configurations since input model. And a json file for inference settings for the candidate model if the candidate model is an ONNX model.

#### SampleCode
Olive will only provide sample codes for ONNX model. Sample code supports 3 different programming languages: `C++`, `C#` and `Python`. Each programming language sample code folder includes an ONNXRuntime package file with the same version that was used by Olive Engine in this run. And a code snippet introducing how to use Olive output artifacts to inference candidate model with recommended inference configurations.


## How to package Olive artifacts
Olive packaging configuration is configured in `PackagingConfig` in Engine configuration. If not specified, Olive will not package artifacts.

* `PackagingConfig`
    * `type [PackagingType]`:
      Olive packaging type. Olive will package different artifacts based on `type`.
    * `name [str]`:
      For `PackagingType.Zipfile` type, Olive will generate a ZIP file with `name` prefix: `<name>.zip`. By default, the output artifacts will be named as `OutputModels.zip`.

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
    "packaging_config": {
        "type": "Zipfile",
        "name": "OutputModels"
    },
    "clean_cache": true,
    "cache_dir": "cache"
}
```
