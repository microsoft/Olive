# How to Package Output Model for Deployment

## What is Olive Packaging

Olive will output multiple candidate models based on metrics priorities. It also can package output artifacts when the user requires. Olive packaging can be used in different scenarios. There are 4 packaging types: `Zipfile` and `Dockerfile`.

### Zipfile

Zipfile packaging will generate a ZIP file which includes 3 folders: `CandidateModels`, `SampleCode` and a `models_rank.json` file in the `output_dir` folder (from Engine Configuration):

* `CandidateModels`: top ranked output model set
  * Model file
  * Olive Pass run history configurations for candidate model
  * Inference settings (`onnx` model only)
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

### Dockerfile

Dockerfile packaging will generate a Dockerfile. You can simple run `docker build` for this Dockerfile to build a docker image which includes first ranked output model.

## How to package Olive artifacts

Olive packaging configuration is configured in `PackagingConfig` in Engine configuration. `PackagingConfig` can be a single packaging configuration. Alternatively, if you want to apply multiple packaging types, you can also define a list of packaging configurations.

If not specified, Olive will not package artifacts.

* `PackagingConfig`
  * `type [PackagingType]`:
    Olive packaging type. Olive will package different artifacts based on `type`.
  * `name [str]`:
    For `PackagingType.Zipfile` type, Olive will generate a ZIP file with `name` prefix: `<name>.zip`.
  * `config [dict]`:
    The packaging config.
    * `Zipfile`
      * `export_in_mlflow_format [bool]`:
        Export model in mlflow format. This is `false` by default.
    * `Dockerfile`
      * `requirements_file [str]`:
        `requirements.txt` file path. The packages will be installed to docker image.

You can add different types `PackagingConfig` as a list to Engine configurations. e.g.:

```json
"engine": {
    "search_strategy": {
        "execution_order": "joint",
        "sampler": "tpe",
        "max_samples": 5,
        "seed": 0
    },
    "evaluator": "common_evaluator",
    "host": "local_system",
    "target": "local_system",
    "packaging_config": [
        {
            "type": "Zipfile",
            "name": "OutputModels"
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
  "3_OrtSessionParamsTuning-2-a843d77ae4964c04e145b83567fb5b05-cpu-cpu": {
    //...
  }
}
```

### Metrics file

The metrics file is a json file including input model metrics and output model metrics.
