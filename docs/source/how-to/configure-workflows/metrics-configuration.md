# How to Define Evaluation Metrics

This document describes how to configure the different types of Metrics.

## Metric Types

### Accuracy Metric
```json
{
    "name": "accuracy",
    "type": "accuracy",
    "data_config": "accuracy_data_config",
    "sub_types": [
        {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
        {"name": "f1_score"},
        {"name": "auroc"}
    ]
}
```

### Latency Metric
```json
{
    "name": "latency",
    "type": "latency",
    "data_config": "latency_data_config",
    "sub_types": [
        {"name": "avg", "priority": 1, "goal": {"type": "percent-min-improvement", "value": 20}}
    ]
}
```

### Throughput Metric
```json
{
    "name": "throughput",
    "type": "throughput",
    "data_config": "throughput_data_config",
    "sub_types": [
        {"name": "avg", "priority": 1, "goal": {"type": "percent-min-improvement", "value": 20}}
    ]
}
```

### Custom Metric

You can define your own metric by using the `custom` type. Your customized metric evaluation function will be defined in your own `user_script.py`,
specify its name in `evaluate_func` field, and Olive will call your function to evaluate the model.

```json
{
    "name": "accuracy",
    "type": "custom",
    "sub_types": [
        {
            "name": "accuracy_custom",
            "priority": 1,
            "higher_is_better": true,
            "goal": {"type": "max-degradation", "value": 0.01}
        }
    ],
    "user_config": {
        "user_script": "user_script.py",
        "evaluate_func": "eval_accuracy",
        "evaluate_func_kwargs": {
            "data_dir": "data",
            "batch_size": 16,
        }
    }
}
```

In your `user_script.py`, you need to define a function that takes in an Olive model, the data directory, and the batch size, and returns a metric value:

```python
def eval_accuracy(model, device, execution_providers):
    # load data
    # evaluate model
    # return metric value
```

```{Note}
Please refer to [this `user_script.py`](https://github.com/microsoft/olive-recipes/blob/main/intel-bert-base-uncased-mrpc/aitk/user_script.py) for a detailed example of how to set up a custom metric.
```


Alternatively, if you only need Olive to run the inference and you will calculate the metric by yourself, you can specify `metric_func: "None"` in the metric configuration.
Olive will run inference with the data you provided, and return the inference results to you. You can then calculate the metric by yourself:

```python
def metric_func(model_output, targets):
    # model_output[0]: preds, model_output[1]: logits
    # calculate metric
    # return metric value
```

If you provide both `evaluate_func` and `metric_func`, Olive will call `evaluate_func` only.

## Configure multiple metrics

If you have multiple metrics to evaluate, you can configure them in the following way:

```json
{
    "metrics":[
        {
            "name": "accuracy",
            "type": "accuracy",
            "sub_types": [
                {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
                {"name": "f1_score"},
                {"name": "auroc"}
            ]
        },
        {
            "name": "latency",
            "type": "latency",
            "sub_types": [
                {"name": "avg", "priority": 2, "goal": {"type": "percent-min-improvement", "value": 20}},
                {"name": "max"},
                {"name": "min"}
            ]
        }
    ]
}
```

```{Note}
If you have more than one metric, you need to specify `priority: {RANK}`, which Olive will use to determine the best model.
```

## Speech Evaluation Metrics (WER and RTFx)

Olive supports Word Error Rate (WER) and Real-Time Factor (RTFx) as built-in accuracy sub-types for evaluating speech/ASR models.

### Using WER with the accuracy metric type

WER can be used as an accuracy sub-type when your data pipeline returns text predictions and references:

```json
{
    "name": "speech_accuracy",
    "type": "accuracy",
    "data_config": "speech_data_config",
    "sub_types": [
        {"name": "wer", "priority": 1, "higher_is_better": false},
        {"name": "rtfx", "priority": 2, "higher_is_better": true}
    ]
}
```

```{Note}
- `wer` (Word Error Rate): Measures transcription errors. Lower is better (defaults to `higher_is_better: false`).
- `rtfx` (Real-Time Factor): Ratio of audio duration to inference time. Higher means faster (defaults to `higher_is_better: true`).
```

## Vision Evaluation Metrics

Olive provides three built-in accuracy sub-types for evaluating vision/multimodal models:

| Metric | Task Type | Description | Suitable Benchmarks |
|--------|-----------|-------------|---------------------|
| `exact_match` | `vision-vqa` | Case-insensitive string equality | AI2D, ScienceQA, TextVQA, MMMU |
| `relaxed_accuracy` | `vision-chart-qa` | ±5% numeric tolerance for numbers | ChartQA |
| `word_sort_ratio` | `vision-ocr` | Word-level overlap ratio | OCR benchmarks |

### Example: VQA with TextVQA (exact_match)

```json
{
    "data_configs": [
        {
            "name": "textvqa_data",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {
                "data_name": "facebook/textvqa",
                "split": "validation"
            },
            "pre_process_data_config": {
                "type": "vision_vqa_pre_process",
                "image_col": "image",
                "question_col": "question",
                "answer_col": "answers",
                "limit": 100
            },
            "dataloader_config": {
                "batch_size": 1
            }
        }
    ],
    "metrics": [
        {
            "name": "vqa_accuracy",
            "type": "accuracy",
            "data_config": "textvqa_data",
            "sub_types": [
                {"name": "exact_match", "priority": 1, "goal": {"type": "max-degradation", "value": 0.05}}
            ]
        }
    ]
}
```

### Example: ChartQA with relaxed_accuracy

```json
{
    "data_configs": [
        {
            "name": "chartqa_data",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {
                "data_name": "HuggingFaceM4/ChartQA",
                "split": "test"
            },
            "pre_process_data_config": {
                "type": "vision_vqa_pre_process",
                "image_col": "image",
                "question_col": "question",
                "answer_col": "answer",
                "limit": 100
            },
            "dataloader_config": {
                "batch_size": 1
            }
        }
    ],
    "metrics": [
        {
            "name": "chart_accuracy",
            "type": "accuracy",
            "data_config": "chartqa_data",
            "sub_types": [
                {"name": "relaxed_accuracy", "priority": 1, "goal": {"type": "max-degradation", "value": 0.05}}
            ]
        }
    ]
}
```

### Example: OCR with DocumentVQA (word_sort_ratio)

```json
{
    "data_configs": [
        {
            "name": "docvqa_data",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {
                "data_name": "HuggingFaceM4/DocumentVQA",
                "split": "validation"
            },
            "pre_process_data_config": {
                "type": "vision_vqa_pre_process",
                "image_col": "image",
                "question_col": "question",
                "answer_col": "answers",
                "limit": 100
            },
            "dataloader_config": {
                "batch_size": 1
            }
        }
    ],
    "metrics": [
        {
            "name": "ocr_accuracy",
            "type": "accuracy",
            "data_config": "docvqa_data",
            "sub_types": [
                {"name": "word_sort_ratio", "priority": 1, "goal": {"type": "max-degradation", "value": 0.05}}
            ]
        }
    ]
}
```

```{Note}
- Vision metrics compare predicted answer strings to ground truth. The model's `post_func` must decode model output into text.
- Use `batch_size: 1` since images have variable sizes.
- Multiple valid answers (lists) are joined with `|` and the metric matches against any valid answer.
- For ONNX models, provide a custom pre-process that applies the processor/tokenizer to produce numeric tensors.
```

## Standard Multimodal Benchmarks with lmms-eval

Use `LMMSEvaluator` when a public image or audio generation benchmark is available
in [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). It delegates task
loading, prompting, and scoring to lmms-eval so results use the benchmark's
declared protocol and metric, such as DocVQA ANLS, TextVQA accuracy, WER, or
BLEU.

Olive's existing `OnnxEvaluator` and custom evaluator support remain the
appropriate choices for proprietary tasks, lightweight smoke tests, and
workflows that should not depend on lmms-eval. `LMMSEvaluator` does not
deprecate those paths.

Install Olive and the pinned upstream lmms-eval dependency before using this
evaluator:

```bash
pip install olive-ai
pip install \
  "lmms-eval[audio,metrics] @ git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@3e675904f8cba6793de12b91979b04d91754bdf3"
```

Olive cannot publish a package extra containing a direct Git dependency. The
installation command therefore pins the upstream lmms-eval commit containing
[the wheel package-data fix](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1390).
The published `0.7.2` wheel omits extensionless task templates and cannot load
its default task registry. A normal `olive-ai[lmms-eval]` extra can replace this
command after upstream publishes a release containing that fix.

For an ONNX input, also install the ORT-GenAI package for the target provider,
such as `onnxruntime-genai` or `onnxruntime-genai-cuda`.

`LMMSEvaluator` accepts:

- an `HfModelHandler`, dispatched to an upstream lmms-eval model wrapper; or
- an `ONNXModelHandler` that represents a complete ORT-GenAI package containing
  `genai_config.json` and all referenced model components.

`MobiusBuilder` returns a `CompositeModelHandler`. Add
`CompositeToOnnxPackage` after it to preserve the package layout while exposing
an `ONNXModelHandler` to the evaluator:

```json
{
    "passes": {
        "build": {
            "type": "MobiusBuilder"
        },
        "as_onnx_package": {
            "type": "CompositeToOnnxPackage"
        }
    },
    "evaluators": {
        "multimodal_benchmarks": {
            "type": "LMMSEvaluator",
            "tasks": ["ai2d"],
            "batch_size": 1,
            "log_samples": true,
            "output_path": "results/lmms_eval.json"
        }
    },
    "evaluator": "multimodal_benchmarks"
}
```

Use `include_path` with one directory or a list of directories to load custom
lmms-eval tasks. Custom task names must not collide with built-in task names.
Public benchmark fixes should be contributed to lmms-eval rather than hidden
behind a colliding local task.

The following limitations apply:

- Raw single-file ONNX models are not supported because they do not contain the
  ORT-GenAI multimodal preprocessing pipeline.
- The ORT-GenAI adapter processes requests individually. It does not implement
  batched generation, video, multi-round or interleaved generation, beam
  search, multiple return sequences, or loglikelihood tasks.
- The adapter is registered through lmms-eval's legacy model registry for
  in-process use by Olive. It is not an lmms-eval command-line entry point.
- Automatic Hugging Face dispatch is limited to model wrappers in the pinned
  upstream lmms-eval release. Set `model_class` explicitly for another wrapper
  that is registered in the installed lmms-eval environment.
- `HfModelHandler.adapter_path` and handler `load_kwargs` are rejected rather
  than silently ignored. Merge an adapter into the checkpoint first, and pass
  wrapper-supported constructor options through `hf_model_kwargs`.

`image_serialization_profile` accepts `lossless` (PNG, the default) or
`jpeg85`. ONNX audio uses the sample rates declared by the ORT-GenAI package;
`audio_target_sample_rate` is an explicit host-resampling override.
