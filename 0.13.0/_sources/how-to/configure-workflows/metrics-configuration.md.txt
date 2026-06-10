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
