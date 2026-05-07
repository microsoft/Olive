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

The data config should use the `speech_transcription_pre_process` pre-processor:

```json
{
    "name": "speech_data_config",
    "type": "HuggingfaceContainer",
    "load_dataset_config": {
        "type": "huggingface_dataset",
        "params": {
            "data_name": "hf-audio/esb-datasets-test-only-sorted",
            "subset": "librispeech",
            "split": "test.clean"
        }
    },
    "pre_process_data_config": {
        "type": "speech_transcription_pre_process",
        "params": {
            "audio_col": "audio",
            "text_col": "text",
            "sample_rate": 16000,
            "max_samples": 100
        }
    }
}
```

### Using WER with the custom metric type

For models that require custom inference logic (e.g., streaming ASR with onnxruntime-genai), use the `custom` metric type with an `evaluate_func`:

```json
{
    "name": "speech_wer",
    "type": "custom",
    "sub_types": [
        {"name": "wer", "priority": 1, "higher_is_better": false},
        {"name": "rtfx", "priority": 2, "higher_is_better": true}
    ],
    "user_config": {
        "user_script": "my_eval_script.py",
        "evaluate_func": "evaluate_speech_wer"
    }
}
```

In your `my_eval_script.py`:

```python
import time
import numpy as np
import jiwer
from datasets import Audio, load_dataset


def evaluate_speech_wer(model, device, execution_providers):
    """Evaluate speech model and return WER and RTFx metrics."""
    # Load dataset
    dataset = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted", "librispeech",
        split="test.clean", streaming=False,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.select(range(min(100, len(dataset))))

    predictions, references = [], []
    total_audio_s, total_inference_s = 0.0, 0.0

    for sample in dataset:
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        reference_text = sample["text"].lower()
        audio_dur = len(audio_array) / 16000

        # Replace with your model's inference logic
        t0 = time.time()
        predicted_text = transcribe(model, audio_array, device, execution_providers)
        total_inference_s += time.time() - t0

        predictions.append(predicted_text)
        references.append(reference_text)
        total_audio_s += audio_dur

    wer = jiwer.wer(references, predictions)
    rtfx = total_audio_s / max(total_inference_s, 1e-9)
    return {"wer": wer, "rtfx": rtfx}


def transcribe(model, audio_array, device, execution_providers):
    """Implement model-specific transcription logic here."""
    raise NotImplementedError("Implement for your specific ASR model.")
```

```{Note}
For a complete working example with the Nemotron streaming ASR model, see the
[olive-recipes Nemotron evaluation](https://github.com/microsoft/olive-recipes/tree/main/nvidia-nemotron-speech-streaming-en-0.6b/cpu).
```
