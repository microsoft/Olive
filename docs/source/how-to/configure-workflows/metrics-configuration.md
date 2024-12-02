# How To Configure Metrics

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
Please refer to [this `user_script.py`](https://github.com/microsoft/Olive/blob/main/examples/resnet/user_script.py) for a detailed example of how to set up a custom metric.
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
