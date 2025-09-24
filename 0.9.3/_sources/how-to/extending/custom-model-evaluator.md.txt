# How to Add Custom Model Evaluator

This document describes how to add custom model evaluator to Olive.

Olive implements evaluation logic via class (`OliveEvaluator`)[https://github.com/microsoft/Olive/blob/main/olive/evaluator/olive_evaluator.py#L59]. Users can extend this implementation and override the `evaluate` method to implement their own custom logic.

Arguments to the `OliveEvaluator.evaluate` function:
- `model: OliveModelHandler`: Model to evaluate.
- `metrics: List[Metric]`: List of metrics to evaluate for.
- `device: Device`: Target device to evaluate on.
- `execution_providers: Union[str, List[str]]`: Evaluation provider(s) to use for the target device.

Here is an example of how to subclass (`OliveEvaluator`)[https://github.com/microsoft/Olive/blob/main/olive/evaluator/olive_evaluator.py#L59]:

```python
from typing import List, Union

from olive.evaluator.registry import Registry
from olive.evaluator import (
    Metric,
    MetricResult,
    OliveEvaluator,
    flatten_metric_result,
)
from olive.hardware import Device
from olive.model.handler import OliveModelHandler

@Registry.register("my_custom_evaluator")
def my_custom_evaluator(**kwargs):
    return MyCustomEvaluator(**kwargs)

class MyCustomEvaluator(OliveEvaluator):
  def evaluate(
        self,
        model: OliveModelHandler,
        metrics: List[Metric],
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
      # Your custom evaluation logic goes in here

      # Olive expects the return to be flattened
      return flatten_metric_result(metrics)
```

For detailed information on supported metric types, refer to [`Metric`](../configure-workflows/metrics-configuration.md#metric-types). `MetricResult` holds the final result of evaluation. Snippet below demonstrates how to build the final results from computed metrics.

```json
{
    "metrics":[
        {
            "name": "accuracy",
            "type": "accuracy",
            "sub_types": [
                {"name": "acc", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
                {"name": "f1"},
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
```python
final_metric_results = {}
for metric_name, eval_results in results.items():
    sub_metrics = {}
    for sub_metric_name, sub_metric_result in eval_results.items():
        sub_metrics[m] = SubMetricResult(value=sub_metric_result, priority=-1, higher_is_better=True)

    final_metric_results[metric_name] = MetricResult.parse_obj(sub_metrics)

return flatten_metric_result(final_metric_results)
```

`LMEvaluator` in (olive_evaluator.py)[https://github.com/microsoft/Olive/blob/main/olive/evaluator/olive_evaluator.py#L1068] is a good example that demonstrate evaluating a HuggingFace model using (`lm_eval`)[https://github.com/EleutherAI/lm-evaluation-harness] harness.
