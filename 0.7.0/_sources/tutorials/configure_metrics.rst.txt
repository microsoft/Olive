How To Configure Metric
=======================

This document describes how to configure the different types of Metrics.

Metric Types
-------------

Accuracy Metric
~~~~~~~~~~~~~~~

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

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

    .. tab:: Python Class

        .. code-block:: python

            from olive.evaluator.metric_config import MetricGoal
            from olive.evaluator.metric import AccuracySubType, Metric, MetricType
            from olive.data.config import DataComponentConfig, DataConfig

            accuracy_metric = Metric(
                name="accuracy",
                type=MetricType.ACCURACY,
                data_config=DataConfig(
                    user_script="user_script.py",
                    dataloader_config=DataComponentConfig(type="create_dataloader", params={"batch_size": 1}),
                    post_process_data_config=DataComponentConfig(type="post_process"),
                ),
                sub_types=[{
                    "name": MetricType.ACCURACY,
                    "priority": 1,
                    "goal": MetricGoal(type="max-degradation", value=0.01),
                }],
                goal={"type": "max-degradation", "value": 0.01}
            )

Please refer to this `example <https://github.com/microsoft/Olive/blob/main/examples/bert/user_script.py>`__
for :code:`"user_script.py"`.

Latency Metric
~~~~~~~~~~~~~~~

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "latency",
                "type": "latency",
                "data_config": "latency_data_config",
                "sub_types": [
                    {"name": "avg", "priority": 1, "goal": {"type": "percent-min-improvement", "value": 20}}
                ]
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.evaluator.metric_config import MetricGoal
            from olive.evaluator.metric import LatencySubType, Metric, MetricType
            from olive.data.config import DataComponentConfig, DataConfig

            latency_metric = Metric(
                name="latency",
                type=MetricType.LATENCY,
                data_config=DataConfig(
                    user_script="user_script.py",
                    dataloader_config=DataComponentConfig(type="create_dataloader", params={"batch_size": 1}),
                ),
                sub_types=sub_types = [{
                    "name": LatencySubType.AVG,
                    "goal": MetricGoal(type="percent-min-improvement", value=20),
                }],
            )

Throughput Metric
~~~~~~~~~~~~~~~~~

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "throughput",
                "type": "throughput",
                "data_config": "throughput_data_config",
                "sub_types": [
                    {"name": "avg", "priority": 1, "goal": {"type": "percent-min-improvement", "value": 20}}
                ]
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.evaluator.metric_config import MetricGoal
            from olive.evaluator.metric import ThroughputSubType, Metric, MetricType
            from olive.data.config import DataComponentConfig, DataConfig

            sub_types =
            throughput_metric = Metric(
                name="throughput",
                type=MetricType.THROUGHPUT,
                data_config=DataConfig(
                    user_script="user_script.py",
                    dataloader_config=DataComponentConfig(type="create_dataloader", params={"batch_size": 1}),
                ),
                sub_types=[{
                    "name": ThroughputSubType.AVG,
                    "goal": MetricGoal(type="percent-min-improvement", value=20),
                }],
            )


Custom Metric
~~~~~~~~~~~~~

You can define your own metric by using the :code:`"custom"` type. Your customized metric evaluation function will be defined in your own :code:`"user_script.py"`,
specify its name in :code:`"evaluate_func"` field, and Olive will call your function to evaluate the model.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

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

    .. tab:: Python Class

        .. code-block:: python

            from olive.evaluator.metric_config import MetricGoal
            from olive.evaluator.metric import Metric, MetricType

            accuracy_metric = Metric(
                name="accuracy",
                type=MetricType.CUSTOM,
                sub_types=[{
                    "name": "accuracy_custom",
                    "priority": 1,
                    "higher_is_better": True,
                    "goal": MetricGoal(type="max-degradation", value=0.01),
                }],
                user_config={
                    "user_script": "user_script.py",
                    "evaluate_func": "eval_accuracy",
                    "evaluate_func_kwargs": {
                        "data_dir": "data",
                        "batch_size": 16,
                    }
                }
            )

Please refer to this `example <https://github.com/microsoft/Olive/blob/main/examples/resnet/user_script.py>`__
for :code:`"user_script.py"`.

Here is an example of the :code:`"eval_accuracy"` function in :code:`"user_script.py"`:
In your :code:`"user_script.py"`, you need to define a function that takes in an Olive model, the data directory, and the batch size, and returns a metric value::

        def eval_accuracy(model, device, execution_providers):
            # load data
            # evaluate model
            # return metric value

Alternatively, if you only need Olive run the inference and you will calculate the metric by yourself, you can specify :code:`"metric_func": "None"` in the metric configuration.
Olive will run the inference with you model with the data you provided, and return the inference results to you. You can then calculate the metric by yourself::

        def metric_func(model_output, targets):
            # model_output[0]: preds, model_output[1]: logits
            # calculate metric
            # return metric value

If you provide both :code:`"evaluate_func"` and :code:`"metric_func"`, Olive will call :code:`"evaluate_func"` only.

Multi Metrics configuration
----------------------------
If you have multiple metrics to evaluate, you can configure them in the following way::

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

You need to specify :code:`"priority": <rank>` for the metrics if you have multiple metrics.
Olive will use the priorities of the metrics to determine the best model.
If you only have one metric, you can omit :code:`"priority": 1`.
