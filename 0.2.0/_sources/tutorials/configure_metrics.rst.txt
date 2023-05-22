Configuring Metric
===================

This document describes how to configure the different types of Metrics.

Metric Types
---------

Accuracy Metric
~~~~~~~~~~~~~~~

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "accuracy",
                "type": "accuracy",
                "sub_type": "accuracy_score",
                "user_config": {
                    "post_processing_func": "post_process",
                    "user_script": "user_script.py",
                    "dataloader_func": "create_dataloader",
                    "batch_size": 1
                },
                "goal": {
                    "type": "max-degradation",
                    "value": 0.01
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.evaluator.metric import AccuracySubType, Metric, MetricType

            accuracy_metric = Metric(
                name="accuracy",
                type=MetricType.ACCURACY,
                sub_type=AccuracySubType.ACCURACY_SCORE,
                user_config={
                    "user_script": "user_script.py",
                    "post_processing_func": "post_process",
                    "dataloader_func": "create_dataloader",
                    "batch_size": 1,
                },
                goal={"type": "max-degradation", "value": 0.01}
            )

Please refer to this `example <https://github.com/microsoft/Olive/blob/main/examples/bert_ptq_cpu/user_script.py>`_
for :code:`"user_script.py"`.

Latency Metric
~~~~~~~~~~~~~~~

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "latency",
                "type": "latency",
                "sub_type": "avg",
                "user_config": {
                    "user_script": "user_script.py",
                    "dataloader_func": "create_dataloader",
                    "batch_size": 1
                },
                "goal": {
                    "type": "percent-min-improvement",
                    "value": 20
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.evaluator.metric import LatencySubType, Metric, MetricType

            latency_metric = Metric(
                name="latency",
                type=MetricType.LATENCY,
                sub_type=LatencySubType.AVG,
                user_config={
                    "user_script": user_script,
                    "dataloader_func": "create_dataloader",
                    "batch_size": 1,
                },
                goal={"type": "percent-min-improvement", "value": 20},
            )

Please refer to this `example <https://github.com/microsoft/Olive/blob/main/examples/bert_ptq_cpu/user_script.py>`_
for :code:`"user_script.py"`.

Custom Metric
~~~~~~~~~~~~~

You can define your own metric by using the :code:`"custom"` type. Your custome metric evaluation function will be defined in your own :code:`"user_script.py"`,
sepcify its name in :code:`"evaluate_func"` field, and Olive will call your function to evaluate the model.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "accuracy",
                "type": "custom",
                "user_config": {
                    "user_script": "user_script.py",
                    "data_dir": "data",
                    "batch_size": 16,
                    "evaluate_func": "eval_accuracy",
                },
                "goal": {
                    "type": "max-degradation",
                    "value": 0.01
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.evaluator.metric import Metric, MetricType

            accuracy_metric = Metric(
                name="accuracy",
                type=MetricType.CUSTOM,
                higher_is_better=True,
                user_config={
                    "user_script": "user_script.py",
                    "data_dir": "data",
                    "batch_size": 16,
                    "evaluate_func": "eval_accuracy",
                }
                goal={"type": "max-degradation", "value": 0.01},
            )

Please refer to this `example <https://github.com/microsoft/Olive/blob/main/examples/resnet_ptq_cpu/user_script.py>`_
for :code:`"user_script.py"`.

Here is an example of the :code:`"eval_accuracy"` function in :code:`"user_script.py"`:
In your :code:`"user_script.py"`, you need to define a function that takes in an Olive model, the data directory, and the batch size, and returns a metric value::

        def eval_accuracy(model, data_dir, batch_size):
            # load data
            # evaluate model
            # return metric value


Multi Metrics configuration
---------
If you have multiple metrics to evaluate, you can configure them in the following way::

        {
            "metrics": [
                {
                    "name": "accuracy",
                    "type": "accuracy",
                    "sub_type": "accuracy_score",
                    "priority_rank": 1,
                    "user_config": {
                        "post_processing_func": "post_process",
                        "user_script": "user_script.py",
                        "dataloader_func": "create_dataloader",
                        "batch_size": 1
                    },
                    "goal": {
                        "type": "max-degradation",
                        "value": 0.01
                    }
                },
                {
                    "name": "latency",
                    "type": "latency",
                    "sub_type": "avg",
                    "priority_rank": 2,
                    "user_config": {
                        "user_script": "user_script.py",
                        "dataloader_func": "create_dataloader",
                        "batch_size": 1
                    },
                    "goal": {
                        "type": "percent-min-improvement",
                        "value": 20
                    }
                }
            ]
        }

You need to specify :code:`"priority_rank": <rank>` for the metrics if you have multiple metrics.
Olive will use the priority_ranks of the metrics to determine the best model.
If you only have one metric, you can omit :code:`"priority_rank": 1`.
