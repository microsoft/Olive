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


Multi Metrics configuration
---------
If you have multiple metrics to evaluate, you can configure them in the following way::

        {
            "metrics": [
                {
                    "name": "accuracy",
                    "type": "accuracy",
                    "sub_type": "accuracy_score",
                    "is_first_priority": true,
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

You need to specify :code:`"is_first_priority": true` for the first priority metric if you have multiple metrics.
The first priority metric is the metric that Olive will use to determine the best model.
If you only have one metric, you can omit :code:`"is_first_priority": true`.
