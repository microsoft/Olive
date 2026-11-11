# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import pytest
import torch

from olive.evaluator.accuracy import AUROC, Perplexity
from olive.evaluator.metric import Metric
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveModelOutput


@pytest.mark.parametrize("as_input", [list, np.asarray, torch.tensor], ids=["list", "numpy", "tensor"])
@pytest.mark.parametrize(
    ("preds", "targets", "metric_config"),
    [
        ([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0], {"task": "binary"}),
        ([-2.0, 2.0, -1.0, 1.0], [0, 1, 0, 1], {"task": "binary"}),
        ([0.8, 0.7, 0.2, 0.1], [1, 0, 0, 0], {"task": "binary", "threshold": 0.75}),
        (
            [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7]],
            [1, 0, 2],
            {"task": "multiclass", "num_classes": 3},
        ),
        ([[0.9, 0.1], [0.2, 0.8]], [[1, 0], [0, 1]], {"task": "multilabel", "num_labels": 2}),
        ([1, 0, 1, 0], [1, 0, 1, 0], {"task": "binary"}),
        ([0.0, 1.0, 2.0], [0, 1, 2], {"task": "multiclass", "num_classes": 3}),
        ([[1.0, 0.0], [0.0, 1.0]], [[1, 0], [0, 1]], {"task": "multilabel", "num_labels": 2}),
    ],
    ids=[
        "binary-probabilities",
        "binary-logits",
        "custom-threshold",
        "multiclass",
        "multilabel",
        "integer-labels",
        "multiclass-float-labels",
        "multilabel-float-labels",
    ],
)
def test_classification_scores_preserve_prediction_values(preds, targets, metric_config, as_input):
    names = ["accuracy_score", "f1_score", "precision", "recall"]
    metric = Metric(
        name="classification",
        type="accuracy",
        sub_types=[{"name": name, "metric_config": dict(metric_config)} for name in names],
    )

    results = OliveEvaluator.compute_accuracy(metric, OliveModelOutput(as_input(preds), None), targets)

    for name in names:
        assert results[name].value == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.uint16, np.uint32, np.uint64, np.int64])
@pytest.mark.parametrize("as_tensor", [False, True], ids=["numpy", "tensor"])
def test_classification_hard_labels_keep_integer_conversion(dtype, as_tensor):
    preds = np.asarray([1, 0, 1, 0], dtype=dtype)
    if as_tensor:
        preds = torch.tensor(preds)
    metric = Metric(
        name="classification",
        type="accuracy",
        sub_types=[{"name": name} for name in ("accuracy_score", "f1_score", "precision", "recall")],
    )

    results = OliveEvaluator.compute_accuracy(metric, OliveModelOutput(preds, None), [1, 0, 1, 0])

    for result in results.values():
        assert result.value == pytest.approx(1.0)


def test_classification_probabilities_keep_fractional_scores():
    expected = {"accuracy_score": 2 / 5, "f1_score": 2 / 5, "precision": 1 / 3, "recall": 1 / 2}
    metric = Metric(name="classification", type="accuracy", sub_types=[{"name": name} for name in expected])

    results = OliveEvaluator.compute_accuracy(
        metric, OliveModelOutput([0.9, 0.8, 0.7, 0.1, 0.2], None), [1, 0, 0, 1, 0]
    )

    for name, value in expected.items():
        assert results[name].value == pytest.approx(value)


@pytest.mark.parametrize(
    ("average", "expected"),
    [
        ("micro", {"accuracy_score": 4 / 5, "f1_score": 4 / 5, "precision": 4 / 5, "recall": 4 / 5}),
        ("macro", {"accuracy_score": 5 / 6, "f1_score": 37 / 45, "precision": 8 / 9, "recall": 5 / 6}),
    ],
)
def test_multiclass_probabilities_honor_average(average, expected):
    metric = Metric(
        name="classification",
        type="accuracy",
        sub_types=[
            {"name": name, "metric_config": {"task": "multiclass", "num_classes": 3, "average": average}}
            for name in expected
        ],
    )
    output = OliveModelOutput(
        [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], None
    )

    results = OliveEvaluator.compute_accuracy(metric, output, [0, 0, 1, 1, 2])

    for name, value in expected.items():
        assert results[name].value == pytest.approx(value)


def test_auroc_keeps_explicit_float_conversion_for_two_class_logits():
    metric = AUROC({"task": "binary"})
    output = OliveModelOutput(None, [[1, 4], [4, 1], [2, 5], [5, 2]])

    assert metric.measure(output, [1, 0, 1, 0]) == pytest.approx(1.0)


def test_perplexity_keeps_explicit_logit_and_target_conversion():
    metric = Perplexity()
    output = OliveModelOutput([[[0, 0], [0, 0], [0, 0]]], None)

    assert metric.measure(output, [[0, 1, 0]]) == pytest.approx(2.0)
