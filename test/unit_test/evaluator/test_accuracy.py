# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from olive.evaluator.accuracy import AUROC, AccuracyScore, F1Score, Perplexity, Precision, Recall
from olive.evaluator.olive_evaluator import OliveModelOutput


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
@pytest.mark.parametrize(
    "metric_config",
    [
        {"task": "binary"},
        {"task": "binary", "kwargs": {"test": 2}},
    ],
)
def test_evaluate_accuracyscore(mock_torchmetrics, mock_torch_tensor, metric_config):
    # setup
    acc = AccuracyScore(metric_config)
    assert "kwargs" in acc.config.dict()
    assert "kwargs" not in acc.config_dict
    model_output = OliveModelOutput([1, 0, 1, 1], None)
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Accuracy().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.preds, dtype=torch.int32)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_f1score(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = F1Score()
    model_output = OliveModelOutput([1, 0, 1, 1], None)
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.F1Score().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.preds, dtype=torch.int32)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_precision(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = Precision()
    model_output = OliveModelOutput([1, 0, 1, 1], None)
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Precision().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.preds, dtype=torch.int32)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_recall(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = Recall()
    model_output = OliveModelOutput([1, 0, 1, 1], None)
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Recall().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.preds, dtype=torch.int32)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_auc(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = AUROC()
    model_output = OliveModelOutput(None, [1, 0, 1, 1])
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.AUROC().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.logits, dtype=torch.float)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_perplexity(mock_torchmetrics, mock_torch_tensor):
    # setup
    Perplexity()
    batch = 2
    seqlen = 3
    vocab_size = 10
    model_output = OliveModelOutput(np.random.rand(batch, seqlen, vocab_size).tolist(), None)
    targets = np.random.randint(0, vocab_size, (batch, seqlen)).tolist()
    expected_res = 20.0
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.text.perplexity.Perplexity().compute.return_value = mock_res

    # execute
    actual_res = Perplexity().measure(model_output, targets)

    # assert
    for i in range(batch):
        mock_torch_tensor.assert_any_call(model_output.preds[i], dtype=torch.float)
        mock_torch_tensor.assert_any_call(targets[i], dtype=torch.long)
    assert actual_res == expected_res
