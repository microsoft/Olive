# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import numpy as np

from olive.evaluator.accuracy import AUC, AccuracyScore, F1Score, Perplexity, Precision, Recall


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_accuracyscore(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = AccuracyScore()
    preds = [1, 0, 1, 1]
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Accuracy().return_value = mock_res

    # execute
    actual_res = acc.measure(preds, targets)

    # assert
    mock_torch_tensor.tensor.called_once_with(preds)
    mock_torch_tensor.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_f1score(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = F1Score()
    preds = [1, 0, 1, 1]
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.F1Score().return_value = mock_res

    # execute
    actual_res = acc.measure(preds, targets)

    # assert
    mock_torch_tensor.tensor.called_once_with(preds)
    mock_torch_tensor.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_precision(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = Precision()
    preds = [1, 0, 1, 1]
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Precision().return_value = mock_res

    # execute
    actual_res = acc.measure(preds, targets)

    # assert
    mock_torch_tensor.tensor.called_once_with(preds)
    mock_torch_tensor.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_recall(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = Recall()
    preds = [1, 0, 1, 1]
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Recall().return_value = mock_res

    # execute
    actual_res = acc.measure(preds, targets)

    # assert
    mock_torch_tensor.tensor.called_once_with(preds)
    mock_torch_tensor.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_auc(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = AUC()
    preds = [1, 0, 1, 1]
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.functional.auc.return_value = mock_res

    # execute
    actual_res = acc.measure(preds, targets)

    # assert
    mock_torch_tensor.tensor.called_once_with(preds)
    mock_torch_tensor.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_perplexity(mock_torchmetrics, mock_torch_tensor):
    # setup
    Perplexity()
    batch = 2
    seqlen = 3
    vocab_size = 10
    preds = np.random.rand(batch, seqlen, vocab_size).tolist()
    targets = np.random.randint(0, vocab_size, (batch, seqlen)).tolist()
    expected_res = 20.0
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.text.perplexity.Perplexity().compute.return_value = mock_res

    # execute
    actual_res = Perplexity().measure(preds, targets)

    # assert
    for i in range(batch):
        mock_torch_tensor.tensor.called_once_with(preds[i])
        mock_torch_tensor.tensor.called_once_with(targets[i])
    assert actual_res == expected_res
