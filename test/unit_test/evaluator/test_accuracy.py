# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

from olive.evaluator.accuracy import AUC, AccuracyScore, F1Score, Precision, Recall


@patch("olive.evaluator.accuracy.torch")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_accuracyscore(mock_torchmetrics, mock_torch):
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
    mock_torch.tensor.called_once_with(preds)
    mock_torch.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_f1score(mock_torchmetrics, mock_torch):
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
    mock_torch.tensor.called_once_with(preds)
    mock_torch.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_precision(mock_torchmetrics, mock_torch):
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
    mock_torch.tensor.called_once_with(preds)
    mock_torch.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_recall(mock_torchmetrics, mock_torch):
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
    mock_torch.tensor.called_once_with(preds)
    mock_torch.tensor.called_once_with(targets)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_auc(mock_torchmetrics, mock_torch):
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
    mock_torch.tensor.called_once_with(preds)
    mock_torch.tensor.called_once_with(targets)
    assert actual_res == expected_res
