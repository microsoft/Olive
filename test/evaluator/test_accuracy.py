# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from olive.evaluator.accuracy import (
    AUROC,
    AccuracyScore,
    F1Score,
    Perplexity,
    Precision,
    RealTimeFactor,
    Recall,
    WordErrorRate,
)
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
    assert "kwargs" in acc.config.model_dump()
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


class TestWordErrorRate:
    def test_perfect_transcription(self):
        wer = WordErrorRate({})
        model_output = OliveModelOutput(preds=["hello world", "test sentence"], logits=None)
        targets = ["hello world", "test sentence"]
        result = wer.measure(model_output, targets)
        assert result == 0.0

    def test_completely_wrong(self):
        wer = WordErrorRate({})
        model_output = OliveModelOutput(preds=["completely wrong words here"], logits=None)
        targets = ["the correct reference text"]
        result = wer.measure(model_output, targets)
        assert result > 0.0

    def test_single_string_input(self):
        """Test that a single string is wrapped in a list, not split into chars."""
        wer = WordErrorRate({})
        model_output = OliveModelOutput(preds="hello world", logits=None)
        targets = "hello world"
        result = wer.measure(model_output, targets)
        assert result == 0.0

    def test_partial_error(self):
        wer = WordErrorRate({})
        model_output = OliveModelOutput(preds=["hello world"], logits=None)
        targets = ["hello earth"]
        result = wer.measure(model_output, targets)
        assert 0.0 < result < 1.0


class TestRealTimeFactor:
    def test_rtfx_computation(self):
        rtfx = RealTimeFactor({})
        # 10 seconds of audio processed in 2 seconds = RTFx 5.0
        timing = {"total_audio_duration": 10.0, "total_inference_time": 2.0}
        model_output = OliveModelOutput(preds=["some text"], logits=timing)
        result = rtfx.measure(model_output, ["some text"])
        assert result == 5.0

    def test_rtfx_realtime(self):
        rtfx = RealTimeFactor({})
        # 5 seconds of audio processed in 5 seconds = RTFx 1.0
        timing = {"total_audio_duration": 5.0, "total_inference_time": 5.0}
        model_output = OliveModelOutput(preds=["text"], logits=timing)
        result = rtfx.measure(model_output, ["text"])
        assert result == 1.0

    def test_rtfx_missing_metadata(self):
        rtfx = RealTimeFactor({})
        model_output = OliveModelOutput(preds=["text"], logits=None)
        with pytest.raises(ValueError, match="RTFx metric requires timing metadata"):
            rtfx.measure(model_output, ["text"])


class TestExactMatch:
    def test_perfect_match(self):
        from olive.evaluator.accuracy import ExactMatch

        metric = ExactMatch({})
        model_output = OliveModelOutput(preds=["A", "B", "C"], logits=None)
        targets = ["A", "B", "C"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_case_insensitive(self):
        from olive.evaluator.accuracy import ExactMatch

        metric = ExactMatch({})
        model_output = OliveModelOutput(preds=["Hello World", "TEST"], logits=None)
        targets = ["hello world", "test"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_whitespace_normalization(self):
        from olive.evaluator.accuracy import ExactMatch

        metric = ExactMatch({})
        model_output = OliveModelOutput(preds=["  hello   world  "], logits=None)
        targets = ["hello world"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_partial_match(self):
        from olive.evaluator.accuracy import ExactMatch

        metric = ExactMatch({})
        model_output = OliveModelOutput(preds=["A", "wrong", "C"], logits=None)
        targets = ["A", "B", "C"]
        result = metric.measure(model_output, targets)
        assert abs(result - 2.0 / 3.0) < 1e-6

    def test_no_match(self):
        from olive.evaluator.accuracy import ExactMatch

        metric = ExactMatch({})
        model_output = OliveModelOutput(preds=["X", "Y"], logits=None)
        targets = ["A", "B"]
        result = metric.measure(model_output, targets)
        assert result == 0.0

    def test_single_string_input(self):
        from olive.evaluator.accuracy import ExactMatch

        metric = ExactMatch({})
        model_output = OliveModelOutput(preds="answer", logits=None)
        targets = "answer"
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_length_mismatch_raises(self):
        from olive.evaluator.accuracy import ExactMatch

        metric = ExactMatch({})
        model_output = OliveModelOutput(preds=["A", "B"], logits=None)
        targets = ["A"]
        with pytest.raises(ValueError, match="does not match"):
            metric.measure(model_output, targets)


class TestRelaxedAccuracy:
    def test_exact_numeric_match(self):
        from olive.evaluator.accuracy import RelaxedAccuracy

        metric = RelaxedAccuracy({})
        model_output = OliveModelOutput(preds=["42.0"], logits=None)
        targets = ["42.0"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_within_tolerance(self):
        from olive.evaluator.accuracy import RelaxedAccuracy

        metric = RelaxedAccuracy({})
        # 41 is within 5% of 42 (42 * 0.05 = 2.1, |42-41| = 1 < 2.1)
        model_output = OliveModelOutput(preds=["41"], logits=None)
        targets = ["42"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_outside_tolerance(self):
        from olive.evaluator.accuracy import RelaxedAccuracy

        metric = RelaxedAccuracy({})
        # 35 is outside 5% of 42 (42 * 0.05 = 2.1, |42-35| = 7 > 2.1)
        model_output = OliveModelOutput(preds=["35"], logits=None)
        targets = ["42"]
        result = metric.measure(model_output, targets)
        assert result == 0.0

    def test_string_exact_match(self):
        from olive.evaluator.accuracy import RelaxedAccuracy

        metric = RelaxedAccuracy({})
        model_output = OliveModelOutput(preds=["yes"], logits=None)
        targets = ["yes"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_string_no_match(self):
        from olive.evaluator.accuracy import RelaxedAccuracy

        metric = RelaxedAccuracy({})
        model_output = OliveModelOutput(preds=["yes"], logits=None)
        targets = ["no"]
        result = metric.measure(model_output, targets)
        assert result == 0.0

    def test_percentage_values(self):
        from olive.evaluator.accuracy import RelaxedAccuracy

        metric = RelaxedAccuracy({})
        # 50% parsed as 50, within 5% of 50
        model_output = OliveModelOutput(preds=["51%"], logits=None)
        targets = ["50%"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_zero_target(self):
        from olive.evaluator.accuracy import RelaxedAccuracy

        metric = RelaxedAccuracy({})
        model_output = OliveModelOutput(preds=["0"], logits=None)
        targets = ["0"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_custom_tolerance(self):
        from olive.evaluator.accuracy import RelaxedAccuracy

        metric = RelaxedAccuracy({"tolerance": 0.1})
        # 38 is within 10% of 42 (42 * 0.1 = 4.2, |42-38| = 4 < 4.2)
        model_output = OliveModelOutput(preds=["38"], logits=None)
        targets = ["42"]
        result = metric.measure(model_output, targets)
        assert result == 1.0


class TestWordSortRatio:
    def test_perfect_match(self):
        from olive.evaluator.accuracy import WordSortRatio

        metric = WordSortRatio({})
        model_output = OliveModelOutput(preds=["hello world"], logits=None)
        targets = ["hello world"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_reordered_words(self):
        from olive.evaluator.accuracy import WordSortRatio

        metric = WordSortRatio({})
        # Same words, different order → ratio = 1.0 (sorted comparison)
        model_output = OliveModelOutput(preds=["world hello"], logits=None)
        targets = ["hello world"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_partial_overlap(self):
        from olive.evaluator.accuracy import WordSortRatio

        metric = WordSortRatio({})
        # "hello" matches, "earth" doesn't match "world"
        model_output = OliveModelOutput(preds=["hello earth"], logits=None)
        targets = ["hello world"]
        result = metric.measure(model_output, targets)
        assert 0.0 < result < 1.0

    def test_no_overlap(self):
        from olive.evaluator.accuracy import WordSortRatio

        metric = WordSortRatio({})
        model_output = OliveModelOutput(preds=["foo bar"], logits=None)
        targets = ["hello world"]
        result = metric.measure(model_output, targets)
        assert result == 0.0

    def test_case_insensitive(self):
        from olive.evaluator.accuracy import WordSortRatio

        metric = WordSortRatio({})
        model_output = OliveModelOutput(preds=["HELLO WORLD"], logits=None)
        targets = ["hello world"]
        result = metric.measure(model_output, targets)
        assert result == 1.0

    def test_empty_reference(self):
        from olive.evaluator.accuracy import WordSortRatio

        metric = WordSortRatio({})
        model_output = OliveModelOutput(preds=[""], logits=None)
        targets = [""]
        result = metric.measure(model_output, targets)
        assert result == 1.0
