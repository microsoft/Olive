# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import abstractmethod
from inspect import isfunction, signature
from typing import Any, Callable, ClassVar, Optional, Union

import torch
import torchmetrics

from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.config_utils import ConfigParam
from olive.data.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


class AccuracyBase(AutoConfigClass):
    registry: ClassVar[dict[str, type["AccuracyBase"]]] = {}
    metric_cls_map: ClassVar[dict[str, Union[torchmetrics.Metric, Callable]]] = {
        "accuracy_score": torchmetrics.Accuracy,
        "f1_score": torchmetrics.F1Score,
        "precision": torchmetrics.Precision,
        "recall": torchmetrics.Recall,
        "auroc": torchmetrics.AUROC,
        "perplexity": torchmetrics.text.perplexity.Perplexity,
        "wer": torchmetrics.text.WordErrorRate,
    }

    def __init__(self, config: Optional[Union[ConfigBase, dict[str, Any]]] = None) -> None:
        super().__init__(config or {})
        self.resolve_kwargs()

    def resolve_kwargs(self):
        config_dict = self.config.model_dump()
        kwargs = config_dict.pop("kwargs", {})
        config_dict.update(kwargs or {})
        self.config_dict = config_dict

    @classmethod
    def _metric_config_from_torch_metrics(cls):
        metric_module = cls.metric_cls_map[cls.name]
        params = signature(metric_module).parameters
        # if the metrics is calculated by torchmetrics.functional, we should filter the label data out
        ignore_idx = 0
        if isfunction(metric_module):
            ignore_idx = 2
            logger.debug("Will ignore the first two params of torchmetrics.functional.")
        metric_config = {}
        for param, info in params.items():
            if ignore_idx > 0:
                ignore_idx -= 1
                continue
            annotation = info.annotation if info.annotation != info.empty else None
            default_value, required = (info.default, False) if info.default != info.empty else (None, True)
            if info.kind in (info.VAR_KEYWORD, info.VAR_POSITIONAL):
                required = False
            metric_config[param] = ConfigParam(type_=annotation, required=required, default_value=default_value)
        if "task" in metric_config:
            metric_config["task"].default_value = "binary"
            metric_config["task"].required = False
        return metric_config

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        return cls._metric_config_from_torch_metrics()

    @staticmethod
    def prepare_tensors(preds, target, dtypes: Union[torch.dtype, list[torch.dtype], tuple[torch.dtype]] = torch.int):
        dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes, dtypes]
        assert len(dtypes) == 2, "dtypes should be a list or tuple with two elements."
        preds = torch.tensor(preds, dtype=dtypes[0]) if not isinstance(preds, torch.Tensor) else preds.to(dtypes[0])
        target = torch.tensor(target, dtype=dtypes[1]) if not isinstance(target, torch.Tensor) else target.to(dtypes[1])
        return preds, target

    @abstractmethod
    def measure(self, model_output, target):
        raise NotImplementedError


class AccuracyScore(AccuracyBase):
    name: Optional[str] = "accuracy_score"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        accuracy = torchmetrics.Accuracy(**self.config_dict)
        result = accuracy(preds_tensor, target_tensor)
        return result.item()


class F1Score(AccuracyBase):
    name: Optional[str] = "f1_score"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        f1 = torchmetrics.F1Score(**self.config_dict)
        result = f1(preds_tensor, target_tensor)
        return result.item()


class Precision(AccuracyBase):
    name: Optional[str] = "precision"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        precision = torchmetrics.Precision(**self.config_dict)
        result = precision(preds_tensor, target_tensor)
        return result.item()


class Recall(AccuracyBase):
    name: Optional[str] = "recall"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        recall = torchmetrics.Recall(**self.config_dict)
        result = recall(preds_tensor, target_tensor)
        return result.item()


class AUROC(AccuracyBase):
    name: Optional[str] = "auroc"

    def measure(self, model_output, target):
        logits_tensor, target_tensor = self.prepare_tensors(model_output.logits, target, [torch.float, torch.int32])
        if self.config_dict.get("task") == "binary" and len(logits_tensor.shape) > 1 and logits_tensor.shape[-1] == 2:
            logits_tensor = torch.softmax(logits_tensor, dim=-1)[:, 1]
        auroc = torchmetrics.AUROC(**self.config_dict)
        target_tensor = target_tensor.flatten()
        result = auroc(logits_tensor, target_tensor)
        return result.item()


class Perplexity(AccuracyBase):
    name: Optional[str] = "perplexity"

    def measure(self, model_output, target):
        # update ignore_index if not set
        config = self.config_dict
        if config["ignore_index"] is None:
            config["ignore_index"] = IGNORE_INDEX

        # create perplexity metric
        perplexity = torchmetrics.text.perplexity.Perplexity(**config)

        # loop through samples
        # the logits are large matrix, so converting all to tensors at once is slow
        num_samples = len(model_output.preds)
        for i in range(num_samples):
            logits, targets = self.prepare_tensors(model_output.preds[i], target[i], dtypes=[torch.float, torch.long])
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)
            # shift targets to the right by one, and drop the last token of logits
            logits = logits[..., :-1, :]
            targets = targets[..., 1:]
            perplexity.update(logits, targets)
        result = perplexity.compute()
        return result.item()


class WordErrorRate(AccuracyBase):
    """Word Error Rate metric for speech/ASR evaluation.

    Expects model_output.preds to be a list of predicted transcription strings
    and target to be a list of reference transcription strings.
    """

    name: Optional[str] = "wer"

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        return {}

    def measure(self, model_output, target):
        preds = model_output.preds
        refs = target
        # Ensure inputs are lists of strings
        if isinstance(preds, str):
            preds = [preds]
        elif not isinstance(preds, list):
            preds = list(preds)
        if isinstance(refs, str):
            refs = [refs]
        elif not isinstance(refs, list):
            refs = list(refs)

        wer = torchmetrics.text.WordErrorRate(**self.config_dict)
        result = wer(preds, refs)
        return result.item()


class RealTimeFactor(AccuracyBase):
    """Real-Time Factor (RTFx) metric for speech/ASR evaluation.

    RTFx = total_audio_duration / total_inference_time.
    A value > 1 means faster than real-time (e.g., RTFx=5 means 5x faster).
    Timing metadata is provided via model_output.logits dict.
    """

    name: Optional[str] = "rtfx"

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        return {}

    def measure(self, model_output, target):
        timing = model_output.logits
        if not isinstance(timing, dict) or "total_audio_duration" not in timing:
            raise ValueError(
                "RTFx metric requires timing metadata from text-based inference path. "
                "Ensure the metric is used with speech evaluation (WER + RTFx together)."
            )
        total_audio = timing["total_audio_duration"]
        total_inference = timing["total_inference_time"]
        if total_inference == 0:
            return float("inf")
        return round(total_audio / total_inference, 2)


class ExactMatch(AccuracyBase):
    """Exact match metric for vision VQA evaluation.

    Compares predicted answer strings to ground truth answers using
    case-insensitive, whitespace-normalized string equality.
    Returns the fraction of samples with an exact match.
    """

    name: Optional[str] = "exact_match"

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        return {}

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison: lowercase and collapse whitespace."""
        return " ".join(text.strip().lower().split())

    def measure(self, model_output, target):
        preds = model_output.preds
        refs = target
        if isinstance(preds, str):
            preds = [preds]
        elif not isinstance(preds, list):
            preds = list(preds)
        if isinstance(refs, str):
            refs = [refs]
        elif not isinstance(refs, list):
            refs = list(refs)

        if len(preds) != len(refs):
            raise ValueError(
                f"Number of predictions ({len(preds)}) does not match "
                f"number of references ({len(refs)}) for exact_match metric."
            )

        correct = sum(1 for p, r in zip(preds, refs) if self._normalize(str(p)) == self._normalize(str(r)))
        return correct / len(refs) if refs else 0.0


class RelaxedAccuracy(AccuracyBase):
    """Relaxed accuracy metric for chart/math VQA evaluation.

    For numeric answers, allows a ±5% tolerance (standard for ChartQA).
    For non-numeric answers, falls back to exact string match.
    Returns the fraction of samples that match within tolerance.
    """

    name: Optional[str] = "relaxed_accuracy"

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        return {
            "tolerance": ConfigParam(type_=float, required=False, default_value=0.05),
        }

    @staticmethod
    def _try_parse_number(text: str):
        """Try to parse text as a number. Returns (True, value) or (False, None)."""
        text = text.strip().replace(",", "").replace("%", "")
        try:
            return True, float(text)
        except ValueError:
            return False, None

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def measure(self, model_output, target):
        preds = model_output.preds
        refs = target
        if isinstance(preds, str):
            preds = [preds]
        elif not isinstance(preds, list):
            preds = list(preds)
        if isinstance(refs, str):
            refs = [refs]
        elif not isinstance(refs, list):
            refs = list(refs)

        if len(preds) != len(refs):
            raise ValueError(
                f"Number of predictions ({len(preds)}) does not match "
                f"number of references ({len(refs)}) for relaxed_accuracy metric."
            )

        tolerance = self.config_dict.get("tolerance", 0.05)
        correct = 0
        for pred, ref in zip(preds, refs):
            pred_str = str(pred)
            ref_str = str(ref)
            pred_is_num, pred_val = self._try_parse_number(pred_str)
            ref_is_num, ref_val = self._try_parse_number(ref_str)

            if pred_is_num and ref_is_num:
                # Numeric comparison with tolerance
                if ref_val == 0:
                    if pred_val == 0:
                        correct += 1
                elif abs(pred_val - ref_val) / abs(ref_val) <= tolerance:
                    correct += 1
            else:
                # String comparison (exact match, case-insensitive)
                if self._normalize(pred_str) == self._normalize(ref_str):
                    correct += 1

        return correct / len(refs) if refs else 0.0


class WordSortRatio(AccuracyBase):
    """Word sort ratio metric for OCR evaluation.

    Computes the ratio of matching words between prediction and reference
    after sorting words alphabetically. This measures word-level overlap
    regardless of word order.
    Returns the average ratio across all samples.
    """

    name: Optional[str] = "word_sort_ratio"

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        return {}

    @staticmethod
    def _compute_word_sort_ratio(pred: str, ref: str) -> float:
        """Compute word sort ratio between two strings."""
        pred_words = sorted(pred.strip().lower().split())
        ref_words = sorted(ref.strip().lower().split())

        if not ref_words:
            return 1.0 if not pred_words else 0.0

        # Count matching words using multiset intersection
        from collections import Counter

        pred_counter = Counter(pred_words)
        ref_counter = Counter(ref_words)
        intersection = sum((pred_counter & ref_counter).values())
        total = max(len(pred_words), len(ref_words))
        return intersection / total if total > 0 else 0.0

    def measure(self, model_output, target):
        preds = model_output.preds
        refs = target
        if isinstance(preds, str):
            preds = [preds]
        elif not isinstance(preds, list):
            preds = list(preds)
        if isinstance(refs, str):
            refs = [refs]
        elif not isinstance(refs, list):
            refs = list(refs)

        if len(preds) != len(refs):
            raise ValueError(
                f"Number of predictions ({len(preds)}) does not match "
                f"number of references ({len(refs)}) for word_sort_ratio metric."
            )

        total_ratio = sum(self._compute_word_sort_ratio(str(p), str(r)) for p, r in zip(preds, refs))
        return total_ratio / len(refs) if refs else 0.0
