# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from inspect import isfunction, signature
from typing import Any, ClassVar, Dict, Type, Union

import torch
import torchmetrics

from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.config_utils import ConfigParam
from olive.data.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


class AccuracyBase(AutoConfigClass):
    registry: ClassVar[Dict[str, Type["AccuracyBase"]]] = {}
    metric_module: ClassVar[torchmetrics.Metric] = None

    def __init__(self, config: Union[ConfigBase, Dict[str, Any]] = None):
        super().__init__(config)

        config_dict = self.config.dict()
        kwargs = config_dict.pop("kwargs", {})
        config_dict.update(kwargs or {})
        self.metric = self.metric_module(**config_dict)  # pylint: disable=not-callable

    @classmethod
    def _metric_config_from_torch_metrics(cls):
        params = signature(cls.metric_module).parameters
        # if the metrics is calculated by torchmetrics.functional, we should filter the label data out
        ignore_idx = 0
        if isfunction(cls.metric_module):
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
        if "ignore_index" in metric_config:
            metric_config["ignore_index"].default_value = IGNORE_INDEX
            metric_config["ignore_index"].required = False
        return metric_config

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return cls._metric_config_from_torch_metrics()

    @staticmethod
    def prepare_tensors(preds, target, dtypes=torch.int):
        dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes, dtypes]
        assert len(dtypes) == 2, "dtypes should be a list or tuple with two elements."
        preds = torch.tensor(preds, dtype=dtypes[0]) if not isinstance(preds, torch.Tensor) else preds.to(dtypes[0])
        target = torch.tensor(target, dtype=dtypes[1]) if not isinstance(target, torch.Tensor) else target.to(dtypes[1])
        return preds, target

    def update(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        self.metric.update(preds_tensor, target_tensor)

    def compute(self) -> float:
        return self.metric.compute().item()


class AccuracyScore(AccuracyBase):
    name: str = "accuracy_score"
    metric_module = torchmetrics.Accuracy


class F1Score(AccuracyBase):
    name: str = "f1_score"
    metric_module = torchmetrics.F1Score


class Precision(AccuracyBase):
    name: str = "precision"
    metric_module = torchmetrics.Precision


class Recall(AccuracyBase):
    name: str = "recall"
    metric_module = torchmetrics.Recall


class AUROC(AccuracyBase):
    name: str = "auroc"
    metric_module = torchmetrics.AUROC

    def update(self, model_output, target):
        logits_tensor, target_tensor = self.prepare_tensors(model_output.logits, target, [torch.float, torch.int32])
        if self.config.task == "binary" and len(logits_tensor.shape) > 1 and logits_tensor.shape[-1] == 2:
            logits_tensor = torch.softmax(logits_tensor, dim=-1)[:, 1]
        self.metric.update(logits_tensor, target_tensor.flatten())


class Perplexity(AccuracyBase):
    name: str = "perplexity"
    metric_module = torchmetrics.text.Perplexity

    def update(self, model_output, target):
        logits, targets = self.prepare_tensors(model_output.preds, target, dtypes=[torch.float, torch.long])
        self.metric.update(logits[..., :-1, :], targets[..., 1:])
