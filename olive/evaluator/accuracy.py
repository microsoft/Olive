# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import abstractmethod
from inspect import isfunction, signature
from typing import Any, Callable, Dict, Union

import numpy as np
import torch
import torchmetrics

from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.config_utils import ConfigParam

logger = logging.getLogger(__name__)


class AccuracyBase(AutoConfigClass):
    registry: Dict[str, "AccuracyBase"] = {}
    metric_cls_map: Dict[str, Union[torchmetrics.Metric, Callable]] = {
        "accuracy_score": torchmetrics.Accuracy,
        "f1_score": torchmetrics.F1Score,
        "precision": torchmetrics.Precision,
        "recall": torchmetrics.Recall,
        "auc": torchmetrics.functional.auc,
    }

    @classmethod
    @property
    @abstractmethod
    def name(cls):
        raise NotImplementedError

    def __init__(self, config: Union[ConfigBase, Dict[str, Any]] = None) -> None:
        super().__init__(config)

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
            if info.kind == info.VAR_KEYWORD or info.kind == info.VAR_POSITIONAL:
                required = False
            metric_config[param] = ConfigParam(type_=annotation, required=required, default_value=default_value)
        return metric_config

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return cls._metric_config_from_torch_metrics()

    @abstractmethod
    def measure(self, preds, target):
        raise NotImplementedError()


class AccuracyScore(AccuracyBase):
    name: str = "accuracy_score"

    def measure(self, preds, target):
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        accuracy = torchmetrics.Accuracy(**self.config.dict())
        result = accuracy(preds_tensor, target_tensor)
        return result.item()


class F1Score(AccuracyBase):
    name: str = "f1_score"

    def measure(self, preds, target):
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        f1 = torchmetrics.F1Score(**self.config.dict())
        result = f1(preds_tensor, target_tensor)
        return result.item()


class Precision(AccuracyBase):
    name: str = "precision"

    def measure(self, preds, target):
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        precision = torchmetrics.Precision(**self.config.dict())
        result = precision(preds_tensor, target_tensor)
        return result.item()


class Recall(AccuracyBase):
    name: str = "recall"

    def measure(self, preds, target):
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        recall = torchmetrics.Recall(**self.config.dict())
        result = recall(preds_tensor, target_tensor)
        return result.item()


class AUC(AccuracyBase):
    name: str = "auc"

    def measure(self, preds, target):
        preds = np.array(preds).flatten()
        target = np.array(target).flatten()
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        result = torchmetrics.functional.auc(preds_tensor, target_tensor, self.config.reorder)
        return result.item()
