# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from typing import Any, Dict, Union

import numpy as np
import torch
import torchmetrics

from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.config_utils import ConfigParam


class AccuracyBase(AutoConfigClass):
    registry: Dict[str, "AccuracyBase"] = {}

    @classmethod
    @property
    @abstractmethod
    def name(cls):
        raise NotImplementedError

    def __init__(self, config: Union[ConfigBase, Dict[str, Any]] = None) -> None:
        super().__init__(config)

    @staticmethod
    def _default_config() -> Dict[str, ConfigParam]:
        return {
            "num_classes": ConfigParam(type_=int),
            "threshold": ConfigParam(type_=float, default_value=0.5),
            "average": ConfigParam(type_=str, default_value="micro"),
            "ignore_index": ConfigParam(type_=list, default_value=None),
            "top_k": ConfigParam(type_=int, default_value=None),
            "mdmc_average": ConfigParam(type_=str, default_value="global"),
        }

    @abstractmethod
    def evaluate(self, preds, target):
        raise NotImplementedError()


class AccuracyScore(AccuracyBase):
    name: str = "accuracy_score"

    def evaluate(self, preds, target):
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        accuracy = torchmetrics.Accuracy(**self.config.dict())
        result = accuracy(preds_tensor, target_tensor)
        return result.item()


class F1Score(AccuracyBase):
    name: str = "f1_score"

    def evaluate(self, preds, target):
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        f1 = torchmetrics.F1Score(**self.config.dict())
        result = f1(preds_tensor, target_tensor)
        return result.item()


class Precision(AccuracyBase):
    name: str = "precision"

    def evaluate(self, preds, target):
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        precision = torchmetrics.Precision(**self.config.dict())
        result = precision(preds_tensor, target_tensor)
        return result.item()


class Recall(AccuracyBase):
    name: str = "recall"

    def evaluate(self, preds, target):
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        recall = torchmetrics.Recall(**self.config.dict())
        result = recall(preds_tensor, target_tensor)
        return result.item()


class AUC(AccuracyBase):
    name: str = "auc"

    @staticmethod
    def _default_config():
        return {"reorder": ConfigParam(type_=bool, default_value=False)}

    def evaluate(self, preds, target):
        preds = np.array(preds).flatten()
        target = np.array(target).flatten()
        preds_tensor = torch.tensor(preds, dtype=torch.int)
        target_tensor = torch.tensor(target, dtype=torch.int)
        result = torchmetrics.functional.auc(preds_tensor, target_tensor, self.config.reorder)
        return result.item()
