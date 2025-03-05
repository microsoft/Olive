# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from copy import deepcopy
from typing import List, Optional

from olive.auto_optimizer.regulate_mixins import RegulatePassConfigMixin
from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import validator
from olive.common.utils import StrEnumBase
from olive.data.config import DataConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ModelConfig

logger = logging.getLogger(__name__)


class Precision(StrEnumBase):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class AutoOptimizerConfig(ConfigBase):
    # opt_level
    # 0: specific pass_flows graph for auto optimizer to search
    # TODO(anyone): support more opt_level
    opt_level: int = 0

    # used to disable auto optimizer when user only want to evaluate input model
    disable_auto_optimizer: bool = False

    # precisions lists: [fp32, fp16, int8] which means the output model can be fp32, fp16 or int8
    # which will be used to filter the passes that are not supported by the precision
    # None for no precision restriction
    precisions: Optional[List[Precision]] = None

    # TODO(trajep): should distinguish model fine-tune and model inference?
    # if fine_tune is True, we will not suggest the training related pass, like: QLora
    # fine_tune: bool = False

    excluded_passes: Optional[List[str]] = None

    @validator("opt_level", pre=True)
    def check_opt_level(cls, v):
        if v != 0:
            raise ValueError("opt_level 0 is the only supported value for now")
        return v


class AutoOptimizer(RegulatePassConfigMixin):
    def __init__(
        self,
        input_model_config: ModelConfig,
        evaluator_config: OliveEvaluatorConfig,
        accelerator_spec: AcceleratorSpec,
        auto_optimizer_config: Optional[AutoOptimizerConfig] = None,
        data_configs: Optional[List[DataConfig]] = None,
    ):
        self.input_model_config = input_model_config
        self.evaluator_config = evaluator_config
        self.accelerator_spec = accelerator_spec
        self.auto_optimizer_config = auto_optimizer_config or AutoOptimizerConfig()
        self.data_configs = data_configs or []
        self._initialize()

    def _initialize(self):
        # 1. input model config
        self.model_config = deepcopy(self.input_model_config)

        # if user can tolerate accuracy drop, we can enable more optimization
        default_precisions = [Precision.FP32]
        if self.evaluator_config and self.evaluator_config.is_accuracy_drop_tolerant:
            default_precisions = [Precision.FP32, Precision.FP16, Precision.INT8, Precision.INT4]
        self.auto_optimizer_config.precisions = self.auto_optimizer_config.precisions or default_precisions

    def suggest(self):
        """Return a tuple of (suggested pass config, suggested pass flows).

        e.g.
        pass_config = {"pass_name1": {"param1": "value1", "param2": "value2"}}
        pass_flows = [["pass_name1", "pass_name2"], ["pass_name3", "pass_name4"]].
        """
        return self._regulate(self._suggest_pass_flows())

    def _suggest_pass_flows(self):
        pass_flows = []
        if self.auto_optimizer_config.opt_level == 0:
            pass_flows = self._suggest_pass_flows_from_template()

        return pass_flows

    def _suggest_pass_flows_from_template(self):
        from olive.auto_optimizer.template_mapping import get_pass_flows_by_accelerator_ep_precision

        assert self.auto_optimizer_config.opt_level <= 1, "opt_level must be 0 for suggest_pass_flows_from_template"

        pass_flows_by_precision = {}
        for precision in self.auto_optimizer_config.precisions or []:
            pass_flows_by_precision[precision] = get_pass_flows_by_accelerator_ep_precision(
                self.auto_optimizer_config.opt_level,
                self.accelerator_spec.accelerator_type.value,
                self.accelerator_spec.execution_provider,
                precision,
                self.auto_optimizer_config.excluded_passes,
            )
        return pass_flows_by_precision

    def _regulate(self, pass_flows_by_precision):
        # step1: regulate the pass name which may be different in different passes
        # for example: OrtTransformersOptimization_cuda_fp16 and OrtTransformersOptimization
        # are for the case of fp16 and fp32 respectively
        pass_config, pass_flows = self.regulate_pass_flows_dict(pass_flows_by_precision)

        # step2: fill the data_config for the passes that need data_config
        pass_config, pass_flows = self.regulate_data_config(pass_config, pass_flows)

        return pass_config, pass_flows
