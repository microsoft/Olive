# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional

from pydantic import validator

import olive.cache as cache_utils
from olive.common.config_utils import ConfigBase
from olive.data.config import DataConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ModelConfig

logger = logging.getLogger(__name__)


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class AutoOptimizerConfig(ConfigBase):
    # opt_level
    # 0: look up template mapping
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

    @validator("opt_level", pre=True)
    def check_opt_level(cls, v):
        if v != 0:
            raise ValueError("opt_level 0 is the only supported value for now")
        if v not in [0, 1]:
            raise ValueError("opt_level must be in [0, 1]")
        return v


class AutoOptimizer:
    def __init__(
        self,
        input_model_config: ModelConfig,
        evaluator_config: OliveEvaluatorConfig,
        accelerator_spec: AcceleratorSpec,
        auto_optimizer_config: Optional[AutoOptimizerConfig] = None,
        data_configs: Optional[Dict[str, DataConfig]] = None,
    ):
        # TODO(trajep): how to get model_attributes if the model is in AML workspace? Maybe we can ignore this for now?
        self.input_model_config = input_model_config
        self.evaluator_config = evaluator_config
        self.accelerator_spec = accelerator_spec
        self.auto_optimizer_config = auto_optimizer_config or AutoOptimizerConfig()
        self.data_configs = data_configs or {}
        self.initialize()

    def initialize(self):
        # 1. input model config
        # TODO(trajep): move duplicate part of _prepare_non_local_model somewhere, maybe OliveSystem?
        model_config = deepcopy(self.input_model_config)
        resource_paths = model_config.get_resource_paths()
        for resource_name, resource_path in resource_paths.items():
            if not resource_path or resource_path.is_local_resource_or_string_name():
                continue
            downloaded_resource_path = cache_utils.download_resource(resource_path, self._config.cache_dir)
            if downloaded_resource_path:
                # set local resource path
                model_config.config[resource_name] = downloaded_resource_path
        model = model_config.create_model()
        self.model_attr = model.model_attributes or {}

        # 2. evaluator config
        self.is_accuracy_drop_tolerance = self.evaluator_config.is_accuracy_drop_tolerance
        # if user can tolerate accuracy drop, we can enable more optimization
        default_precisions = [Precision.FP32]
        if self.is_accuracy_drop_tolerance:
            # ignore int4 for now as it is not supported very well in onnxruntime
            # enable it only when user explicitly set it
            # default_precisions = [Precision.FP16, Precision.INT8, Precision.INT4]
            default_precisions = [Precision.FP16, Precision.INT8]
        self.auto_optimizer_config.precisions = self.auto_optimizer_config.precisions or default_precisions

        # 3. accelerator spec
        self.is_gpu = self.accelerator_spec.accelerator_type == Device.GPU

    def suggest(self):
        """Return a tuple of (suggested pass config, suggested pass flows).

        e.g.
        pass_config = {"pass_name1": {"param1": "value1", "param2": "value2"}}
        pass_flows = [["pass_name1", "pass_name2"], ["pass_name3", "pass_name4"]].
        """
        return self.regulate(self.suggest_pass_flows())

    def suggest_pass_flows(self):
        pass_flows_by_precision = []
        if self.auto_optimizer_config.opt_level == 0:
            pass_flows_by_precision = self.suggest_pass_flows_from_template()

        return pass_flows_by_precision

    def suggest_pass_flows_from_template(self):
        from olive.auto_optimizer.template_mapping import get_pass_flows_by_accelerator_ep_precision

        assert self.auto_optimizer_config.opt_level <= 1, "opt_level must be 0 for suggest_pass_flows_from_template"

        pass_flows_by_precision = {}
        for precision in self.auto_optimizer_config.precisions or []:
            pass_flows_by_precision[precision] = get_pass_flows_by_accelerator_ep_precision(
                self.accelerator_spec.accelerator_type.value,
                self.accelerator_spec.execution_provider,
                precision,
            )
        return pass_flows_by_precision

    def regulate(self, pass_flows_by_precision):
        # step1: regulate the pass name which may be different in different passes
        # for example: OrtTransformersOptimization_cuda_fp16 and OrtTransformersOptimization
        # are for the case of fp16 and fp32 respectively
        pass_config, pass_flows = self.regulate_pass_flows_dict(pass_flows_by_precision)

        # step2: fill the data_config for the passes that need data_config
        return self.regulate_data_config(pass_config, pass_flows)

    def regulate_pass_flows_dict(self, pass_flows_dict):
        # special passes: OrtTransformerOptimization and OrtPerfTuning can be used for both fp16 and fp32
        # we need assign different pass name for them
        # for example: gpu_cuda_fp16, we need rename OrtTransformerOptimization to OrtTransformerOptimization_cuda_fp16
        pass_flows_by_fp16 = pass_flows_dict.get(Precision.FP16, [])
        pass_config, pass_flows_16 = self.regulate_fp16(None, pass_flows_by_fp16)

        # flatten pass_flows_dict to pass_flows and generate the default pass_configs
        pass_flows = []
        unique_pass_flows = set()
        if pass_flows_16:
            pass_flows_dict[Precision.FP16] = pass_flows_16
        for pfs in pass_flows_dict.values():
            for pf in pfs:
                if tuple(pf) not in unique_pass_flows:
                    pass_flows.append(pf)
                unique_pass_flows.add(tuple(pf))
                for p in pf:
                    if p not in pass_config:
                        pass_config.update({p: {"type": p, "config": {}}})

        return pass_config, pass_flows

    def regulate_fp16(self, pass_config, pass_flows):
        pass_config = pass_config or {}
        if not self.is_gpu or not self.is_accuracy_drop_tolerance:
            return {}, []

        is_cuda_ep = self.accelerator_spec.execution_provider == "CUDAExecutionProvider"
        is_trt_ep = self.accelerator_spec.execution_provider == "TensorrtExecutionProvider"
        assert (
            is_cuda_ep or is_trt_ep
        ), "can not support CUDAExecutionProvider and TensorrtExecutionProvider at the same time"

        customized_fp16 = self.allow_precision(Precision.FP16)
        cuda_fp16 = customized_fp16 and is_cuda_ep
        trt_fp16 = customized_fp16 and not cuda_fp16

        trans_opt = "OrtTransformersOptimization"
        perf_tuning = "OrtPerfTuning"
        trans_opt_fp16 = "OrtTransformerOptimization_cuda_fp16"
        perf_tuning_fp16 = "OrtPerfTuning_trt_fp16"

        for i, pf in enumerate(pass_flows):
            new_pf = deepcopy(pf)
            if "OrtMixedPrecision" not in pf:
                for j, p in enumerate(pf):
                    if trans_opt == p:
                        new_pf[j] = trans_opt_fp16 if cuda_fp16 else p
                        pass_config.update(
                            {
                                new_pf[j]: {
                                    "type": trans_opt,
                                    "config": {
                                        "float16": cuda_fp16,
                                        "use_gpu": True,
                                    },
                                }
                            }
                        )
                    if perf_tuning == p:
                        new_pf[j] = perf_tuning_fp16 if trt_fp16 else p
                        pass_config.update(
                            {
                                new_pf[j]: {
                                    "type": perf_tuning,
                                    "config": {
                                        "trt_fp16_enable": trt_fp16,
                                        "enable_cuda_graph": cuda_fp16,
                                        "io_bind": True,
                                    },
                                }
                            }
                        )

            pass_flows[i] = new_pf

        return pass_config, pass_flows

    def regulate_data_config(self, pass_config, pass_flows):
        from olive.workflows.run.config import INPUT_MODEL_DATA_CONFIG

        data_configs = self.data_configs.get(INPUT_MODEL_DATA_CONFIG)
        if not data_configs:
            return pass_config, pass_flows

        # data_config
        passes_require_data_config = ["OnnxQuantization", "OrtPerfTuning"]
        for p in passes_require_data_config:
            # TODO(anyone): support multi data_config for different passes, pass_flows
            p_names = self.find_pass_name_in_pass_flow(p, pass_flows)
            for pn in p_names:
                pass_config[pn]["config"]["data_config"] = data_configs.to_json()
        return pass_config, pass_flows

    def allow_precision(self, precision: Precision):
        if not self.auto_optimizer_config.precisions:
            # empty precisions means no precision restriction
            return True

        return precision in self.auto_optimizer_config.precisions

    def find_pass_name_in_pass_flow(self, pass_name, pass_flows):
        passes = set()
        for pf in pass_flows:
            for p in pf:
                if p.startswith(pass_name):
                    passes.add(p)
        return passes
