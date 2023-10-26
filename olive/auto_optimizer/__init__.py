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


quant_passes = [
    "OnnxQuantization",
    "OnnxStaticQuantization",
    "OnnxDynamicQuantization",
    "IncQuantization",
    "VitisAIQuantization",
    "QuantizationAwareTraining",
]
passes_may_cause_accuracy_drop = ["SparseGPT", "OrtMixedPrecision", *quant_passes]


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    # TODO(trajep): add more precision bf16, int4 and etc.
    # ort int4 kernel is not fully optimized, so we disable it for now even we have the int4 quantization


class AutoOptimizerConfig(ConfigBase):
    # opt_level
    # 0: look up template mapping
    # TODO(trajep): add more opt_level
    # TODO(trajep): 1 pass search with model related config pruner
    # TODO(trajep): 99 full optimization: pass search + pass config search
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

    # required pass config like {"pass_name": {"param1": "value1", "param2": "value2"}
    disabled_passes: Optional[List] = None
    customized_pass_config: Optional[Dict] = None

    # TODO(trajep): high level switch to enable/disable quant/compress/distill/sparse etc.

    @validator("opt_level", pre=True)
    def check_opt_level(cls, v):
        if v != 0:
            raise ValueError("opt_level 0 is the only supported value for now")
        if v not in [0, 1, 99]:
            raise ValueError("opt_level must be in [0, 1, 99]")
        return v

    @validator("customized_pass_config")
    def check_pass_config(cls, v, values):
        if not v:
            return v
        disabled_passes = values.get("disabled_passes", [])
        # customized_pass_config and disabled_pass_config should not have same pass config
        for pass_name in v:
            if pass_name in disabled_passes:
                raise ValueError(f"{pass_name} is both required and disabled")
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
        self.auto_optimizer_config.customized_pass_config = self.auto_optimizer_config.customized_pass_config or {}
        self.auto_optimizer_config.disabled_passes = self.auto_optimizer_config.disabled_passes or []
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

        # 3. accelerator spec
        self.is_gpu = self.accelerator_spec.accelerator_type == Device.GPU

    def suggest(self):
        """Return a tuple of (suggested pass config, suggested pass flows).

        e.g.
        pass_config = {"pass_name": {"param1": "value1", "param2": "value2"}}
        pass_flows = [["pass_name1", "pass_name2"], ["pass_name3", "pass_name4"]].
        """
        return self.regulate(*self.suggest_passes())

    def suggest_passes(self):
        pass_config = {}
        pass_flows = []
        if self.auto_optimizer_config.opt_level == 0:
            pass_config, pass_flows = self.suggest_passes_from_template()

        # TODO(trajep): define default template for all models.
        assert pass_config and pass_flows, "current model is not supported by auto optimizer, please disable it"
        return pass_config, pass_flows

    def regulate(self, pass_config, pass_flows):
        # can be called separately when to user want to regulate defined pass config
        # decide which passes to be skipped
        self.regulate_pass_flows(pass_flows)

        # clean no-reference pass configs
        to_del_passes = set(pass_config.keys()) - {p for pf in pass_flows for p in pf}
        for p in to_del_passes:
            pass_config.pop(p)

        self.regulate_pass_configs(pass_config)
        return pass_config, pass_flows

    def suggest_passes_from_template(self):
        from olive.auto_optimizer.template_mapping import default_onnx_opt_pass_flows

        assert self.auto_optimizer_config.opt_level == 0, "opt_level must be 0 for suggest_pass_flows_from_template"

        pass_flows = default_onnx_opt_pass_flows()
        pass_config = {}

        for pass_item in pass_flows:
            for p_name in pass_item:
                user_defined_config = self.auto_optimizer_config.customized_pass_config.get(p_name, {})
                if p_name not in pass_config:
                    pass_config[p_name] = user_defined_config
                else:
                    pass_config[p_name].update(user_defined_config)
        # if there are duplicated passes in template, we should raise error
        assert all(len(p) == len(set(p)) for p in pass_flows), f"duplicated passes in template: {pass_flows}"
        return pass_config, pass_flows

    def regulate_pass_flows(self, pass_flows):
        # remove passes that are not supported by execution provider
        to_del_passes = set(self.auto_optimizer_config.disabled_passes) or set()

        # TODO(trajep): centralize the filter logic to config
        if self.is_gpu or not self.allow_precision(Precision.INT8):
            to_del_passes.quant_passes(quant_passes)

        if not self.is_accuracy_drop_tolerance:
            to_del_passes.update(passes_may_cause_accuracy_drop)

        # del passes
        for p in to_del_passes:
            for flow in pass_flows:
                if p in flow:
                    flow.remove(p)

        # after removing passes, there may be empty/duplicated pass flows, we should remove them
        to_del_pf_ids = []
        list_duplicated_check = set()
        for idx, flow in enumerate(pass_flows):
            if not flow or str(flow) in list_duplicated_check:
                to_del_pf_ids.append(idx)
                break
            list_duplicated_check.add(str(flow))
        for idx in to_del_pf_ids:
            pass_flows.pop(idx)

    def regulate_pass_configs(self, pass_config):
        self.regulate_fp16(pass_config)
        self.regulate_data_config(pass_config)

    def regulate_fp16(self, pass_config):
        if not self.is_gpu or not self.is_accuracy_drop_tolerance:
            return

        is_cuda_ep = self.accelerator_spec.execution_provider == "CUDAExecutionProvider"
        is_trt_ep = self.accelerator_spec.execution_provider == "TensorrtExecutionProvider"
        assert (
            is_cuda_ep or is_trt_ep
        ), "can not support CUDAExecutionProvider and TensorrtExecutionProvider at the same time"

        customized_fp16 = self.allow_precision(Precision.FP16)
        cuda_fp16 = customized_fp16 and is_cuda_ep
        trt_fp16 = customized_fp16 and not cuda_fp16

        if "OrtTransformersOptimization" in pass_config:
            # if cuda ep, enable fp16 for transformers optimization but not for tensorrt
            self.set_config_value_if_not_exist(pass_config["OrtTransformersOptimization"], "float16", cuda_fp16)
            self.set_config_value_if_not_exist(pass_config["OrtTransformersOptimization"], "use_gpu", True)
            if "OrtPerfTuning" in pass_config:
                # if trt ep, disable fp16 for transformers optimization but enable fp16 for perf_tuning
                self.set_config_value_if_not_exist(pass_config["OrtPerfTuning"], "trt_fp16_enable", trt_fp16)
                # enable_cuda_graph only for cuda ep
                self.set_config_value_if_not_exist(pass_config["OrtPerfTuning"], "enable_cuda_graph", cuda_fp16)
                self.set_config_value_if_not_exist(pass_config["OrtPerfTuning"], "io_bind", True)

    def regulate_data_config(self, pass_config):
        from olive.workflows.run.config import INPUT_MODEL_DATA_CONFIG

        # data_config
        passes_require_data_config = ["OnnxQuantization", "OrtPerfTuning"]
        for p in passes_require_data_config:
            if p not in pass_config:
                continue
            custom_data_config_name = pass_config.get(p, {}).get("data_config", None)
            if custom_data_config_name is not None:
                if isinstance(custom_data_config_name, str):
                    pass_config[p]["data_config"] = self.data_configs[custom_data_config_name].to_json()
            else:
                pass_config[p]["data_config"] = self.data_configs[INPUT_MODEL_DATA_CONFIG].to_json()

    def set_config_value_if_not_exist(self, config, key, value):
        if key not in config:
            config[key] = value

    def allow_precision(self, precision: Precision):
        if not self.auto_optimizer_config.precisions:
            return True

        return precision in self.auto_optimizer_config.precisions
