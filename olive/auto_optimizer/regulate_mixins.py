# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy

from olive.hardware.accelerator import Device


class RegulatePassConfigMixin:
    def regulate_pass_flows_dict(self, pass_flows_dict):
        # remove useless passes according to the olive model type, for example if onnx model
        # the conversion pass will be removed
        if self.input_model_config.type.lower().endswith("onnxmodel"):
            for pfs in pass_flows_dict.values():
                for pf in pfs:
                    pf.remove("OnnxConversion")

        # special passes: OrtTransformerOptimization and OrtPerfTuning can be used for both fp16 and fp32
        # we need assign different pass name for them
        # for example: gpu_cuda_fp16, we need rename OrtTransformerOptimization to OrtTransformerOptimization_cuda_fp16
        pass_flows_by_fp16 = pass_flows_dict.get("fp16", [])
        pass_config, pass_flows_16 = self._regulate_fp16(None, pass_flows_by_fp16)

        # flatten pass_flows_dict to pass_flows and generate the default pass_configs
        pass_flows = []
        unique_pass_flows = set()
        if pass_flows_16:
            pass_flows_dict["fp16"] = pass_flows_16
        for pfs in pass_flows_dict.values():
            for pf in pfs:
                if tuple(pf) not in unique_pass_flows:
                    pass_flows.append(pf)
                unique_pass_flows.add(tuple(pf))
                for p in pf:
                    if p not in pass_config:
                        pass_config.update({p: {"type": p, "config": {}}})
        # disable pass search when search strategy is None/False
        if not self.evaluator_config:
            for pass_name in pass_config:
                pass_config[pass_name]["disable_search"] = True
        return pass_config, pass_flows

    def _regulate_fp16(self, pass_config, pass_flows):
        pass_config = pass_config or {}
        is_gpu = self.accelerator_spec.accelerator_type == Device.GPU and self.accelerator_spec.execution_provider in [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "TensorrtExecutionProvider",
        ]
        if not is_gpu or not self.is_accuracy_drop_tolerance:
            return {}, []

        is_cuda_ep = self.accelerator_spec.execution_provider != "TensorrtExecutionProvider"
        is_trt_ep = self.accelerator_spec.execution_provider == "TensorrtExecutionProvider"
        assert (
            not is_cuda_ep or not is_trt_ep
        ), "can not support CUDA/DmlExecutionProvider and TensorrtExecutionProvider at the same time"

        customized_fp16 = self._allow_precision("fp16")
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
                    elif perf_tuning == p:
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
            p_names = self._find_pass_name_in_pass_flow(p, pass_flows)
            for pn in p_names:
                pass_config[pn]["config"]["data_config"] = data_configs.to_json()
        return pass_config, pass_flows

    def _allow_precision(self, precision):
        if not self.auto_optimizer_config.precisions:
            # empty precisions means no precision restriction
            return True

        return precision in self.auto_optimizer_config.precisions

    def _find_pass_name_in_pass_flow(self, pass_name, pass_flows):
        passes = set()
        for pf in pass_flows:
            for p in pf:
                if p.startswith(pass_name):
                    passes.add(p)
        return passes
