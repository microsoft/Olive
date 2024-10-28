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
            to_remove_passes = ["OnnxConversion", "ModelBuilder"]
            for pfs in pass_flows_dict.values():
                for pf in pfs:
                    for p in to_remove_passes:
                        if p in pf:
                            pf.remove(p)

        # special passes: ModelBuilder, OrtTransformerOptimization and OrtSessionParamsTuning can
        # be used for both fp16 and fp32 we need assign different pass name for them
        # for example: gpu_cuda_fp16, we need rename OrtTransformerOptimization to OrtTransformerOptimization_cuda_fp16
        pass_config, pass_flows_dict = self._regulate_precision(None, pass_flows_dict)

        # flatten pass_flows_dict to pass_flows and generate the default pass_configs
        pass_flows = []
        unique_pass_flows = set()
        for pfs in pass_flows_dict.values():
            for pf in pfs:
                if tuple(pf) not in unique_pass_flows:
                    unique_pass_flows.add(tuple(pf))
                    pass_flows.append(pf)
                for p in pf:
                    if p not in pass_config:
                        pass_config.update({p: {"type": p, "config": {}}})
        return pass_config, pass_flows

    def _fill_precision_for_model_builder(self, pass_config, pass_flows):
        for precision, pfs in pass_flows.items():
            for pass_flow in pfs:
                for i, p in enumerate(pass_flow):
                    if p == "ModelBuilder":
                        pass_flow[i] = f"ModelBuilder_{precision}"
                        pass_config.update(
                            {
                                pass_flow[i]: {
                                    "type": "ModelBuilder",
                                    "config": {
                                        "precision": precision,
                                    },
                                }
                            }
                        )

    def _regulate_precision(self, pass_config, pass_flows):
        pass_config = pass_config or {}
        # if it is model builder, we need to add suffix for all precisions to distinguish them
        self._fill_precision_for_model_builder(pass_config, pass_flows)
        is_gpu = self.accelerator_spec.accelerator_type == Device.GPU and self.accelerator_spec.execution_provider in [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "TensorrtExecutionProvider",
        ]
        if not is_gpu:
            return pass_config, pass_flows

        is_cuda_ep = self.accelerator_spec.execution_provider != "TensorrtExecutionProvider"
        is_trt_ep = self.accelerator_spec.execution_provider == "TensorrtExecutionProvider"
        assert (
            not is_cuda_ep or not is_trt_ep
        ), "can not support CUDA/DmlExecutionProvider and TensorrtExecutionProvider at the same time"

        customized_fp16 = self._allow_precision("fp16")
        cuda_fp16 = customized_fp16 and is_cuda_ep
        trt_fp16 = customized_fp16 and not cuda_fp16

        trans_opt = "OrtTransformersOptimization"
        session_params_tuning = "OrtSessionParamsTuning"
        trans_opt_fp16 = "OrtTransformerOptimization_cuda_fp16"
        session_params_tuning_fp16 = "OrtSessionParamsTuning_trt_fp16"
        pass_flows_by_fp16 = pass_flows.get("fp16", [])
        for i, pf in enumerate(pass_flows_by_fp16):
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
                    elif session_params_tuning == p:
                        new_pf[j] = session_params_tuning_fp16 if trt_fp16 else p
                        pass_config.update(
                            {
                                new_pf[j]: {
                                    "type": session_params_tuning,
                                    "config": {
                                        "trt_fp16_enable": trt_fp16,
                                        "enable_cuda_graph": cuda_fp16,
                                        "io_bind": True,
                                    },
                                }
                            }
                        )
            pass_flows_by_fp16[i] = new_pf
        if pass_flows_by_fp16:
            pass_flows["fp16"] = pass_flows_by_fp16

        for flow_list in pass_flows.values():
            for p in flow_list:
                if trans_opt in p:
                    if trans_opt not in pass_config:
                        pass_config[trans_opt] = {"type": trans_opt, "config": {}}
                    if "config" not in pass_config[trans_opt]:
                        pass_config[trans_opt]["config"] = {}
                    pass_config[trans_opt]["config"]["use_gpu"] = True

        return pass_config, pass_flows

    def regulate_data_config(self, pass_config, pass_flows):
        if not self.auto_optimizer_config or self.auto_optimizer_config.disable_auto_optimizer:
            return pass_config, pass_flows

        passes_require_data_config = ["OrtSessionParamsTuning", "IncQuantization", "OnnxQuantization"]
        if not self.data_configs:
            # remove the passes which require data_config
            for pass_flow in pass_flows:
                for p in passes_require_data_config:
                    p_names = self._find_pass_name_in_pass_flow(p, [pass_flow])
                    for pn in p_names:
                        pass_flow.remove(pn)
                        pass_config.pop(pn, None)
                for p in pass_flow:
                    if p.lower().startswith("onnxquantization"):
                        pass_config[p]["config"]["quant_mode"] = "dynamic"
        else:
            if len(self.data_configs) != 1:
                raise ValueError("AutoOptimizer expects exactly one data config.")

            for p in passes_require_data_config:
                # TODO(anyone): support multi data_config for different passes, pass_flows
                p_names = self._find_pass_name_in_pass_flow(p, pass_flows)
                for pn in p_names:
                    pass_config[pn]["config"]["data_config"] = self.data_configs[0]

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
