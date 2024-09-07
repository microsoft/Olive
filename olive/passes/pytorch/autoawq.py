# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict, Union

import torch

from olive.common.utils import StrEnumBase
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam, get_user_script_data_config
from olive.passes.pytorch.common import inherit_hf_from_hf, inherit_pytorch_from_hf

logger = logging.getLogger(__name__)


class AutoAWQQuantizer(Pass):
    """AWQ quantization."""

    class ModelDtype(StrEnumBase):
        # input model's data type, we can assume the model is all float type
        # sometime, the model is in double type, but we can convert it to float type
        # before quantization
        FP32 = "fp32"
        FP16 = "fp16"
        FP64 = "fp64"

        def get_torch_dtype(self):
            return {
                AutoAWQQuantizer.ModelDtype.FP32: torch.float32,
                AutoAWQQuantizer.ModelDtype.FP16: torch.float16,
                AutoAWQQuantizer.ModelDtype.FP64: torch.float64,
            }[self]

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "input_model_dtype": PassConfigParam(
                type_=AutoAWQQuantizer.ModelDtype,
                default_value=AutoAWQQuantizer.ModelDtype.FP16,
                description="The input model data type.",
            ),
            "zero_point": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether to use zero point quantization to calculate the scales and zeros. "
                    "If False, it use the symmetric quantization."
                ),
            ),
            "q_group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description=(
                    "The group size to use for quantization. Recommended value is "
                    "128 and -1 uses per-column quantization."
                ),
            ),
            "w_bit": PassConfigParam(
                type_=int,
                default_value=4,
                description="The number of bits to quantize to.",
            ),
            "version": PassConfigParam(
                type_=str,
                default_value="gemm",
                description=(
                    "The version of the quantization algorithm to use. gemm is better "
                    "for big batch_size (e.g. >= 8) otherwise, gemv is better (e.g. < 8 ). "
                    "gemm models are compatible with Exllama kernels."
                ),
            ),
            "duo_scaling": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Whether to scale using both w/x(True) or just x(False).",
            ),
            "modules_to_not_convert": PassConfigParam(
                type_=list,
                default_value=[],
                description=(
                    "The list of modules to not quantize, useful for quantizing models that explicitly "
                    "require to have some modules left in their original precision (e.g. Whisper encoder, "
                    "Llava encoder, Mixtral gate layers). Please refer to AutoAWQ documentation for "
                    "quantizing HF models."
                ),
            ),
            "export_compatible": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "If True, this argument avoids real quantization by only applying "
                    "the scales quantizing down to FP16."
                ),
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                default_value=None,
                description="Data config for quantization. Default value is None.",
            ),
            "pack_model_for_onnx_conversion": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to pack the model for ONNX conversion. If True, the model will be saved as a PyTorch model"
                    " with custom quantized linears. If False, the model will be saved as a HF model with quantized"
                    " weights. Default value is False."
                ),
            ),
            "accuracy_level_4_aware": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Calibrate the weight scales to be aware about the input being quantized to 8bits when using"
                    " MatMulNBits with accuracy_level=4. Default value is False."
                ),
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        from awq import AutoAWQForCausalLM
        from awq.models import base as awq_model_base
        from awq.quantize.quantizer import AwqQuantizer as PyAutoAWQQuantizer

        if not torch.cuda.is_available():
            raise ValueError("Please use GPU to run AWQ quantization.")
        elif self.host_device != Device.GPU:
            logger.debug(
                "AWQ quantization requires GPU but the host device is %s, will ignore the host device",
                self.host_device,
            )

        data_kwargs = {}
        if config["data_config"]:
            # set default values for data config
            data_kwargs.update(
                {
                    "calib_data": config["data_config"].load_dataset_params.get("data_name"),
                    "split": config["data_config"].load_dataset_params.get("split"),
                    "text_column": config["data_config"].pre_process_params.get("input_cols"),
                }
            )

        # pack_model_for_onnx_conversion is a flag to switch between the two quantizers
        # 1. PyAutoAWQQuantizer is a quantizer implemented by autoawq package which is used by default
        #    for quantizing the model. But it does not work with ONNX conversion since there are some
        #    operations that are not supported by ONNX. So, we have to pack the model for ONNX conversion
        # 2. That is why we have another quantizer method self._pack_model_for_onnx_conversion(config) to
        #    return OrtAutoAWQQuantizer which is used for ONNX conversion.
        quantizer = (
            self._pack_model_for_onnx_conversion(config)
            if config["pack_model_for_onnx_conversion"]
            else (self._awq_with_quant_activation() if config["accuracy_level_4_aware"] else PyAutoAWQQuantizer)
        )

        # autoawq load the model with fp16 by default and they did not expose the interface to change it
        awq_model = AutoAWQForCausalLM.from_pretrained(
            model.model_name_or_path, **self._resolve_load_args(model.get_load_kwargs())
        )
        tokenizer = model.get_hf_tokenizer()
        try:
            awq_model_base.AwqQuantizer = quantizer
            awq_model.quantize(
                tokenizer,
                quant_config={
                    "zero_point": config["zero_point"],
                    "q_group_size": config["q_group_size"],
                    "w_bit": config["w_bit"],
                    "version": config["version"],
                    "modules_to_not_convert": config["modules_to_not_convert"],
                },
                duo_scaling=config["duo_scaling"],
                export_compatible=config["export_compatible"],
                **data_kwargs,
            )
        finally:
            awq_model_base.AwqQuantizer = PyAutoAWQQuantizer

        if config["pack_model_for_onnx_conversion"]:
            output_model_path = normalize_path_suffix(output_model_path, "model.pt")
            torch.save(awq_model.model, output_model_path)

            return inherit_pytorch_from_hf(model, output_model_path)

        # save_quantized also saves the metadata, so we just save the tokenizer
        tokenizer.save_pretrained(output_model_path)
        awq_model.save_quantized(output_model_path)

        # return HfModelHandler with updated model path
        new_load_kwargs = deepcopy(model.load_kwargs.dict()) if model.load_kwargs else {}
        # model is saved in safetensors format so need to enable safetensors load
        if new_load_kwargs.get("extra_args") and new_load_kwargs["extra_args"].get("use_safetensors") is False:
            new_load_kwargs["extra_args"]["use_safetensors"] = True
        return inherit_hf_from_hf(model, output_model_path, load_kwargs=new_load_kwargs)

    def _pack_model_for_onnx_conversion(self, config):
        from awq.quantize.quantizer import AwqQuantizer as PyAutoAWQQuantizer
        from awq.quantize.quantizer import clear_memory, get_best_device, set_op_by_name

        from olive.passes.pytorch.quant_utils import QuantLinear

        class OrtAutoAWQQuantizer(PyAutoAWQQuantizer):
            def _apply_quant(self, module, named_linears: Dict[str, torch.nn.Linear]):
                for name, old_linear_layer in named_linears.items():
                    # NOTE: small regression in perplexity if linear layer uses .cpu().float()
                    linear_layer = old_linear_layer.to(get_best_device()).half()
                    linear_layer.weight.data, _, _ = self.pseudo_quantize_tensor(linear_layer.weight.data)
                    q_linear = QuantLinear(
                        bits=config["w_bit"],
                        groupsize=config["q_group_size"],
                        infeatures=linear_layer.in_features,
                        outfeatures=linear_layer.out_features,
                        bias=linear_layer.bias is not None,
                        input_model_dtype=config["input_model_dtype"].get_torch_dtype(),
                    )
                    linear_layer.cpu()
                    q_linear.to(next(module.parameters()).device)
                    set_op_by_name(module, name, q_linear)
                    clear_memory()

        return OrtAutoAWQQuantizer

    def _awq_with_quant_activation(self):
        from awq.quantize.quantizer import AwqQuantizer as PyAutoAWQQuantizer

        from olive.passes.pytorch.quant_utils import symmetric_quantize_dequantize

        class AwqQuantizerWithQuantActivation(PyAutoAWQQuantizer):
            # meant to make the weight scales be aware about the input being quantized
            # to 8bits when using MatMulNBits with accuracy_level=4
            @torch.no_grad()
            def _search_best_scale(self, module, prev_op, layers, inp, module2inspect=None, kwargs=None):
                # quantize dequantize to mimic accuracy level 4 impl
                inp = symmetric_quantize_dequantize(inp, 8, self.group_size)
                return super()._search_best_scale(
                    module, prev_op, layers, inp, module2inspect=module2inspect, kwargs=(kwargs or {})
                )

        return AwqQuantizerWithQuantActivation

    def _resolve_load_args(self, hf_loading_args):
        loading_args = {}
        # default value for `safetensors` is True in auto AWQ
        loading_args["safetensors"] = hf_loading_args.get("use_safetensors", True)
        if device_map := hf_loading_args.get("device_map"):
            loading_args["device_map"] = device_map
        if trust_remote_code := hf_loading_args.get("trust_remote_code"):
            loading_args["trust_remote_code"] = trust_remote_code
        return loading_args
