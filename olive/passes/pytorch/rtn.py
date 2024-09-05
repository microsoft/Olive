# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict, Union

import torch

from olive.common.utils import StrEnumBase
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam, get_user_script_data_config
from olive.passes.pytorch.common import inherit_hf_from_hf

logger = logging.getLogger(__name__)


class RTNQuantizer(Pass):
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
                RTNQuantizer.ModelDtype.FP32: torch.float32,
                RTNQuantizer.ModelDtype.FP16: torch.float16,
                RTNQuantizer.ModelDtype.FP64: torch.float64,
            }[self]

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "input_model_dtype": PassConfigParam(
                type_=RTNQuantizer.ModelDtype,
                default_value=RTNQuantizer.ModelDtype.FP16,
                description="The input model data type.",
            ),
            "zero_point": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to use zero point quantization to calculate the scales and zeros. "
                    "If False, it use the symmetric quantization."
                ),
            ),
            "q_group_size": PassConfigParam(
                type_=int,
                default_value=32,
                description="The group size to use for quantization.",
            ),
            "w_bit": PassConfigParam(
                type_=int,
                default_value=4,
                description="The number of bits to quantize to.",
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
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        from awq import AutoAWQForCausalLM
        from awq.models import base as awq_model_base
        from awq.quantize.quantizer import AwqQuantizer as PyAutoAWQQuantizer

        from olive.passes.pytorch.quant_utils import AutoRTNQuantizer

        if not torch.cuda.is_available():
            raise ValueError("Please use GPU to run AWQ quantization.")
        elif self.host_device != Device.GPU:
            logger.debug(
                "AWQ quantization requires GPU but the host device is %s, will ignore the host device",
                self.host_device,
            )

        # autoawq load the model with fp16 by default and they did not expose the interface to change it
        awq_model = AutoAWQForCausalLM.from_pretrained(
            model.model_name_or_path, **self._resolve_load_args(model.get_load_kwargs())
        )
        tokenizer = model.get_hf_tokenizer()
        try:
            awq_model_base.AwqQuantizer = AutoRTNQuantizer
            awq_model.quantize(
                tokenizer,
                quant_config={
                    "zero_point": config["zero_point"],
                    "q_group_size": config["q_group_size"],
                    "w_bit": config["w_bit"],
                    "version": "gemm",
                    "modules_to_not_convert": config["modules_to_not_convert"],
                },
            )
        finally:
            awq_model_base.AwqQuantizer = PyAutoAWQQuantizer

        # save_quantized also saves the metadata, so we just save the tokenizer
        tokenizer.save_pretrained(output_model_path)
        awq_model.save_quantized(output_model_path)

        # return HfModelHandler with updated model path
        new_load_kwargs = deepcopy(model.load_kwargs.dict()) if model.load_kwargs else {}
        # model is saved in safetensors format so need to enable safetensors load
        if new_load_kwargs.get("extra_args") and new_load_kwargs["extra_args"].get("use_safetensors") is False:
            new_load_kwargs["extra_args"]["use_safetensors"] = True
        return inherit_hf_from_hf(model, output_model_path, load_kwargs=new_load_kwargs)

    def _resolve_load_args(self, hf_loading_args):
        loading_args = {}
        # default value for `safetensors` is True in auto AWQ
        loading_args["safetensors"] = hf_loading_args.get("use_safetensors", True)
        if device_map := hf_loading_args.get("device_map"):
            loading_args["device_map"] = device_map
        if trust_remote_code := hf_loading_args.get("trust_remote_code"):
            loading_args["trust_remote_code"] = trust_remote_code
        return loading_args
