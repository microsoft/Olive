# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict, Union

import torch

from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class AwqQuantizer(Pass):
    """AWQ quantization."""

    _requires_user_script = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "zero_point": PassConfigParam(
                type_=bool,
                default_value=True,
                # TODO(anyone): update description
                description="",
            ),
            "q_group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="",
            ),
            "w_bit": PassConfigParam(
                type_=int,
                default_value=4,
                description="",
            ),
            "version": PassConfigParam(
                type_=str,
                default_value="gemm",
                description="",
            ),
            "duo_scaling": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Whether to scale using both w/x or just x.",
            ),
            "export_compatible": PassConfigParam(
                type_=bool,
                default_value=False,
                description="This argument avoids real quantization by only applying"
                " the scales without quantizing down to FP16.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                default_value=None,
                description="""
                    Data config for quantization. Default value is None.
                """,
            ),
            "pack_model_for_onnx_conversion": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to pack the model for ONNX conversion. Default is False.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        from awq import AutoAWQForCausalLM
        from awq.models import base as awq_model_base
        from awq.quantize.quantizer import AwqQuantizer as AutoAwqQuantizer
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise ValueError("Please use GPU to run gptq quantization.")
        elif self.host_device != Device.GPU:
            logger.warning(
                "GPTQ quantization requires GPU but the host device is %s, will ignore the host device",
                self.host_device,
            )

        if config["data_config"]:
            # TODO(trajep): Implement data config for quantization
            ...

        quantizer = (
            self._pack_model_for_onnx_conversion(config)
            if config["pack_model_for_onnx_conversion"]
            else AutoAwqQuantizer
        )

        loading_args = self._resolve_load_args(model.hf_config.get_loading_args_from_pretrained())
        model_path = model.model_path or model.hf_config.model_name
        awq_model = AutoAWQForCausalLM.from_pretrained(model_path, **loading_args)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **loading_args)
        try:
            awq_model_base.AwqQuantizer = quantizer
            awq_model.quantize(
                tokenizer,
                quant_config={
                    "zero_point": config["zero_point"],
                    "q_group_size": config["q_group_size"],
                    "w_bit": config["w_bit"],
                    "version": config["version"],
                },
            )
        finally:
            awq_model_base.AwqQuantizer = AutoAwqQuantizer

        output_model_path = normalize_path_suffix(output_model_path, "model.pt")
        torch.save(awq_model.model, output_model_path)

        model_config = model.to_json()["config"]
        model_config["model_path"] = output_model_path
        model_config.pop("model_loader", None)
        if model.hf_config is not None:
            hf_config = model.get_hf_model_config()
            del model_config["hf_config"]
            model_config["model_attributes"] = hf_config.to_dict()

        return PyTorchModelHandler(
            **model_config,
        )

    def _pack_model_for_onnx_conversion(self, config):
        from awq.quantize.quantizer import AwqQuantizer as AutoAwqQuantizer
        from awq.quantize.quantizer import clear_memory, get_best_device, set_op_by_name

        from olive.passes.pytorch.gptq_utils import QuantLinearORT

        class OrtAwqQuantizer(AutoAwqQuantizer):
            def _apply_quant(self, module, named_linears: Dict[str, torch.nn.Linear]):
                for name, old_linear_layer in named_linears.items():
                    # NOTE: small regression in perplexity if linear layer uses .cpu().float()
                    linear_layer = old_linear_layer.to(get_best_device()).half()
                    linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(linear_layer.weight.data)
                    q_linear = QuantLinearORT(
                        bits=config["w_bit"],
                        groupsize=config["q_group_size"],
                        infeatures=linear_layer.in_features,
                        outfeatures=linear_layer.out_features,
                        bias=linear_layer.bias is not None,
                    )
                    linear_layer.cpu()
                    q_linear.to(next(module.parameters()).device)
                    set_op_by_name(module, name, q_linear)
                    clear_memory()

        return OrtAwqQuantizer

    def _resolve_load_args(self, hf_loading_args):
        loading_args = {}
        if safetensors := hf_loading_args.get("use_safetensors"):
            loading_args["safetensors"] = safetensors
        if device_map := hf_loading_args.get("device_map"):
            loading_args["device_map"] = device_map
        if trust_remote_code := hf_loading_args.get("trust_remote_code"):
            loading_args["trust_remote_code"] = trust_remote_code
        return loading_args
