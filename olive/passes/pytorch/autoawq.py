# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict, Type, Union

import torch
from packaging import version

from olive.common.utils import StrEnumBase, get_attr
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config
from olive.passes.pytorch.common import inherit_hf_from_hf

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
                description="Data config for quantization. If not provided, pile validation data will be used.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        from awq import AutoAWQForCausalLM

        if not torch.cuda.is_available():
            raise ValueError("Please use GPU to run AWQ quantization.")

        data_kwargs = {}
        if config.data_config:
            # set default values for data config
            data_kwargs.update(
                {
                    "calib_data": config.data_config.load_dataset_params.get("data_name"),
                    "split": config.data_config.load_dataset_params.get("split"),
                    "text_column": config.data_config.pre_process_params.get("input_cols"),
                }
            )

        adapter_path = None
        if model.adapter_path:
            logger.info(
                "Model has adapters but AWQ does not support adapters. Quantizing without adapters. The original"
                " adapters will be used as is with the quantized base model."
            )
            # TODO(jambayk): should we copy the adapter? what about non-local adapters?
            adapter_path = model.adapter_path

        # autoawq load the model with fp16 by default and they did not expose the interface to change it
        awq_model = AutoAWQForCausalLM.from_pretrained(
            model.model_name_or_path, **self._resolve_load_args(model.get_load_kwargs())
        )
        awq_model = self._maybe_patch_awq_model(awq_model)
        tokenizer = model.get_hf_tokenizer()

        # quantize the model
        awq_model.quantize(
            tokenizer,
            quant_config={
                "zero_point": config.zero_point,
                "q_group_size": config.q_group_size,
                "w_bit": config.w_bit,
                "version": config.version,
                "modules_to_not_convert": config.modules_to_not_convert,
            },
            duo_scaling=config.duo_scaling,
            export_compatible=config.export_compatible,
            **data_kwargs,
        )

        awq_model.save_quantized(output_model_path)
        model.save_metadata(output_model_path)

        # return HfModelHandler with updated model path
        new_load_kwargs = deepcopy(model.load_kwargs.dict()) if model.load_kwargs else {}
        # model is saved in safetensors format so need to enable safetensors load
        if new_load_kwargs.get("extra_args") and new_load_kwargs["extra_args"].get("use_safetensors") is False:
            new_load_kwargs["extra_args"]["use_safetensors"] = True
        return inherit_hf_from_hf(model, output_model_path, adapter_path=adapter_path, load_kwargs=new_load_kwargs)

    def _resolve_load_args(self, hf_loading_args: Dict[str, Any]):
        return {
            # want to default to using safetensors like in AutoAWQ
            "safetensors": hf_loading_args.get("use_safetensors", True),
            # only trust remote code if the user has explicitly set it
            "trust_remote_code": hf_loading_args.get("trust_remote_code"),
            # Not much to be gained my using "auto" device map, so default to None
            "device_map": hf_loading_args.get("device_map"),
        }

    def _maybe_patch_awq_model(self, awq_model):
        from awq import __version__ as autoawq_version
        from transformers import __version__ as transformers_version

        if version.parse(transformers_version) >= version.parse("4.43") and version.parse(
            autoawq_version
        ) <= version.parse("0.2.6"):
            original_move_embed = awq_model.move_embed

            def new_move_embed(model, device):
                original_move_embed(model, "cuda")
                # almost all model types have rotary embeddings at model.model.rotary_emb so won't keep a mapping
                if rotary_embed_module := get_attr(model, "model.rotary_emb"):
                    rotary_embed_module.to(device)

            awq_model.move_embed = new_move_embed

        return awq_model
