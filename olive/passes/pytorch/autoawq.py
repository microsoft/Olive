# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import torch

from olive.common.utils import StrEnumBase, copy_dir
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam, get_user_script_data_config
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
                description="Data config for quantization. Default value is None.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        from awq import AutoAWQForCausalLM

        if not torch.cuda.is_available():
            raise ValueError("Please use GPU to run AWQ quantization.")

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

        adapter_path = None
        # copy over adapter path if it exists
        if model.adapter_path:
            logger.info(
                "Model has adapters but AWQ does not support adapters. Quantizing without adapters. Adapters"
                " will be copied as is."
            )
            adapter_path = Path(output_model_path) / "adapter"
            adapter_path.mkdir(parents=True, exist_ok=True)
            copy_dir(model.adapter_path, adapter_path)

            output_model_path = str(Path(output_model_path) / "model")
            # TODO(jambayk): should we update the base_model_name_or_path in the adapter_config json?

        # autoawq load the model with fp16 by default and they did not expose the interface to change it
        awq_model = AutoAWQForCausalLM.from_pretrained(
            model.model_name_or_path, **self._resolve_load_args(model.get_load_kwargs())
        )
        tokenizer = model.get_hf_tokenizer()

        # quantize the model
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

        # save_quantized also saves the metadata, so we just save the tokenizer
        tokenizer.save_pretrained(output_model_path)
        awq_model.save_quantized(output_model_path)

        # return HfModelHandler with updated model path
        new_load_kwargs = deepcopy(model.load_kwargs.dict()) if model.load_kwargs else {}
        # model is saved in safetensors format so need to enable safetensors load
        if new_load_kwargs.get("extra_args") and new_load_kwargs["extra_args"].get("use_safetensors") is False:
            new_load_kwargs["extra_args"]["use_safetensors"] = True
        return inherit_hf_from_hf(model, output_model_path, adapter_path=adapter_path, load_kwargs=new_load_kwargs)

    def _resolve_load_args(self, hf_loading_args):
        loading_args = {}
        # default value for `safetensors` is True in auto AWQ
        loading_args["safetensors"] = hf_loading_args.get("use_safetensors", True)
        if device_map := hf_loading_args.get("device_map"):
            loading_args["device_map"] = device_map
        if trust_remote_code := hf_loading_args.get("trust_remote_code"):
            loading_args["trust_remote_code"] = trust_remote_code
        return loading_args
