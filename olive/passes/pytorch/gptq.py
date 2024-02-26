# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict

import torch

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModelHandler
from olive.model.handler.onnx import ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.gptq_utils import QuantLinearORT

logger = logging.getLogger(__name__)

ROUNDTRIP_CHECK = False


class GptqQuantizer(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "nsamples": PassConfigParam(
                type_=int,
                default_value=128,
                description="number of samples in calibration dataset to apply quantization. Default value is 128",
            ),
            "bits": PassConfigParam(
                type_=int,
                default_value=4,
                description="quantization bits. Default value is 4",
            ),
            "dataset": PassConfigParam(
                type_=str,
                default_value="wikitext2",
                description="Calibration dataset. Default value is wikitext2",
            ),
            "block_name_to_quantize": PassConfigParam(
                type_=str,
                default_value=None,
                description="Block name to quantize. Default value is model.decoder.layers.",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Block size for quantization. Default value is 128.",
            ),
            "batch_size": PassConfigParam(
                type_=int,
                default_value=1,
                description="Batch size for quantization. Default value is 1.",
            ),
            "seed": PassConfigParam(
                type_=int,
                default_value=0,
                description="Random seed for sampling calibration dataset. Default value is 0.",
            ),
            "damp_percent": PassConfigParam(
                type_=float,
                default_value=0.01,
                description="Damping factor for quantization. Default value is 0.01.",
            ),
            "static_groups": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use static groups for quantization. Default value is False.",
            ),
            "true_sequential": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use true sequential for quantization. Default value is False.",
            ),
            "desc_act": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use descriptive activation for quantization. Default value is False.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Symmetric quantization. Default value is True.",
            ),
            "gpu": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Use GPU for quantization. Default value is True.",
            ),
            "export_fp16": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Export FP16 onnx model. Default value is True.",
            ),
            "export_optimization": PassConfigParam(
                type_=str,
                default_value=None,
                description="Export optimization level. Default value is None.",
            ),
            "enable_transformers_specific_optimizations": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Enable transformers specific optimizations. Default value is False.",
            ),
        }
        config.update(get_external_data_config())
        return config

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        from optimum.exporters.onnx import onnx_export_from_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

        tokenizer = AutoTokenizer.from_pretrained(model.hf_config.model_name, use_fast=False)

        gptq_config = GPTQConfig(
            bits=config["bits"],
            dataset=config["dataset"],
            group_size=config["group_size"],
            damp_percent=config["damp_percent"],
            static_groups=config["static_groups"],
            true_sequential=config["true_sequential"],
            desc_act=config["desc_act"],
            sym=config["sym"],
            tokenizer=tokenizer,
        )

        from unittest import mock

        def get_onnx_quant_linear(*args, **kwargs):
            return QuantLinearORT

        # GPTQ Quantization in transformers use auto_gptq under the hood, so we
        # replace QuantLinear in auto_gptq with QuantLinearORT for quant linear layer packing
        with mock.patch("auto_gptq.utils.import_utils.dynamically_import_QuantLinear", get_onnx_quant_linear):
            quantized_model = AutoModelForCausalLM.from_pretrained(
                model.hf_config.model_name,
                quantization_config=gptq_config,
                attn_implementation="eager",
                torch_dtype=torch.half,
            )

        onnx_export_from_model(
            model=quantized_model,
            output=output_model_path,
            monolith=False,
            do_validation=True,
            model_kwargs=None,
            device="cuda" if config["gpu"] else "cpu",
            preprocessors=None,
            task="text-generation-with-past",
            fp16=config["export_fp16"],
            optimize=config["export_optimization"],
            enable_transformers_specific_optimizations=config["enable_transformers_specific_optimizations"],
        )

        return ONNXModelHandler(model_path=output_model_path, onnx_file_name="model.onnx")
