# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# -------------------------------------------------------------------------
import logging
import sys
from typing import Any, Dict, Union

import torch
import transformers
from torch.utils.data import DataLoader, SubsetRandomSampler

from olive.common.config_utils import validate_config
from olive.common.utils import StrEnumBase
from olive.constants import ModelFileFormat
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pytorch.common import inherit_pytorch_from_hf

logger = logging.getLogger(__name__)


class QuaRot(Pass):
    """A new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end.

    See https://arxiv.org/pdf/2404.00456 for more details on the algorithm.

    This pass only supports HfModelHandler.
    """

    class ModelDtype(StrEnumBase):
        # input model's data type, we can assume the model is all float type
        # sometime, the model is in double type, but we can convert it to float type
        # before quantization
        FP32 = "fp32"
        FP16 = "fp16"
        FP64 = "fp64"

        def get_torch_dtype(self):
            return {
                QuaRot.ModelDtype.FP32: torch.float32,
                QuaRot.ModelDtype.FP16: torch.float16,
                QuaRot.ModelDtype.FP64: torch.float64,
            }[self]

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "input_model_dtype": PassConfigParam(
                type_=QuaRot.ModelDtype,
                default_value=QuaRot.ModelDtype.FP16,
                description="Input model's data type.",
            ),
            "calibration_data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                required=False,
                description="Data config for Dataset to calibrate and calculate perplexity on.",
            ),
            "calibration_nsamples": PassConfigParam(
                type_=int,
                required=False,
                default_value=128,
                description="Number of samples of the calibration data to load.",
            ),
            "calibration_batch_size": PassConfigParam(
                type_=int,
                required=False,
                default_value=16,
                description="Batch size for loading the calibration data.",
            ),
            "seed": PassConfigParam(
                type_=int,
                required=False,
                default_value=42,
                description="Seed for sampling the calibration data.",
            ),
            # Rotation configs
            "rotate": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Apply QuaRot/Hadamard rotation to the model.",
            ),
            "rotation_seed": PassConfigParam(
                type_=int,
                default_value=0,
                description="Seed for generating random matrix. Use 0 to replicate paper results.",
            ),
            # weight quantization configs
            "w_rtn": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Apply RTN quantization to the weights.",
            ),
            "w_gptq": PassConfigParam(
                type_=bool,
                default_value=False,
                description="""
                    Apply GPTQ quantization to the weights.
                    It requires flash_attention_2 which only supports Ampere GPUs or newer.
                """,
            ),
            "gptq_damping": PassConfigParam(
                type_=float,
                default_value=0.01,
                description="Damping factor for GPTQ. (ignored for RTN quantization)",
            ),
            "gptq_opt_scales": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Optimize scales for GPTQ (ignored for RTN quantization)",
            ),
            "w_bits": PassConfigParam(
                type_=int,
                default_value=16,
                description="Number of bits for quantizing weights.",
            ),
            "w_asym": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Asymmetric weight quantization (else symmetric by default).",
            ),
            "w_groupsize": PassConfigParam(
                type_=int,
                default_value=None,
                description="Group size for groupwise weight quantization.",
            ),
            # activation quantization configs
            "a_bits": PassConfigParam(
                type_=int,
                default_value=16,
                description="Number of bits for quantizing activations.",
            ),
            "a_asym": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Asymmetric activation quantization (else symmetric by default).",
            ),
            "a_clip_ratio": PassConfigParam(
                type_=float,
                default_value=1.0,
                description="Clip ratio for activation quantization: new_max = max * clip_ratio.",
            ),
            "a_groupsize": PassConfigParam(
                type_=int,
                default_value=None,
                description="Group size for groupwise activation quantization, default is None.",
            ),
            # kv quantization configs
            "k_bits": PassConfigParam(
                type_=int,
                default_value=16,
                description="Number of bits to quantize the keys to.",
            ),
            "k_clip_ratio": PassConfigParam(
                type_=float,
                default_value=1.0,
                description="Clip ratio for keys quantization: new_max = max * clip_ratio.",
            ),
            "k_groupsize": PassConfigParam(
                type_=int,
                default_value=None,
                description="Group size for groupwise key quantization.",
            ),
            "v_bits": PassConfigParam(
                type_=int,
                default_value=16,
                description="Number of bits to quantize the values to.",
            ),
            "v_clip_ratio": PassConfigParam(
                type_=float,
                default_value=1.0,
                description="Clip ratio for values quantization: new_max = max * clip_ratio.",
            ),
            "v_groupsize": PassConfigParam(
                type_=int,
                default_value=None,
                description="Group size for groupwise value quantization.",
            ),
            # Scale Quantization Arguments
            "s_bits": PassConfigParam(
                type_=int,
                default_value=16,
                description="Number of bits to quantize the values to.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        if sys.version_info < (3, 10):
            raise ValueError("QuaRot requires python3.10 or higher")

        from quarot import gptq, hf_utils, quant_utils, rotate, rtn
        from quarot.hf_utils import get_quarot_model, get_quarot_model_adapter, quarot_model_config
        from slicegpt import layernorm_fusion

        # Renaming variables to match their contextual use
        model_handler = model
        model = None

        # convert config to pass config class
        # this will validate the config and convert to the correct types
        config = self._config_class(**config)

        model_adapter, _ = hf_utils.get_model_and_tokenizer(
            model_handler.model_name_or_path,
            dtype=config.input_model_dtype.get_torch_dtype(),
        )
        model_handler.model = model_adapter.model
        model = model_adapter.model

        size_in_mb = quant_utils.count_bytes(model) / 1024 / 1024
        logger.info("Input Model size: %.2f MB", size_in_mb)

        # replace and fuse layers
        if config.rotate:
            layernorm_fusion.fuse_modules(model_adapter)
            rotate.rotate_model(model_adapter, config.rotation_seed)

        quarot_model_dtype = torch.float16
        quantization_keys = (
            "w_bits",
            "w_groupsize",
            "w_asym",
            "a_bits",
            "a_groupsize",
            "a_asym",
            "a_clip_ratio",
            "k_bits",
            "k_clip_ratio",
            "v_bits",
            "v_clip_ratio",
        )
        quantization_kwargs = {key: getattr(config, key) for key in quantization_keys if key in config.dict()}
        model_config = quarot_model_config(
            model_handler.model_name_or_path, dtype=quarot_model_dtype, quantization_kwargs=quantization_kwargs
        )
        with transformers.modeling_utils.no_init_weights():
            # initialize quarot model
            quarot_model = get_quarot_model(
                model_name_or_path=model_handler.model_name_or_path,
                rotate=config.rotate,
                model_config=model_config,
            )

            quarot_model = quarot_model.to(quarot_model_dtype)

            # load the rotated weights into the quarot model
            quarot_model.load_state_dict(model_adapter.model.state_dict(), strict=False)

        quarot_model_adapter = get_quarot_model_adapter(quarot_model)

        if config.w_rtn:
            logger.info("Quantizing weights to INT%d using RTN.", config.w_bits)
            rtn.quantize_model_rtn(
                quarot_model,
                bits=config.w_bits,
                bits_scales=config.s_bits,
                groupsize=config.w_groupsize,
                symmetric=not config.w_asym,
            )
        elif config.w_gptq:
            data_config = validate_config(config.calibration_data_config, DataConfig)
            dataloader = data_config.to_data_container().create_dataloader()
            dataset = [
                {
                    "input_ids": data[0]["input_ids"].squeeze(),
                    "attention_mask": data[0]["attention_mask"].squeeze(),
                    "labels": data[1].squeeze(),
                }
                for data in dataloader
            ]
            torch.manual_seed(config.seed)
            sampler = SubsetRandomSampler(torch.randperm(len(dataset))[: config.calibration_nsamples])
            train_loader = DataLoader(dataset, batch_size=config.calibration_batch_size, sampler=sampler)

            logger.info("Quantizing weights to INT%d using GPTQ.", config.w_bits)
            gptq.quantize_model_gptq(
                quarot_model_adapter,
                train_loader,
                bits=config.w_bits,
                bits_scales=config.s_bits,
                symmetric=not config.w_asym,
                damping=config.gptq_damping,
                groupsize=config.w_groupsize,
                optimize_scales=config.gptq_opt_scales,
            )
        else:
            logger.info("No weight quantization performed")

        size_in_mb = quant_utils.count_bytes(quarot_model) / 1024 / 1024
        logger.info("Quantized Model size: %.2f MB", size_in_mb)

        quarot_model.save_pretrained(output_model_path)

        # return PyTorchModelHandler
        return inherit_pytorch_from_hf(
            model_handler,
            output_model_path,
            model_file_format=ModelFileFormat.PYTORCH_ENTIRE_MODEL,
            model_name=model_handler.model_name_or_path,
        )
