# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch

from olive.common.hf.utils import get_tokenizer
from olive.constants import PrecisionBits
from olive.data.config import DataConfig
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.train_utils import get_calibration_dataset

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec


logger = logging.getLogger(__name__)


def _get_embed_quant_config(config: type[BasePassConfig]):
    if config.embed_quant_mode is None:
        return None

    from gptqmodel import QuantizeEmbed, QuantizeEmbedConfig

    return QuantizeEmbedConfig(
        embed_quant_mode=QuantizeEmbed(config.embed_quant_mode),
        embed_only=config.embed_only,
    )


class GptqModel(Pass):
    """GPTQ quantization using Hugging Face Optimum and export model with onnxruntime optimized kernel."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS4,
                description="quantization bits. Default value is 4",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=-1,
                description="Block size for quantization. Default value is -1.",
            ),
            "damp_percent": PassConfigParam(
                type_=float,
                default_value=0.05,
                description="Damping factor for quantization. Default value is 0.05.",
            ),
            "damp_auto_increment": PassConfigParam(
                type_=float,
                default_value=0.01,
                description="Damping increment for quantization. Default value is 0.01.",
            ),
            "static_groups": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use static groups for quantization. Default value is False.",
            ),
            "true_sequential": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Use true sequential for quantization. Default value is True.",
            ),
            "desc_act": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Reorder weights based on activation importance. Default value is True.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Symmetric quantization. Default value is True.",
            ),
            "mse": PassConfigParam(
                type_=float,
                default_value=0.0,
                description="mean square error calculation. Default value is 0.0.",
            ),
            "lm_head": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to quantized lm_head or not. Default value is False.",
            ),
            "embed_quant_mode": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description=(
                    "Quantize input embeddings, output embeddings (lm_head), or both using GPTQModel's official "
                    "embedding quantization flow. Accepted values are 'input', 'output', and 'both'. GPTQModel "
                    "unties shared input/output embeddings before quantizing them independently. Defaults to None "
                    "to preserve decoder-only quantization and ModelBuilder compatibility. Input embedding "
                    "quantization is not compatible with ModelBuilder until its GPTQ loader supports packed inputs."
                ),
            ),
            "embed_only": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Only quantize the endpoints selected by embed_quant_mode. Defaults to False so decoder layers "
                    "and selected embedding endpoints are quantized in one invocation. Ignored when "
                    "embed_quant_mode is None."
                ),
            ),
            "device": PassConfigParam(
                type_=str,
                default_value="cpu",
                description="Whether to run quantization on cpu or gpu. Accepted values are 'cpu' and 'cuda'.",
            ),
            "dynamic": PassConfigParam(
                type_=dict[str, dict[str, Union[int, bool]]],
                default_value=None,
                description="Dynamic quantization configuration. Default value is None.",
            ),
            "rotation": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description="Rotation configuration for quantization. Values supported ['hadamard', 'random'].",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value=None,
                description=(
                    "Data config for quantization. If not provided, wikitest train data will be used for HfModels."
                    " Required for PyTorch models."
                ),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.embed_quant_mode not in {None, "input", "output", "both"}:
            logger.info("embed_quant_mode must be None or one of 'input', 'output', or 'both'.")
            return False

        if config.lm_head and config.embed_quant_mode in {"output", "both"}:
            logger.info("lm_head must be False when embed_quant_mode already includes the output embedding.")
            return False

        return True

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler | PyTorchModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> PyTorchModelHandler:
        from gptqmodel import QuantizeConfig
        from gptqmodel.models.auto import MODEL_MAP, BaseGPTQModel

        dataset = get_calibration_dataset(model, config.data_config)

        adapter_path = None
        trust_remote_code = False
        if isinstance(model, HfModelHandler) and model.adapter_path:
            logger.info(
                "Model has adapters but GPTQ does not support adapters. Quantizing without adapters. The original"
                " adapters will be used as is with the quantized base model."
            )
            # TODO(jambayk): should we copy the adapter? what about non-local adapters?
            adapter_path = model.adapter_path
            trust_remote_code = model.get_load_kwargs().get("trust_remote_code", False)

            # create a new input model with the adapter path removed
            model.model = None
            model = deepcopy(model)
            model.set_resource("adapter_path", None)

        pytorch_model = model.load_model(cache_model=False)
        model_type = pytorch_model.config.model_type if hasattr(pytorch_model, "config") else ""

        quantize_config = QuantizeConfig(
            bits=config.bits.value,
            group_size=config.group_size,
            sym=config.sym,
            mse=config.mse,
            lm_head=config.lm_head,
            dynamic=config.dynamic,
            device=config.device,
            rotation=config.rotation,
            static_groups=config.static_groups,
            desc_act=config.desc_act,
            damp_percent=config.damp_percent,
            damp_auto_increment=config.damp_auto_increment,
            true_sequential=config.true_sequential,
        )

        model_class = MODEL_MAP.get(model_type, BaseGPTQModel)
        quantized_model: BaseGPTQModel = model_class(
            pytorch_model,
            False,
            quantize_config,
            trust_remote_code=trust_remote_code,
            model_local_path=model.model_path,
        )

        # quantize the model
        quantized_model.quantize(
            dataset,
            tokenizer=get_tokenizer(model.model_path),
            embed_quant_config=_get_embed_quant_config(config),
        )

        # save quantized model and metadata
        quantized_model.save_quantized(output_model_path)
        model.save_metadata(output_model_path)

        # need to disable exllama to be able to load on cpu
        # should we do this using load kwargs? It works but transformers prints a warning
        config_json_path = Path(output_model_path) / "config.json"
        with open(config_json_path, encoding="utf-8") as f:
            model_config = json.load(f)

        model_config["quantization_config"]["use_exllama"] = False
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        # return HfModelHandler with updated model path
        new_load_kwargs = deepcopy(model.load_kwargs.model_dump()) if model.load_kwargs else {}
        # model is saved in safetensors format so need to enable safetensors load
        if new_load_kwargs.get("extra_args") and new_load_kwargs["extra_args"].get("use_safetensors") is False:
            new_load_kwargs["extra_args"]["use_safetensors"] = True
        return inherit_hf_from_hf(model, output_model_path, adapter_path=adapter_path, load_kwargs=new_load_kwargs)
