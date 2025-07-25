# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import torch

from olive.common.config_utils import validate_config
from olive.common.hf.utils import get_tokenizer
from olive.constants import PrecisionBits
from olive.data.config import DataConfig
from olive.data.template import huggingface_data_config_template
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf

logger = logging.getLogger(__name__)


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

    @torch.no_grad()
    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: type[BasePassConfig], output_model_path: str
    ) -> PyTorchModelHandler:
        from gptqmodel import QuantizeConfig
        from gptqmodel.models.auto import MODEL_MAP, BaseGPTQModel

        dataset = self.get_dataset(model, config)

        adapter_path = None
        if isinstance(model, HfModelHandler) and model.adapter_path:
            logger.info(
                "Model has adapters but GPTQ does not support adapters. Quantizing without adapters. The original"
                " adapters will be used as is with the quantized base model."
            )
            # TODO(jambayk): should we copy the adapter? what about non-local adapters?
            adapter_path = model.adapter_path

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
            pytorch_model, False, quantize_config, trust_remote_code=True, model_local_path=model.model_path
        )

        # quantize the model
        quantized_model.quantize(dataset, tokenizer=get_tokenizer(model.model_path))

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
        new_load_kwargs = deepcopy(model.load_kwargs.dict()) if model.load_kwargs else {}
        # model is saved in safetensors format so need to enable safetensors load
        if new_load_kwargs.get("extra_args") and new_load_kwargs["extra_args"].get("use_safetensors") is False:
            new_load_kwargs["extra_args"]["use_safetensors"] = True
        return inherit_hf_from_hf(model, output_model_path, adapter_path=adapter_path, load_kwargs=new_load_kwargs)

    def get_dataset(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get the dataset for quantization."""
        data_config = config.data_config
        if not data_config and isinstance(model, HfModelHandler):
            data_config = self.get_calibration_data_config(
                model.model_name_or_path, trust_remote_code=model.get_load_kwargs().get("trust_remote_code", None)
            )
        elif not data_config:
            raise ValueError("Data config is required for PyTorch model.")
        data_config = validate_config(data_config, DataConfig)
        dataloader = data_config.to_data_container().create_dataloader()
        # each batch consists of (input_data, labels)
        dataset = [data[0] for data in dataloader]
        if (
            not dataset
            or not isinstance(dataset, list)
            or not isinstance(dataset[0], dict)
            or ("input_ids" not in dataset[0] or "attention_mask" not in dataset[0])
        ):
            raise ValueError(
                "Provided dataset is invalid. The returned datasets is a list of tokenized data "
                "(e.g. [{ 'input_ids': [[ 1, 100, 15, ... ]],'attention_mask': [[ 1, 1, 1, ... ]]},...])"
            )

        return dataset

    @staticmethod
    def get_calibration_data_config(model_name_or_path: str, trust_remote_code: Optional[bool] = None):
        return huggingface_data_config_template(
            model_name=model_name_or_path,
            task="text-generation",
            load_dataset_config={
                "data_name": "wikitext",
                "subset": "wikitext-2-raw-v1",
                # only require 128 samples for calibration
                "split": "train[:1000]",
                "trust_remote_code": trust_remote_code,
            },
            pre_process_data_config={
                # should we randomize the data?
                "add_special_tokens": False,
                "max_seq_len": 2048,
                "max_samples": 128,
                "trust_remote_code": trust_remote_code,
            },
        )
