# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from packaging import version

from olive.common.config_utils import validate_config
from olive.common.hf.mappings import MODEL_INSIDE_LAYER_MODULES, MODEL_OUTSIDE_LAYER_MODULES, MODELS_TO_LAYERS_MAPPING
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam, get_user_script_data_config
from olive.passes.pytorch.common import inherit_hf_from_hf, inherit_pytorch_from_pytorch

logger = logging.getLogger(__name__)


class GptqQuantizer(Pass):
    """GPTQ quantization using Hugging Face Optimum and export model with onnxruntime optimized kernel."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "bits": PassConfigParam(
                type_=int,
                default_value=4,
                description="quantization bits. Default value is 4",
            ),
            "layers_block_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Block name to quantize. "
                    "For models can't be auto filled, you can refer this link to fill these parameters.\n"
                    "https://github.com/AutoGPTQ/AutoGPTQ/blob/896d8204bc89a7cfbda42bf3314e13cf4ce20b02/auto_gptq/modeling/llama.py#L19-L26"
                ),
            ),
            "outside_layer_modules": PassConfigParam(
                type_=List[str],
                default_value=None,
                description=(
                    "Names of other nn modules that in the same level as the transformer layer block. "
                    "Default value is None."
                ),
            ),
            "inside_layer_modules": PassConfigParam(
                type_=List[List[str]],
                default_value=None,
                description="Names of linear layers in transformer layer module. Default value is None.",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Block size for quantization. Default value is 128.",
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
                description="Symmetric quantization. Default value is False.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                default_value=None,
                description="""
                    Data config for quantization. Default value is None.
                """,
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        from auto_gptq import BaseQuantizeConfig, __version__
        from auto_gptq.modeling import BaseGPTQForCausalLM
        from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP

        if not torch.cuda.is_available():
            # Autogpq quantize_model currently only support cuda device. It accepts model on cpu but
            # will move each block(layer) to cuda before quantization and move back to cpu when finished.
            raise ValueError("Please use GPU to run gptq quantization.")

        dataset = None
        if config["data_config"]:
            data_config = validate_config(config["data_config"], DataConfig)
            dataloader = data_config.to_data_container().create_dataloader()
            dataset = [data[0] for data in dataloader]

        if (
            not dataset
            or not isinstance(dataset, list)
            or not isinstance(dataset[0], dict)
            or ("input_ids" not in dataset[0] or "attention_mask" not in dataset[0])
        ):
            raise ValueError(
                "Provided dataset is invalid. The returned datasets is a list of tokenized data "
                "(e.g. [{ 'input_ids': [ 1, 100, 15, ... ],'attention_mask': [ 1, 1, 1, ... ]},...])"
            )

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
        quantize_config = BaseQuantizeConfig(
            bits=config["bits"],
            group_size=config["group_size"],
            damp_percent=config["damp_percent"],
            static_groups=config["static_groups"],
            true_sequential=config["true_sequential"],
            desc_act=config["desc_act"],
            sym=config["sym"],
            # this is so that the weight gets saved as "model.safetensors"
            model_file_base_name="model",
        )

        model_type = pytorch_model.config.model_type if hasattr(pytorch_model, "config") else ""
        model_class = GPTQ_CAUSAL_LM_MODEL_MAP.get(model_type, BaseGPTQForCausalLM)
        quantized_model: BaseGPTQForCausalLM = model_class(pytorch_model, False, quantize_config)

        fields_to_set = {
            "outside_layer_modules": MODEL_OUTSIDE_LAYER_MODULES,
            "inside_layer_modules": MODEL_INSIDE_LAYER_MODULES,
            "layers_block_name": MODELS_TO_LAYERS_MAPPING,
        }
        for key, value in fields_to_set.items():
            if config[key]:
                setattr(quantized_model, key, config[key])
            elif model_type not in GPTQ_CAUSAL_LM_MODEL_MAP:
                if model_type in value:
                    setattr(quantized_model, key, value[model_type])
                else:
                    raise ValueError(f"Can't get {key} to quantize automatically, please provide it in config.")

        quantized_model.quantize(dataset)

        # until https://github.com/AutoGPTQ/AutoGPTQ/pull/602, bias was always present
        # in the quantized model, so we need to remove it
        if version.parse(__version__) < version.parse("0.8.0"):
            from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

            qlinear_class = dynamically_import_QuantLinear(
                use_triton=False,
                desc_act=config["desc_act"],
                group_size=config["group_size"],
                bits=config["bits"],
                disable_exllama=False,
                disable_exllamav2=True,
            )

            for module in quantized_model.modules():
                if not isinstance(module, qlinear_class) or module.bias is None:
                    continue

                if all(module.bias == 0):
                    module.bias = None

        # TODO(anyone): Is pytorch model support needed? auto-awq only works with transformers like models
        if isinstance(model, PyTorchModelHandler):
            pytorch_model = quantized_model.model
            # add quantization related attributes to the model for downstream usage
            pytorch_model.quantization_method = "gptq"
            if hasattr(pytorch_model, "config"):
                pytorch_model.config.quantization_config = Namespace(quantized_model.quantize_config.to_dict())
            else:
                pytorch_model.config = Namespace(
                    quantization_config=Namespace(quantized_model.quantize_config.to_dict())
                )

            output_model_path = normalize_path_suffix(output_model_path, "model.pt")
            torch.save(quantized_model, output_model_path)

            return inherit_pytorch_from_pytorch(model, output_model_path)

        # save quantized model and metadata
        model.save_metadata(output_model_path)
        quantized_model.save_quantized(output_model_path)

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
