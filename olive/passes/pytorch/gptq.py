# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam, get_user_script_data_config
from olive.passes.pytorch.common import inherit_hf_from_hf, inherit_pytorch_from_hf, inherit_pytorch_from_pytorch

logger = logging.getLogger(__name__)


class GptqQuantizer(Pass):
    """GPTQ quantization using Hugging Face Optimum and export model with onnxruntime optimized kernel."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
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
            "layers_block_name": PassConfigParam(
                type_=str,
                default_value="model.layers",
                description=(
                    "Block name to quantize. Default value is model.layers. "
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
                description="Symmetric quantization. Default value is False.",
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
                description=(
                    "Whether to pack the model for ONNX conversion. If True, the model will be saved as a PyTorch model"
                    " with custom quantized layers. If False, the model will be saved in the same format as the input"
                    " model."
                ),
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        # pylint: disable=protected-access
        from auto_gptq import BaseQuantizeConfig
        from auto_gptq.modeling import BaseGPTQForCausalLM
        from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP

        from olive.passes.pytorch.quant_utils import QuantLinear

        if not torch.cuda.is_available():
            raise ValueError("Please use GPU to run gptq quantization.")
        elif self.host_device != Device.GPU:
            logger.debug(
                "GPTQ quantization requires GPU but the host device is %s, will ignore the host device",
                self.host_device,
            )

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

        pytorch_model = model.load_model()
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

        def get_onnx_quant_linear(*args, **kwargs):
            return QuantLinear

        if hasattr(pytorch_model, "config") and pytorch_model.config.model_type in GPTQ_CAUSAL_LM_MODEL_MAP:
            model_type = pytorch_model.config.model_type
            model_class = GPTQ_CAUSAL_LM_MODEL_MAP[model_type]
            quantized_model = model_class(pytorch_model, False, quantize_config)
        else:
            quantized_model = BaseGPTQForCausalLM(pytorch_model, False, quantize_config)
            if not (config["layers_block_name"] and config["outside_layer_modules"] and config["inside_layer_modules"]):
                raise ValueError(
                    "Can't get layers_block_name to quantize automatically, "
                    "please set layers_block_name, outside_layer_modules and inside_layer_modules in config."
                )
            quantized_model.layers_block_name = config["layers_block_name"]
            quantized_model.outside_layer_modules = config["outside_layer_modules"]
            quantized_model.inside_layer_modules = config["inside_layer_modules"]

        import auto_gptq

        original = auto_gptq.modeling._utils.dynamically_import_QuantLinear

        quantizer = get_onnx_quant_linear if config["pack_model_for_onnx_conversion"] else original
        try:
            # Replace QuantLinear in autogptq with QuantLinear for quant linear layer packing
            auto_gptq.modeling._utils.dynamically_import_QuantLinear = quantizer

            # Autogpq quantize_model currently only support cuda device. It accepts model on cpu but
            # will move each block(layer) to cuda before quantization and move back to cpu when finished.
            quantized_model.quantize(dataset)
        finally:
            auto_gptq.modeling._utils.dynamically_import_QuantLinear = original

        if config["pack_model_for_onnx_conversion"] or isinstance(model, PyTorchModelHandler):

            quantized_model = quantized_model.model

            output_model_path = normalize_path_suffix(output_model_path, "model.pt")
            torch.save(quantized_model, output_model_path)

            if isinstance(model, HfModelHandler):
                return inherit_pytorch_from_hf(model, output_model_path)

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
        return inherit_hf_from_hf(model, output_model_path, load_kwargs=new_load_kwargs)
