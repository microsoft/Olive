# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
from packaging import version
from transformers import PreTrainedModel

from olive.common.config_utils import validate_config
from olive.common.hf.wrapper import ModelWrapper
from olive.common.utils import get_attr
from olive.data.config import DataConfig
from olive.data.template import huggingface_data_config_template
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config
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
                description=(
                    "Data config for quantization. If not provided, wikitest train data will be used for HfModels."
                    " Required for PyTorch models."
                ),
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Type[BasePassConfig], output_model_path: str
    ) -> PyTorchModelHandler:
        from auto_gptq import BaseQuantizeConfig, __version__
        from auto_gptq.modeling import BaseGPTQForCausalLM
        from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP

        if not torch.cuda.is_available():
            # Autogpq quantize_model currently only support cuda device. It accepts model on cpu but
            # will move each block(layer) to cuda before quantization and move back to cpu when finished.
            raise ValueError("Please use GPU to run gptq quantization.")

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

        # create model adapter if needed
        model_wrapper = None
        if isinstance(pytorch_model, PreTrainedModel) and model_type not in GPTQ_CAUSAL_LM_MODEL_MAP:
            model_wrapper = ModelWrapper.from_model(pytorch_model)

        quantize_config = BaseQuantizeConfig(
            bits=config.bits,
            group_size=config.group_size,
            damp_percent=config.damp_percent,
            static_groups=config.static_groups,
            true_sequential=config.true_sequential,
            desc_act=config.desc_act,
            sym=config.sym,
            # this is so that the weight gets saved as "model.safetensors"
            model_file_base_name="model",
        )

        model_class = GPTQ_CAUSAL_LM_MODEL_MAP.get(model_type, BaseGPTQForCausalLM)
        quantized_model: BaseGPTQForCausalLM = model_class(pytorch_model, False, quantize_config)

        for key in ["outside_layer_modules", "inside_layer_modules", "layers_block_name"]:
            v = getattr(config, key, None)
            if v:
                # user provided value
                setattr(quantized_model, key, v)
            elif model_type in GPTQ_CAUSAL_LM_MODEL_MAP:
                # gptq supports the model type
                pass
            elif model_wrapper:
                # try to get the value from the model adapter
                setattr(quantized_model, key, self.get_gptq_info(model_wrapper, key))
            else:
                raise ValueError(f"Can't get {key} to quantize automatically, please provide it in config.")

        # quantize the model
        with self._maybe_patch_gptq_model(quantized_model) as quantized_model:
            quantized_model.quantize(dataset)

        # until https://github.com/AutoGPTQ/AutoGPTQ/pull/602, bias was always present
        # in the quantized model, so we need to remove it
        if version.parse(__version__) < version.parse("0.8.0"):
            from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

            qlinear_class = dynamically_import_QuantLinear(
                use_triton=False,
                desc_act=config.desc_act,
                group_size=config.group_size,
                bits=config.bits,
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
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
    def get_gptq_info(model_wrapper: ModelWrapper, name: str) -> List[str]:
        """Get the GPTQ info from the model wrapper."""
        if name == "outside_layer_modules":
            return [*model_wrapper.get_embeds()[1], model_wrapper.get_pre_head_layernorm()[1]]
        if name == "inside_layer_modules":
            layer_wrapper = model_wrapper.get_layer_wrappers()[0]
            return [
                layer_wrapper.get_attention_inputs()[1],
                layer_wrapper.get_attention_outputs()[1],
                layer_wrapper.get_mlp_inputs()[1],
                layer_wrapper.get_mlp_outputs()[1],
            ]
        if name == "layers_block_name":
            return model_wrapper.get_layers()[1]

        raise ValueError(f"Unknown key {name}")

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

    @contextmanager
    def _maybe_patch_gptq_model(self, gptq_model):
        from auto_gptq import __version__ as autogptq_version
        from transformers import __version__ as transformers_version

        # almost all model types have rotary embeddings at model.model.rotary_emb so won't keep a mapping
        rotary_embed_module_name = "model.rotary_emb"
        if (
            version.parse(transformers_version) >= version.parse("4.43")
            and version.parse(autogptq_version).release < version.parse("0.8.0").release
            and get_attr(gptq_model.model, rotary_embed_module_name)
        ):
            rotary_embed_module = get_attr(gptq_model.model, rotary_embed_module_name)

            if rotary_embed_module_name not in gptq_model.outside_layer_modules:
                gptq_model.outside_layer_modules.append(rotary_embed_module_name)

            # add a dummy parameter to the module so that it gets moved to device
            rotary_embed_module.register_parameter("dummy", torch.nn.Parameter(torch.zeros(0), requires_grad=False))

            yield gptq_model

            # remove the dummy parameter
            del rotary_embed_module.dummy
            return

        yield gptq_model
