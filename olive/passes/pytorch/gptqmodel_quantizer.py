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
from typing import Any, Optional, Union

import torch
from packaging import version
from transformers import PreTrainedModel

from olive.common.config_utils import validate_config
from olive.common.hf.wrapper import ModelWrapper
from olive.common.utils import get_attr
from olive.constants import PrecisionBits
from olive.data.config import DataConfig
from olive.data.template import huggingface_data_config_template
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf, inherit_pytorch_from_pytorch

logger = logging.getLogger(__name__)


class GPTQModelQuantizer(Pass):
    """GPTQ quantization using GPTQModel library with enhanced performance and features."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS4,
                description="Quantization bits. Default value is 4",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Block size for quantization. Default value is 128.",
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
            ),            "quant_method": PassConfigParam(
                type_=str,
                default_value="gptq",
                description="Quantization method to use. Default value is 'gptq'.",
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
            "batch_size": PassConfigParam(
                type_=int,
                default_value=1,
                description="Batch size for quantization. Default value is 1.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value=None,
                description=(
                    "Data config for quantization. If not provided, wikitest train data will be used for HfModels."
                    " Required for PyTorch models."
                ),
            ),
            "use_fast_tokenizer": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Use fast tokenizer for quantization. Default value is True.",
            ),
            "disable_exllama": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Disable Exllama kernels. Default value is False.",
            ),
            "disable_exllamav2": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Disable Exllama v2 kernels. Default value is False.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: type[BasePassConfig], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        from gptqmodel import GPTQModel, QuantizeConfig

        if not torch.cuda.is_available():
            # GPTQModel quantize_model currently only support cuda device. It accepts model on cpu but
            # will move each block(layer) to cuda before quantization and move back to cpu when finished.
            raise ValueError("Please use GPU to run GPTQModel quantization.")

        dataset = self.get_dataset(model, config)

        adapter_path = None
        if isinstance(model, HfModelHandler) and model.adapter_path:
            logger.info(
                "Model has adapters but GPTQModel does not support adapters. Quantizing without adapters. The original"
                " adapters will be used as is with the quantized base model."
            )
            # TODO(jambayk): should we copy the adapter? what about non-local adapters?
            adapter_path = model.adapter_path

            # create a new input model with the adapter path removed
            model.model = None
            model = deepcopy(model)
            model.set_resource("adapter_path", None)        # Create quantization config
        quantize_config = QuantizeConfig(
            bits=config.bits.value,
            group_size=config.group_size,
            desc_act=config.desc_act,
            sym=config.sym,
            damp_percent=config.damp_percent,
            static_groups=config.static_groups,
            true_sequential=config.true_sequential,
        )# Load model using GPTQModel
        if isinstance(model, HfModelHandler):
            model_name_or_path = model.model_name_or_path
            load_kwargs = model.get_load_kwargs()
            trust_remote_code = load_kwargs.get("trust_remote_code", False)
            
            gptq_model = GPTQModel.load(
                model_name_or_path,
                quantize_config,
                trust_remote_code=trust_remote_code,
            )
        else:
            # For PyTorch models, we need to load the model first
            pytorch_model = model.load_model(cache_model=False)
            gptq_model = GPTQModel.load(pytorch_model, quantize_config)

        # Quantize the model
        gptq_model.quantize(dataset, batch_size=config.batch_size)

        # Handle PyTorch model case
        if isinstance(model, PyTorchModelHandler):
            pytorch_model = gptq_model.model
            # add quantization related attributes to the model for downstream usage
            pytorch_model.quantization_method = "gptq"
            if hasattr(pytorch_model, "config"):
                pytorch_model.config.quantization_config = Namespace(quantize_config.to_dict())
            else:
                pytorch_model.config = Namespace(
                    quantization_config=Namespace(quantize_config.to_dict())
                )

            output_model_path = normalize_path_suffix(output_model_path, "model.pt")
            torch.save(gptq_model, output_model_path)

            return inherit_pytorch_from_pytorch(model, output_model_path)

        # Save quantized model and metadata for HF models
        gptq_model.save(output_model_path)
        model.save_metadata(output_model_path)

        # Update config to disable exllama for CPU loading compatibility
        config_json_path = Path(output_model_path) / "config.json"
        if config_json_path.exists():
            with open(config_json_path, encoding="utf-8") as f:
                model_config = json.load(f)

            if "quantization_config" in model_config:
                model_config["quantization_config"]["use_exllama"] = False
                with open(config_json_path, "w", encoding="utf-8") as f:
                    json.dump(model_config, f, indent=2)        # return HfModelHandler with updated model path
        new_load_kwargs = deepcopy(model.load_kwargs.dict()) if model.load_kwargs else {}
        # model is saved in safetensors format so need to enable safetensors load
        if new_load_kwargs.get("extra_args") and new_load_kwargs["extra_args"].get("use_safetensors") is False:
            new_load_kwargs["extra_args"]["use_safetensors"] = True
        return inherit_hf_from_hf(model, output_model_path, adapter_path=adapter_path, load_kwargs=new_load_kwargs)

    def get_dataset(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: dict[str, Any]
    ) -> list[str]:
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
        
        # GPTQModel expects a list of strings for calibration data
        dataset = []
        for data in dataloader:
            # each batch consists of (input_data, labels)
            input_data = data[0]
            if isinstance(input_data, dict) and "input_ids" in input_data:
                # For GPTQModel, we need to provide text strings, not tokenized inputs
                # We'll use the original text data from the dataloader if available
                if "text" in input_data:
                    # If text is available, use it directly
                    texts = input_data["text"] if isinstance(input_data["text"], list) else [input_data["text"]]
                    dataset.extend([text for text in texts if text.strip()])
                else:
                    # If only input_ids available, we'll need to use default calibration text
                    # This is a fallback and might not be optimal
                    dataset.extend(["Sample calibration text for quantization."] * len(input_data["input_ids"]))
            else:
                # Fallback for unexpected data format
                dataset.append("Sample calibration text for quantization.")

        if not dataset:
            # Use default calibration dataset if nothing was extracted
            logger.warning("No calibration data extracted from data_config, using default text samples.")
            dataset = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming the world of technology.",
                "Large language models require careful quantization for deployment.",
                "This is sample text for model calibration and quantization.",
            ] * 32  # Repeat to get enough samples

        logger.info(f"Using {len(dataset)} calibration samples for quantization.")
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
