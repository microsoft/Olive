# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on original implementation at
# https://github.com/artidoro/qlora/blob/main/qlora.py
# https://arxiv.org/abs/2305.14314
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict, List, Union

import torch
import transformers
from packaging import version

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModel
from olive.model.hf_utils import get_peft_task_type_from_task, load_huggingface_model_from_task
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"


class QLoRA(Pass):
    """
    Run QLoRA fine-tuning on a Hugging Face PyTorch model.
    See https://arxiv.org/abs/2305.14314 for more details on the method.

    This pass only supports PyTorchModel with hf_config. The transformers model type
    must be one of [gpt_neox]
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            # quantization parameters
            "compute_dtype": PassConfigParam(
                type_=str,
                default_value="bfloat16",
                description=(
                    "The computation data type used by the quantized model. It is also the data type used for the LoRA"
                    " weights. Should be one of `bfloat16`, `float16` or `float32`."
                ),
            ),
            "double_quant": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether tonested quantization where the quantization constants from the first quantization are"
                    " quantized again"
                ),
            ),
            "quant_type": PassConfigParam(
                type_=str,
                default_value="nf4",
                description="Quantization data type to use. Should be one of `fp4` or `nf4`.",
            ),
            # LoRA parameters
            "lora_r": PassConfigParam(type_=int, default_value=64, description="Lora r"),
            "lora_alpha": PassConfigParam(type_=float, default_value=16, description="Lora alpha"),
            "lora_dropout": PassConfigParam(type_=float, default_value=0.0, description="Lora dropout"),
            # data parameters
            "train_data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                # required=True,
                description=(
                    "Data config for fine-tuning training. If `eval_data_config` is not provided and"
                    " `eval_dataset_size` is not None, the data will be split into train and eval. Otherwise, the data"
                    " will be used for training only."
                ),
            ),
            "eval_data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                description=(
                    "Data config for fine-tuning evaluation. Optional if `eval_dataset_size` is provided or evaluation"
                    " is not needed."
                ),
            ),
            "eval_dataset_size": PassConfigParam(
                type_=float,
                default_value=None,
                description=(
                    "Size of the validation dataset. Should be either positive and smaller than the number of train"
                    " sample or a float in the (0, 1) range. If `eval_data_config` is provided, this parameter will be"
                    " ignored."
                ),
            ),
        }

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModel:
        transformers_version = transformers.__version__
        if version.parse(transformers_version) < version.parse("4.30.0"):
            raise RuntimeError(f"QLoRA pass only supports transformers >= 4.30.0, but {transformers_version} is used.")

        # get model and tokenizer
        pytorch_model, tokenizer = self.get_model_tokenizer(model, config)

        # get datasets
        train_dataset, eval_dataset = self.get_datasets(config, data_root)

        return pytorch_model, train_dataset, eval_dataset

    def get_model_tokenizer(self, model: PyTorchModel, config: Dict[str, Any]) -> torch.nn.Module:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from peft.tuners.lora import LoraLayer
        from transformers import AutoTokenizer, BitsAndBytesConfig

        if not model.hf_config:
            raise ValueError("QLoRA pass only supports PyTorchModel with hf_config.")

        model_name = model.hf_config.model_name
        model_path = model.model_path  # can be None, if so, model_name is used to load model
        task = model.hf_config.task

        # get peft task type
        peft_task_type = get_peft_task_type_from_task(task, fail_on_not_found=True)

        # compute_dtype
        supported_dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        assert (
            config["compute_dtype"] in supported_dtypes
        ), f"compute_dtype must be one of {list(supported_dtypes.keys())} but got {config['compute_dtype']}"
        compute_dtype = supported_dtypes[config["compute_dtype"]]

        # load model
        pytorch_model = load_huggingface_model_from_task(
            task=task,
            name=model_path or model_name,
            **{
                # TODO: Worry about `use_multi_gpu` and distributed training later
                # this uses all available GPUs, model parallel
                # "device_map": "auto",
                "device_map": {"": 1},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=config["double_quant"],
                    bnb_4bit_quant_type=config["quant_type"],
                ),
                "torch_dtype": compute_dtype,
            },
        )
        # set model_parallel and is_parallelizable to True
        # we are using "auto" device_map, so model_parallel is True or doing DDP
        # don't want the trainer to do Data Parallel
        setattr(pytorch_model, "model_parallel", True)
        setattr(pytorch_model, "is_parallelizable", True)

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # if there is no pad token, add to tokenizer and model
        self.smart_tokenizer_and_embedding_resize(
            special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN}, tokenizer=tokenizer, model=pytorch_model
        )
        # TODO: need to see if we still need this line https://github.com/artidoro/qlora/blob/main/qlora.py#L362

        # prepare model for kbit training
        # Note: this also converts all float16 and bfloat16 parameters to float32
        # TODO: add gradient checkpointing arg
        pytorch_model = prepare_model_for_kbit_training(pytorch_model)

        # cast float32 linear and embedding layers back to compute_dtype
        for name, module in pytorch_model.named_modules():
            if (
                isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Embedding)
            ) and module.weight.dtype == torch.float32:
                # TODO: why only cast for bfloat16?
                logger.debug(f"Casting {name} to {compute_dtype}")
                module.to(compute_dtype)

        # add lora modules
        logger.debug("Adding LoRA modules")
        # this doesn't pick up the embedding layer and projection layer since those are not quantized
        # this is good since we don't want to touch those, LoRA might not work with input output embedding layers
        modules = self.find_all_linear_names(pytorch_model)
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=modules,
            bias="none",
            task_type=peft_task_type,
        )
        pytorch_model = get_peft_model(pytorch_model, lora_config)

        # cast lora modules to compute_dtype
        for name, module in pytorch_model.named_modules():
            if isinstance(module, LoraLayer):
                # TODO: why only cast for bfloat16? https://github.com/artidoro/qlora/blob/main/qlora.py#L397
                module.to(compute_dtype)

        return pytorch_model, tokenizer

    @staticmethod
    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel
    ):
        """
        Resize the tokenizer and the model embedding layer to take into account new special tokens.
        """
        # resize tokenizer
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        # resize model embedding layer
        model.resize_token_embeddings(len(tokenizer))
        if num_new_tokens > 0:
            logger.info(f"Added {num_new_tokens} new tokens to tokenizer and resized model embedding layer.")
            input_embeddings_data = model.get_input_embeddings().weight.data
            output_embeddings_data = model.get_output_embeddings().weight.data

            # average the embeddings of the pre-existing tokens
            input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

            # set the new embeddings to the average
            input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
            output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

    @staticmethod
    def find_all_linear_names(model: torch.nn.Module) -> List[str]:
        """
        Find all linear layers in a model.
        """
        import bitsandbytes as bnb

        linear_cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, linear_cls):
                lora_module_names.add(name.split(".")[-1])
        return list(lora_module_names)

    def get_datasets(self, config: Dict[str, Any], data_root: str) -> tuple:
        """
        Load training and evaluation datasets.
        """
        train_data_config = validate_config(config["train_data_config"], DataConfig)
        eval_data_config = config["eval_data_config"]
        eval_data_config = validate_config(eval_data_config, DataConfig) if eval_data_config else None
        eval_dataset_size = config["eval_dataset_size"]

        # load training dataset
        train_data_container = train_data_config.to_data_container()
        train_dataset = train_data_container.pre_process(train_data_container.load_dataset(data_root))
        train_dataset = train_dataset.to_hf_dataset(label_name="labels")

        # load evaluation dataset if needed
        if eval_data_config:
            # eval data config has been provided
            eval_data_container = eval_data_config.to_data_container()
            eval_dataset = eval_data_container.pre_process(eval_data_container.load_dataset(data_root))
            eval_dataset = eval_dataset.to_hf_dataset(label_name="labels")
        elif eval_dataset_size:
            if eval_dataset_size >= 1:
                # when eval_dataset_size is an integer, it is the number of samples
                eval_dataset_size = int(eval_dataset_size)
            # eval data config has not been provided, but eval_dataset_size has been provided
            split_data = train_dataset.train_test_split(test_size=eval_dataset_size, shuffle=True, seed=42)
            train_dataset = split_data["train"]
            eval_dataset = split_data["test"]
        else:
            # eval data config has not been provided, and eval_dataset_size has not been provided
            eval_dataset = None

        return train_dataset, eval_dataset
