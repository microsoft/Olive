# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on original implementation at
# LoRA: https://huggingface.co/docs/diffusers/training/lora
# QLoRA: https://github.com/artidoro/qlora/blob/main/qlora.py
#        https://arxiv.org/abs/2305.14314
# --------------------------------------------------------------------------
import dataclasses
import logging
import os
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import transformers
from packaging import version
from pydantic import Field, validator
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from olive.common.config_utils import ConfigBase, ConfigWithExtraArgs
from olive.common.utils import find_submodules
from olive.data.config import DataConfig
from olive.data.constants import IGNORE_INDEX
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModel
from olive.model.hf_utils import HFModelLoadingArgs, get_peft_task_type_from_task
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"


# ruff: noqa: B010
# creating a Config class since transformers.TrainingArguments is a dataclass
# pydantic handles dataclasses differently and causes issues with validation
# this also allows us to handle and validate extra_args better
class HFTrainingArguments(ConfigWithExtraArgs):
    """Training arguments for transformers.Trainer.

    Has the same fields as transformers.TrainingArguments with recommended default values for QLoRA fine-tuning.
    """

    seed: int = Field(42, description="Random seed for initialization.")
    data_seed: int = Field(42, description="Random seed to be used with data samplers.")
    optim: str = Field("paged_adamw_32bit", description="The optimizer to use.")
    per_device_train_batch_size: int = Field(1, description="The batch size per GPU for training.")
    per_device_eval_batch_size: int = Field(1, description="The batch size per GPU for evaluation.")
    gradient_accumulation_steps: int = Field(
        16,
        description=(
            "Number of updates steps to accumulate the gradients for, before performing a backward/update pass."
        ),
    )
    max_steps: int = Field(10000, description="The total number of training steps to perform.")
    # use lora dropout instead for regularization if needed
    weight_decay: float = Field(0.0, description="The L2 weight decay rate of AdamW")
    learning_rate: float = Field(0.0002, description="The initial learning rate for AdamW.")
    gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing. Recommended.")
    lr_scheduler_type: str = Field(
        "constant",
        description="Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis.",
    )
    warmup_ratio: float = Field(0.03, description="Fraction of steps to do a warmup for.")
    logging_steps: int = Field(10, description="Number of update steps between two logs.")
    evaluation_strategy: str = Field(
        "no", description="The evaluation strategy to use. Will be forced to 'no' if there is no eval dataset."
    )
    eval_steps: float = Field(
        None,
        description=(
            "Number of update steps between two evaluations if `evaluation_strategy='steps'`. Will default to the same"
            " value as `logging_steps` if not set"
        ),
    )
    group_by_length: bool = Field(
        True, description="Whether or not to group samples of roughly the same length together when batching."
    )
    report_to: Union[str, List[str]] = Field(
        "none", description="The list of integrations to report the results and logs to."
    )
    output_dir: str = Field(None, description="The output dir for logs and checkpoints. If None, will use a temp dir.")
    extra_args: Dict[str, Any] = Field(
        None,
        description=(
            "Extra arguments to pass to the trainer. Values can be provided directly to this field as a dict or as"
            " keyword arguments to the config. See transformers.TrainingArguments for more details on the available"
            " arguments."
        ),
    )

    @validator("extra_args", pre=True, always=True)
    def validate_extra_args(cls, v):
        if v is None:
            v = {}
        # make sure extra args are fields of transformers.Trainer
        training_args_fields = {f.name for f in dataclasses.fields(transformers.TrainingArguments) if f.init}
        # use_module_with_loss is a field of optimum.onnxruntime.ORTTrainingArguments
        training_args_fields.add("use_module_with_loss")
        for k in list(v):  # need a copy of the keys since we are mutating the dict
            if k == "fp16":
                logger.warning(f"Extra arg {k} is not allowed. Please use `torch_dtype` instead.")
                del v[k]
            elif k not in training_args_fields:
                logger.warning(f"Extra arg {k} is not a field of transformers.TrainingArguments. Ignoring.")
                del v[k]
        return v

    def create_training_args(self, use_ort_trainer: bool) -> transformers.TrainingArguments:
        args = self.dict()
        if not args["output_dir"]:
            raise ValueError("output_dir must be provided.")
        extra_args = args.pop("extra_args")
        if use_ort_trainer:
            from optimum.onnxruntime import ORTTrainingArguments

            training_args_cls = ORTTrainingArguments
        else:
            training_args_cls = transformers.TrainingArguments
            if "use_module_with_loss" in extra_args:
                logger.warning("use_module_with_loss is not supported by transformers.TrainingArguments. Ignoring.")
                extra_args.pop("use_module_with_loss")
        return training_args_cls(**args, **extra_args)


class LoRABase(Pass):
    """Base class for LoRA and QLoRA fine-tuning passes."""

    # these are the attributes of the model (in hf_config) that will be overwritten by the pass
    # values from the input model will be ignored and new values will be set based on the pass config
    model_overwrites: ClassVar[tuple] = ("torch_dtype", "device_map")

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "use_ort_trainer": PassConfigParam(
                type_=bool, default_value=False, description="Whether or not to use ORTTrainer."
            ),
            "lora_r": PassConfigParam(type_=int, default_value=64, description="Lora attention dimension."),
            "lora_alpha": PassConfigParam(
                type_=float, default_value=16, description="The alpha parameter for Lora scaling."
            ),
            "lora_dropout": PassConfigParam(
                type_=float, default_value=0.0, description="The dropout probability for Lora layers."
            ),
            "bias": PassConfigParam(type_=str, default_value="none", description="Bias type for Lora"),
            "modules_to_save": PassConfigParam(
                type_=None,
                default_value=None,
                description=(
                    "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
                ),
            ),
            "torch_dtype": PassConfigParam(
                type_=str,
                default_value="bfloat16",
                description=(
                    "Data type to use for training. Should be one of `bfloat16`, `float16` or `float32`. If `float16`"
                    " will use fp16 mixed-precision training."
                ),
            ),
            "allow_tf32": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether or not to allow TF32 on Ampere GPUs. "
                    "Can be used to speed up training. For more information, "
                    "see 'https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices'"
                ),
            ),
            # data parameters
            "train_data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                required=True,
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
            # training parameters
            "training_args": PassConfigParam(
                type_=Union[HFTrainingArguments, Dict],
                default_value=None,
                description=(
                    "Training arguments. If None, will use default arguments. See HFTrainingArguments for more details."
                ),
            ),
        }

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})
        if search_point.get("use_ort_trainer") and search_point.get("training_args", {}).get("gradient_checkpointing"):
            logger.info(
                "gradient_checkpointing is not supported by onnxruntime-training. Please set gradient_checkpointing"
                " to False."
            )
            return False
        return True

    @classmethod
    def check_dependencies(cls, config: ConfigBase, is_qlora: bool = False):
        """Check dependencies for the pass."""
        if config.use_ort_trainer:
            # check for ort trainer dependencies
            try:
                from optimum.onnxruntime import ORTTrainer  # noqa: F401
                from optimum.onnxruntime.utils import is_onnxruntime_training_available
                from torch_ort import ORTModule  # noqa: F401

                assert is_onnxruntime_training_available(), "onnxruntime-training is not available."
            except (ImportError, AssertionError):
                raise ImportError(
                    "Please install `olive-ai[optimum,ort-training]` or `onnxruntime-training optimum torch-ort` to use"
                    f" {cls.__name__} pass with use_ort_trainer=True."
                ) from None

            # check if model uses bfloat16
            uses_bf16 = cls.get_torch_dtype(config.torch_dtype) == torch.bfloat16
            if is_qlora and config.compute_dtype:
                # qlora compute dtype might be different from torch dtype
                uses_bf16 |= cls.get_torch_dtype(config.compute_dtype) == torch.bfloat16

            from onnxruntime import __version__ as OrtVersion

            # onnxruntime-training doesn't support bfloat16 fully until 1.17.0
            if uses_bf16 and version.parse(OrtVersion) < version.parse("1.17.0"):
                raise ImportError(
                    f"Please install onnxruntime >= 1.17.0 to use {cls.__name__} with bfloat16 and"
                    " use_ort_trainer=True."
                )

            # set the opset version to 16 if using bfloat16
            if uses_bf16:
                original_opset_version = os.environ.get("ORTMODULE_ONNX_OPSET_VERSION", None)
                # will try to convert to int, if not possible, will set to -1
                try:
                    original_opset_version = int(original_opset_version)
                except (ValueError, TypeError):
                    # ValueError: original_opset_version is not a string representation of an integer. E.g. "dummy"
                    # TypeError: original_opset_version is None
                    original_opset_version = -1
                if original_opset_version < 16:
                    logger.debug("Setting ORTMODULE_ONNX_OPSET_VERSION to 16")
                    os.environ["ORTMODULE_ONNX_OPSET_VERSION"] = "16"

        # bitsandbytes quantization only supported after transformers 4.30.0
        if is_qlora and version.parse(transformers.__version__) < version.parse("4.30.0"):
            raise ImportError(f"Please install transformers >= 4.30.0 to use {cls.__name__} pass.")

    @staticmethod
    def collate_batch(batch: List[Dict], tokenizer: transformers.PreTrainedTokenizer) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples into a padded batch of tensors.

        Add padding to the input_ids, attention_mask and labels.
        """
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [sample["input_ids"] for sample in batch]
        attention_mask = None
        if "attention_mask" in batch[0]:
            attention_mask = [sample["attention_mask"] for sample in batch]
        label_col = "labels" if "labels" in batch[0] else "label"
        if label_col not in batch[0]:
            raise ValueError("Batch does not contain 'labels' or 'label' column.")
        labels = [sample[label_col] for sample in batch]

        # apply padding and add to batch
        new_batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
        }
        if attention_mask:
            new_batch["attention_mask"] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return new_batch

    @staticmethod
    def get_datasets(config: ConfigBase, data_root: str) -> tuple:
        """Load training and evaluation datasets."""
        train_data_config = config.train_data_config
        eval_data_config = config.eval_data_config
        eval_dataset_size = config.eval_dataset_size

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
            split_data = train_dataset.train_test_split(
                test_size=eval_dataset_size, shuffle=True, seed=config.training_args.data_seed
            )
            train_dataset = split_data["train"]
            eval_dataset = split_data["test"]
        else:
            # eval data config has not been provided, and eval_dataset_size has not been provided
            eval_dataset = None

        return train_dataset, eval_dataset

    @staticmethod
    def prepare_model_for_lora_finetuning(model: PreTrainedModel, use_gradient_checkpointing: bool):
        """Prepare the model for fine-tuning.

        Freeze base model's layers and prepare model for gradient checkpointing if necessary.
        Similar to peft.prepare_model_for_kbit_training but no casting to fp32 and gradient checkpointing is
        also supported for non-quantized models.
        """
        for param in model.parameters():
            # freeze base model's layers
            param.requires_grad = False

        if use_gradient_checkpointing:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module_, input_, output_):
                    output_.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()

        return model

    def enable_lora(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: str,
        config: ConfigBase,
        target_modules: Optional[List[str]],
    ) -> torch.nn.Module:
        """Run LoRA fine-tuning on a Hugging Face PyTorch model."""
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora import LoraLayer

        logger.debug("Adding LoRA modules")
        # if there is no pad token, add to tokenizer and model
        # TODO(jambayk): Do this in a better way since the embedding size might become unoptimal
        # (not a multiple of 64, etc) perhaps use eos_token as pad_token, but need to ensure the actual eos_token
        # at the end of the sequence is not masked (both in attention mask and loss calculation)
        if not tokenizer.pad_token_id:
            self.smart_tokenizer_and_embedding_resize(
                special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN}, tokenizer=tokenizer, model=model
            )

        if config.training_args.gradient_checkpointing and not model.supports_gradient_checkpointing:
            logger.warning(
                "gradient_checkpointing is True, but model does not support gradient checkpointing! Setting"
                " gradient_checkpoing to False"
            )
            config.training_args.gradient_checkpointing = False

        model = self.prepare_model_for_lora_finetuning(model, config.training_args.gradient_checkpointing)

        # set model_parallel and is_parallelizable to True
        # we are using "auto" device_map, so model_parallel is True or doing DDP
        # don't want the trainer to do Data Parallel
        setattr(model, "model_parallel", True)
        setattr(model, "is_parallelizable", True)

        logger.debug(
            f"The number of trainable parameters in the original model: {self.count_trainable_parameters(model)}"
        )
        peft_task_type = get_peft_task_type_from_task(task, fail_on_not_found=True)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias=config.bias,
            task_type=peft_task_type,
            modules_to_save=config.modules_to_save,
        )

        lora_model = get_peft_model(model, lora_config)
        logger.debug(
            f"The number of trainable parameters in the LoRA model: {self.count_trainable_parameters(lora_model)}"
        )

        # cast lora modules to model's dtype, should be same as torch_dtype
        for module in lora_model.modules():
            if isinstance(module, LoraLayer):
                module.to(lora_model.dtype)

        return lora_model

    def train_and_save_new_model(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ConfigBase,
        data_root: str,
        output_model: PyTorchModel,
        output_model_path: str,
        torch_dtype: torch.dtype,
    ) -> PyTorchModel:
        if torch.cuda.is_available():
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32

        # get datasets
        train_dataset, eval_dataset = self.get_datasets(config, data_root)

        # get training arguments
        if config.training_args.evaluation_strategy is None and eval_dataset is not None:
            logger.info(
                "evaluation_strategy is None, but eval_dataset is not None. Please set evaluation_strategy if"
                " evaluation is needed while training."
            )
        elif config.training_args.evaluation_strategy is not None and eval_dataset is None:
            logger.warning(
                "evaluation_strategy is not None, but eval_dataset is None. Setting evaluation_strategy to 'no'."
            )
            config.training_args.evaluation_strategy = "no"

        # We always create a temp dir even if output_dir is provided because we want the temp dir to be deleted
        # after training or if there is an error
        # With a context manager, the temp dir will be deleted automatically as soon as the context is exited or
        # there is an error
        # If we do `tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")` and there is an error before
        # cleanup or run returns (tmp_dir goes out of scopt), the temp dir will not be deleted until the the exception
        # is handled by the caller (after try except) or the program exits
        # Plus the cleanup after error doesn't work as expected with notebooks
        with tempfile.TemporaryDirectory(prefix="olive_tmp") as temp_dir:
            if not config.training_args.output_dir:
                logger.info("No training_args.output_dir provided. Using a temp dir.")
                config.training_args.output_dir = temp_dir
                # set save_total_limit to 1 since the temp dir will be deleted after training
                config.training_args.extra_args["save_total_limit"] = 1
            if torch_dtype == torch.float16:
                # use fp16 mixed precision training
                config.training_args.extra_args["fp16"] = True
            # create training args
            logger.debug(f"Training args: {config.training_args.dict()}")

            trainer_cls = transformers.Trainer
            if config.use_ort_trainer:
                from optimum.onnxruntime import ORTTrainer

                trainer_cls = ORTTrainer

            # get trainer
            trainer = trainer_cls(
                model=model,
                tokenizer=tokenizer,
                args=config.training_args.create_training_args(config.use_ort_trainer),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=partial(self.collate_batch, tokenizer=tokenizer),
            )
            # TODO(jambayk): trainer callback for saving might be needed for DDP training
            # worry about this later

            # train
            logger.info("Running fine-tuning")
            train_result = trainer.train()
            logger.debug(f"train_result: {train_result}")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        # save adapter weights
        adapter_path = Path(output_model_path) / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_path)

        # remove loaded model
        output_model.model = None
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # remove the device map since we don't want "auto" device map
        output_model.hf_config.model_loading_args.device_map = None
        # remove model_overwrites from model_attributes
        if output_model.model_attributes:
            for k in self.model_overwrites:
                output_model.model_attributes.pop(k, None)

        # set adapter_path
        output_model.set_resource("adapter_path", adapter_path)
        return output_model

    @staticmethod
    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel
    ):
        """Resize the tokenizer and the model embedding layer to take into account new special tokens."""
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
    def get_torch_dtype(torch_dtype: str):
        supported_dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        assert (
            torch_dtype in supported_dtypes
        ), f"torch_dtype must be one of {list(supported_dtypes.keys())} but got {torch_dtype}"
        return supported_dtypes[torch_dtype]

    @classmethod
    def input_model_check(cls, model):
        """Validate the input model and reset model_loading_args and adapter_path."""
        if not model.hf_config:
            raise ValueError(f"{cls.__name__} pass only supports PyTorchModel with hf_config.")

        # load model, reset model_loading_args and adapter_path
        model_loading_args = {}
        if model.hf_config.model_loading_args:
            model_loading_args = model.hf_config.model_loading_args.dict()
            for k in cls.model_overwrites:
                if model_loading_args.get(k) is not None:
                    logger.warning(
                        f"Input model has model_loading_args.{k}. Ignoring. {cls.__name__} will overwrite it based on"
                        " the pass config."
                    )

        if model.get_resource("adapter_path"):
            logger.warning(
                "Input model has adapter_path. Ignoring. QLoRA will save the adapter weights to its own adapter_path."
            )
        model.set_resource("adapter_path", None)
        return model

    @staticmethod
    def count_trainable_parameters(model):
        """Count and return the number of trainable parameters in a model."""
        trainable_params = 0
        all_param = 0
        for param in model.parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return (
            f"trainable params: {trainable_params} || all params: {all_param} "
            f"|| trainable%: {100 * trainable_params / all_param:.2f}"
        )


class LoRA(LoRABase):
    """Run LoRA fine-tuning on a Hugging Face PyTorch model.

    This pass only supports PyTorchModel with hf_config.
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_modules": PassConfigParam(type_=List[str], default_value=None, description="Target modules"),
        }
        config.update(LoRABase._default_config(accelerator_spec))
        return config

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModel:
        # convert config to pass config class
        # this will validate the config and convert to the correct types
        config = self._config_class(**config)

        # check dependencies
        self.check_dependencies(config)

        # use default training args if not provided
        config.training_args = config.training_args or HFTrainingArguments()

        # create a copy of the input model, will modify this model
        model.model = None
        new_model = self.input_model_check(deepcopy(model))

        torch_dtype = self.get_torch_dtype(config.torch_dtype)
        # will use mixed precision since full fp16 is unstable
        model_dtype = torch_dtype if torch_dtype != torch.float16 else torch.float32

        # load model, reset model_loading_args and adapter_path
        model_loading_args = (
            new_model.hf_config.model_loading_args.dict() if new_model.hf_config.model_loading_args else {}
        )
        model_loading_args.update(
            {"torch_dtype": model_dtype, "device_map": "auto" if not config.use_ort_trainer else None}
        )
        new_model.hf_config.model_loading_args = HFModelLoadingArgs(**model_loading_args)
        pytorch_model = new_model.load_model()
        if torch.cuda.is_available() and config.use_ort_trainer:
            # put the model on GPU since device_map is None and the model will be on CPU
            pytorch_model.to("cuda")
        pytorch_model.config.torch_dtype = model_dtype

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            new_model.hf_config.model_name,
            trust_remote_code=new_model.hf_config.model_loading_args.trust_remote_code,
        )

        # add lora modules
        pytorch_model = self.enable_lora(
            pytorch_model, tokenizer, new_model.hf_config.task, config, config.target_modules
        )

        # train and return new model
        return self.train_and_save_new_model(
            pytorch_model, tokenizer, config, data_root, new_model, output_model_path, torch_dtype
        )


class QLoRA(LoRABase):
    """Run QLoRA fine-tuning on a Hugging Face PyTorch model.

    This pass only supports PyTorchModel with hf_config.
    """

    model_overwrites: ClassVar[tuple] = ("torch_dtype", "device_map", "quantization_method", "quantization_config")

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            # quantization parameters
            "double_quant": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to use nested quantization where the quantization constants from the first quantization"
                    " are quantized again."
                ),
            ),
            "quant_type": PassConfigParam(
                type_=str,
                default_value="nf4",
                description="Quantization data type to use. Should be one of `fp4` or `nf4`.",
            ),
            "compute_dtype": PassConfigParam(
                type_=str,
                description=(
                    "Computation data type for the quantized modules. If not provided, will use the same dtype as"
                    " torch_dtype"
                ),
            ),
        }
        config.update(LoRABase._default_config(accelerator_spec))
        return config

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModel:
        # convert config to pass config class
        # this will validate the config and convert to the correct types
        config = self._config_class(**config)

        # check dependencies
        self.check_dependencies(config, is_qlora=True)

        # MatMulBnb4 contrib op doesn't support double quantization so the trainer falls back to PythonOp
        # which uses more memory and is slower
        if config.use_ort_trainer and config.double_quant:
            logger.warning(
                "double_quant is set to True but it is inefficient with onnxruntime-training! Consider setting it to"
                " False."
            )

        # use default training args if not provided
        config.training_args = config.training_args or HFTrainingArguments()

        # get models and tokenizer
        new_model, pytorch_model, tokenizer, quantized_modules, torch_dtype = self.get_model_tokenizer(model, config)

        # train and get new model
        output_model = self.train_and_save_new_model(
            pytorch_model, tokenizer, config, data_root, new_model, output_model_path, torch_dtype
        )
        # add quantized_modules attributes
        output_model.model_attributes["quantized_modules"] = quantized_modules
        return output_model

    def get_model_tokenizer(
        self, model: PyTorchModel, config: ConfigBase
    ) -> Tuple[PyTorchModel, PreTrainedModel, PreTrainedTokenizer, List[str], torch.dtype]:
        """Get the Olive model, PyTorch model and tokenizer for QLoRA fine-tuning."""
        import bitsandbytes as bnb

        # don't want the original loaded model
        # also frees gpu memory if original model is on gpu
        model.model = None
        # create copy of the input model, will modify this model
        new_model = self.input_model_check(deepcopy(model))

        torch_dtype = self.get_torch_dtype(config.torch_dtype)
        # will use mixed precision since full fp16 is unstable
        model_dtype = torch_dtype if torch_dtype != torch.float16 else torch.float32

        # load model, reset model_loading_args and adapter_path
        model_loading_args = (
            new_model.hf_config.model_loading_args.dict() if new_model.hf_config.model_loading_args else {}
        )
        model_loading_args.update(
            {
                "torch_dtype": model_dtype,
                # TODO(jambayk): Worry about `use_multi_gpu` and distributed training later
                # this uses all available GPUs, model parallel
                # ORTTrainer falls back to pytorch when model parallel is used
                # use `None` device_map to only use one GPU
                "device_map": "auto" if not config.use_ort_trainer else None,
                "quantization_method": "bitsandbytes",
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": self.get_torch_dtype(config.compute_dtype or config.torch_dtype),
                    "bnb_4bit_use_double_quant": config.double_quant,
                    "bnb_4bit_quant_type": config.quant_type,
                },
            }
        )
        new_model.hf_config.model_loading_args = HFModelLoadingArgs(**model_loading_args)
        pytorch_model = new_model.load_model()
        pytorch_model.config.torch_dtype = model_dtype

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            new_model.hf_config.model_name,
            trust_remote_code=new_model.hf_config.model_loading_args.trust_remote_code,
        )

        # TODO(jambayk): need to see if we still need this line
        # https://github.com/artidoro/qlora/blob/main/qlora.py#L362

        # add lora modules
        # this doesn't pick up the embedding layer and projection layer since those are not quantized
        # this is good since we don't want to touch those, LoRA might not work with input output embedding layers
        target_modules = find_submodules(pytorch_model, bnb.nn.Linear4bit)
        pytorch_model = self.enable_lora(pytorch_model, tokenizer, new_model.hf_config.task, config, target_modules)

        return new_model, pytorch_model, tokenizer, target_modules, torch_dtype
