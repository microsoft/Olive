# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on original implementation at
# https://github.com/artidoro/qlora/blob/main/qlora.py
# https://arxiv.org/abs/2305.14314
# --------------------------------------------------------------------------
import dataclasses
import logging
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import transformers
from packaging import version
from pydantic import Field, validator

from olive.common.config_utils import ConfigBase, ConfigWithExtraArgs
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
        for k in list(v):  # need a copy of the keys since we are mutating the dict
            if k == "output_dir":
                logger.warning(f"Extra arg {k} is not allowed. Please use `training_output_dir` instead.")
                del v[k]
            elif k not in training_args_fields:
                logger.warning(f"Extra arg {k} is not a field of transformers.TrainingArguments. Ignoring.")
                del v[k]
        return v

    def create_training_args(self) -> transformers.TrainingArguments:
        args = self.dict()
        if not args["output_dir"]:
            raise ValueError("output_dir must be provided.")
        extra_args = args.pop("extra_args")
        return transformers.TrainingArguments(**args, **extra_args)


class QLoRA(Pass):
    """Run QLoRA fine-tuning on a Hugging Face PyTorch model.

    See https://arxiv.org/abs/2305.14314 for more details on the method.

    This pass only supports PyTorchModel with hf_config.
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

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModel:
        transformers_version = transformers.__version__
        if version.parse(transformers_version) < version.parse("4.30.0"):
            raise RuntimeError(f"QLoRA pass only supports transformers >= 4.30.0, but {transformers_version} is used.")

        if torch.cuda.is_available():
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

        # convert config to pass config class
        # this will validate the config and convert to the correct types
        config = self._config_class(**config)

        # use default training args if not provided
        config.training_args = config.training_args or HFTrainingArguments()

        # get model and tokenizer
        new_model, pytorch_model, tokenizer = self.get_model_tokenizer(model, config)

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
                logger.info("No training_output_dir provided. Using a temp dir.")
                config.training_args.output_dir = temp_dir
                # set save_total_limit to 1 since the temp dir will be deleted after training
                config.training_args.extra_args["save_total_limit"] = 1

            # get trainer
            trainer = transformers.Trainer(
                model=pytorch_model,
                tokenizer=tokenizer,
                args=config.training_args.create_training_args(),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=partial(self.collate_batch, tokenizer=tokenizer),
            )
            # TODO(jambayk): trainer callback for saving might be needed for DDP training
            # worry about this later

            # train
            logger.info("Running QLoRA fine-tuning")
            train_result = trainer.train()
            logger.debug(f"train_result: {train_result}")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        # save adapter weights
        adapter_path = Path(output_model_path) / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        pytorch_model.save_pretrained(adapter_path)

        # remove loaded model
        new_model.model = None
        # remove the device map since we don't want "auto" device map
        new_model.hf_config.model_loading_args.device_map = None
        # set adapter_path
        new_model.set_resource("adapter_path", adapter_path)

        return new_model

    @classmethod
    def get_model_tokenizer(
        cls, model: PyTorchModel, config: ConfigBase
    ) -> Tuple[PyTorchModel, transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Get the Olive model, PyTorch model and tokenizer for QLoRA fine-tuning."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from peft.tuners.lora import LoraLayer
        from transformers import AutoTokenizer

        # don't want the original loaded model
        # also frees gpu memory if original model is on gpu
        model.model = None
        # create copy of the input model, will modify this model
        new_model = deepcopy(model)

        if not new_model.hf_config:
            raise ValueError("QLoRA pass only supports PyTorchModel with hf_config.")

        model_name = new_model.hf_config.model_name
        task = new_model.hf_config.task

        # get peft task type
        peft_task_type = get_peft_task_type_from_task(task, fail_on_not_found=True)

        # compute_dtype
        supported_dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        assert (
            config.compute_dtype in supported_dtypes
        ), f"compute_dtype must be one of {list(supported_dtypes.keys())} but got {config['compute_dtype']}"
        compute_dtype = supported_dtypes[config.compute_dtype]

        # load model, reset model_loading_args and adapter_path
        if new_model.hf_config.model_loading_args:
            logger.warning(
                "Input model has model_loading_args. Ignoring. QLoRA will use its own model_loading_args based on the"
                " pass config."
            )
        new_model.hf_config.model_loading_args = HFModelLoadingArgs(
            torch_dtype=compute_dtype,
            # TODO(jambayk): Worry about `use_multi_gpu` and distributed training later
            # this uses all available GPUs, model parallel
            device_map="auto",
            quantization_method="bitsandbytes",
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": compute_dtype,
                "bnb_4bit_use_double_quant": config.double_quant,
                "bnb_4bit_quant_type": config.quant_type,
            },
        )
        if new_model.get_resource("adapter_path"):
            logger.warning(
                "Input model has adapter_path. Ignoring. QLoRA will save the adapter weights to its own adapter_path."
            )
        new_model.set_resource("adapter_path", None)
        pytorch_model = new_model.load_model()
        # set model_parallel and is_parallelizable to True
        # we are using "auto" device_map, so model_parallel is True or doing DDP
        # don't want the trainer to do Data Parallel
        setattr(pytorch_model, "model_parallel", True)
        setattr(pytorch_model, "is_parallelizable", True)

        pytorch_model.config.torch_dtype = compute_dtype

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # if there is no pad token, add to tokenizer and model
        # TODO(jambayk): Do this in a better way since the embedding size might become unoptimal
        # (not a multiple of 64, etc) perhaps use eos_token as pad_token, but need to ensure the actual eos_token
        # at the end of the sequence is not masked (both in attention mask and loss calculation)
        if not tokenizer.pad_token_id:
            cls.smart_tokenizer_and_embedding_resize(
                special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN}, tokenizer=tokenizer, model=pytorch_model
            )
        # TODO(jambayk): need to see if we still need this line
        # https://github.com/artidoro/qlora/blob/main/qlora.py#L362

        # prepare model for kbit training
        # Note: this also converts all float16 and bfloat16 parameters to float32
        pytorch_model = prepare_model_for_kbit_training(
            pytorch_model, use_gradient_checkpointing=config.training_args.gradient_checkpointing
        )

        # TODO(jambayk): should we make this optional? fp16 is unstable?
        # https://github.com/artidoro/qlora/blob/main/qlora.py#L396 doesn't work for all models
        # mismatch between dtypes
        # we will just undo the float32 casting from prepare_model_for_kbit_training and cast to compute_dtype
        for param in pytorch_model.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(compute_dtype)

        # add lora modules
        logger.debug("Adding LoRA modules")
        # this doesn't pick up the embedding layer and projection layer since those are not quantized
        # this is good since we don't want to touch those, LoRA might not work with input output embedding layers
        modules = cls.find_all_linear_names(pytorch_model)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=modules,
            bias="none",
            task_type=peft_task_type,
        )
        pytorch_model = get_peft_model(pytorch_model, lora_config)

        # cast lora modules to compute_dtype
        for module in pytorch_model.modules():
            if isinstance(module, LoraLayer):
                # TODO(jambayk): why only cast for bfloat16? https://github.com/artidoro/qlora/blob/main/qlora.py#L397
                module.to(compute_dtype)

        return new_model, pytorch_model, tokenizer

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
    def find_all_linear_names(model: torch.nn.Module) -> List[str]:
        """Find all linear layers in a model."""
        import bitsandbytes as bnb

        linear_cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, linear_cls):
                lora_module_names.add(name.split(".")[-1])
        return list(lora_module_names)

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
