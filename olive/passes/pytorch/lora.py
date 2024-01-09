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
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import transformers
from packaging import version
from transformers import AutoTokenizer

from olive.common.config_utils import ConfigBase, ConfigWithExtraArgs
from olive.common.pydantic_v1 import Field, validator
from olive.common.utils import find_submodules
from olive.data.config import DataConfig
from olive.data.constants import IGNORE_INDEX
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModelHandler
from olive.model.config.hf_config import HfFromPretrainedArgs
from olive.model.utils.hf_utils import get_peft_task_type_from_task
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizer

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
            "ortmodule_onnx_opset_version": PassConfigParam(
                type_=int,
                default_value=16,
                description=(
                    "The opset version to use for ONNX export when using ORTTrainer. Only used if use_ort_trainer is"
                    " True. 16+ is required when using bfloat16 and model has operators such as Where."
                ),
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

            assert config.ortmodule_onnx_opset_version > 0, "ortmodule_onnx_opset_version must be a positive integer."
            # ops such as Where only support bfloat16 from opset 16
            if uses_bf16 and config.ortmodule_onnx_opset_version < 16:
                logger.warning(
                    f"ortmodule_onnx_opset_version is {config.ortmodule_onnx_opset_version} but training with bfloat16"
                    " might not work properly with opset versions < 16"
                )
            os.environ["ORTMODULE_ONNX_OPSET_VERSION"] = str(config.ortmodule_onnx_opset_version)

        # bitsandbytes quantization only supported after transformers 4.30.0
        if is_qlora and version.parse(transformers.__version__) < version.parse("4.30.0"):
            raise ImportError(f"Please install transformers >= 4.30.0 to use {cls.__name__} pass.")

    @staticmethod
    def collate_batch(batch: List[Dict], tokenizer: "PreTrainedTokenizer") -> Dict[str, torch.Tensor]:
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
    def prepare_model_for_lora_finetuning(
        model: "PreTrainedModel", use_gradient_checkpointing: bool
    ) -> "PreTrainedModel":
        """Prepare the model for fine-tuning.

        Freeze base model's layers and prepare model for gradient checkpointing if necessary.
        Similar to peft.prepare_model_for_kbit_training but no casting to fp32 and gradient checkpointing is
        also supported for non-quantized models.

        :param model: The Hugging Face PyTorch model to prepare for fine-tuning.
        :param use_gradient_checkpointing: Whether to use gradient checkpointing.
        :return: The prepared model.
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

    def create_and_load_new_model(
        self, model_handler: PyTorchModelHandler, config: ConfigBase, **kwargs
    ) -> Tuple[PyTorchModelHandler, "PreTrainedModel"]:
        """Clone the input model handler and update the model from_pretrained_args.

        :param model_handler: The input model handler.
        :param config: The config for the pass run.
        :param kwargs: Additional arguments to update from_pretrained_args with.
        :return: The new model handler and the new loaded pytorch model.
        """
        # don't want the original loaded model
        # also frees gpu memory if original model is on gpu
        model_handler.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # create copy of the input model, will modify this model
        # also resets adapter_path
        new_model_handler = self.input_model_check(deepcopy(model_handler))

        torch_dtype = self.get_torch_dtype(config.torch_dtype)
        # will use mixed precision since full fp16 is unstable
        model_dtype = torch_dtype if torch_dtype != torch.float16 else torch.float32

        # load model, reset from_pretrained_args and adapter_path
        from_pretrained_args = (
            new_model_handler.hf_config.from_pretrained_args.dict()
            if new_model_handler.hf_config.from_pretrained_args
            else {}
        )
        from_pretrained_args.update(
            {
                "torch_dtype": model_dtype,
                # TODO(jambayk): Worry about `use_multi_gpu` and distributed training later
                # "auto": uses all available GPUs, model parallel
                # ORTTrainer falls back to pytorch when model parallel is used
                # None: maps to cpu for non-quantized models, first gpu for quantized models
                "device_map": "auto" if not config.use_ort_trainer else None,
            }
        )
        # overwrite from_pretrained_args with kwargs
        from_pretrained_args.update(kwargs)
        new_model_handler.hf_config.from_pretrained_args = HfFromPretrainedArgs(**from_pretrained_args)
        pytorch_model = new_model_handler.load_model()
        pytorch_model.config.torch_dtype = model_dtype

        return new_model_handler, pytorch_model

    def init_lora_adapters(
        self,
        model: "PreTrainedModel",
        task: str,
        config: ConfigBase,
        target_modules: Optional[List[str]] = None,
        use_loftq: Optional[bool] = False,
    ) -> "PeftModel":
        """Initialize LoRA adapters.

        :param model: The Hugging Face PyTorch model to add LoRA adapters to.
        :param task: The task type of the model.
        :param config: The config for the pass run.
        :param target_modules: List of modules to target for LoRA fine-tuning.
        :param use_loftq: Whether to use LoftQ to initialize weights.
        :return: The LoRA model.
        """
        from peft import LoraConfig, get_peft_model

        lora_config_kwargs = {}
        if use_loftq:
            from peft import LoftQConfig

            lora_config_kwargs = {
                "init_lora_weights": "loftq",
                "loftq_config": LoftQConfig(loftq_bits=4, loftq_iter=config.loftq_iter),
            }

        peft_task_type = get_peft_task_type_from_task(task, fail_on_not_found=True)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias=config.bias,
            task_type=peft_task_type,
            modules_to_save=config.modules_to_save,
            **lora_config_kwargs,
        )

        return get_peft_model(model, lora_config)

    def enable_lora(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        task: str,
        config: ConfigBase,
        adapter_path: Optional[str] = None,
        target_modules: Optional[List[str]] = None,
    ) -> "PeftModel":
        """Enable LoRA fine-tuning on a Hugging Face PyTorch model.

        Add padding token to tokenizer and resize model embedding layer if needed.
        Prepare model for fine-tuning by freezing master weights and enabling gradient checkpointing if needed.
        Load or initialize LoRA adapters.

        :param model: The Hugging Face PyTorch model to enable LoRA fine-tuning on.
        :param tokenizer: The tokenizer for the model.
        :param task: The task type of the model.
        :param config: The config for the pass run.
        :param adapter_path: Path to the adapter weights. If None, will initialize new adapters.
        :param target_modules: List of modules to target for LoRA fine-tuning. Only used if adapter_path is None.
        :return: The LoRA model.
        """
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer

        logger.debug("Enabling LoRA fine-tuning")
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
        if not adapter_path:
            logger.debug("Initializing LoRA adapters from config")
            lora_model = self.init_lora_adapters(model, task, config, target_modules=target_modules)
        else:
            logger.debug(f"Loading LoRA adapters from {adapter_path}")
            lora_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
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
        model: "PeftModel",
        tokenizer: "PreTrainedTokenizer",
        config: ConfigBase,
        data_root: str,
        output_model: PyTorchModelHandler,
        output_model_path: str,
    ) -> PyTorchModelHandler:
        """Train and save the new model.

        The fine-tuned adapter weights will be saved and updated in the output model handler.

        :param model: The prepared LoRA model to train.
        :param tokenizer: The tokenizer for the model.
        :param config: The config for the pass run.
        :param data_root: The root directory for the data.
        :param output_model: The output model handler.
        :param output_model_path: The path to save the output model to.
        :return: The output model handler.
        """
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
            if self.get_torch_dtype(config.torch_dtype) == torch.float16:
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
        output_model.hf_config.from_pretrained_args.device_map = None
        # remove model_overwrites from model_attributes
        if output_model.model_attributes:
            for k in self.model_overwrites:
                output_model.model_attributes.pop(k, None)

        # set adapter_path
        output_model.set_resource("adapter_path", adapter_path)
        return output_model

    @staticmethod
    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict, tokenizer: "PreTrainedTokenizer", model: "PreTrainedModel"
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
    def get_torch_dtype(torch_dtype: str) -> torch.dtype:
        """Get the torch dtype from the string."""
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
    def input_model_check(cls, model: PyTorchModelHandler) -> PyTorchModelHandler:
        """Validate the input model and reset from_pretrained_args and adapter_path."""
        if not model.hf_config:
            raise ValueError(f"{cls.__name__} pass only supports PyTorchModelHandler with hf_config.")

        # load model, reset from_pretrained_args and adapter_path
        from_pretrained_args = {}
        if model.hf_config.from_pretrained_args:
            from_pretrained_args = model.hf_config.from_pretrained_args.dict()
            for k in cls.model_overwrites:
                if from_pretrained_args.get(k) is not None:
                    logger.warning(
                        f"Input model has from_pretrained_args.{k}. Ignoring. "
                        f"{cls.__name__} will overwrite it based on the pass config."
                    )

        if model.get_resource("adapter_path"):
            logger.warning(
                "Input model has adapter_path. Ignoring. QLoRA will save the adapter weights to its own adapter_path."
            )
        model.set_resource("adapter_path", None)
        return model

    @staticmethod
    def count_trainable_parameters(model) -> str:
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

    This pass only supports PyTorchModelHandler with hf_config.
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_modules": PassConfigParam(type_=List[str], default_value=None, description="Target modules"),
        }
        config.update(LoRABase._default_config(accelerator_spec))
        return config

    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        # convert config to pass config class
        # this will validate the config and convert to the correct types
        config = self._config_class(**config)

        # check dependencies
        self.check_dependencies(config)

        # use default training args if not provided
        config.training_args = config.training_args or HFTrainingArguments()

        # get new model
        new_model_handler, pytorch_model = self.create_and_load_new_model(model, config)
        if torch.cuda.is_available() and pytorch_model.model.device.type == "cpu":
            # put the model on GPU since model was loaded on CPU with device_map=None
            pytorch_model.to("cuda")

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            new_model_handler.hf_config.model_name,
            trust_remote_code=new_model_handler.hf_config.from_pretrained_args.trust_remote_code,
        )

        # add lora modules
        pytorch_model = self.enable_lora(
            pytorch_model, tokenizer, new_model_handler.hf_config.task, config, target_modules=config.target_modules
        )

        # train and return new model
        return self.train_and_save_new_model(
            pytorch_model, tokenizer, config, data_root, new_model_handler, output_model_path
        )


class QLoRABase(LoRABase):
    """Base class for QLoRA and LoftQ fine-tuning passes."""

    model_overwrites: ClassVar[tuple] = ("torch_dtype", "device_map", "quantization_method", "quantization_config")

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            # quantization parameters
            "compute_dtype": PassConfigParam(
                type_=str,
                description=(
                    "Computation data type for the quantized modules. If not provided, will use the same dtype as"
                    " torch_dtype"
                ),
            )
        }
        config.update(LoRABase._default_config(accelerator_spec))
        return config

    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        # convert config to pass config class
        # this will validate the config and convert to the correct types
        config = self._config_class(**config)

        # check dependencies
        self.check_dependencies(config, is_qlora=True)

        # MatMulBnb4 contrib op doesn't support double quantization so the trainer falls back to PythonOp
        # which uses more memory and is slower
        if config.use_ort_trainer and getattr(config, "double_quant", False):
            logger.warning(
                "double_quant is set to True but it is inefficient with onnxruntime-training! Consider setting it to"
                " False."
            )

        # use default training args if not provided
        config.training_args = config.training_args or HFTrainingArguments()

        # get models and tokenizer
        new_model_handler, pytorch_model, tokenizer, quantized_modules = self.get_model_tokenizer(
            model, config, output_model_path
        )

        # train and get new model
        output_model = self.train_and_save_new_model(
            pytorch_model, tokenizer, config, data_root, new_model_handler, output_model_path
        )
        # add quantized_modules attributes
        output_model.model_attributes["quantized_modules"] = quantized_modules
        return output_model

    @abstractmethod
    def get_model_tokenizer(
        self, model: PyTorchModelHandler, config: ConfigBase, output_model_path: str
    ) -> Tuple[PyTorchModelHandler, "PreTrainedModel", "PreTrainedTokenizer", List[str]]:
        """Get the model handler, LoRA model and tokenizer for fine-tuning."""
        raise NotImplementedError


class QLoRA(QLoRABase):
    """Run QLoRA fine-tuning on a Hugging Face PyTorch model.

    This pass only supports PyTorchModelHandler with hf_config.
    """

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
        }
        config.update(QLoRABase._default_config(accelerator_spec))
        return config

    def get_model_tokenizer(
        self, model: PyTorchModelHandler, config: ConfigBase, output_model_path: str
    ) -> Tuple[PyTorchModelHandler, "PreTrainedModel", "PreTrainedTokenizer", List[str]]:
        """Get the model handler, LoRA model and tokenizer for QLoRA fine-tuning.

        :param model: The input model handler.
        :param config: The config for the pass run.
        :param output_model_path: The path to save the output model to.
        :return: The new model handler, LoRA model, tokenizer and list of quantized modules.
        """
        import bitsandbytes as bnb

        # get new model
        bnb_quant_config = {
            "quantization_method": "bitsandbytes",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": self.get_torch_dtype(config.compute_dtype or config.torch_dtype),
                "bnb_4bit_use_double_quant": config.double_quant,
                "bnb_4bit_quant_type": config.quant_type,
            },
        }
        new_model_handler, pytorch_model = self.create_and_load_new_model(model, config, **bnb_quant_config)

        # find the quantized modules
        # this doesn't pick up the embedding layer and projection layer since those are not quantized
        # this is good since we don't want to touch those, LoRA might not work with input output embedding layers
        quantized_modules = find_submodules(pytorch_model, bnb.nn.Linear4bit)
        logger.debug(f"Quantized modules: {quantized_modules}")

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            new_model_handler.hf_config.model_name,
            trust_remote_code=new_model_handler.hf_config.from_pretrained_args.trust_remote_code,
        )

        # enable lora fine-tuning with new lora modules
        pytorch_model = self.enable_lora(
            pytorch_model, tokenizer, new_model_handler.hf_config.task, config, target_modules=quantized_modules
        )

        return new_model_handler, pytorch_model, tokenizer, quantized_modules


class LoftQ(QLoRA):
    """Run LoftQ fine-tuning on a Hugging Face PyTorch model.

    This pass only supports PyTorchModelHandler with hf_config.
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            # quantization parameters
            "loftq_iter": PassConfigParam(
                type_=int,
                default_value=1,
                description="Number of LoftQ iterations.",
            ),
        }
        config.update(QLoRABase._default_config(accelerator_spec))
        return config

    @classmethod
    def check_dependencies(cls, config: ConfigBase, is_qlora: bool = False):
        """Check dependencies for the pass."""
        super().check_dependencies(config, is_qlora=is_qlora)

        from peft import __version__ as peft_version

        # LoftQ is only supported after peft 0.7.0
        if version.parse(peft_version) < version.parse("0.7.0"):
            raise ImportError(f"Please install peft >= 0.7.0 to use {cls.__name__} pass.")

    def get_model_tokenizer(
        self, model: PyTorchModelHandler, config: ConfigBase, output_model_path: str
    ) -> Tuple[PyTorchModelHandler, "PreTrainedModel", "PreTrainedTokenizer", List[str]]:
        """Get the model handler, LoRA model and tokenizer for LoftQ fine-tuning.

        :param model: The input model handler.
        :param config: The config for the pass run.
        :param output_model_path: The path to save the output model to.
        :return: The new model handler, LoRA model, tokenizer and list of quantized modules.
        """
        import bitsandbytes as bnb

        # get new quantized model
        bnb_quant_config = {
            "quantization_method": "bitsandbytes",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": self.get_torch_dtype(config.compute_dtype or config.torch_dtype),
                "bnb_4bit_use_double_quant": False,
                "bnb_4bit_quant_type": "nf4",
            },
        }
        new_model_handler, pytorch_model = self.create_and_load_new_model(model, config, **bnb_quant_config)

        # find the quantized modules
        quantized_modules = find_submodules(pytorch_model, bnb.nn.Linear4bit)
        logger.debug(f"Quantized modules: {quantized_modules}")

        # only need the quantized module to find the quantized modules
        # delete quantized model to free memory
        del pytorch_model
        new_model_handler.model = None

        # get the original base model
        _, pytorch_model = self.create_and_load_new_model(
            model, config, device_map="auto", quantization_method=None, quantization_config=None
        )
        # get loftq initialized lora model
        logger.debug("Initializing LoRA with LoftQ")
        pytorch_model = self.init_lora_adapters(
            pytorch_model, new_model_handler.hf_config.task, config, quantized_modules, use_loftq=True
        )
        # change adapter config since we don't want to apply loftq again
        pytorch_model.peft_config["default"].base_model_name_or_path = "../model"
        pytorch_model.peft_config["default"].init_lora_weights = True

        output_model_path = Path(output_model_path)

        # save the loftq initialized adapter weights
        loftq_init_adapter_path = output_model_path / "loftq_init_adapter"
        loftq_init_adapter_path.mkdir(parents=True, exist_ok=True)
        pytorch_model.save_pretrained(loftq_init_adapter_path)

        # unload adapter and get the base model with new weights
        pytorch_model: "PreTrainedModel" = pytorch_model.unload()

        # save the new master weights
        new_master_weights_path = output_model_path / "model"
        new_master_weights_path.mkdir(parents=True, exist_ok=True)
        pytorch_model.save_pretrained(new_master_weights_path)

        # update the model path in new model handler
        new_model_handler.set_resource("model_path", new_master_weights_path)

        # load the quantized model with new master weights
        pytorch_model: "PreTrainedModel" = new_model_handler.load_model()
        pytorch_model.config.torch_dtype = pytorch_model.dtype

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            new_model_handler.hf_config.model_name,
            trust_remote_code=new_model_handler.hf_config.from_pretrained_args.trust_remote_code,
        )
        tokenizer.save_pretrained(new_master_weights_path)

        # enable lora fine-tuning with the loftq initialized adapter weights
        pytorch_model = self.enable_lora(
            pytorch_model, tokenizer, new_model_handler.hf_config.task, config, adapter_path=loftq_init_adapter_path
        )

        return new_model_handler, pytorch_model, tokenizer, quantized_modules
