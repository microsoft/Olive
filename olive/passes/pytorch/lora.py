# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on original implementation at
# LoRA: https://huggingface.co/docs/diffusers/training/lora
# QLoRA: https://github.com/artidoro/qlora/blob/main/qlora.py
#        https://arxiv.org/abs/2305.14314
# LoHa: https://arxiv.org/abs/2108.06098
# LoKr: https://arxiv.org/abs/2309.14859
# --------------------------------------------------------------------------
import logging
import tempfile
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union

import transformers
from packaging import version

from olive.common.hf.mappings import MODELS_TO_LORA_TARGET_MODULES_MAPPING
from olive.common.hf.utils import get_peft_task_type_from_task
from olive.common.pydantic_v1 import Field, validator
from olive.common.utils import StrEnumBase, find_submodules, resolve_torch_dtype
from olive.data.config import DataConfig
from olive.data.constants import IGNORE_INDEX
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.model.config.hf_config import HfLoadKwargs
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pass_config import BasePassConfig
from olive.passes.pytorch.train_utils import (
    BaseHFTrainingArguments,
    count_trainable_parameters,
    get_training_dataset,
    load_hf_base_model,
    prepare_model_for_finetuning,
)
from olive.search.search_parameter import Categorical

if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DeviceMap(StrEnumBase):
    AUTO = "auto"
    CURRENT_DEVICE = "current_device"


class HFTrainingArguments(BaseHFTrainingArguments):
    """Training arguments for transformers.Trainer.

    Has the same fields as transformers.TrainingArguments with recommended default values for QLoRA fine-tuning.
    """

    # TODO(jambayk): is this default optim required? does it work for regular lora? what about lr_scheduler_type?
    optim: str = Field("paged_adamw_32bit", description="The optimizer to use.")
    learning_rate: float = Field(0.0002, description="The initial learning rate for AdamW.")
    lr_scheduler_type: str = Field(
        "constant",
        description="Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis.",
    )
    warmup_ratio: float = Field(0.03, description="Fraction of steps to do a warmup for.")
    evaluation_strategy: str = Field(
        None,
        description=(
            "The evaluation strategy to use. Forced to 'no' if eval_dataset is not provided. Otherwise, 'steps' unless"
            " set to 'epoch'."
        ),
    )
    overwrite_output_dir: bool = Field(
        False,
        description=(
            "If True, overwrite the content of output_dir. Otherwise, will continue training if `output_dir` points to"
            " a checkpoint directory."
        ),
    )
    resume_from_checkpoint: str = Field(
        None,
        description=(
            "The path to a folder with a valid checkpoint for the model. Supercedes any checkpoint found in output_dir."
        ),
    )

    @validator("extra_args", pre=True, always=True)
    def validate_torch_dtype(cls, v):
        if v and "fp16" in v:
            logger.warning("Extra arg 'fp16' is not allowed. Please use `torch_dtype` instead.")
            del v["fp16"]
        return v


class LoRA(Pass):
    """Run LoRA fine-tuning on a Hugging Face PyTorch model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "r": PassConfigParam(
                type_=int,
                default_value=64,
                search_defaults=Categorical([16, 32, 64]),
                description="R dimension.",
            ),
            "alpha": PassConfigParam(type_=float, default_value=16, description="The alpha parameter for scaling."),
            "lora_dropout": PassConfigParam(
                type_=float, default_value=0.05, description="The dropout probability for Lora layers."
            ),
            "target_modules": PassConfigParam(type_=List[str], default_value=None, description="Target modules"),
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
            "device_map": PassConfigParam(
                type_=Optional[DeviceMap],
                default_value=DeviceMap.AUTO,
                description="Device map to use to load the model.",
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
            "ephemeral_gpu_offload": PassConfigParam(
                type_=bool, default_value=False, description="Ephemeral GPU offload"
            ),
            # data parameters
            "train_data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                required=True,
                description="Data config for fine-tuning training.",
            ),
            "eval_data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                description="Data config for fine-tuning evaluation. Optional if evaluation is not needed.",
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

    @classmethod
    def check_dependencies(cls, config: Type[BasePassConfig], is_qlora: bool = False):
        """Check dependencies for the pass."""
        # bitsandbytes quantization only supported after transformers 4.30.0
        if is_qlora and version.parse(transformers.__version__) < version.parse("4.30.0"):
            raise ImportError(f"Please install transformers >= 4.30.0 to use {cls.__name__} pass.")

        if config.training_args:
            # check if output_dir is a valid directory
            # must be a directory with checkpoints
            output_dir = config.training_args.output_dir
            if config.training_args.overwrite_output_dir or not output_dir or not Path(output_dir).exists():
                return
            # find the last checkpoint in output_dir
            checkpoint = transformers.trainer_utils.get_last_checkpoint(output_dir)
            if not checkpoint and len(list(Path(output_dir).iterdir())) > 0:
                raise ValueError(
                    f"Output directory ({output_dir}) already exists and is not empty. Set overwrite_output_dir to True"
                    " to overwrite or provide a new output_dir."
                )

    # TODO(jambayk): consider introducing a data collator component for data container
    @staticmethod
    def collate_batch(batch: List[Dict], tokenizer: "PreTrainedTokenizer") -> Dict[str, "torch.Tensor"]:
        """Collate a batch of samples into a padded batch of tensors.

        Add padding to the input_ids, attention_mask and labels.
        Each example can be a dictionary with inputs (and optionally labels).
        """
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [sample["input_ids"] for sample in batch]
        attention_mask = None
        if "attention_mask" in batch[0]:
            attention_mask = [sample["attention_mask"] for sample in batch]
        label_col = "labels" if "labels" in batch[0] else "label"
        if label_col not in batch[0]:
            # labels is the same as input_ids, the trainer left shifts the labels when computing loss
            labels = [input_id.clone() for input_id in input_ids]
        else:
            labels = [sample[label_col] for sample in batch]

        # apply padding and add to batch
        # need to worry about left or right padding?
        new_batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            # pad the labels with IGNORE_INDEX so that they are not used in loss computation
            # don't want to clone batched input_ids and ignore padding tokens in case eos token is used
            # as padding token
            "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
        }
        if attention_mask:
            new_batch["attention_mask"] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return new_batch

    @staticmethod
    def get_datasets(config: Type[BasePassConfig]) -> Tuple["Dataset", Optional["Dataset"]]:
        """Load training and evaluation datasets."""
        # we return dataset.Dataset object since the trainer works better with it
        # load training dataset
        train_dataset = get_training_dataset(config.train_data_config)

        # load evaluation dataset if needed
        eval_dataset = None
        if config.eval_data_config:
            eval_dataset = get_training_dataset(config.eval_data_config)

        return train_dataset, eval_dataset

    def _run_for_config(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        return self._run_lora_training(model, config, output_model_path)

    def _run_lora_training(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str, use_dora: bool = False
    ) -> HfModelHandler:
        # check dependencies
        self.check_dependencies(config)

        # use default training args if not provided
        config.training_args = config.training_args or HFTrainingArguments()

        # check if peft or olive has target modules for the model
        config.target_modules = config.target_modules or self.get_target_modules(model)

        # get new model
        pytorch_model = self.load_base_pytorch_model(model, config)
        # NOTE: quantized model support
        # awq: requires awq cuda extension or triton for backward pass, scale must be fp16
        # gptq: there is no custom backend. works fine when using naive dequantize + matmul
        # no issue with single precision. mix precision depends on autocast as there is no input cast
        # gradient might not be correct when using cuda/exllama deqauntize kernels
        # we load in fp32/bf16 so cuda kernels are disabled by default. Might need extra work to
        # disable exllama (gptq pass disables it)

        # add lora modules
        pytorch_model = self.enable_lora(pytorch_model, config, model.task, use_dora=use_dora)

        # train and return new model
        return self.train_and_save_new_model(
            pytorch_model, model.get_hf_tokenizer(), config, deepcopy(model), output_model_path
        )

    def load_base_pytorch_model(
        self, model_handler: HfModelHandler, config: Type[BasePassConfig], **kwargs
    ) -> "PreTrainedModel":
        """Load a base PyTorch model for fine-tuning.

        :param model_handler: The input model handler.
        :param config: The config for the pass run.
        :param kwargs: Additional arguments to update load_kwargs with.
        :return: The new loaded pytorch model
        """
        import torch

        torch_dtype = self.get_torch_dtype(config.torch_dtype)
        # will use mixed precision since full fp16 is unstable
        model_dtype = torch_dtype if torch_dtype != torch.float16 else torch.float32

        # TODO(jambayk): Worry about `use_multi_gpu` and distributed training later
        # "auto": uses all available GPUs, model parallel
        device_map = config.device_map
        if device_map == DeviceMap.CURRENT_DEVICE:
            device_map = {"": torch.cuda.current_device()}
        return load_hf_base_model(model_handler, torch_dtype=model_dtype, device_map=device_map, **kwargs)

    def init_adapters(
        self,
        model: "PreTrainedModel",
        config: Type[BasePassConfig],
        *,
        task: Optional[str] = None,
        use_loftq: Optional[bool] = False,
        use_dora: Optional[bool] = False,
    ) -> "PeftModel":
        """Initialize LoRA adapters.

        :param model: The Hugging Face PyTorch model to add LoRA adapters to.
        :param config: The config for the pass run.
        :param task: The task type of the model.
        :param use_loftq: Whether to use LoftQ to initialize weights.
        :param use_dora: Whether to use DoRA to initialize weights.
        :return: The LoRA model.
        """
        config_kwargs = {}
        if use_loftq:
            from peft import LoftQConfig

            config_kwargs = {
                "init_lora_weights": "loftq",
                "loftq_config": LoftQConfig(loftq_bits=4, loftq_iter=config.loftq_iter),
            }
        if use_dora:
            config_kwargs = {
                "use_dora": True,
            }
        if task:
            config_kwargs.update({"task_type": get_peft_task_type_from_task(task, fail_on_not_found=True)})

        return self.get_peft_model(model, config, config_kwargs)

    def enable_lora(
        self,
        model: "PreTrainedModel",
        config: Type[BasePassConfig],
        task: Optional[str] = None,
        use_dora: bool = False,
        adapter_path: Optional[str] = None,
    ) -> "PeftModel":
        """Enable LoRA fine-tuning on a Hugging Face PyTorch model.

        Add padding token to tokenizer and resize model embedding layer if needed.
        Prepare model for fine-tuning by freezing master weights and enabling gradient checkpointing if needed.
        Load or initialize LoRA adapters.

        :param model: The Hugging Face PyTorch model to enable LoRA fine-tuning on.
        :param config: The config for the pass run.
        :param task: The task type of the model.
        :param use_dora: Whether to use DoRA to train adapters.
        :param adapter_path: Path to the adapter weights. If None, will initialize new adapters.
        :return: The LoRA model.
        """
        logger.debug("Enabling LoRA fine-tuning")
        prepare_model_for_finetuning(model, config.training_args)

        # set model_parallel and is_parallelizable to True
        # we are using "auto" device_map, so model_parallel is True or doing DDP
        # don't want the trainer to do Data Parallel
        model.model_parallel = True
        model.is_parallelizable = True
        # TODO(team): ERROR: forward() got an unexpected keyword argument 'num_items_in_batch'
        # Temporary fix by disabling loss kwargs
        # https://github.com/huggingface/transformers/issues/35838
        model.accepts_loss_kwargs = False

        if not adapter_path:
            logger.debug("Initializing LoRA adapters from config")
            lora_model = self.init_adapters(model, config, task=task, use_dora=use_dora)
        else:
            from peft import PeftModel

            logger.debug("Loading LoRA adapters from %s", adapter_path)
            lora_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        logger.debug("The number of trainable parameters in the LoRA model: %s", count_trainable_parameters(lora_model))
        # no need to cast lora modules to model's dtype, we dont do peft.prepare_model_for_kbit_training so the modules
        # are already in the same dtype as the model
        # casting to dtype is risky since for awq quant linear, it also casts the scales to dtype and but the qlinear
        # expects scales to be in half
        return lora_model

    def train_and_save_new_model(
        self,
        model: "PeftModel",
        tokenizer: "PreTrainedTokenizer",
        config: Type[BasePassConfig],
        output_model: HfModelHandler,
        output_model_path: str,
    ) -> HfModelHandler:
        """Train and save the new model.

        The fine-tuned adapter weights will be saved and updated in the output model handler.

        :param model: The prepared LoRA model to train.
        :param tokenizer: The tokenizer for the model.
        :param config: The config for the pass run.
        :param output_model: The output model handler.
        :param output_model_path: The path to save the output model to.
        :return: The output model handler.
        """
        import torch

        if torch.cuda.is_available():
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32

        # get datasets
        train_dataset, eval_dataset = self.get_datasets(config)

        # get training arguments
        orig_eval_strat = config.training_args.evaluation_strategy
        config.training_args.evaluation_strategy = "no"
        if eval_dataset:
            # default to "steps" if eval dataset is provided
            config.training_args.evaluation_strategy = "steps" if orig_eval_strat in {None, "no"} else orig_eval_strat

        # We always create a temp dir even if output_dir is provided because we want the temp dir to be deleted
        # after training or if there is an error
        # With a context manager, the temp dir will be deleted automatically as soon as the context is exited or
        # there is an error
        # If we do `tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")` and there is an error before
        # cleanup or run returns (tmp_dir goes out of scopt), the temp dir will not be deleted until the the exception
        # is handled by the caller (after try except) or the program exits
        # Plus the cleanup after error doesn't work as expected with notebooks
        with tempfile.TemporaryDirectory(prefix="olive_tmp") as temp_dir:
            checkpoint = config.training_args.resume_from_checkpoint
            if not config.training_args.output_dir:
                logger.debug("No training_args.output_dir provided. Using a temp dir.")
                config.training_args.output_dir = temp_dir
                # set save_total_limit to 1 since the temp dir will be deleted after training
                config.training_args.extra_args["save_total_limit"] = 1
            elif (
                not checkpoint
                and not config.training_args.overwrite_output_dir
                and Path(config.training_args.output_dir).exists()
            ):
                # find the last checkpoint in output_dir
                checkpoint = transformers.trainer_utils.get_last_checkpoint(config.training_args.output_dir)
                if checkpoint:
                    logger.info(
                        "Checkpoint detected in output_dir. Resuming training at %s. To avoid this behavior and train"
                        " from scratch, change `output_dir` or set `overwrite_output_dir` to True.",
                        checkpoint,
                    )

            if self.get_torch_dtype(config.torch_dtype) == torch.float16:
                # use fp16 mixed precision training
                config.training_args.extra_args["fp16"] = True
            # create training args
            logger.debug("Training args: %s", config.training_args.dict())

            # get trainer'
            trainer = transformers.Trainer(
                model=model,
                args=config.training_args.create_training_args(),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=partial(self.collate_batch, tokenizer=tokenizer),
            )

            # train
            logger.info("Running fine-tuning")
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            logger.debug("train_result: %s", train_result)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # lgtm

        # save adapter weights
        adapter_path = Path(output_model_path) / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        # don't save embedding layers since only adapter weights are trained
        # if we don't provide as False, it defaults to "auto" which checks if the vocab size changed
        model.save_pretrained(adapter_path, save_embedding_layers=False)

        # remove loaded model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # set adapter_path
        output_model.set_resource("adapter_path", adapter_path)
        return output_model

    @staticmethod
    def get_torch_dtype(torch_dtype: str) -> "torch.dtype":
        """Get the torch dtype from the string."""
        supported_dtypes = ("bfloat16", "float16", "float32")
        assert torch_dtype in supported_dtypes, f"torch_dtype must be one of {supported_dtypes} but got {torch_dtype}"
        return resolve_torch_dtype(torch_dtype)

    @staticmethod
    def get_target_modules(model: HfModelHandler) -> Optional[List[str]]:
        """Get the target modules for the model."""
        from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

        model_type = model.get_hf_model_type()
        if model_type not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            if model_type in MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                return MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
            else:
                raise ValueError(
                    f"Model type {model_type} is not recognized by peft or olive. Please provide 'target_modules'."
                )
        return None

    @staticmethod
    def get_peft_model(
        model: "PreTrainedModel", config: Type[BasePassConfig], config_kwargs: Dict = None
    ) -> "PeftModel":
        """Get the PEFT model for LoRA fine-tuning."""
        from peft import LoraConfig, LoraRuntimeConfig, get_peft_model

        if config_kwargs is None:
            config_kwargs = {}

        lora_config = LoraConfig(
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            modules_to_save=config.modules_to_save,
            runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=config.ephemeral_gpu_offload),
            **config_kwargs,
        )

        return get_peft_model(model, lora_config)


class DoRA(LoRA):
    """Run DoRA fine-tuning on a Hugging Face PyTorch model."""

    def _run_for_config(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        return self._run_lora_training(model, config, output_model_path, use_dora=True)


class LoRAVariant(LoRA):
    """Run LoRA variant fine-tuning on a Hugging Face PyTorch model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "rank_dropout": PassConfigParam(
                type_=float,
                default_value=0.0,
                description="The dropout probability for rank dimension during training.",
            ),
            "module_dropout": PassConfigParam(
                type_=float,
                default_value=0.0,
                description="The dropout probability for disabling modules during training.",
            ),
            "use_effective_conv2d": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Use parameter effective decomposition for Conv2d with ksize > 1.",
            ),
            "exclude_modules": PassConfigParam(
                type_=Optional[Union[List[str], str]], default_value=None, description="Modules to exclude from tuning."
            ),
            "init_weights": PassConfigParam(
                type_=bool, default_value=True, description="Whether to perform initialization of adapter weights."
            ),
            "layers_to_transform": PassConfigParam(
                type_=List[int], default_value=None, description="The layer indices to transform."
            ),
            "layers_pattern": PassConfigParam(
                type_=List[str],
                default_value=None,
                description="The layer pattern name, used only if layers_to_transform is different from None.",
            ),
            "rank_pattern": PassConfigParam(
                type_=Dict,
                default_value={},
                description="The mapping from layer names or regexp expression "
                "to ranks which are different from the default rank specified by r.",
            ),
            "alpha_pattern": PassConfigParam(
                type_=Dict,
                default_value={},
                description="The mapping from layer names or regexp expression "
                "to alphas which are different from the default alpha specified by alpha.",
            ),
        }
        config.update(super()._default_config(accelerator_spec))
        return config


class LoHa(LoRAVariant):
    """Run LoHa fine-tuning on a Hugging Face PyTorch model."""

    @staticmethod
    def get_peft_model(
        model: "PreTrainedModel", config: Type[BasePassConfig], config_kwargs: Dict = None
    ) -> "PeftModel":
        """Get the PEFT model for LoHa fine-tuning."""
        from peft import LoHaConfig, get_peft_model

        target_modules = config.target_modules or "all-linear"
        config = LoHaConfig(
            r=config.r,
            alpha=config.alpha,
            rank_dropout=config.rank_dropout,
            module_dropout=config.module_dropout,
            use_effective_conv2d=config.use_effective_conv2d,
            target_modules=target_modules,
            exclude_modules=config.exclude_modules,
            init_weights=config.init_weights,
            layers_to_transform=config.layers_to_transform,
            layers_pattern=config.layers_pattern,
            rank_pattern=config.rank_pattern,
            alpha_pattern=config.alpha_pattern,
            modules_to_save=config.modules_to_save,
        )

        return get_peft_model(model, config)

    @classmethod
    def check_dependencies(cls, config: Type[BasePassConfig], is_qlora: bool = False):
        """Check dependencies for the pass."""
        super().check_dependencies(config, is_qlora=is_qlora)

        from peft import __version__ as peft_version

        # LoHa is only supported after peft 0.7.0
        if version.parse(peft_version) < version.parse("0.7.0"):
            raise ImportError(f"Please install peft >= 0.7.0 to use {cls.__name__} pass.")


class LoKr(LoRAVariant):
    """Run LoKr fine-tuning on a Hugging Face PyTorch model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "decompose_both": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Perform rank decomposition of left kronecker product matrix.",
            ),
            "decompose_factor": PassConfigParam(
                type_=int,
                default_value=-1,
                description="Kronecker product decomposition factor.",
            ),
            "rank_dropout_scale": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to scale the rank dropout while training.",
            ),
        }
        config.update(super()._default_config(accelerator_spec))
        return config

    @staticmethod
    def get_peft_model(
        model: "PreTrainedModel", config: Type[BasePassConfig], config_kwargs: Dict = None
    ) -> "PeftModel":
        """Get the PEFT model for LoKr fine-tuning."""
        from peft import LoKrConfig, get_peft_model

        target_modules = config.target_modules or "all-linear"
        config = LoKrConfig(
            r=config.r,
            alpha=config.alpha,
            rank_dropout=config.rank_dropout,
            module_dropout=config.module_dropout,
            decompose_both=config.decompose_both,
            decompose_factor=config.decompose_factor,
            rank_dropout_scale=config.rank_dropout_scale,
            use_effective_conv2d=config.use_effective_conv2d,
            target_modules=target_modules,
            exclude_modules=config.exclude_modules,
            init_weights=config.init_weights,
            layers_to_transform=config.layers_to_transform,
            layers_pattern=config.layers_pattern,
            rank_pattern=config.rank_pattern,
            alpha_pattern=config.alpha_pattern,
            modules_to_save=config.modules_to_save,
        )

        return get_peft_model(model, config)

    @classmethod
    def check_dependencies(cls, config: Type[BasePassConfig], is_qlora: bool = False):
        """Check dependencies for the pass."""
        super().check_dependencies(config, is_qlora=is_qlora)

        from peft import __version__ as peft_version

        # LoHa is only supported after peft 0.7.0
        if version.parse(peft_version) < version.parse("0.7.0"):
            raise ImportError(f"Please install peft >= 0.7.0 to use {cls.__name__} pass.")


class QLoRABase(LoRA):
    """Base class for QLoRA and LoftQ fine-tuning passes."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            # quantization parameters
            "compute_dtype": PassConfigParam(
                type_=str,
                description=(
                    "Computation data type for the quantized modules. If not provided, will use the same dtype as"
                    " torch_dtype"
                ),
            ),
            "save_quant_config": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether to save the output model with the bitsandbytes quantization config. If False, the base"
                    " model will be in the original precision. If True, the base model will be quantized on load."
                ),
            ),
        }
        config.update(super()._default_config(accelerator_spec))
        return config

    def _run_for_config(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        # check dependencies
        self.check_dependencies(config, is_qlora=True)

        # use default training args if not provided
        config.training_args = config.training_args or HFTrainingArguments()

        # model cannot be quantized
        model_config = model.get_hf_model_config()
        if hasattr(model_config, "quantization_config"):
            raise ValueError("Model is already quantized. Please provide a non-quantized model or use LoRA pass.")

        # get models and tokenizer
        new_model_handler, pytorch_model, bnb_quant_config, quantized_modules = self.get_quant_model(
            model, config, output_model_path
        )
        if config.save_quant_config:
            load_kwargs = new_model_handler.load_kwargs.dict() if new_model_handler.load_kwargs else {}
            load_kwargs.update(bnb_quant_config)
            new_model_handler.load_kwargs = HfLoadKwargs(**load_kwargs)
            new_model_handler.model_attributes["quantized_modules"] = quantized_modules

        # train and return new model
        return self.train_and_save_new_model(
            pytorch_model, new_model_handler.get_hf_tokenizer(), config, new_model_handler, output_model_path
        )

    @abstractmethod
    def get_quant_model(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> Tuple[HfModelHandler, "PreTrainedModel", Dict, List[str]]:
        """Get the model handler, LoRA model for QLoRA fine-tuning.

        :param model: The input model handler.
        :param config: The config for the pass run.
        :param output_model_path: The path to save the output model to.
        :return: The new model handler, LoRA model, quantization config and list of quantized modules.
        """
        raise NotImplementedError


class QLoRA(QLoRABase):
    """Run QLoRA fine-tuning on a Hugging Face PyTorch model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
        config.update(super()._default_config(accelerator_spec))
        return config

    def get_quant_model(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> Tuple[HfModelHandler, "PreTrainedModel", Dict, List[str]]:
        """Get the model handler, LoRA model for QLoRA fine-tuning.

        :param model: The input model handler.
        :param config: The config for the pass run.
        :param output_model_path: The path to save the output model to.
        :return: The new model handler, LoRA model, quantization config and list of quantized modules.
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
        pytorch_model = self.load_base_pytorch_model(model, config, **bnb_quant_config)

        # find the quantized modules, only linears
        quantized_modules = find_submodules(pytorch_model, bnb.nn.Linear4bit)
        logger.debug("Quantized modules: %s", quantized_modules)

        # enable lora fine-tuning with new lora modules
        config.target_modules = quantized_modules
        pytorch_model = self.enable_lora(pytorch_model, config, model.task)

        return deepcopy(model), pytorch_model, bnb_quant_config, quantized_modules


class LoftQ(QLoRABase):
    """Run LoftQ fine-tuning on a Hugging Face PyTorch model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            # quantization parameters
            "loftq_iter": PassConfigParam(
                type_=int,
                default_value=1,
                description="Number of LoftQ iterations.",
            ),
        }
        config.update(super()._default_config(accelerator_spec))
        return config

    @classmethod
    def check_dependencies(cls, config: Type[BasePassConfig], is_qlora: bool = False):
        """Check dependencies for the pass."""
        super().check_dependencies(config, is_qlora=is_qlora)

        from peft import __version__ as peft_version

        # LoftQ is only supported after peft 0.7.0
        if version.parse(peft_version) < version.parse("0.7.0"):
            raise ImportError(f"Please install peft >= 0.7.0 to use {cls.__name__} pass.")

    def get_quant_model(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> Tuple[HfModelHandler, "PreTrainedModel", Dict, List[str]]:
        """Get the model handler, LoRA model for QLoRA fine-tuning.

        :param model: The input model handler.
        :param config: The config for the pass run.
        :param output_model_path: The path to save the output model to.
        :return: The new model handler, LoRA model, quantization config and list of quantized modules.
        """
        import torch

        # get the original base model
        pytorch_model = self.load_base_pytorch_model(model, config)
        # find all modules that will be quantized, all linears except the lm_head
        quantized_modules = [
            module for module in find_submodules(pytorch_model, torch.nn.Linear) if module != "lm_head"
        ]

        # get loftq initialized lora model
        logger.debug("Initializing LoRA with LoftQ")
        config.target_modules = quantized_modules
        pytorch_model = self.init_adapters(pytorch_model, config, task=model.task, use_loftq=True)

        output_model_path = Path(output_model_path)

        # save the loftq initialized adapter weights
        loftq_init_adapter_path = output_model_path / "loftq_init_adapter"
        loftq_init_adapter_path.mkdir(parents=True, exist_ok=True)
        # change adapter config since we don't want to apply loftq again
        pytorch_model.peft_config["default"].init_lora_weights = True
        pytorch_model.save_pretrained(loftq_init_adapter_path)

        # unload adapter and get the base model with new weights
        pytorch_model: PreTrainedModel = pytorch_model.unload()

        # save the new master weights
        new_master_weights_path = output_model_path / "model"
        new_master_weights_path.mkdir(parents=True, exist_ok=True)
        pytorch_model.save_pretrained(new_master_weights_path)
        model.save_metadata(new_master_weights_path)

        del pytorch_model

        # create new model handler
        new_model_handler = deepcopy(model)
        # update the model path in new model handler
        new_model_handler.set_resource("model_path", new_master_weights_path)

        # get the quantized base model
        bnb_quant_config = {
            "quantization_method": "bitsandbytes",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": self.get_torch_dtype(config.compute_dtype or config.torch_dtype),
                "bnb_4bit_use_double_quant": False,
                "bnb_4bit_quant_type": "nf4",
            },
        }
        pytorch_model = self.load_base_pytorch_model(new_model_handler, config, **bnb_quant_config)

        # enable lora fine-tuning with the loftq initialized adapter weights
        pytorch_model = self.enable_lora(
            pytorch_model, config, new_model_handler.task, adapter_path=loftq_init_adapter_path
        )

        return new_model_handler, pytorch_model, bnb_quant_config, quantized_modules
