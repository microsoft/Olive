# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import dataclasses
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import transformers

from olive.common.config_utils import NestedConfig
from olive.common.pydantic_v1 import Field, validator
from olive.common.utils import cleanup_memory
from olive.model import HfModelHandler
from olive.model.config.hf_config import HfLoadKwargs

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


# creating a Config class since transformers.TrainingArguments is a dataclass
# pydantic handles dataclasses differently and causes issues with validation
# this also allows us to handle and validate extra_args better
class BaseHFTrainingArguments(NestedConfig):
    """Training arguments for transformers.Trainer."""

    _nested_field_name = "extra_args"

    report_to: Union[str, List[str]] = Field(
        "none", description="The list of integrations to report the results and logs to."
    )
    output_dir: str = Field(None, description="The output dir for logs and checkpoints. If None, will use a temp dir.")
    deepspeed: Union[bool, str, Dict] = Field(
        None,
        description=(
            "Use [Deepspeed](https://github.com/microsoft/deepspeed). If True, will use default deepspeed config. Else,"
            " it is a path to a deepspeed config file or a dict with deepspeed config."
        ),
    )
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
            if k not in training_args_fields:
                logger.warning("Extra arg %s is not a field of transformers.TrainingArguments. Ignoring.", k)
                del v[k]
        return v

    def create_training_args(self) -> transformers.TrainingArguments:
        args = self.dict()
        if not args["output_dir"]:
            raise ValueError("output_dir must be provided.")
        if args["deepspeed"] is True:
            args["deepspeed"] = deepcopy(DEFAULT_DEEPSPEED_CONFIG)
        elif args["deepspeed"] is False:
            del args["deepspeed"]
        extra_args = args.pop("extra_args")
        return transformers.TrainingArguments(**args, **extra_args)


def load_hf_base_model(
    model_handler: HfModelHandler,
    torch_dtype: Optional["torch.dtype"] = None,
    device_map: Optional[Union[int, str, Dict]] = None,
    **kwargs
) -> "PreTrainedModel":
    """Load a base PyTorch model.

    :param model_handler: The input model handler.
    :param torch_dtype: The torch dtype to load the model with.
    :param device_map: The device map to load the model with.
    :param kwargs: Additional arguments to update load_kwargs with.
    :return: The new loaded pytorch model
    """
    # model cannot have it's own adapter
    if model_handler.adapter_path:
        raise ValueError("Model already has an adapter. Please provide a model without an adapter.")

    # don't want the original loaded model
    # also frees gpu memory if original model is on gpu
    model_handler.model = None
    cleanup_memory()

    # create copy of the input model, will modify this model
    # also resets adapter_path
    new_model_handler = deepcopy(model_handler)

    # load model, reset load_kwargs and adapter_path
    load_kwargs = new_model_handler.load_kwargs.dict() if new_model_handler.load_kwargs else {}
    load_kwargs.update(
        {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
    )
    # overwrite load_kwargs with kwargs
    load_kwargs.update(kwargs)
    new_model_handler.load_kwargs = HfLoadKwargs(**load_kwargs)

    return new_model_handler.load_model(cache_model=False)


DEFAULT_DEEPSPEED_CONFIG = {
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": True,
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": "auto",
        "offload_param": {
            "device": "cpu",
        },
        "offload_optimizer": {
            "device": "cpu",
        },
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
}
