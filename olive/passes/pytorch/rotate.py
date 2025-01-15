# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional, Union

import torch

from olive.common.hf.adapter import ModelAdapter
from olive.common.pydantic_v1 import Field
from olive.common.utils import StrEnumBase, replace_submodules, set_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.utils.train import (
    BaseHFTrainingArguments,
    count_trainable_parameters,
    load_hf_base_model,
    prepare_model_for_finetuning,
)

logger = logging.getLogger(__name__)

# ruff: noqa: N806


class RotateBase(Pass):
    """Base class for rotation passes."""

    class RotateMode(StrEnumBase):
        HADAMARD = "hadamard"
        RANDOM = "random"

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "seed": PassConfigParam(
                type_=int,
                default_value=0,
                description="Random seed for rotation. Default value is 0.",
            ),
            "rotate_mode": PassConfigParam(
                type_=RotateBase.RotateMode,
                default_value=RotateBase.RotateMode.HADAMARD,
                description="Rotation method to use. Default value is 'hadamard'.",
            ),
        }

    @torch.no_grad()
    def rotate_model(
        self,
        model: HfModelHandler,
        rotate_mode: str,
        seed: int,
        training_args: Optional[BaseHFTrainingArguments] = None,
    ):
        """Create a new model with rotate modules.

        :param model: HfModelHandler: The model to rotate.
        :param rotate_mode: str: The rotation method to use.
        :param seed: int: The random seed for the rotation.
        :param training_args: Optional[BaseHFTrainingArguments]: The training arguments for the model.
        :return: ModelAdapter with the rotated model, rotation parameters, and save replacements.
        """
        from olive.passes.pytorch.utils.rotate import RotateEmbed, RotateLinear, fuse_layer_norms, get_orthogonal_matrix

        if model.adapter_path:
            logger.info(
                "Model has adapters but %s does not support adapters. Rotating without adapters. The original"
                " adapters will be used as is with the rotated base model.",
                self.__class__.__name__,
            )

            # create a new input model with the adapter path removed
            model.model = None
            model = deepcopy(model)
            model.set_resource("adapter_path", None)

        # create model adapter and
        model_adapter = ModelAdapter(model.get_hf_model_config())

        # load pytorch model
        torch_dtype = None
        if training_args:
            torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        pytorch_model = load_hf_base_model(model, torch_dtype=torch_dtype)

        # prepare model for finetuning
        if training_args:
            prepare_model_for_finetuning(pytorch_model, training_args)

        model_adapter.set_model(pytorch_model)
        model_adapter.model.eval()

        # fuse layernorms into adjacent linear layers
        fuse_layer_norms(model_adapter)

        # rotate the model
        torch.manual_seed(seed)
        rotation_params = {}
        R1 = torch.nn.Parameter(
            get_orthogonal_matrix(model_adapter.hidden_size, rotate_mode, model_adapter.model.device)
        )
        rotation_params["R1"] = R1

        # rotate embeddings and lm_head
        for embed, embed_name in zip(*model_adapter.get_embeds(return_name=True)):
            # embed_tokens @ R1
            set_attr(model_adapter.model, embed_name, RotateEmbed(embed, R1))
        # R1^-1 @ Whead
        lm_head, lm_head_name = model_adapter.get_lm_head(return_name=True)
        set_attr(model_adapter.model, lm_head_name, RotateLinear(lm_head, Q_pre=R1))

        # need v_proj to be rotated separately, so unpack if necessary
        model_adapter.maybe_unpack_qkv()

        # rotate the hidden layers
        for i, layer_adapter in enumerate(model_adapter.get_layer_adapters()):
            R2 = None
            for linear_idx, (linear, linear_name) in enumerate(
                zip(*layer_adapter.get_attention_inputs(return_name=True))
            ):
                # R1^-1 @ Wq, R1^-1 @ Wk, R1^-1 @ Wv @ R2
                if linear_idx == 2 and getattr(linear, "bias", None) is None:
                    # original implementation ignores bias but output doesn't match both when bias is
                    # rotated headwise and when it is not, so we skip it for now
                    # not really an issue since bias is not present in most models
                    R2 = torch.nn.Parameter(
                        get_orthogonal_matrix(model_adapter.head_dim, rotate_mode, model_adapter.model.device)
                    )
                    rotation_params[f"R2_{i}"] = R2
                set_attr(
                    layer_adapter.layer,
                    linear_name,
                    RotateLinear(linear, Q_pre=R1, Q_post=R2 if linear_idx == 2 else None),
                )

            for linear, linear_name in zip(*layer_adapter.get_attention_outputs(return_name=True)):
                # R2^-1 @ Wo @ R1
                set_attr(layer_adapter.layer, linear_name, RotateLinear(linear, Q_pre=R2, Q_post=R1))

            for linear, linear_name in zip(*layer_adapter.get_mlp_inputs(return_name=True)):
                # R1^-1 @ Wup, R1^-1 @ Wgate
                set_attr(layer_adapter.layer, linear_name, RotateLinear(linear, Q_pre=R1))

            for linear, linear_name in zip(*layer_adapter.get_mlp_outputs(return_name=True)):
                # Wdown @ R1
                set_attr(layer_adapter.layer, linear_name, RotateLinear(linear, Q_post=R1))

        if training_args:
            logger.debug(
                "The number of trainable parameters in the trainable model: %s",
                count_trainable_parameters(model_adapter.model),
            )

        return (
            model_adapter,
            rotation_params,
            [((RotateEmbed, RotateLinear), lambda x: x.create_merged("cuda" if torch.cuda.is_available() else "cpu"))],
        )


class QuaRot(RotateBase):
    """Rotate model using QuaRot.

    See https://arxiv.org/pdf/2404.00456 for more details on the algorithm. Only offline weight rotation is supported.

    This pass only supports HfModelHandler.
    """

    @torch.no_grad()
    def _run_for_config(self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str) -> HfModelHandler:
        model_adapter, _, save_replacements = self.rotate_model(model, config["rotate_mode"], config["seed"])

        # save the model
        model_adapter.save_model(output_model_path, replacements=save_replacements)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)


class HFTrainingArguments(BaseHFTrainingArguments):
    """Training arguments for transformers.Trainer.

    Has the same fields as transformers.TrainingArguments with recommended default values for SpinQuant
    """

    bf16 = Field(
        True, description="Whether to use bfloat16 precision. Recommended for performance on supported hardware."
    )
    per_device_train_batch_size: int = Field(
        1,
        description="The batch size per GPU. Only effective batch size 8 has been tested.",
    )
    gradient_accumulation_steps: int = Field(
        1,
        description="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    learning_rate: float = Field(1.5, description="The initial learning rate for AdamW.")
    weight_decay: float = Field(0.0, description="Weight decay to apply.")
    lr_scheduler_type: str = Field(
        "cosine",
        description="Learning rate schedule.",
    )
    num_train_epochs: int = Field(
        1,
        description="Number of training epochs.",
    )


class SpinQuant(RotateBase):
    """Rotate model using SpinQuant.

    See https://arxiv.org/pdf/2405.16406 for more details on the algorithm. Only offline weight rotation is supported.

    This pass only supports HfModelHandler.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = super()._default_config(accelerator_spec)
        config.update(
            {
                "a_bits": PassConfigParam(
                    type_=int,
                    default_value=16,
                    description="Number of bits for dynamic quantization of activations.",
                ),
                "a_symmetric": PassConfigParam(
                    type_=bool,
                    default_value=True,
                    description="Whether to use symmetric quantization for activations.",
                ),
                "a_per_token": PassConfigParam(
                    type_=bool,
                    default_value=True,
                    description="Whether to quantize activations per token. If False, quantize activations per tensor.",
                ),
                # training parameters
                "training_args": PassConfigParam(
                    type_=Union[HFTrainingArguments, Dict],
                    default_value=None,
                    description=("Training arguments. If None, will use default arguments."),
                ),
            }
        )
        return config

    def _run_for_config(self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str) -> HfModelHandler:
        from olive.passes.pytorch.utils.quant import ActQuantLinear
        from olive.passes.pytorch.utils.rotate import RotateLinear

        training_args = HFTrainingArguments.parse_obj(config["training_args"] or {})

        # rotate the model
        model_adapter, rotation_params, save_replacements = self.rotate_model(
            model, config["rotate_mode"], config["seed"], training_args
        )

        # add activation quantization to the layer linear modules
        replace_submodules(
            model_adapter.get_layers(),
            RotateLinear,
            partial(
                ActQuantLinear, bits=config["a_bits"], symmetric=config["a_symmetric"], per_token=config["a_per_token"]
            ),
        )
        save_replacements = [(ActQuantLinear, lambda x: x.linear), *save_replacements]

        # save the model
        model_adapter.save_model(output_model_path, replacements=save_replacements)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)
