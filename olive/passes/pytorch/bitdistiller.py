# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on original implementation at
# BitDistiller: https://github.com/OpenBitSys/BitDistiller
#               https://arxiv.org/abs/2402.10631
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
import tempfile
from functools import partial
from typing import TYPE_CHECKING, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer

from olive.common.pydantic_v1 import Field
from olive.common.quant.hf_utils import replace_matching_submodules
from olive.data.config import DataConfig
from olive.data.constants import IGNORE_INDEX
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.quant_utils import finalize, get_quantizer_config, prepare_model
from olive.passes.pytorch.train_utils import (
    BaseHFTrainingArguments,
    count_trainable_parameters,
    enable_gradient_checkpointing,
    get_training_dataset,
    load_hf_base_model,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from olive.common.quant.utils import WeightQuantizer
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler

logger = logging.getLogger(__name__)


# =====================================================================
# Fake Quantized Linear Wrapper (STE-based QAT)
# =====================================================================


class STERound(torch.autograd.Function):
    """``torch.round`` with Straight-Through Estimator gradient."""

    # pylint: disable=W0223,W0221
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _ste_fake_quantize(weight: torch.Tensor, quantizer: WeightQuantizer) -> torch.Tensor:
    """STE fake-quantize a weight tensor: quantize-dequantize round-trip with straight-through gradient.

    The scale and zero-point computation matches ``WeightQuantizer.find_qparams`` exactly,
    while the rounding step uses a Straight-Through Estimator to allow gradient flow.
    """
    weight = weight.to(torch.float32)
    orig_shape = weight.shape
    q = quantizer

    # Reshape weight to groups (matching WeightQuantizer._reshape_tensor)
    group_sz = q.group_size if q.group_size > 0 else orig_shape[1]
    num_groups = orig_shape[1] // group_sz
    w = weight.reshape(orig_shape[0], num_groups, -1)

    # Compute scales and zero_points (matching WeightQuantizer.find_qparams)
    # No torch.no_grad: gradients flow through scales/zeros back to weight
    # just like in SteInt2AsymQuantizer
    tmp = torch.zeros(w.shape[:-1], device=w.device, dtype=w.dtype)
    min_val = torch.minimum(w.min(-1)[0], tmp)
    max_val = torch.maximum(w.max(-1)[0], tmp)

    if q.symmetric:
        max_val = torch.maximum(abs(min_val), max_val)
        tmp = min_val < 0
        if torch.any(tmp):
            min_val = torch.where(tmp, -max_val, min_val)

    dead = (min_val == 0) & (max_val == 0)
    min_val = torch.where(dead, torch.tensor(-1.0, device=w.device, dtype=w.dtype), min_val)
    max_val = torch.where(dead, torch.tensor(1.0, device=w.device, dtype=w.dtype), max_val)

    scales = (max_val - min_val).clamp(min=1e-5) / (q.maxq - q.minq)
    if q.symmetric:
        zero_points = torch.full_like(scales, q.midq)
    else:
        zero_points = torch.round(q.minq - min_val / scales).clamp(q.minq, q.maxq)

    # STE quantize-dequantize: round(w/scale + zp) clamped, then (q - zp) * scale
    q_w = torch.clamp(
        STERound.apply(w / scales.unsqueeze(-1) + zero_points.unsqueeze(-1)),
        q.minq,
        q.maxq,
    )
    dq_w = (q_w - zero_points.unsqueeze(-1)) * scales.unsqueeze(-1)
    return dq_w.reshape(orig_shape)


class FakeQuantLinear(nn.Module):
    """Linear module wrapper that applies fake quantization to the weight during forward pass."""

    def __init__(self, linear: nn.Linear, quantizer: WeightQuantizer):
        super().__init__()
        self.linear = linear
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = _ste_fake_quantize(self.linear.weight, self.quantizer)
        return F.linear(x, q_weight.to(x.dtype), self.linear.bias)


class FakeQuantEmbedding(nn.Module):
    """Embedding module wrapper that applies fake quantization to the weight during forward pass."""

    def __init__(self, embedding: nn.Embedding, quantizer: WeightQuantizer):
        super().__init__()
        self.embedding = embedding
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = _ste_fake_quantize(self.embedding.weight, self.quantizer)
        return F.embedding(
            x,
            q_weight.to(self.embedding.weight.dtype),
            padding_idx=self.embedding.padding_idx,
            max_norm=self.embedding.max_norm,
            norm_type=self.embedding.norm_type,
            scale_grad_by_freq=self.embedding.scale_grad_by_freq,
            sparse=self.embedding.sparse,
        )


# =====================================================================
# Knowledge Distillation Loss (CAKLD)
# =====================================================================


def _cakld_loss(
    labels: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Context-Aware KL Divergence: convex combination of forward and reverse KL."""
    mask = labels != IGNORE_INDEX

    # reverse KL: KL(student || teacher)
    teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
    student_prob = F.softmax(student_logits, dim=-1)
    reverse_kl = F.kl_div(teacher_log_prob, student_prob, reduction="none").sum(-1)

    # forward KL: KL(teacher || student)
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    forward_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(-1)

    kl = beta * reverse_kl + (1 - beta) * forward_kl
    kl = kl * mask
    return kl.sum(-1).mean()


# =====================================================================
# KD Trainer
# =====================================================================


class KDTrainer(Trainer):
    """Trainer subclass that computes CAKLD knowledge distillation loss.

    Runs the teacher model in ``no_grad``, computes student outputs, and returns the KD loss.
    """

    def __init__(self, teacher_model, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        teacher_logits = teacher_outputs.get("logits")
        del teacher_outputs

        student_outputs = model(**inputs)
        student_logits = student_outputs.get("logits")

        loss = self.loss_fn(inputs["labels"], student_logits, teacher_logits)

        del teacher_logits
        if not return_outputs:
            del student_logits, student_outputs
            return loss
        return loss, student_outputs


# =====================================================================
# CAKLD beta computation
# =====================================================================


@torch.no_grad()
def _compute_cakld_beta(
    teacher_model: PreTrainedModel,
    train_dataset: Dataset,
    collate_fn,
    batch_size: int = 4,
    max_steps: int = 100,
) -> torch.Tensor:
    """Compute the mean max probability coefficient for CAKLD loss.

    Runs the teacher model on a subset of the training data and computes the average maximum token
    probability, which is used as the *beta* mixing coefficient for CAKLD.
    """
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )

    device = next(teacher_model.parameters()).device
    prob_sum = torch.tensor(0.0, device=device)
    steps = 0
    for batch in dataloader:
        if steps >= max_steps:
            break
        batch_data = {k: v.to(device) for k, v in batch.items()}
        outputs = teacher_model(**batch_data)
        logits = outputs.get("logits").contiguous()
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1).values
        prob_sum += max_probs.mean()
        steps += 1

    if steps == 0:
        return torch.tensor(0.5, device=device)
    return prob_sum / steps


# =====================================================================
# Training Arguments
# =====================================================================


class HFTrainingArguments(BaseHFTrainingArguments):
    """Training arguments for BitDistiller KD-QAT.

    Has the same fields as ``transformers.TrainingArguments`` with recommended defaults for
    low-bit knowledge distillation fine-tuning.
    """

    bf16: bool = Field(True, description="Whether to use bfloat16 precision.")
    per_device_train_batch_size: int = Field(4, description="Batch size per device for training.")
    gradient_accumulation_steps: int = Field(2, description="Gradient accumulation steps.")
    learning_rate: float = Field(5e-5, description="Initial learning rate.")
    weight_decay: float = Field(0.0, description="Weight decay.")
    lr_scheduler_type: str = Field("cosine", description="Learning rate schedule.")
    num_train_epochs: int = Field(1, description="Number of training epochs.")
    logging_steps: int = Field(10, description="Log every n steps.")
    warmup_ratio: float = Field(0.03, description="Fraction of steps to do a warmup for.")
    eval_strategy: str = Field(
        None,
        description=(
            "Evaluation strategy. Forced to 'no' if no eval dataset is provided. Otherwise 'steps' unless set to"
            " 'epoch'."
        ),
    )


# =====================================================================
# BitDistiller Pass
# =====================================================================


class BitDistiller(Pass):
    """BitDistiller: Knowledge-Distilled Quantization-Aware Training for low-bit LLMs.

    This pass performs quantization-aware training (QAT) using knowledge distillation from a
    full-precision teacher model. Linear layers are wrapped with fake quantization using a
    Straight-Through Estimator so that the quantization error is minimized during training.

    After training, the model weights are finalized into packed quantized format.

    See https://arxiv.org/abs/2402.10631 for more details.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_quantizer_config(allow_embeds=True),
            "cakld_max_steps": PassConfigParam(
                type_=int,
                default_value=100,
                description=(
                    "Number of steps to compute the mean max probability coefficient (beta) for the CAKLD loss."
                    " Default is 100."
                ),
            ),
            "torch_dtype": PassConfigParam(
                type_=str,
                default_value="bfloat16",
                description=(
                    "Data type for training. One of 'bfloat16', 'float16' or 'float32'. Default is 'bfloat16'."
                ),
            ),
            "train_data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                required=True,
                description="Data config for training.",
            ),
            "eval_data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                description="Data config for evaluation. Optional.",
            ),
            "training_args": PassConfigParam(
                type_=Union[HFTrainingArguments, dict],
                default_value=None,
                description="Training arguments. If None, will use default arguments.",
            ),
        }

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        # use default training args if not provided
        training_args = config.training_args or HFTrainingArguments()
        if isinstance(training_args, dict):
            training_args = HFTrainingArguments.parse_obj(training_args)

        # ── Prepare student model with quant_info ──
        wrapper, qcfg, retie_word_embeddings = prepare_model(model, config, allow_quantized=False)

        # ── Replace marked modules with FakeQuant wrappers ──
        def should_wrap(module: nn.Module, _: str) -> bool:
            return isinstance(module, (nn.Linear, nn.Embedding)) and hasattr(module, "quant_info")

        def wrap_with_fake_quant(module: nn.Module, _: str) -> nn.Module:
            quantizer = module.quant_info.quantizer
            if isinstance(module, nn.Embedding):
                return FakeQuantEmbedding(module, quantizer)
            return FakeQuantLinear(module, quantizer)

        replace_matching_submodules(
            wrapper.model,
            should_wrap,
            wrap_with_fake_quant,
            description="Wrapping modules with fake quantization",
        )

        student_model = wrapper.model

        # Enable gradient checkpointing for memory efficiency (but do NOT freeze params;
        # all student weights are trained, matching the original BitDistiller implementation)
        if training_args.gradient_checkpointing and not enable_gradient_checkpointing(student_model):
            training_args.gradient_checkpointing = False

        # set model_parallel flags so Trainer doesn't try DataParallel
        student_model.model_parallel = True
        student_model.is_parallelizable = True
        student_model.accepts_loss_kwargs = False

        logger.debug("Trainable parameters in student model: %s", count_trainable_parameters(student_model))

        # ── Load teacher model ──
        teacher_model = load_hf_base_model(
            model,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            device_map="auto",
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False

        # ── Load datasets ──
        tokenizer = model.get_hf_tokenizer()
        train_dataset, eval_dataset = self._get_datasets(config)
        collate_fn = partial(self._collate_batch, tokenizer=tokenizer)

        # ── Build loss function ──
        loss_fn = self._build_loss_fn(config, teacher_model, train_dataset, collate_fn, training_args)

        # ── Set up training args ──
        orig_eval_strat = training_args.eval_strategy
        training_args.eval_strategy = "no"
        if eval_dataset:
            training_args.eval_strategy = "steps" if orig_eval_strat in {None, "no"} else orig_eval_strat

        with tempfile.TemporaryDirectory(prefix="olive_tmp") as temp_dir:
            checkpoint = getattr(training_args, "resume_from_checkpoint", None)
            if not training_args.output_dir:
                training_args.output_dir = temp_dir
                training_args.extra_args = training_args.extra_args or {}
                training_args.extra_args["save_total_limit"] = 1

            # fp16 mixed precision
            torch_dtype = self._get_torch_dtype(config.torch_dtype)
            if torch_dtype == torch.float16:
                training_args.extra_args = training_args.extra_args or {}
                training_args.extra_args["fp16"] = True

            logger.debug("Training args: %s", training_args.dict())

            trainer = KDTrainer(
                teacher_model=teacher_model,
                loss_fn=loss_fn,
                model=student_model,
                args=training_args.create_training_args(),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collate_fn,
            )

            logger.info("Running BitDistiller KD-QAT training")
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            logger.debug("train_result: %s", train_result)

        # ── Clean up teacher model ──
        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Unwrap FakeQuant wrappers back to nn.Linear / nn.Embedding ──
        def should_unwrap(module: nn.Module, _: str) -> bool:
            return isinstance(module, (FakeQuantLinear, FakeQuantEmbedding))

        def unwrap(module: nn.Module, _: str) -> nn.Module:
            if isinstance(module, FakeQuantEmbedding):
                return module.embedding
            return module.linear

        replace_matching_submodules(
            student_model,
            should_unwrap,
            unwrap,
            description="Unwrapping FakeQuantLinear modules",
        )

        # Determine device for finalization
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return finalize(
            model,
            output_model_path,
            wrapper,
            qcfg,
            device,
            retie_word_embeddings=retie_word_embeddings,
        )

    # ── Helper methods ──

    @staticmethod
    def _get_datasets(config: type[BasePassConfig]) -> tuple[Dataset, Dataset | None]:
        """Load training and optionally evaluation datasets."""
        train_dataset = get_training_dataset(config.train_data_config)
        eval_dataset = None
        if config.eval_data_config:
            eval_dataset = get_training_dataset(config.eval_data_config)
        return train_dataset, eval_dataset

    @staticmethod
    def _collate_batch(batch: list[dict], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
        """Collate a batch with padding for input_ids, attention_mask, and labels."""
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [sample["input_ids"] for sample in batch]
        attention_mask = None
        if "attention_mask" in batch[0]:
            attention_mask = [sample["attention_mask"] for sample in batch]
        label_col = "labels" if "labels" in batch[0] else "label"
        if label_col not in batch[0]:
            labels = [input_id.clone() for input_id in input_ids]
        else:
            labels = [sample[label_col] for sample in batch]

        new_batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
        }
        if attention_mask:
            new_batch["attention_mask"] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return new_batch

    @staticmethod
    def _build_loss_fn(config, teacher_model, train_dataset, collate_fn, training_args):
        """Build the CAKLD loss function with computed beta."""
        beta = _compute_cakld_beta(
            teacher_model,
            train_dataset,
            collate_fn,
            batch_size=training_args.per_device_train_batch_size,
            max_steps=config.cakld_max_steps,
        )
        logger.info("Computed CAKLD beta (mean max prob): %s", beta.item())
        return partial(_cakld_loss, beta=beta)

    @staticmethod
    def _get_torch_dtype(torch_dtype: str) -> torch.dtype:
        """Resolve a string dtype to a torch.dtype."""
        from olive.common.utils import resolve_torch_dtype

        supported = ("bfloat16", "float16", "float32")
        assert torch_dtype in supported, f"torch_dtype must be one of {supported} but got {torch_dtype}"
        return resolve_torch_dtype(torch_dtype)
