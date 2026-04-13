# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on original implementation at
# https://github.com/OpenBitSys/BitDistiller/blob/main/quantization/autoclip.py
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Union

import torch

from olive.data.config import DataConfig
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.quant_utils import (
    get_quantizer_config,
    prepare_model,
    run_layerwise_quantization,
)
from olive.passes.pytorch.train_utils import get_calibration_data_config

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler


logger = logging.getLogger(__name__)


class AutoClip(Pass):
    """AutoClip quantization-aware clipping for weights."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_quantizer_config(),
            "n_grid": PassConfigParam(
                type_=int,
                default_value=20,
                description="Number of grid steps to search for clipping.",
            ),
            "max_shrink": PassConfigParam(
                type_=float,
                default_value=0.5,
                description="Maximum shrink ratio for clip bounds.",
            ),
            "n_sample_token": PassConfigParam(
                type_=int,
                default_value=512,
                description="Number of token samples for input feature selection.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value=None,
                description=(
                    "Data config for clipping calibration. If not provided, wikitest train data will be used for"
                    " HfModels. Required for PyTorch models."
                ),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.group_size <= 0 and config.group_size != -1:
            logger.info("group_size must be -1 or greater than 0")
            return False

        bits = config.bits.value if hasattr(config.bits, "value") else config.bits
        if bits not in [2, 4, 8]:
            logger.info("bits must be 2, 4, or 8")
            return False

        return True

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        wrapper, _, _ = prepare_model(model, config, exclude_attn_inputs=True)

        data_config = config.data_config or get_calibration_data_config(
            model.model_name_or_path,
            trust_remote_code=model.get_load_kwargs().get("trust_remote_code", None),
            data_name="mit-han-lab/pile-val-backup",
            subset=None,
            split="validation[:1000]",
            max_seq_len=1024,
            max_samples=128,
        )
        process_module = partial(
            self.process_module,
            n_grid=config.n_grid,
            max_shrink=config.max_shrink,
            n_sample_token=config.n_sample_token,
        )
        run_layerwise_quantization(
            model,
            wrapper,
            data_config,
            input_hook=self.accumulate_inputs,
            process_module=process_module,
            update_before_process=True,
            include_lm_head=config.lm_head,
        )

        # TODO(jambayk): explore whether we should tie the embedding with lm_head after lm_head is clipped

        wrapper.model.save_pretrained(output_model_path)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)

    @staticmethod
    def _get_oc_batch_size(out_features: int) -> int:
        for candidate in [256, 128, 64, 32, 16]:
            if out_features % candidate == 0:
                return candidate
        return out_features

    @staticmethod
    def accumulate_inputs(module: torch.nn.Module, inputs: tuple, _: torch.Tensor) -> None:
        if module.quant_info.data is None:
            module.quant_info.data = {"inputs": []}
        module.quant_info.data["inputs"].append(inputs[0].detach().cpu())

    @classmethod
    def process_module(
        cls,
        module: torch.nn.Module,
        device: str,
        n_grid: int,
        max_shrink: float,
        n_sample_token: int,
    ) -> None:
        if module.quant_info.data is None or not module.quant_info.data.get("inputs"):
            raise ValueError(f"Module {module} does not have cached inputs initialized!")

        input_feat = torch.cat(module.quant_info.data["inputs"], dim=0)
        module.quant_info.data = None

        module.to(device)
        cls._auto_clip_layer(
            module,
            input_feat,
            n_grid,
            max_shrink,
            n_sample_token,
        )
        module.to("cpu")

    @classmethod
    def _auto_clip_layer(
        cls,
        module: torch.nn.Module,
        input_feat: torch.Tensor,
        n_grid: int,
        max_shrink: float,
        n_sample_token: int,
    ) -> None:
        weight = module.weight.data
        if weight.dim() != 2:
            raise ValueError("AutoClip expects a 2D linear weight tensor.")

        quantizer = module.quant_info.quantizer
        effective_group_size = weight.shape[1] if quantizer.group_size <= 0 else quantizer.group_size
        if weight.shape[1] % effective_group_size != 0:
            raise ValueError("Weight in_features must be divisible by group_size.")

        input_feat = input_feat.view(-1, input_feat.shape[-1])
        if input_feat.shape[0] > n_sample_token:
            step = max(1, input_feat.shape[0] // n_sample_token)
            input_feat = input_feat[::step]
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, effective_group_size)

        weight = weight.reshape(weight.shape[0], 1, -1, effective_group_size)

        oc_batch_size = cls._get_oc_batch_size(weight.shape[0])
        best_max_val_all = []
        best_min_val_all = []

        input_feat = input_feat.to(weight.device)
        for i_b in range(0, weight.shape[0], oc_batch_size):
            w_block = weight[i_b : i_b + oc_batch_size]

            org_max_val = w_block.amax(dim=-1, keepdim=True)
            org_min_val = w_block.amin(dim=-1, keepdim=True)

            best_max_val = org_max_val.clone()
            best_min_val = org_min_val.clone()
            min_errs = torch.full_like(org_max_val, 1e9)

            org_out = (input_feat * w_block).sum(dim=-1)

            for i_s_p in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s_p / n_grid)
                for i_s_n in range(int(max_shrink * n_grid)):
                    min_val = org_min_val * (1 - i_s_n / n_grid)
                    cur_w = torch.clamp(w_block, min_val, max_val)
                    q_w = quantizer.fake_quantize(cur_w.reshape(cur_w.shape[0], -1)).reshape(cur_w.shape)

                    cur_out = (input_feat * q_w).sum(dim=-1)
                    err = (cur_out - org_out).pow(2).mean(dim=1).reshape(min_errs.shape)

                    cur_best = err < min_errs
                    min_errs[cur_best] = err[cur_best]
                    best_max_val[cur_best] = max_val[cur_best]
                    best_min_val[cur_best] = min_val[cur_best]

            best_max_val_all.append(best_max_val)
            best_min_val_all.append(best_min_val)

        best_max_val = torch.cat(best_max_val_all, dim=0).squeeze(1)
        best_min_val = torch.cat(best_min_val_all, dim=0).squeeze(1)
        original_shape = module.weight.data.shape
        clipped = module.weight.data.reshape(best_max_val.shape[0], best_max_val.shape[1], -1)
        clipped = torch.clamp(clipped, best_min_val, best_max_val)
        module.weight.data = clipped.reshape(original_shape)
