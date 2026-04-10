# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Union

import torch

from olive.common.quant.utils import WeightQuantizer
from olive.data.config import DataConfig
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.quant_utils import (
    finalize,
    get_quantizer_config,
    prepare_model,
    run_layerwise_quantization,
)

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler


logger = logging.getLogger(__name__)

# ruff: noqa: N806


class Gptq(Pass):
    """GPTQ quantization."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_quantizer_config(),
            "damp_percent": PassConfigParam(
                type_=float,
                default_value=0.01,
                description="Damping factor for quantization. Default value is 0.01.",
            ),
            "desc_act": PassConfigParam(
                type_=bool,
                default_value=None,
                description=(
                    "Whether to use act-order (also called desc-act) scheme. True is only supported when group_size is"
                    " -1. Default is None, which is equivalent to True for group_size -1 and False for other group"
                    " sizes."
                ),
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value=None,
                description=(
                    "Data config for quantization. If not provided, wikitest train data will be used for HfModels."
                    " Required for PyTorch models."
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

        if config.desc_act is True and config.group_size != -1:
            logger.info("desc_act can only be True when group_size is -1.")
            return False

        return True

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        """Run GPTQ quantization on the model.

        Args:
            model: The HuggingFace model to quantize.
            config: Configuration object containing quantization parameters.
            output_model_path: Path where the quantized model will be saved.

        Returns:
            HfModelHandler for the quantized model.

        """
        wrapper, qcfg, _ = prepare_model(model, config)
        device = run_layerwise_quantization(
            model,
            wrapper,
            config.data_config,
            input_hook=self.accumulate_hessian,
            process_module=lambda module, _: self.process_module(
                module, percdamp=config.damp_percent, actorder=config.desc_act
            ),
            update_before_process=False,
            include_lm_head=config.lm_head,
        )

        return finalize(model, output_model_path, wrapper, qcfg, device)

    @staticmethod
    def accumulate_hessian(module: torch.nn.Module, inp: tuple, _: Any) -> None:
        """Accumulate Hessian matrix for GPTQ quantization.

        Args:
            module: The linear module to accumulate Hessian for.
            inp: Input tensors to the module.
            _: Unused output parameter.

        """
        if module.quant_info.data is None:
            module.quant_info.data = {
                "H": torch.zeros((module.in_features, module.in_features), device=inp[0].device),
                "N": 0,
            }

        batch_size = inp[0].shape[0]
        inp = inp[0].reshape(-1, module.in_features).t()

        module.quant_info.data["H"] *= module.quant_info.data["N"] / (module.quant_info.data["N"] + batch_size)
        module.quant_info.data["N"] += batch_size
        inp = math.sqrt(2 / module.quant_info.data["N"]) * inp.float()
        module.quant_info.data["H"] += inp.matmul(inp.t())

    @staticmethod
    def process_module(
        module: torch.nn.Module, blocksize: int = 128, percdamp: float = 0.01, actorder: bool | None = False
    ) -> None:
        """Process a module for GPTQ quantization using the accumulated Hessian.

        Args:
            module: The linear module to quantize.
            blocksize: Block size for processing weights.
            percdamp: Damping factor for numerical stability.
            actorder: Whether to use act-order quantization scheme.

        """
        if module.quant_info.data is None:
            raise ValueError(f"Module {module} does not have quant_info.data initialized!")

        if actorder is None:
            actorder = module.quant_info.quantizer.group_size == -1
        elif actorder is True:
            assert module.quant_info.quantizer.group_size == -1, (
                "actorder can only be True when group_size is -1, but got group_size="
                f"{module.quant_info.quantizer.group_size}"
            )

        H = module.quant_info.data["H"]
        W = module.weight.data.clone().float().to(H.device)
        num_cols = H.shape[0]

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(num_cols, device=H.device)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)  # pylint: disable=not-callable
        del H
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)  # pylint: disable=not-callable

        all_scales = []
        all_zp = []
        now_idx = 1
        # create a per-channel quantizer
        quantizer = WeightQuantizer(
            bits=module.quant_info.quantizer.bits, symmetric=module.quant_info.quantizer.symmetric, group_size=-1
        )
        if module.quant_info.quantizer.group_size == -1:
            # this can be before or after actorder permutation since there's only one group
            active_scale, active_zp = quantizer.find_qparams(W)
        else:
            active_scale, active_zp = None, None

        for i1 in range(0, num_cols, blocksize):
            i2 = min(i1 + blocksize, num_cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if module.quant_info.quantizer.group_size != -1:
                    if (i1 + i) % module.quant_info.quantizer.group_size == 0:
                        active_scale, active_zp = quantizer.find_qparams(
                            W[:, (i1 + i) : (i1 + i + module.quant_info.quantizer.group_size)]
                        )

                    if ((i1 + i) // module.quant_info.quantizer.group_size) - now_idx == -1:
                        all_scales.append(active_scale)
                        all_zp.append(active_zp)
                        now_idx += 1

                q = quantizer.fake_quantize(w.unsqueeze(1), active_scale, active_zp).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        if not all_scales:
            all_scales.append(active_scale)
            all_zp.append(active_zp)

        module.weight.data = Q.to(module.weight.data.device).to(module.weight.data.dtype)
        module.quant_info.scales = torch.cat(all_scales, dim=1).to("cpu")
        module.quant_info.zero_points = torch.cat(all_zp, dim=1).to("cpu")

        module.quant_info.data = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
