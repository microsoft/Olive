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
from olive.passes.pytorch.moe_calib import DEFAULT_MOE_FALLBACK_THRESHOLD, MoeCalibrationSession
from olive.passes.pytorch.quant_utils import (
    _module_weight_has_quant_info,
    finalize,
    get_quantizer_config,
    module_quant_info_param_names,
    prepare_model,
    run_layerwise_quantization,
)
from olive.search.search_parameter import Categorical

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
            **get_quantizer_config(allow_moe=True),
            "damp_percent": PassConfigParam(
                type_=float,
                default_value=0.01,
                search_defaults=Categorical([0.001, 0.01, 0.1]),
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
            "moe_fallback_threshold": PassConfigParam(
                type_=float,
                default_value=DEFAULT_MOE_FALLBACK_THRESHOLD,
                description=(
                    "Only used when moe=True. Fraction of the calibration tokens reaching an experts"
                    " module below which an individual expert is quantized with round-to-nearest"
                    " instead of GPTQ. MoE routing sends only a fraction of the tokens to each"
                    " expert, so cold experts get a rank-deficient (or missing) Hessian for which"
                    " GPTQ's correction is driven by the damping prior rather than by data."
                    " Default is 0.005 (0.5%, matching GPTQModel's default)."
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

        if not 0 <= config.moe_fallback_threshold < 1:
            logger.info("moe_fallback_threshold must be in [0, 1).")
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
        moe_session = (
            MoeCalibrationSession.create(wrapper, fallback_threshold=config.moe_fallback_threshold)
            if getattr(config, "moe", False)
            else None
        )
        device = run_layerwise_quantization(
            model,
            wrapper,
            config.data_config,
            input_hook=self.accumulate_hessian,
            process_module=lambda module, _: self.process_module(
                module,
                percdamp=config.damp_percent,
                actorder=config.desc_act,
                moe_fallback_threshold=config.moe_fallback_threshold,
            ),
            update_before_process=False,
            include_lm_head=config.lm_head,
            moe_session=moe_session,
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
        if module.weight.quant_info.data is None:
            module.weight.quant_info.data = {
                "H": torch.zeros((module.in_features, module.in_features), device=inp[0].device),
                "N": 0,
            }

        batch_size = inp[0].shape[0]
        inp = inp[0].reshape(-1, module.in_features).t()

        module.weight.quant_info.data["H"] *= module.weight.quant_info.data["N"] / (
            module.weight.quant_info.data["N"] + batch_size
        )
        module.weight.quant_info.data["N"] += batch_size
        inp = math.sqrt(2 / module.weight.quant_info.data["N"]) * inp.float()
        module.weight.quant_info.data["H"] += inp.matmul(inp.t())

    @staticmethod
    def process_module(
        module: torch.nn.Module,
        blocksize: int = 128,
        percdamp: float = 0.01,
        actorder: bool | None = False,
        moe_fallback_threshold: float = DEFAULT_MOE_FALLBACK_THRESHOLD,
    ) -> None:
        """Process a module for GPTQ quantization using the accumulated calibration data.

        Dispatches on how the module's selected parameters were calibrated:

        * ``nn.Linear`` / ``nn.Embedding`` ``weight`` -- a single ``(K, K)`` Hessian
          collected by :meth:`accumulate_hessian` from a forward hook;
        * fused-3D MoE experts parameters (``gate_up_proj`` / ``down_proj``) -- one
          independent ``(K, K)`` Hessian *per expert*, collected by
          :mod:`olive.passes.pytorch.moe_calib`. Experts that saw too few calibration
          tokens fall back to RTN.

        Args:
            module: The module to quantize.
            blocksize: Block size for processing weights.
            percdamp: Damping factor for numerical stability.
            actorder: Whether to use act-order quantization scheme.
            moe_fallback_threshold: Fraction of the calibration tokens reaching an experts
                module below which an expert is quantized with the RTN fallback.

        """
        if _module_weight_has_quant_info(module):
            Gptq._process_dense_module(module, blocksize=blocksize, percdamp=percdamp, actorder=actorder)
        else:
            for pname in module_quant_info_param_names(module):
                Gptq._process_moe_param(
                    module,
                    pname,
                    blocksize=blocksize,
                    percdamp=percdamp,
                    actorder=actorder,
                    fallback_threshold=moe_fallback_threshold,
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _process_dense_module(
        module: torch.nn.Module, blocksize: int = 128, percdamp: float = 0.01, actorder: bool | None = False
    ) -> None:
        """Quantize ``module.weight`` with GPTQ using its accumulated Hessian."""
        info = module.weight.quant_info
        if info.data is None:
            raise ValueError(f"Module {module} does not have quant_info.data initialized!")

        Q, scales, zero_points = gptq_quantize_weight(
            module.weight.data.clone().float().to(info.data["H"].device),
            info.data["H"],
            info.quantizer,
            blocksize=blocksize,
            percdamp=percdamp,
            actorder=actorder,
        )

        module.weight.data = Q.to(module.weight.data.device).to(module.weight.data.dtype)
        info.scales = scales.to("cpu")
        info.zero_points = zero_points.to("cpu")
        info.data = None

    @staticmethod
    def _process_moe_param(
        module: torch.nn.Module,
        pname: str,
        blocksize: int = 128,
        percdamp: float = 0.01,
        actorder: bool | None = False,
        fallback_threshold: float = DEFAULT_MOE_FALLBACK_THRESHOLD,
    ) -> None:
        """Quantize one fused-3D MoE parameter, expert by expert.

        Each expert is quantized from its *own* Hessian. Experts whose observed token count
        is below ``fallback_threshold`` x (calibration tokens reaching this experts module)
        -- including experts that were never routed to and therefore have no Hessian at all
        -- are quantized with RTN instead: GPTQ on a rank-deficient Hessian is driven by the
        damping prior rather than by data, so the fallback is never worse than plain RTN for
        that expert.
        """
        param = module._parameters[pname]  # pylint: disable=protected-access
        info = param.quant_info
        data = info.data
        if not data or not data.get("moe"):
            raise ValueError(
                f"MoE parameter '{pname}' of {type(module).__name__} has no per-expert calibration "
                "data. This usually means the experts forward was never intercepted during "
                "calibration."
            )

        weight = param.data
        num_experts = weight.shape[0]
        threshold = fallback_threshold * data["tokens_seen"]
        observed = data["token_counts"]

        expert_weights, expert_scales, expert_zero_points = [], [], []
        fallback_experts = []
        for expert_idx in range(num_experts):
            entry = data["experts"].get(expert_idx)
            W = weight[expert_idx].clone().float()
            if entry is None or entry["N"] < threshold:
                fallback_experts.append(expert_idx)
                # RTN: derive qparams straight from the float weight and fake-quantize with
                # them. Fake-quantizing here (rather than deferring all rounding to
                # ``finalize``) keeps the true-sequential invariant: the post-quantization
                # re-run of the layer sees on-grid weights for *every* expert, GPTQ or
                # fallback. ``finalize`` is idempotent on an already-on-grid tensor, so the
                # result is still bit-identical to what the Rtn pass would have produced.
                scales, zero_points = info.quantizer.find_qparams(W)
                Q = info.quantizer.fake_quantize(W, scales, zero_points)
            else:
                Q, scales, zero_points = gptq_quantize_weight(
                    W.to(entry["H"].device),
                    entry["H"],
                    info.quantizer,
                    blocksize=blocksize,
                    percdamp=percdamp,
                    actorder=actorder,
                )
            expert_weights.append(Q.to(weight.device).to(weight.dtype))
            expert_scales.append(scales.to("cpu"))
            expert_zero_points.append(zero_points.to("cpu"))

        if fallback_experts:
            logger.info(
                "GPTQ MoE fallback for '%s': %d/%d experts quantized with RTN "
                "(observed tokens below threshold %.1f = %.2f%% of the %d calibration tokens "
                "reaching this module; observed min=%d max=%d). Experts: %s",
                type(module).__name__ + "." + pname,
                len(fallback_experts),
                num_experts,
                threshold,
                100 * fallback_threshold,
                data["tokens_seen"],
                min(observed, default=0),
                max(observed, default=0),
                fallback_experts,
            )

        param.data = torch.stack(expert_weights, dim=0)
        info.scales = torch.stack(expert_scales, dim=0)
        info.zero_points = torch.stack(expert_zero_points, dim=0)
        info.data = None


@torch.no_grad()
def gptq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    quantizer: WeightQuantizer,
    blocksize: int = 128,
    percdamp: float = 0.01,
    actorder: bool | None = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the GPTQ column sweep on a single 2D weight matrix.

    Args:
        W: Float weight of shape ``(out_features, K)``; quantization runs along the last dim.
        H: The ``(K, K)`` Hessian accumulated from this weight's calibration inputs. Modified
            in place (dead-column patching + damping).
        quantizer: The target :class:`WeightQuantizer` (supplies bits / symmetric / group_size).
        blocksize: Column block size for the error-compensated sweep.
        percdamp: Damping factor for numerical stability.
        actorder: Act-order (desc_act) scheme. ``None`` means "True iff per-channel".

    Returns:
        ``(Q, scales, zero_points)`` -- the fake-quantized weight and its qparams, with
        ``scales``/``zero_points`` of shape ``(out_features, num_groups)``.

    """
    group_size = quantizer.group_size
    if actorder is None:
        actorder = group_size == -1
    elif actorder is True:
        assert group_size == -1, f"actorder can only be True when group_size is -1, but got group_size={group_size}"

    W = W.to(H.device)
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
    per_channel_quantizer = WeightQuantizer(
        bits=quantizer.bits,
        symmetric=quantizer.symmetric,
        group_size=-1,
    )
    if group_size == -1:
        # this can be before or after actorder permutation since there's only one group
        active_scale, active_zp = per_channel_quantizer.find_qparams(W)
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

            if group_size != -1:
                if (i1 + i) % group_size == 0:
                    active_scale, active_zp = per_channel_quantizer.find_qparams(W[:, (i1 + i) : (i1 + i + group_size)])

                if ((i1 + i) // group_size) - now_idx == -1:
                    all_scales.append(active_scale)
                    all_zp.append(active_zp)
                    now_idx += 1

            q = per_channel_quantizer.fake_quantize(w.unsqueeze(1), active_scale, active_zp).flatten()
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

    return Q, torch.cat(all_scales, dim=1), torch.cat(all_zp, dim=1)
