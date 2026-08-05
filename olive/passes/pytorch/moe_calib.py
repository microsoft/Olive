# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Per-expert calibration for MoE (Mixture-of-Experts) weight quantization.

Calibrated passes (GPTQ) need the *activations that actually reach each expert*. For fused
MoE architectures routing happens inside a single experts-module forward call, so a plain
``register_forward_hook`` on the experts module sees one undifferentiated activation batch
and cannot attribute rows to experts.

Interception therefore goes through transformers' own experts-implementation registry
(``ALL_EXPERTS_FUNCTIONS`` / ``@use_experts_implementation``, transformers >= 5.0): we
register one generic recording implementation and point the model at it for the duration of
calibration. Every decorated experts class dispatches to it with no per-model branching,
which is why this file contains no architecture-specific code paths -- only an allow-list of
the architectures whose fused weights are stored ``(num_experts, out_features, in_features)``
(K last), matching GPTQ's Hessian layout.

Transposed-layout architectures (gpt_oss, llama4, aria) store ``(num_experts, in, out)``.
GPTQ's ``(K, K)`` Hessian math assumes K is the last dim, so they are refused with a clear
error rather than silently mis-quantized; supporting them requires a layout-normalization
step that is deliberately out of scope here.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from olive.common.hf.wrapper import ModelWrapper

logger = logging.getLogger(__name__)


#: ``model_type``s whose fused expert weights are stored ``(num_experts, out, in)`` (K last)
#: *and* whose experts class carries transformers' ``@use_experts_implementation`` decorator.
#: Verified against transformers 5.14.1. Anything else is refused (fail closed).
SUPPORTED_MOE_MODEL_TYPES = frozenset(
    {
        "deepseek_v3",
        "granitemoe",
        "jamba",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    }
)

#: Key under which the recording forward is registered in ``ALL_EXPERTS_FUNCTIONS``.
OLIVE_MOE_CALIB_IMPLEMENTATION = "olive_moe_calib"

#: Minimum transformers version exposing ``ALL_EXPERTS_FUNCTIONS`` / ``set_experts_implementation``.
MIN_TRANSFORMERS_VERSION = "5.0.0"

#: Fraction of the calibration tokens reaching an experts module below which an expert is
#: quantized by the RTN fallback instead of GPTQ. Matches GPTQModel's ``"0.5%"`` default.
DEFAULT_MOE_FALLBACK_THRESHOLD = 0.005


class MoeCalibrationError(ValueError):
    """Raised when calibrated MoE quantization cannot be performed correctly."""


# ---------------------------------------------------------------------------
# recording forward implementation
# ---------------------------------------------------------------------------

# id(experts_module) -> _ExpertsRecorder, populated only while a layer is being calibrated.
_ACTIVE_RECORDERS: dict[int, _ExpertsRecorder] = {}


def olive_moe_calib_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Reference per-expert experts forward that also records per-expert calibration inputs.

    Registered in transformers' ``ALL_EXPERTS_FUNCTIONS`` registry, so it replaces the
    experts forward of *every* decorated experts module while the calibration
    implementation is active. It reproduces the canonical eager loop shared by every
    allow-listed architecture (``F.linear`` on ``W[e]`` of shape ``(out, in)``, gated
    activation, routing-weighted scatter-add), so outputs are correct on both the
    Hessian-collection pass and the post-quantization re-run of the true-sequential loop.

    Recording is driven by :data:`_ACTIVE_RECORDERS`: only experts modules registered by
    :meth:`MoeCalibrationSession.record` are recorded, and only while that context is
    active -- this is what keeps the second (post-quantization) ``run_layer`` from
    double-counting into the Hessians.
    """
    recorder = _ACTIVE_RECORDERS.get(id(self))

    if hidden_states.dim() != 2:
        raise MoeCalibrationError(
            "Olive's MoE calibration expects the experts forward to receive 2D "
            f"(num_tokens, hidden_dim) hidden states, got shape {tuple(hidden_states.shape)}."
        )

    num_experts = self.num_experts
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    if recorder is not None:
        recorder.note_tokens(hidden_states.shape[0])

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        intermediate = self._apply_gate(  # pylint: disable=protected-access
            torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
        )
        if recorder is not None:
            recorder.record(int(expert_idx), {"gate_up_proj": current_state, "down_proj": intermediate})
        current_hidden_states = torch.nn.functional.linear(intermediate, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def _register_calib_implementation() -> None:
    """Register :func:`olive_moe_calib_experts_forward` in transformers' experts registry."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    if OLIVE_MOE_CALIB_IMPLEMENTATION not in ALL_EXPERTS_FUNCTIONS:
        ALL_EXPERTS_FUNCTIONS.register(OLIVE_MOE_CALIB_IMPLEMENTATION, olive_moe_calib_experts_forward)


# ---------------------------------------------------------------------------
# per-expert recording
# ---------------------------------------------------------------------------


class _ExpertsRecorder:
    """Accumulates one independent Hessian per (parameter, expert) for a single experts module.

    Each expert gets its *own* ``(K, K)`` Hessian -- activations are never pooled across
    experts, because different experts see systematically different activation
    distributions by routing design. Hessians are allocated lazily on an expert's first
    sample, so experts that never get routed simply have no entry (and are handed to the
    RTN fallback).

    The accumulated state is written straight onto each parameter's
    ``quant_info.data`` so the quantization math (``Gptq.process_module``) reads it the
    same way it reads the dense path's ``{"H": ..., "N": ...}``.
    """

    def __init__(self, experts: torch.nn.Module, pnames: list[str]):
        self.experts = experts
        self.pnames = pnames
        self.num_experts = int(experts.num_experts)
        self.tokens_seen = 0
        # pname -> {expert_idx: {"H": (K, K) tensor, "N": int}}
        self.hessians: dict[str, dict[int, dict]] = {pname: {} for pname in pnames}
        self.token_counts: list[int] = [0] * self.num_experts

    def note_tokens(self, num_tokens: int) -> None:
        self.tokens_seen += int(num_tokens)

    @torch.no_grad()
    def record(self, expert_idx: int, inputs: dict[str, torch.Tensor]) -> None:
        """Accumulate one expert's activation slice into that expert's Hessian."""
        for pname in self.pnames:
            inp = inputs.get(pname)
            if inp is None:
                continue
            if pname == "gate_up_proj":
                # counted once per (token, expert) pair, from the model's own routing decision
                self.token_counts[expert_idx] += int(inp.shape[0])
            self._accumulate(pname, expert_idx, inp)

    @torch.no_grad()
    def _accumulate(self, pname: str, expert_idx: int, inp: torch.Tensor) -> None:
        num_cols = inp.shape[-1]
        entry = self.hessians[pname].get(expert_idx)
        if entry is None:
            entry = {"H": torch.zeros((num_cols, num_cols), device=inp.device, dtype=torch.float32), "N": 0}
            self.hessians[pname][expert_idx] = entry

        num_rows = inp.shape[0]
        if num_rows == 0:
            return
        x = inp.reshape(-1, num_cols).t()
        entry["H"] *= entry["N"] / (entry["N"] + num_rows)
        entry["N"] += num_rows
        x = math.sqrt(2 / entry["N"]) * x.float()
        entry["H"] += x.matmul(x.t())

    def publish(self) -> None:
        """Write the collected state onto the parameters' ``quant_info.data``."""
        for pname in self.pnames:
            param = self.experts._parameters[pname]  # pylint: disable=protected-access
            param.quant_info.data = {
                "moe": True,
                "experts": self.hessians[pname],
                "tokens_seen": self.tokens_seen,
                "token_counts": list(self.token_counts),
            }


# ---------------------------------------------------------------------------
# coverage report
# ---------------------------------------------------------------------------


@dataclass
class LayerCoverage:
    """Per-layer routing coverage collected during calibration."""

    layer_name: str
    num_experts: int
    tokens_seen: int
    token_counts: list[int]
    threshold: float

    @property
    def unseen(self) -> int:
        return sum(1 for c in self.token_counts if c == 0)

    @property
    def starved(self) -> int:
        return sum(1 for c in self.token_counts if 0 < c < self.threshold)

    @property
    def covered(self) -> int:
        return self.num_experts - self.unseen - self.starved

    def format(self) -> str:
        counts = sorted(self.token_counts)
        median = counts[len(counts) // 2] if counts else 0
        return (
            f"MoE coverage [{self.layer_name}]: {self.covered}/{self.num_experts} experts covered, "
            f"{self.starved} starved (< {self.threshold:.1f} tokens), {self.unseen} unseen; "
            f"tokens/expert min={min(counts, default=0)} median={median} max={max(counts, default=0)} "
            f"(calibration tokens reaching the layer: {self.tokens_seen})"
        )


@dataclass
class CoverageReport:
    """Aggregated routing coverage across every MoE layer of a calibration run."""

    layers: list[LayerCoverage] = field(default_factory=list)

    def add(self, coverage: LayerCoverage) -> None:
        self.layers.append(coverage)
        logger.info("%s", coverage.format())

    def format_summary(self) -> str:
        if not self.layers:
            return "MoE coverage summary: no MoE layers were calibrated."
        total = sum(lc.num_experts for lc in self.layers)
        starved = sum(lc.starved for lc in self.layers)
        unseen = sum(lc.unseen for lc in self.layers)
        return (
            f"MoE coverage summary: {len(self.layers)} MoE layers, {total} experts total, "
            f"{starved} starved ({100 * starved / total:.1f}%), {unseen} unseen "
            f"({100 * unseen / total:.1f}%). Starved/unseen experts are quantized with the "
            "RTN fallback instead of GPTQ."
        )

    def log_summary(self) -> None:
        message = self.format_summary()
        if any(lc.starved or lc.unseen for lc in self.layers):
            logger.warning("%s", message)
        else:
            logger.info("%s", message)


# ---------------------------------------------------------------------------
# support gating
# ---------------------------------------------------------------------------


def _transformers_supports_experts_registry() -> bool:
    try:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS  # noqa: F401
    except ImportError:
        return False
    return True


def _unsupported_model_type_message(model_type: str) -> str:
    return (
        f"Calibrated MoE quantization (moe=True) is not supported for model_type='{model_type}'. "
        f"Supported architectures: {sorted(SUPPORTED_MOE_MODEL_TYPES)}. Architectures such as "
        "gpt_oss / llama4 / aria store fused expert weights transposed "
        "((num_experts, in_features, out_features)), which is incompatible with GPTQ's (K, K) "
        "Hessian layout and is not handled by this pass. Re-run with moe=False (experts stay in "
        "full precision), or quantize the experts with the Rtn pass instead."
    )


def check_moe_gptq_support(model_type: str, experts_modules: list[torch.nn.Module]) -> None:
    """Fail closed unless calibrated MoE quantization is known to be correct for this model.

    Raises :class:`MoeCalibrationError` when

    * ``model_type`` is not in :data:`SUPPORTED_MOE_MODEL_TYPES`;
    * the installed ``transformers`` predates the experts-implementation registry; or
    * an experts module does not carry the ``@use_experts_implementation`` metadata (probed
      via ``is_transposed``) or declares a layout this pass does not handle.

    No weight is touched before this returns, mirroring the fail-closed guards in
    :mod:`olive.common.quant.selection`.
    """
    if model_type not in SUPPORTED_MOE_MODEL_TYPES:
        raise MoeCalibrationError(_unsupported_model_type_message(model_type))

    if not _transformers_supports_experts_registry():
        raise MoeCalibrationError(
            "Calibrated MoE quantization (moe=True) requires transformers >= "
            f"{MIN_TRANSFORMERS_VERSION}, which provides the experts-implementation registry "
            "(transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS) used to collect per-expert "
            "activations. Upgrade transformers, or re-run with moe=False."
        )

    for experts in experts_modules:
        if not hasattr(experts, "is_transposed"):
            raise MoeCalibrationError(
                f"Experts module '{type(experts).__name__}' does not expose the "
                "'@use_experts_implementation' metadata that Olive needs to intercept per-expert "
                "activations (attribute 'is_transposed' is missing). This usually means the "
                "installed transformers version predates the experts-implementation registry "
                f"(>= {MIN_TRANSFORMERS_VERSION} required) or this architecture has not adopted "
                "it yet. Upgrade transformers, or re-run with moe=False."
            )
        if experts.is_transposed:
            raise MoeCalibrationError(
                f"Experts module '{type(experts).__name__}' stores transposed fused weights "
                "((num_experts, in_features, out_features)), which is incompatible with GPTQ's "
                "(K, K) Hessian layout. Re-run with moe=False, or quantize the experts with the "
                "Rtn pass."
            )
        if getattr(experts, "has_bias", False):
            raise MoeCalibrationError(
                f"Experts module '{type(experts).__name__}' declares expert biases, which Olive's "
                "calibrated MoE path does not handle. Re-run with moe=False."
            )
        if not getattr(experts, "has_gate", True):
            raise MoeCalibrationError(
                f"Experts module '{type(experts).__name__}' declares non-gated experts, which "
                "Olive's calibrated MoE path does not handle. Re-run with moe=False."
            )


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


class MoeCalibrationSession:
    """Owns the experts-implementation swap and the per-layer recording lifecycle.

    Usage (see :func:`olive.passes.pytorch.quant_utils.run_layerwise_quantization`)::

        session = MoeCalibrationSession.create(wrapper, fallback_threshold=0.005)
        if session:
            session.start()          # swap in the recording experts implementation
        ...
        with session.record(experts_modules):   # recording ON for one layer
            run_layer(...)                      # Hessian collection pass
        run_layer(...)                          # true-sequential re-run: recording OFF
        ...
        session.finish()             # restore the original implementation + log summary
    """

    def __init__(self, model: torch.nn.Module, fallback_threshold: float = DEFAULT_MOE_FALLBACK_THRESHOLD):
        self.model = model
        self.fallback_threshold = fallback_threshold
        self.report = CoverageReport()
        self.experts_modules: list[torch.nn.Module] = []
        self._saved_implementation = None
        self._active = False

    @classmethod
    def create(
        cls,
        wrapper: ModelWrapper,
        fallback_threshold: float = DEFAULT_MOE_FALLBACK_THRESHOLD,
    ) -> MoeCalibrationSession | None:
        """Validate support and build a session, or return ``None`` when the model has no experts."""
        experts_modules = [
            experts
            for lw in wrapper.get_layer_wrappers()
            if (experts := lw.get_experts(return_name=False)) is not None
        ]
        if not experts_modules:
            return None

        check_moe_gptq_support(wrapper.model_type, experts_modules)
        session = cls(wrapper.model, fallback_threshold=fallback_threshold)
        session.experts_modules = experts_modules
        return session

    def start(self) -> None:
        """Swap the model's experts implementation to the recording one."""
        if not hasattr(self.model, "set_experts_implementation"):
            raise MoeCalibrationError(
                "Calibrated MoE quantization (moe=True) requires transformers >= "
                f"{MIN_TRANSFORMERS_VERSION}: the loaded model has no "
                "'set_experts_implementation'. Upgrade transformers, or re-run with moe=False."
            )
        _register_calib_implementation()
        self._saved_implementation = self.model.get_experts_implementation()
        self.model.set_experts_implementation(OLIVE_MOE_CALIB_IMPLEMENTATION)
        self._active = True

        # ``set_experts_implementation`` silently no-ops when transformers' source-inspection
        # heuristic decides the class isn't switchable. Verify the swap actually reached every
        # experts module rather than silently collecting zero Hessians.
        stale = [
            type(experts).__name__
            for experts in self.experts_modules
            if getattr(experts.config, "_experts_implementation", None) != OLIVE_MOE_CALIB_IMPLEMENTATION
        ]
        if stale:
            raise MoeCalibrationError(
                "Olive could not switch the experts implementation to "
                f"'{OLIVE_MOE_CALIB_IMPLEMENTATION}' for {sorted(set(stale))}; per-expert "
                "calibration data cannot be collected. Re-run with moe=False."
            )
        logger.debug("Switched experts implementation to '%s' for calibration.", OLIVE_MOE_CALIB_IMPLEMENTATION)

    @contextmanager
    def record(self, experts_modules: list[torch.nn.Module]) -> Iterator[None]:
        """Record per-expert Hessians for ``experts_modules`` for the duration of the block."""
        recorders = []
        for experts in experts_modules:
            pnames = [
                pname
                for pname, param in experts.named_parameters(recurse=False)
                if param is not None and hasattr(param, "quant_info")
            ]
            recorder = _ExpertsRecorder(experts, pnames)
            recorders.append(recorder)
            _ACTIVE_RECORDERS[id(experts)] = recorder
        try:
            yield
        finally:
            for experts in experts_modules:
                _ACTIVE_RECORDERS.pop(id(experts), None)
            for recorder in recorders:
                recorder.publish()

    def add_coverage(self, layer_name: str, experts: torch.nn.Module) -> None:
        """Log (and remember) the routing coverage recorded for one experts module."""
        pnames = [
            pname
            for pname, param in experts.named_parameters(recurse=False)
            if param is not None and hasattr(param, "quant_info")
        ]
        if not pnames:
            return
        data = experts._parameters[pnames[0]].quant_info.data  # pylint: disable=protected-access
        if not data:
            return
        self.report.add(
            LayerCoverage(
                layer_name=layer_name,
                num_experts=len(data["token_counts"]),
                tokens_seen=data["tokens_seen"],
                token_counts=data["token_counts"],
                threshold=self.fallback_threshold * data["tokens_seen"],
            )
        )

    def finish(self) -> None:
        """Restore the original experts implementation and log the coverage summary."""
        if self._active:
            self.model.set_experts_implementation(self._saved_implementation)
            self._active = False
        self.report.log_summary()
