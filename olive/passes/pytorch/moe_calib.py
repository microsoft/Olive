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
which is why this file contains no architecture-specific code paths. Layout support is
determined from transformers-owned ``is_transposed`` metadata: fused weights must be stored
``(num_experts, out_features, in_features)`` (K last), matching GPTQ's Hessian layout.

Transposed-layout architectures such as gpt-oss store ``(num_experts, in, out)``. GPTQ's
``(K, K)`` Hessian math assumes K is the last dim, so they are refused with a clear error
rather than silently mis-quantized. Architectures such as llama4 and aria that do not report
``is_transposed`` are also refused because Olive cannot verify their layout; supporting
either case requires work that is deliberately out of scope here.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from olive.passes.pytorch.moe_support import MoeSupportError, check_moe_layout_support

if TYPE_CHECKING:
    from collections.abc import Iterator

    from olive.common.hf.wrapper import ModelWrapper

logger = logging.getLogger(__name__)


#: Key under which the recording forward is registered in ``ALL_EXPERTS_FUNCTIONS``.
OLIVE_MOE_CALIB_IMPLEMENTATION = "olive_moe_calib"

#: Minimum transformers version exposing ``ALL_EXPERTS_FUNCTIONS`` / ``set_experts_implementation``.
MIN_TRANSFORMERS_VERSION = "5.0.0"

#: Fraction of the calibration tokens reaching an experts module below which an expert is
#: quantized by the RTN fallback instead of GPTQ. Matches GPTQModel's ``"0.5%"`` default.
#: Measures routing skew (is this expert under-served relative to its peers?) -- scale
#: invariant, so it does not by itself guarantee an expert's Hessian is well-formed. An
#: expert falls back if it fails this OR ``DEFAULT_MOE_FALLBACK_MIN_K_MULTIPLE``.
DEFAULT_MOE_FALLBACK_THRESHOLD = 0.005

#: Minimum number of calibration tokens an expert must have seen, expressed as a multiple of
#: K (the expert weight's last dimension), below which an expert is quantized by the RTN
#: fallback instead of GPTQ. An expert's Hessian is a (K, K) matrix accumulated from its
#: routed tokens, so rank(H) <= num_tokens_seen: below K tokens the Hessian is *necessarily*
#: rank-deficient (N < K is necessary but not sufficient for a well-conditioned Hessian --
#: damping does not make GPTQ's correction reduce to exactly RTN, it reweights the
#: rank-deficient directions rather than eliminating their influence, so a naive "N=K is the
#: floor" framing overstates what's guaranteed). Empirically, GPTQ has been measured to
#: underperform plain RTN somewhere in the 1x-2x K range and only reliably beat it above
#: roughly 2x K; the default below is set conservatively past that empirical crossover rather
#: than at the bare N>=K rank floor. Measures statistical sufficiency (absolute: more
#: calibration data genuinely helps), unlike DEFAULT_MOE_FALLBACK_THRESHOLD's routing-skew
#: measure. An expert falls back if it fails EITHER condition.
DEFAULT_MOE_FALLBACK_MIN_K_MULTIPLE = 2.0

#: Peak per-layer Hessian working set (bytes) above which :func:`check_moe_gptq_support`
#: warns about a likely out-of-memory during calibration. One float32 ``(K, K)`` Hessian is
#: allocated per (expert, parameter), so a layer needs
#: ``num_experts * (hidden_size**2 + intermediate_size**2) * 4`` bytes on the calibration
#: device, on top of the layer's own weights and activations. 4 GiB is chosen as the
#: threshold because it is small enough to fire well before any mainstream accelerator
#: (16-80 GB) actually OOMs -- note this means even Mixtral-8x7B (~6.6 GB/layer at
#: hidden_size=4096, intermediate_size=14336, num_local_experts=8) is expected to trip this
#: warning; it is not reserved for exceptionally large configs like DeepSeek-V3 (~57 GB/layer).
MOE_HESSIAN_MEMORY_WARN_BYTES = 4 * 1024**3


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
    """Run the reference per-expert experts forward, recording per-expert calibration inputs.

    Registered in transformers' ``ALL_EXPERTS_FUNCTIONS`` registry, so it replaces the
    experts forward of *every* decorated experts module while the calibration
    implementation is active. It reproduces the canonical eager loop shared by every
    supported architecture (``F.linear`` on ``W[e]`` of shape ``(out, in)``, gated
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
        # pylint: disable=not-callable
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    if recorder is not None:
        recorder.note_tokens(hidden_states.shape[0])

    for hit in expert_hit:
        expert_idx = hit[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        intermediate = self._apply_gate(  # pylint: disable=protected-access
            torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx])  # pylint: disable=not-callable
        )
        if recorder is not None:
            recorder.record(int(expert_idx), {"gate_up_proj": current_state, "down_proj": intermediate})
        current_hidden_states = torch.nn.functional.linear(  # pylint: disable=not-callable
            intermediate, self.down_proj[expert_idx]
        )
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def _register_calib_implementation() -> None:
    """Register :func:`olive_moe_calib_experts_forward` in transformers' experts registry.

    The registry is a process-global singleton, so the key alone is not proof that *our*
    function is what will run. Verify identity: if something else already owns the key,
    calibration would silently execute a foreign forward (collecting wrong or zero
    Hessians), so fail closed instead.
    """
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    if OLIVE_MOE_CALIB_IMPLEMENTATION not in ALL_EXPERTS_FUNCTIONS:
        ALL_EXPERTS_FUNCTIONS.register(OLIVE_MOE_CALIB_IMPLEMENTATION, olive_moe_calib_experts_forward)
        return

    registered = ALL_EXPERTS_FUNCTIONS[OLIVE_MOE_CALIB_IMPLEMENTATION]
    if registered is not olive_moe_calib_experts_forward:
        raise MoeCalibrationError(
            f"transformers' experts registry already maps '{OLIVE_MOE_CALIB_IMPLEMENTATION}' to "
            f"{registered!r}, which is not Olive's recording forward "
            f"({olive_moe_calib_experts_forward!r}). Calibration would silently run the wrong "
            "experts forward, so it is refused. Remove the conflicting registration (or restart "
            "the process), or re-run with moe=False."
        )


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

    def note_tokens(self, num_tokens: int) -> None:
        self.tokens_seen += int(num_tokens)

    @torch.no_grad()
    def record(self, expert_idx: int, inputs: dict[str, torch.Tensor]) -> None:
        """Accumulate one expert's activation slice into that expert's Hessian."""
        for pname in self.pnames:
            inp = inputs.get(pname)
            if inp is None:
                continue
            self._accumulate(pname, expert_idx, inp)

    def token_counts(self) -> list[int]:
        """Return per-expert routed-token counts, derived from the recorded sample counts.

        Every recorded parameter sees exactly one activation row per (token, expert)
        routing decision, so the Hessian sample count ``N`` *is* the routed-token count.
        Deriving the counts here (rather than incrementing a counter keyed on a hardcoded
        parameter name) keeps coverage reporting correct when e.g. ``modules_to_not_convert``
        leaves only ``down_proj`` quantized.
        """
        counts = [0] * self.num_experts
        for expert_hessians in self.hessians.values():
            for expert_idx, entry in expert_hessians.items():
                if expert_idx < self.num_experts:
                    counts[expert_idx] = max(counts[expert_idx], int(entry["N"]))
        return counts

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
        token_counts = self.token_counts()
        for pname in self.pnames:
            param = self.experts._parameters[pname]  # pylint: disable=protected-access
            param.quant_info.data = {
                "moe": True,
                "experts": self.hessians[pname],
                "tokens_seen": self.tokens_seen,
                "token_counts": list(token_counts),
            }


# ---------------------------------------------------------------------------
# coverage report
# ---------------------------------------------------------------------------


@dataclass
class LayerCoverage:
    """Per-layer routing coverage collected during calibration.

    An expert counts as "starved" if its observed token count is below EITHER threshold --
    the routing-skew threshold (a fraction of this layer's calibration tokens) or the
    statistical-sufficiency threshold (a multiple of K) -- matching the dual-condition
    fallback gate in ``Gptq._process_moe_param``.
    """

    layer_name: str
    num_experts: int
    tokens_seen: int
    token_counts: list[int]
    skew_threshold: float
    k_threshold: float

    @property
    def threshold(self) -> float:
        """The effective (combined) starved-cutoff: an expert starves below EITHER threshold."""
        return max(self.skew_threshold, self.k_threshold)

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
            f"{self.starved} starved (< {self.threshold:.1f} tokens = "
            f"max(skew {self.skew_threshold:.1f}, sufficiency {self.k_threshold:.1f})), "
            f"{self.unseen} unseen; tokens/expert min={min(counts, default=0)} median={median} "
            f"max={max(counts, default=0)} (calibration tokens reaching the layer: {self.tokens_seen})"
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
        import transformers.integrations.moe as transformers_moe
    except ImportError:
        return False
    return hasattr(transformers_moe, "ALL_EXPERTS_FUNCTIONS")


def check_moe_gptq_support(model_type: str, experts_modules: list[torch.nn.Module]) -> None:
    """Fail closed unless calibrated MoE quantization can safely record this model.

    GPTQ calibration intercepts the experts forward through transformers' experts-
    implementation registry, so that registry must be available. GPTQ groups weights and
    constructs Hessians along the last dimension, so fused experts must report a K-last
    layout through the shared ``is_transposed`` metadata check. Finally, the recording
    forward implements only bias-free experts with a gated activation; modules that declare
    biases or a non-gated activation are refused rather than calibrated with the wrong
    computation.

    Layout support is independent of ``model_type``: transformers-owned
    ``is_transposed=False`` metadata is sufficient for any fused-experts architecture.

    Also logs a warning when the estimated per-layer Hessian memory is large enough to risk
    an out-of-memory during calibration.

    No weight is touched before this returns, mirroring the fail-closed guards in
    :mod:`olive.common.quant.selection`.

    Raises:
        MoeCalibrationError: If the experts registry is unavailable, the fused-experts
            layout cannot be proven K-last, or an experts module declares bias or a
            non-gated activation.

    """
    if not _transformers_supports_experts_registry():
        raise MoeCalibrationError(
            "Calibrated MoE quantization (moe=True) requires transformers >= "
            f"{MIN_TRANSFORMERS_VERSION}, which provides the experts-implementation registry "
            "(transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS) used to collect per-expert "
            "activations. Upgrade transformers, or re-run with moe=False."
        )

    try:
        check_moe_layout_support(
            experts_modules,
            model_type=model_type,
            operation="GPTQ MoE calibration",
        )
    except MoeSupportError as exc:
        raise MoeCalibrationError(str(exc)) from exc

    for experts in experts_modules:
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

    _warn_on_hessian_memory(experts_modules)


def _estimate_layer_hessian_bytes(experts: torch.nn.Module) -> int | None:
    """Estimate the peak per-layer Hessian working set in bytes, or ``None`` if unknown.

    One float32 ``(K, K)`` Hessian is held per (expert, quantized parameter), where ``K`` is
    that parameter's input dim: ``hidden_size`` for ``gate_up_proj`` and
    ``moe_intermediate_size`` for ``down_proj``.
    """
    total_cols_squared = 0
    for pname in ("gate_up_proj", "down_proj"):
        param = getattr(experts, pname, None)
        if param is None or not hasattr(param, "shape") or len(param.shape) != 3:
            return None
        total_cols_squared += int(param.shape[-1]) ** 2
    num_experts = getattr(experts, "num_experts", None)
    if num_experts is None:
        return None
    return int(num_experts) * total_cols_squared * 4  # float32


def _warn_on_hessian_memory(experts_modules: list[torch.nn.Module]) -> None:
    """Warn up front when per-layer Hessian memory is likely to exhaust the device.

    Hessians are allocated per expert on the calibration device and only freed once the
    layer is quantized, so the peak is a whole layer's worth. For DeepSeek-V3-class configs
    (256 experts, hidden 7168, moe_intermediate 2048) this is ~57 GB -- an OOM that would
    otherwise surface as a raw CUDA error minutes into calibration with no explanation.
    """
    estimates = [(experts, nbytes) for experts in experts_modules if (nbytes := _estimate_layer_hessian_bytes(experts))]
    if not estimates:
        return
    experts, peak = max(estimates, key=lambda item: item[1])
    if peak <= MOE_HESSIAN_MEMORY_WARN_BYTES:
        return
    logger.warning(
        "Calibrated MoE quantization will allocate up to %.1f GiB of float32 Hessians for a "
        "single '%s' layer (%d experts x (%d^2 + %d^2) x 4 bytes), held on the calibration "
        "device on top of the layer's weights and activations. This exceeds the %.1f GiB "
        "warning threshold and may run out of memory. Consider calibrating on CPU "
        "(device='cpu'), reducing the number of quantized MoE parameters via "
        "modules_to_not_convert, or re-running with moe=False.",
        peak / 1024**3,
        type(experts).__name__,
        int(experts.num_experts),
        int(experts.gate_up_proj.shape[-1]),
        int(experts.down_proj.shape[-1]),
        MOE_HESSIAN_MEMORY_WARN_BYTES / 1024**3,
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

    def __init__(
        self,
        model: torch.nn.Module,
        fallback_threshold: float = DEFAULT_MOE_FALLBACK_THRESHOLD,
        fallback_min_k_multiple: float = DEFAULT_MOE_FALLBACK_MIN_K_MULTIPLE,
    ):
        self.model = model
        self.fallback_threshold = fallback_threshold
        self.fallback_min_k_multiple = fallback_min_k_multiple
        self.report = CoverageReport()
        self.experts_modules: list[torch.nn.Module] = []
        self._saved_implementation = None
        self._active = False

    @classmethod
    def create(
        cls,
        wrapper: ModelWrapper,
        fallback_threshold: float = DEFAULT_MOE_FALLBACK_THRESHOLD,
        fallback_min_k_multiple: float = DEFAULT_MOE_FALLBACK_MIN_K_MULTIPLE,
    ) -> MoeCalibrationSession | None:
        """Validate support and build a session, or return ``None`` when the model has no experts."""
        experts_modules = [
            experts for lw in wrapper.get_layer_wrappers() if (experts := lw.get_experts(return_name=False)) is not None
        ]
        if not experts_modules:
            return None

        check_moe_gptq_support(wrapper.model_type, experts_modules)
        session = cls(
            wrapper.model, fallback_threshold=fallback_threshold, fallback_min_k_multiple=fallback_min_k_multiple
        )
        session.experts_modules = experts_modules
        return session

    def start(self) -> None:
        """Swap the model's experts implementation to the recording one.

        Re-entrancy is refused: a second ``start()`` would overwrite
        ``_saved_implementation`` with ``"olive_moe_calib"``, so the eventual ``finish()``
        would "restore" the calibration implementation instead of the model's original one.
        Any failure after the swap restores the model before propagating, so a caller that
        aborts here never leaves the model in calibration mode.
        """
        if self._active:
            raise MoeCalibrationError(
                "MoE calibration is already active for this model; MoeCalibrationSession.start() "
                "cannot be nested or called twice (the original experts implementation would be "
                "lost). Call finish() before starting a new session."
            )
        if not hasattr(self.model, "set_experts_implementation"):
            raise MoeCalibrationError(
                "Calibrated MoE quantization (moe=True) requires transformers >= "
                f"{MIN_TRANSFORMERS_VERSION}: the loaded model has no "
                "'set_experts_implementation'. Upgrade transformers, or re-run with moe=False."
            )
        _register_calib_implementation()
        saved_implementation = self.model.get_experts_implementation()
        # transformers returns a dict ({"": impl, <sub_config>: impl}); older/simpler models
        # may return a plain string.
        saved_values = (
            list(saved_implementation.values()) if isinstance(saved_implementation, dict) else [saved_implementation]
        )
        if OLIVE_MOE_CALIB_IMPLEMENTATION in saved_values:
            raise MoeCalibrationError(
                f"The model is already using the '{OLIVE_MOE_CALIB_IMPLEMENTATION}' experts "
                "implementation, which means another MoE calibration session is active on it. "
                "Concurrent/nested calibration sessions are not supported."
            )

        try:
            self.model.set_experts_implementation(OLIVE_MOE_CALIB_IMPLEMENTATION)
        except Exception:
            # transformers can mutate some submodels' configs before raising while
            # configuring a later one (e.g. a heterogeneous/composite model); attempt a
            # best-effort restore of whatever was already swapped, but always propagate the
            # original error so the caller sees why start() actually failed.
            try:
                self.model.set_experts_implementation(saved_implementation)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to restore the original experts implementation after a MoE "
                    "calibration start() error; the model's experts implementation may be left "
                    "in an inconsistent state.",
                    exc_info=True,
                )
            raise
        try:
            # ``set_experts_implementation`` silently no-ops when transformers' source-inspection
            # heuristic decides the class isn't switchable. Verify the swap actually reached every
            # experts module rather than silently collecting zero Hessians.
            stale = sorted(
                {
                    type(experts).__name__
                    for experts in self.experts_modules
                    if getattr(experts.config, "_experts_implementation", None) != OLIVE_MOE_CALIB_IMPLEMENTATION
                }
            )
        except Exception:
            # Best-effort restore: transformers' own setter can itself mutate some submodels'
            # configs before raising on a later one, so even the "verify the swap landed" step
            # (not just the setter call above) can observe a partially-swapped model. Attempt to
            # restore regardless, but propagate the original error either way -- a failed restore
            # attempt here should not mask why ``start()`` failed in the first place.
            try:
                self.model.set_experts_implementation(saved_implementation)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to restore the original experts implementation after a MoE "
                    "calibration start() error; the model's experts implementation may be left "
                    "in an inconsistent state.",
                    exc_info=True,
                )
            raise

        if stale:
            self.model.set_experts_implementation(saved_implementation)
            raise MoeCalibrationError(
                "Olive could not switch the experts implementation to "
                f"'{OLIVE_MOE_CALIB_IMPLEMENTATION}' for {stale}; per-expert "
                "calibration data cannot be collected. Re-run with moe=False."
            )

        self._saved_implementation = saved_implementation
        self._active = True
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
        """Log (and remember) the routing coverage recorded for one experts module.

        NOTE: the sufficiency threshold (``k_threshold``) is derived from ``pnames[0]``'s
        last-dim size only (typically ``gate_up_proj``). Parameters on the same layer with a
        different last-dim size (e.g. ``down_proj``, whose K is the intermediate size rather
        than the hidden size) are NOT separately represented in this report -- the actual
        per-expert fallback gate in ``Gptq._process_moe_param`` does use each parameter's own
        K, so this coverage summary may under- or over-count "starved" experts relative to
        what actually happened for parameters other than ``pnames[0]``.
        """
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
        k = experts._parameters[pnames[0]].shape[-1]  # pylint: disable=protected-access
        self.report.add(
            LayerCoverage(
                layer_name=layer_name,
                num_experts=len(data["token_counts"]),
                tokens_seen=data["tokens_seen"],
                token_counts=data["token_counts"],
                skew_threshold=self.fallback_threshold * data["tokens_seen"],
                k_threshold=self.fallback_min_k_multiple * k,
            )
        )

    def finish(self) -> None:
        """Restore the original experts implementation and log the coverage summary."""
        try:
            if self._active:
                self.model.set_experts_implementation(self._saved_implementation)
        finally:
            # Clear session state and log whatever coverage was recorded regardless of
            # whether the restore above succeeded -- a failed restore should not also
            # suppress the coverage summary or leave ``_active``/``_saved_implementation``
            # in a way that could make a later ``start()`` behave inconsistently.
            self._saved_implementation = None
            self._active = False
            self.report.log_summary()
