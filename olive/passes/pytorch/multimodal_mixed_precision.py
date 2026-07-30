# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Component/modality-aware mixed-precision planning for multimodal models."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.utils import StrEnumBase
from olive.constants import PrecisionBits
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler

logger = logging.getLogger(__name__)


class Component(StrEnumBase):
    """Coarse functional component of a (multimodal) model."""

    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"  # the language decoder body
    PROJECTOR = "projector"  # modality connector / merger between encoder and decoder
    LM_HEAD = "lm_head"
    EMBEDS = "embeds"  # text input embedding table
    OTHER = "other"


# Name-substring heuristics, evaluated in order (first match wins). Each entry maps a set
# of lowercase substrings (any-of) to a component. Ordering matters: more specific /
# higher-priority modality markers (projector, embeds) are checked before the broad
# encoder/decoder roots so e.g. ``visual.merger`` -> PROJECTOR, not VISION.
_DEFAULT_RULES: tuple[tuple[tuple[str, ...], Component], ...] = (
    # modality connectors / mergers (often nested under the vision tower)
    (
        ("multi_modal_projector", "mm_projector", "multimodal_projector", "connector", ".merger", "resampler"),
        Component.PROJECTOR,
    ),
    # vision encoders across common VLMs
    (("vision_tower", "vision_model", "visual", "image_encoder", "vit", "embed_vision"), Component.VISION),
    # audio encoders
    (("audio_tower", "audio_model", "audio_encoder", "speech_encoder", "embed_audio"), Component.AUDIO),
    # language head / text input-embedding fallbacks (used when exact names aren't provided)
    (("lm_head",), Component.LM_HEAD),
    (("embed_tokens", "word_embeddings", "wte"), Component.EMBEDS),
)


def classify_component(
    name: str,
    *,
    lm_head_name: str | None = None,
    embeds_name: str | None = None,
    embeds_names: set[str] | None = None,
    extra_rules: list[tuple[list[str], str]] | None = None,
) -> Component:
    """Classify a module (by its dotted name) into a coarse functional component.

    Classification is name-based so it works uniformly for any HF multimodal model
    without materializing weights. ``lm_head_name`` / ``embeds_names`` (resolved from
    :class:`ModelWrapper`) take precedence so the language head/tables are tagged
    exactly, independent of naming conventions. ``embeds_name`` remains available for
    callers with a single embedding module. ``extra_rules`` lets a recipe extend or
    override the built-in heuristics (checked before the defaults).

    Args:
        name: Dotted module name (as produced by ``model.named_modules()``).
        lm_head_name: Exact name of the language-model head, if known.
        embeds_name: Exact name of the text input-embedding module, if known.
        embeds_names: Exact names of all embedding-component modules, if known.
        extra_rules: Optional list of ``([substrings], component)`` rules, checked first.

    Returns:
        The :class:`Component` the module belongs to.

    """
    if lm_head_name is not None and (name == lm_head_name):
        return Component.LM_HEAD
    if name == embeds_name or name in (embeds_names or set()):
        return Component.EMBEDS

    lowered = name.lower()
    for substrings, component in extra_rules or []:
        if any(s.lower() in lowered for s in substrings):
            return Component(component)
    for substrings, component in _DEFAULT_RULES:
        if any(s in lowered for s in substrings):
            return component

    # Anything not matched by a modality/connector rule and not the head/embeds is
    # treated as the text decoder body (language_model / model.layers / mlp / attn).
    return Component.TEXT


class MultiModalMixedPrecision(Pass):
    """Assign per-component precision for multimodal models.

    Multimodal quantization literature (e.g. MBQ arXiv:2412.19509, Q-VLM
    arXiv:2410.08119, MQuant arXiv:2502.00425) repeatedly finds that vision/audio
    encoders are more quantization-sensitive than the language decoder, so a common
    recipe is: keep the encoders in full precision (or higher bit-width) while
    quantizing the decoder aggressively.

    This pass classifies every module into a coarse component
    (``vision``/``audio``/``text``/``projector``/``lm_head``/``embeds``) and, from a
    per-component precision map, produces a ``mixed_precision_info`` model attribute
    consumed by the downstream weight-quant passes (Rtn/Gptq/Kquant via
    ``prepare_model``):

    * A component set to 16/32 bits is **excluded** from quantization (kept full
      precision) via ``mixed_precision_info["exclude"]``.
    * A component set to 2/4/8 bits that differs from the default is emitted as
      per-module ``overrides``.
    * The default precision (applied to any component not named in the map) is the
      pass's ``bits``/``group_size``/``sym``.

    It does not quantize anything itself; run it before a weight-quant pass. This
    mirrors :class:`SelectiveMixedPrecision` (which is layer-sensitivity based)
    but selects along the modality/component axis instead.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "component_precision": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Mapping of component name to precision bits, e.g. "
                    '{"vision": 16, "audio": 16, "text": 4}. Components are: '
                    "'vision', 'audio', 'text', 'projector', 'lm_head', 'embeds'. "
                    "Bits of 16 or 32 keep the component in full precision (excluded from "
                    "quantization); bits in {2,4,8} quantize the component. Components not "
                    "listed use the pass 'bits' default."
                ),
            ),
            "bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS4,
                description="Default quantization bits for components not named in component_precision.",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Default group size for quantized components.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to use symmetric quantization for quantized components.",
            ),
            "component_map": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Optional extra classification rules to extend/override the built-in "
                    "heuristics, as a mapping of component name to a list of name substrings, "
                    'e.g. {"vision": ["my_custom_tower"]}. Checked before the built-in rules.'
                ),
            ),
        }

    @staticmethod
    def _build_plan(
        modules: list[tuple[str, str]],
        component_precision: dict[str, int],
        default_bits: int,
        *,
        lm_head_name: str | None = None,
        embeds_name: str | None = None,
        embeds_names: set[str] | None = None,
        extra_rules: list[tuple[list[str], str]] | None = None,
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        """Turn a list of quantizable modules into an exclude list + overrides.

        Pure (no model/torch state) so it is straightforward to unit test.

        Args:
            modules: list of ``(module_name, kind)`` where kind is unused today but kept
                for forward compatibility (e.g. distinguishing linear vs embedding).
            component_precision: component -> bits map (16/32 => exclude, {2,4,8} => quantize).
            default_bits: bits applied to components not named in ``component_precision``.
            lm_head_name: exact lm-head module name, if known.
            embeds_name: exact text-embedding module name, if known.
            embeds_names: exact names of all embedding-component modules, if known.
            extra_rules: optional classification rules checked before the built-ins.

        Returns:
            ``(exclude, overrides, component_counts)``.

        """
        exclude: list[str] = []
        overrides: dict[str, dict] = {}
        component_counts: dict[str, int] = {}

        for name, _kind in modules:
            component = classify_component(
                name,
                lm_head_name=lm_head_name,
                embeds_name=embeds_name,
                embeds_names=embeds_names,
                extra_rules=extra_rules,
            )
            component_counts[str(component)] = component_counts.get(str(component), 0) + 1

            bits = component_precision.get(str(component))
            if bits is None:
                continue
            if bits in (16, 32):
                exclude.append(name)
            elif bits in (2, 4, 8):
                if bits != default_bits:
                    overrides[name] = {"bits": bits}
            else:
                raise ValueError(f"Unsupported bits {bits} for component; expected one of 2,4,8,16,32.")

        return exclude, overrides, component_counts

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        component_precision = {str(k).lower(): int(v) for k, v in (config.component_precision or {}).items()}
        valid_components = {str(c) for c in Component if c != Component.OTHER}
        for comp in component_precision:
            if comp not in valid_components:
                raise ValueError(
                    f"Unknown component {comp!r} in component_precision. Valid components: {sorted(valid_components)}."
                )
        component_map = config.component_map or {}
        for component, substrings in component_map.items():
            normalized_component = str(component).lower()
            if normalized_component not in valid_components:
                raise ValueError(
                    f"Unknown component {component!r} in component_map. Valid components: {sorted(valid_components)}."
                )
            if (
                not isinstance(substrings, list)
                or not substrings
                or not all(isinstance(substring, str) and substring.strip() for substring in substrings)
            ):
                raise ValueError(f"component_map[{component!r}] must be a non-empty list of non-empty name substrings.")
        extra_rules = [(substrings, str(component).lower()) for component, substrings in component_map.items()]

        meta_model = self._load_meta_model(model)
        # Best-effort exact resolution of the language head / text-embedding names via
        # ModelWrapper (decoder-centric). If it can't parse this architecture, fall back to
        # the classifier's name-substring rules for lm_head/embeds.
        lm_head_name = None
        embeds_names: set[str] = set()
        try:
            wrapper = ModelWrapper.from_model(meta_model)
            try:
                lm_head_name = wrapper.get_lm_head()[1]
            except Exception:
                lm_head_name = None
            try:
                embeds_names = set(wrapper.get_embeds()[1])
            except Exception:
                embeds_names = set()
        except Exception as e:
            logger.debug("ModelWrapper could not parse model for exact head/embeds names (%s).", e)

        default_bits = config.bits.value if hasattr(config.bits, "value") else int(config.bits)

        modules = [
            (name, "linear" if isinstance(module, torch.nn.Linear) else "embedding")
            for name, module in meta_model.named_modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
        ]
        exclude, overrides, component_counts = self._build_plan(
            modules,
            component_precision,
            default_bits,
            lm_head_name=lm_head_name,
            embeds_names=embeds_names,
            extra_rules=extra_rules,
        )

        logger.info(
            "MultiModalMixedPrecision: quantizable modules per component: %s",
            {k: component_counts[k] for k in sorted(component_counts)},
        )
        logger.info(
            "MultiModalMixedPrecision: default_bits=%d, excluded_modules=%d, overridden_modules=%d",
            default_bits,
            len(exclude),
            len(overrides),
        )
        if component_precision and not exclude and not overrides:
            logger.warning(
                "MultiModalMixedPrecision: component_precision produced no exclusions or overrides. "
                "Check that the component names match the model (see per-component counts above); "
                "you may need 'component_map' to classify custom module names."
            )

        output_model = deepcopy(model)
        output_model.model_attributes = output_model.model_attributes or {}
        output_model.model_attributes["mixed_precision_info"] = {
            "default": {
                "bits": default_bits,
                "group_size": config.group_size,
                "symmetric": config.sym,
            },
            "overrides": overrides,
            "exclude": sorted(exclude),
        }
        return output_model

    @staticmethod
    def _load_meta_model(model: HfModelHandler):
        """Instantiate the model architecture on the meta device (no weights).

        Only module names/types are needed for classification, so weights are never
        materialized. Instantiates the exact architecture named in the config (so the
        vision/audio towers and lm_head are present), falling back to ``AutoModel`` and
        finally to a real (heavier) load.
        """
        import transformers
        from accelerate import init_empty_weights
        from transformers import AutoModel

        hf_config = model.get_hf_model_config()
        trust_remote_code = model.load_kwargs is not None and bool(
            getattr(model.load_kwargs, "trust_remote_code", False)
        )

        architectures = list(getattr(hf_config, "architectures", None) or [])
        for arch in architectures:
            arch_cls = getattr(transformers, arch, None)
            if arch_cls is None:
                continue
            try:
                with init_empty_weights():
                    return arch_cls(hf_config)
            except Exception as e:
                logger.debug("Meta instantiation of %s failed (%s); trying next option.", arch, e)

        try:
            with init_empty_weights():
                return AutoModel.from_config(hf_config, trust_remote_code=trust_remote_code)
        except Exception as e:
            logger.debug("Meta-device instantiation failed (%s); falling back to real load.", e)
            from olive.passes.pytorch.train_utils import load_hf_base_model

            return load_hf_base_model(model)
