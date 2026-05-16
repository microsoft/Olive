# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from olive.common.quant.patterns import match_override
from olive.common.quant.state_dict import buffer_names, install_quant_tensor_param, refresh_quant_tensor_refs
from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import WeightQuantizer
from olive.common.utils import StrEnumBase

if TYPE_CHECKING:
    from tqdm.auto import tqdm
    from transformers import PreTrainedModel


# older transformers expects a StrEnum and accesses .value
class OliveHfQuantizationMethod(StrEnumBase):
    """Enumeration for Olive quantization methods."""

    # Olive quantization method
    OLIVE = "olive"


# TODO(jambayk): standardize this with the default quantization settings
@dataclass
class OliveHfQuantizationOverrideConfig:
    """Configuration overrides for individual modules during Olive quantization.

    Attributes:
        bits: Override for bit width (e.g. 4, 8).
        symmetric: Whether to use symmetric quantization for this module.
        group_size: Size of quantization group for this module.

    """

    bits: int | None = None
    symmetric: bool | None = None
    group_size: int | None = None


@dataclass
class OliveHfQuantizationConfig(QuantizationConfigMixin):
    """Configuration for Olive quantization.

    Extends Hugging Face's QuantizationConfigMixin with Olive-specific settings.

    Attributes:
        bits: Default bit width for quantization (e.g. 4, 8).
        symmetric: Whether to use symmetric quantization.
        group_size: Quantization group size.
            -1 = per-channel, >0 = groupwise.
        lm_head: Whether to quantize the language model head.
        embeds: Whether to quantize the input embeddings.
        moe: Whether to quantize MoE expert modules / parameters.
            When ``False`` (default), every ``nn.Module`` under each
            experts subtree returned by ``LayerWrapper.get_experts()``
            is added to the skip set — this both leaves fused-3D
            experts alone *and* fixes the previous silent quantization
            of per-expert ``nn.Linear``s in ``ModuleList(Expert)``
            blocks (Mixtral, PhiMoE, Qwen2/3-MoE).
        modules_to_not_convert: List of module name patterns to exclude
            from quantization. Plain strings use **substring** matching
            (preserving HF semantics); entries prefixed with ``re:`` use
            ``re.fullmatch``.
        overrides: Per-module overrides for quantization parameters.
            Keys use **literal equality** matching by default; entries
            prefixed with ``re:`` use ``re.fullmatch``. Among matching
            keys, the longest pattern wins (ties broken lexically).

    """

    # pylint: disable
    def __init__(
        self,
        bits: int,
        symmetric: bool,
        group_size: int,
        lm_head: bool = False,
        embeds: bool = False,
        moe: bool = False,
        modules_to_not_convert: list | None = None,
        overrides: dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # pylint: disable=W0231
        self.quant_method = OliveHfQuantizationMethod.OLIVE

        self.bits = bits
        self.symmetric = symmetric
        self.group_size = group_size
        self.lm_head = lm_head
        self.embeds = embeds
        self.moe = moe
        self.modules_to_not_convert = modules_to_not_convert
        self.overrides = {
            module_name: OliveHfQuantizationOverrideConfig(**override)
            for module_name, override in (overrides or {}).items()
        }
        self.tie_word_embeddings = tie_word_embeddings
        self.post_init()

    def post_init(self):
        """Safety checker that arguments are correct."""
        if self.bits not in [2, 4, 8]:
            raise ValueError(f"Only 2-bit, 4-bit and 8-bit quantization supported, got {self.bits}")

    def to_dict(self) -> dict:
        """Serialize this instance to a Python dictionary."""
        output = super().to_dict()
        if self.overrides:
            overrides = {}
            for module_name, override in self.overrides.items():
                # remove None or default values from the override
                cleaned_override = {k: v for k, v in override.__dict__.items() if v is not None and v != output[k]}
                if cleaned_override:
                    overrides[module_name] = cleaned_override
            output["overrides"] = sort_layers_by_name(overrides) or None
        else:
            output["overrides"] = None
        return output

    def get_qlinear_init_args(self, module_name: str) -> dict:
        """Get the initialization arguments for a QuantLinear layer based on the module name.

        Args:
            module_name (str): The name of the module to get initialization args for.

        Returns:
            dict: Initialization arguments for QuantLinear.

        """
        init_args = {
            "bits": self.bits,
            "symmetric": self.symmetric,
            "group_size": self.group_size,
        }
        best = match_override(module_name, list(self.overrides.keys())) if self.overrides else None
        if best is not None:
            override = self.overrides[best]
            init_args.update({k: v for k, v in override.__dict__.items() if v is not None})
        return init_args


def sort_layers_by_name(layers_dict) -> dict:
    """Sort layers dictionary by layer names in natural order."""

    def sort_key(name: str) -> tuple:
        return [int(part) if part.isdigit() else part for part in name.split(".")]

    return dict(sorted(layers_dict.items(), key=lambda item: sort_key(item[0])))


class OliveHfQuantizer(HfQuantizer):
    """Olive quantizer.

    Layout (see ``olive/common/quant/state_dict.py``):

    * Each quantized weight is installed as ``nn.Parameter(QuantTensor)``
      on the original host module (``nn.Linear``, ``nn.Embedding``, or an
      experts module that owns a fused-3D parameter).
    * Sibling buffers ``<pname>_qweight`` / ``_scales`` / ``_qzeros``
      alias the QuantTensor's inner tensors. These are the only things
      written to safetensors; HF's loader fills them via normal dotted
      paths. After load we re-bind the QuantTensor inner refs to point
      at the freshly-loaded buffer storage.
    """

    # only support load and inference, no on-the-fly quantization
    requires_calibration = True

    def _process_model_before_weight_loading(
        self, model: PreTrainedModel, keep_in_fp32_modules: list[str] | None = None, **kwargs
    ):
        from olive.common.quant.selection import iter_quant_targets

        skip_patterns: list[str] = []
        if self.quantization_config.modules_to_not_convert:
            skip_patterns.extend(self.quantization_config.modules_to_not_convert)
        if keep_in_fp32_modules:
            skip_patterns.extend(keep_in_fp32_modules)

        for target in iter_quant_targets(
            model,
            quantize_lm_head=self.quantization_config.lm_head,
            quantize_embeds=self.quantization_config.embeds,
            quantize_moe=self.quantization_config.moe,
            skip_patterns=skip_patterns,
        ):
            qargs = self.quantization_config.get_qlinear_init_args(target.full_name)
            param = target.param
            qt = _build_placeholder_quant_tensor(
                shape=tuple(param.shape),
                bits=qargs["bits"],
                symmetric=qargs["symmetric"],
                group_size=qargs["group_size"],
                dtype=param.dtype,
                device=param.device,
            )
            install_quant_tensor_param(target.module, target.pname, qt)

        if self.quantization_config.tie_word_embeddings:
            # doing first time so that the weight load doesn't complain about missing weights
            tie_quant_word_embeddings(model)

    def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        # HF's loader assigns freshly-loaded buffer tensors in place, so
        # re-bind every QuantTensor parameter to point at the current
        # ``<pname>_qweight`` / ``_scales`` / ``_qzeros`` buffer storages.
        refresh_quant_tensor_refs(model)
        if self.quantization_config.tie_word_embeddings:
            tie_quant_word_embeddings(model)
        return model

    def is_serializable(self, safe_serialization=None) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False


def _build_placeholder_quant_tensor(
    *,
    shape: tuple[int, ...],
    bits: int,
    symmetric: bool,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> QuantTensor:
    """Build a zero-filled ``QuantTensor`` with the correct buffer shapes.

    This is used as a placeholder before ``from_pretrained`` fills in the
    real values; the buffer shapes must match what was written at save
    time so HF's loader assigns into them without complaint.
    """
    quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)
    packing_factor = 8 // bits
    qparam_shape = quantizer.get_qparam_shape(shape)
    qweight_shape = (*shape[:-1], math.ceil(shape[-1] / packing_factor))

    qweight = torch.zeros(qweight_shape, dtype=torch.uint8, device=device)
    scales = torch.zeros(qparam_shape, dtype=dtype, device=device)
    qzeros: torch.Tensor | None
    if symmetric:
        qzeros = None
    else:
        qz_shape = (*qparam_shape[:-1], math.ceil(qparam_shape[-1] / packing_factor))
        qzeros = torch.zeros(qz_shape, dtype=torch.uint8, device=device)

    return QuantTensor.from_packed(
        qweight=qweight,
        scales=scales,
        qzeros=qzeros,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        shape=shape,
        dtype=dtype,
    )


def replace_matching_submodules(
    module: nn.Module,
    condition: Callable[[nn.Module, str], bool],
    transform: Callable[[nn.Module], nn.Module],
    path: list[str] | None = None,
    description: str | None = None,
    pbar: tqdm | None = None,
):
    """Walk the module tree and replace every sub-module that meets a condition.

    Args:
        module: Root ``nn.Module`` to start from.
        condition: ``condition(m, name) -> bool``. Return ``True`` to trigger
            replacement.
        transform: ``transform(m, name) -> nn.Module``. Produces the replacement.
        path: (Internal) List of name segments used to build the dotted path.
        description: (Internal) Description for the progress bar.
        pbar: (Internal) Progress bar to update. If ``None``, a new
            ``tqdm`` progress bar will be created.

    Returns:
        The root module with all replacements applied. If the root matches,
        the returned object is whatever ``transform`` returns.

    """
    is_root = pbar is None
    if is_root:
        from tqdm.auto import tqdm

        # TODO(jambayk): check logging level and only show progress bar if debug
        pbar = tqdm(total=None, desc=description or "Replacing submodules", unit="mod")
    path = [] if path is None else path

    name = ".".join(path)
    if condition(module, name):
        pbar.set_postfix(module=name, refresh=False)
        pbar.update(1)
        # do we need to recurse into children first? no use case for that yet
        return transform(module, name)

    for name, child in module.named_children():
        new_child = replace_matching_submodules(child, condition, transform, [*path, name], description, pbar)
        module.add_module(name, new_child)

    if is_root:
        pbar.close()

    return module


def tie_quant_word_embeddings(model: PreTrainedModel) -> None:
    """Tie the input and output embeddings when both share a quantized weight.

    Both modules' ``weight`` ``nn.Parameter`` is set to the **same**
    ``nn.Parameter(QuantTensor)`` object, and the underlying
    ``weight_qweight`` / ``weight_scales`` / ``weight_qzeros`` buffers
    are tied (aliased to the input embedding's buffers). This preserves
    the standard HF tied-weights semantics for the quantized layout.

    Tying is a no-op unless **both** the input and output embeddings
    are already backed by compatible ``QuantTensor`` weights with
    matching shape and dtype.
    """
    src = model.get_input_embeddings()
    dst = model.get_output_embeddings()
    if src is None or dst is None:
        return

    src_param = src._parameters.get("weight")
    dst_param = dst._parameters.get("weight")
    if (
        src_param is None
        or dst_param is None
        or not isinstance(src_param.data, QuantTensor)
        or not isinstance(dst_param.data, QuantTensor)
    ):
        return
    if src_param.shape != dst_param.shape or src_param.dtype != dst_param.dtype:
        return

    qname, sname, zname = buffer_names("weight")
    # tie buffers
    for n in (qname, sname, zname):
        src_buf = src._buffers.get(n)
        if src_buf is None:
            continue
        dst._buffers[n] = src_buf

    # tie the QuantTensor parameter itself (same Python Parameter object,
    # so both modules see the same .data and the same inner tensors).
    dst._parameters["weight"] = src_param
