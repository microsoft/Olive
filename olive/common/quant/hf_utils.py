# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch.nn as nn
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from olive.common.utils import StrEnumBase

if TYPE_CHECKING:
    from tqdm.auto import tqdm
    from transformers import PreTrainedModel

    from olive.common.quant.nn import QuantModule


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
        modules_to_not_convert : List of module names to exclude from quantization.
        overrides: Per-module overrides for quantization parameters.

    """

    # pylint: disable
    def __init__(
        self,
        bits: int,
        symmetric: bool,
        group_size: int,
        lm_head: bool = False,
        embeds: bool = False,
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
        if override := self.overrides.get(module_name):
            init_args.update({k: v for k, v in override.__dict__.items() if v is not None})
        return init_args


def sort_layers_by_name(layers_dict) -> dict:
    """Sort layers dictionary by layer names in natural order."""

    def sort_key(name: str) -> tuple:
        return [int(part) if part.isdigit() else part for part in name.split(".")]

    return dict(sorted(layers_dict.items(), key=lambda item: sort_key(item[0])))


class OliveHfQuantizer(HfQuantizer):
    """Olive quantizer."""

    # only support load and inference, no on-the-fly quantization
    requires_calibration = True
    modules_to_not_convert: list[str] | None = None

    def _process_model_before_weight_loading(
        self, model: PreTrainedModel, keep_in_fp32_modules: list[str] | None = None, **kwargs
    ):
        from olive.common.quant.nn import QuantEmbedding, QuantLinear

        ids_to_skip = []
        if not self.quantization_config.lm_head:
            ids_to_skip.append(id(model.get_output_embeddings()))
        if not self.quantization_config.embeds:
            ids_to_skip.append(id(model.get_input_embeddings()))
        self.modules_to_not_convert = (
            [name for name, module in model.named_modules() if id(module) in ids_to_skip] if ids_to_skip else []
        )
        if self.quantization_config.modules_to_not_convert:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)
        if keep_in_fp32_modules:
            self.modules_to_not_convert.extend(keep_in_fp32_modules)

        def should_quantize(module: nn.Module, name: str) -> bool:
            """Check if a module should be quantized."""
            return isinstance(module, (nn.Linear, nn.Embedding)) and not any(
                key in name for key in self.modules_to_not_convert
            )

        def create_quantized_module(module: nn.Linear | nn.Embedding, name: str) -> QuantLinear | QuantEmbedding:
            """Create a quantized version of a module."""
            common_kwargs = {
                **self.quantization_config.get_qlinear_init_args(name),
                "device": module.weight.device,
                "dtype": module.weight.dtype,
            }
            if isinstance(module, nn.Embedding):
                return QuantEmbedding(
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    padding_idx=module.padding_idx,
                    **common_kwargs,
                )
            return QuantLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                **common_kwargs,
            )

        replace_matching_submodules(model, should_quantize, create_quantized_module)

        if self.quantization_config.tie_word_embeddings:
            # doing first time so that the weight load doesn't complain about missing weights
            tie_quant_word_embeddings(model)

    def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        if self.quantization_config.tie_word_embeddings:
            # doing again to ensure buffers are tied after loading weights
            tie_quant_word_embeddings(model)
        return model

    def is_serializable(self, safe_serialization=None) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        # TODO(jambayk): investigate what this means (peft, scale+bias, etc.?)
        # need to support peft, scale+bias comes for free since everything is in torch
        return False


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


def _repoint_buffer(src: nn.Module, dst: nn.Module, name: str):
    """Repoint a buffer from src to dst.

    Args:
        src: Source module.
        dst: Destination module.
        name: Name of the buffer to repoint.

    """
    src_buf = getattr(src, name, None)

    # ensure both are None or both exist
    if src_buf is None:
        assert getattr(dst, name, None) is None, f"Output embedding has {name} but input does not."
        return

    # ensure both have the buffer shapes and types match
    dst_buf = getattr(dst, name, None)
    assert src_buf.shape == dst_buf.shape, (
        f"Cannot tie embeddings: input embedding {name} shape {src_buf.shape} "
        f"does not match output embedding shape {dst_buf.shape}."
    )
    assert src_buf.dtype == dst_buf.dtype, (
        f"Cannot tie embeddings: input embedding {name} dtype {src_buf.dtype} "
        f"does not match output embedding dtype {dst_buf.dtype}."
    )

    # tie the buffers
    # pylint: disable=W0212
    dst._buffers[name] = src_buf
    dst._non_persistent_buffers_set.add(name)


def tie_quant_modules(src: QuantModule, dst: QuantModule):
    """Tie the quantization buffers of two QuantModules.

    Args:
        src: Source QuantModule.
        dst: Destination QuantModule.

    """
    for name in ["qweight", "scales", "qzeros"]:
        _repoint_buffer(src, dst, name)


def tie_quant_word_embeddings(model: PreTrainedModel):
    """Tie the word embeddings and output embeddings if they have the same shape.

    Args:
        model: The HuggingFace model to tie embeddings for.

    """
    tie_quant_modules(model.get_input_embeddings(), model.get_output_embeddings())
