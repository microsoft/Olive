# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy

from olive.common.hf.wrapper import ModelWrapper
from olive.common.utils import StrEnumBase
from olive.constants import PrecisionBits
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class SelectiveMixedPrecision(Pass):
    """Annotate the model with mixed precision information.

    This pass is used to annotate the model with mixed precision information, which can be used by other passes
    to quantize the model. The pass will add a model attribute `mixed_precision_info` that contains
    information about the default precision along with overrides for specific layers.
    """

    class Algorithm(StrEnumBase):
        """The algorithm to use for mixed precision."""

        K_QUANT_MIXED = "k_quant_mixed"
        K_QUANT_LAST = "k_quant_last"
        # TODO(jambayk): add other heuristic/algorithms

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "algorithm": PassConfigParam(
                type_=SelectiveMixedPrecision.Algorithm,
                required=True,
                description="The algorithm to use for mixed precision.",
            )
        }

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        """Run the selective mixed precision pass."""
        if not isinstance(model, HfModelHandler):
            raise ValueError("SelectiveMixedPrecision pass currently only supports HfModelHandler.")

        model_wrapper = ModelWrapper.from_model(model.load_model(cache_model=False))

        # TODO(jambayk): make this configurable
        override_config = {"bits": PrecisionBits.BITS8}
        overrides = {model_wrapper.get_lm_head()[1]: override_config}

        if config.algorithm == SelectiveMixedPrecision.Algorithm.K_QUANT_MIXED:
            layer_prefix = model_wrapper.get_layers()[1]
            num_layers = model_wrapper.num_hidden_layers
            for layer_idx, layer_wrapper in enumerate(model_wrapper.get_layer_wrappers()):
                if not (
                    layer_idx < num_layers / 8
                    or layer_idx >= 7 * num_layers / 8
                    or ((layer_idx - num_layers // 8) % 3 == 2)
                ):
                    continue

                # Add qkv
                for attn_input_name in layer_wrapper.get_attention_inputs(return_name=True)[1]:
                    overrides[f"{layer_prefix}.{layer_idx}.{attn_input_name}"] = override_config

                # Add down_proj
                for attn_output_name in layer_wrapper.get_mlp_outputs(return_name=True)[1]:
                    overrides[f"{layer_prefix}.{layer_idx}.{attn_output_name}"] = override_config

        # Create output model with mixed precision info as model attribute
        output_model = deepcopy(model)
        output_model.model_attributes = output_model.model_attributes or {}
        output_model.model_attributes["mixed_precision_info"] = {
            "default": {"bits": PrecisionBits.BITS4},
            "overrides": overrides,
        }
        return output_model
