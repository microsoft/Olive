# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict, Union

import torch
from tqdm import tqdm

from olive.common.hf.mappings import MODELS_TO_LAYERS_MAPPING
from olive.common.hf.peft import is_peft_model
from olive.common.utils import get_attr, set_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam, get_user_script_data_config
from olive.passes.pytorch.common import inherit_pytorch_from_hf, inherit_pytorch_from_pytorch

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class RTNQuantizer2(Pass):
    """AWQ quantization."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "zero_point": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to use zero point quantization to calculate the scales and zeros. "
                    "If False, it use the symmetric quantization."
                ),
            ),
            "q_group_size": PassConfigParam(
                type_=int,
                default_value=32,
                description="The group size to use for quantization.",
            ),
            "w_bit": PassConfigParam(
                type_=int,
                default_value=4,
                description="The number of bits to quantize to.",
            ),
            "modules_to_not_convert": PassConfigParam(
                type_=list,
                default_value=[],
                description=(
                    "The list of modules to not quantize, useful for quantizing models that explicitly "
                    "require to have some modules left in their original precision (e.g. Whisper encoder, "
                    "Llava encoder, Mixtral gate layers). Please refer to AutoAWQ documentation for "
                    "quantizing HF models."
                ),
            ),
            "layers_block_name": PassConfigParam(
                type_=str,
                default_value="model.layers",
                description=(
                    "Block name to quantize. Default value is model.layers. For models can't be auto filled, you can"
                    " refer this link to fill these parameters."
                ),
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        from neural_compressor.torch.algorithms.weight_only.rtn import INCWeightOnlyLinear, quant_tensor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pytorch_model = model.load_model()
        if is_peft_model(pytorch_model):
            pytorch_model = pytorch_model.get_base_model()
        pytorch_model.eval()
        model.model = None

        model_type = None
        if model.model_attributes:
            model_type = model.model_attributes.get("model_type")

        layers = get_layers(pytorch_model, model_type, config["layers_block_name"])
        for i in tqdm(range(len(layers)), desc="RTN"):
            # [STEP 1]: Get layer, extract linear modules
            named_linears = get_named_linears(layers[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(named_linears, config["modules_to_not_convert"])

            # [STEP 2]: Quantize the linear layers
            for name, linear_layer in named_linears.items():
                linear_layer = linear_layer.to(device)  # noqa: PLW2901

                int_weight, scale, zp = quant_tensor(
                    linear_layer.weight.data,
                    bits=config["w_bit"],
                    group_size=config["q_group_size"],
                    scheme="asym" if config["zero_point"] else "sym",
                    return_int=True,
                )

                q_linear = INCWeightOnlyLinear(
                    linear_layer.in_features,
                    linear_layer.out_features,
                    dtype="int",
                    bits=config["w_bit"],
                    group_size=config["q_group_size"],
                    zp=zp is not None,
                    bias=linear_layer.bias is not None,
                    use_optimum_format=True,
                )
                q_linear.pack(int_weight, scale, zp, linear_layer.bias)
                q_linear.weight = q_linear.recover().to(device)

                linear_layer.cpu()
                set_attr(layers[i], name, q_linear)

        if hasattr(pytorch_model, "quantize_config"):
            # cannot save gptq model otherwise
            pytorch_model.quantize_config = None

        output_model_path = normalize_path_suffix(output_model_path, "model.pt")
        torch.save(pytorch_model, output_model_path)

        if isinstance(model, HfModelHandler):
            return inherit_pytorch_from_hf(model, output_model_path)

        return inherit_pytorch_from_pytorch(model, output_model_path)


def get_layers(model, model_type, layer_block_name):
    """Get the layers from model based on model type."""
    layer_block_name = MODELS_TO_LAYERS_MAPPING.get(model_type, layer_block_name)
    return get_attr(model, layer_block_name)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}


def exclude_layers_to_not_quantize(linear_layers, modules_to_not_convert):
    if modules_to_not_convert is None:
        return linear_layers

    return {
        name: linear_layer
        for name, linear_layer in linear_layers.items()
        if not any(key in name for key in modules_to_not_convert)
    }
