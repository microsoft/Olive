# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from olive.common.config_utils import validate_config
from olive.common.utils import get_attr, tensor_data_to_device
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import PyTorchModelHandler
from olive.model.utils.hf_utils import get_model_max_length
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pytorch.sparsegpt_utils import (
    get_layer_submodules,
    get_layers,
    supported_models,
    validate_min_max_layers,
)

logger = logging.getLogger(__name__)


class TorchTRTConversion(Pass):
    """Convert torch.nn.Linear modules in the transformer layers of a HuggingFace PyTorch model to TensorRT modules.

    The conversion would include fp16 precision and sparse weights, if applicable.
    The entire model is saved using `torch.save` and can be loaded using `torch.load`. Loading the model requires
    `torch-tensorrt` and Olive to be installed.

    This pass only supports PyTorchModelHandler with hf_config. The transformers model type
    must be one of [bloom, gpt2, gpt_neox, llama, opt].
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "min_layer": PassConfigParam(
                type_=int, default_value=None, description="Convert all layers with id >= min_layer."
            ),
            "max_layer": PassConfigParam(
                type_=int, default_value=None, description="Convert all layers with id < max_layer."
            ),
            "layer_name_filter": PassConfigParam(
                type_=Union[str, List[str]],
                default_value=None,
                description="Only convert layers whose name contains the given string(s).",
            ),
            "float16": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Convert entire model to fp16. If False, only the sparse modules are converted to fp16.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                required=True,
                description=(
                    "Data config to use for compiling module to TensorRT. The batch size of the compiled module is set"
                    " to the batch size of the first batch of the dataloader."
                ),
            ),
        }

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if accelerator_spec.accelerator_type != Device.GPU:
            logger.info("TorchTRTConversion only supports GPU.")
            return False
        return True

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        from olive.passes.pytorch.trt_utils import compile_trt_model

        model_type = model.model_attributes["model_type"]
        if model_type not in supported_models:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {supported_models}")

        if not torch.cuda.is_available():
            raise ValueError("TorchTRTConversion requires a GPU to run.")
        device = "cuda"

        # load_data
        data_config = validate_config(config["data_config"], DataConfig)
        first_batch = data_config.to_data_container().get_first_batch(data_root_path=data_root)[0]
        first_batch = tensor_data_to_device(first_batch, device=device)
        batch_size = first_batch["input_ids"].shape[0]
        # get max sequence length
        model_name = model.hf_config.model_name
        seqlen = get_model_max_length(model_name, fail_on_not_found=True)

        # load model
        pytorch_model = model.load_model()
        # we will update the model inplace
        # since the models are large, it is expensive to copy and maintain two copies
        # set model.model to None so that the input model doesn't use this same model object when it is loaded
        model.model = None
        # alternative is to copy the model and use the copy
        # pytorch_model = copy.deepcopy(model.model)
        pytorch_model.eval()
        # move model to device
        pytorch_model.to(device=device)
        # convert model to fp16 if needed
        if config["float16"]:
            pytorch_model = pytorch_model.to(dtype=torch.float16)
        # disable cache
        use_cache = pytorch_model.config.use_cache
        pytorch_model.config.use_cache = False

        # get module list of layers
        layers = get_layers(pytorch_model, model_type)

        # get layer information
        min_layer, max_layer = validate_min_max_layers(config["min_layer"], config["max_layer"], len(layers))
        layer_name_filter = config["layer_name_filter"] or []
        if isinstance(layer_name_filter, str):
            layer_name_filter = [layer_name_filter]
        # layer information storage
        layer_info = {}
        # loop over layers
        for i in range(min_layer, max_layer):
            layer = layers[i]
            layer_info[i] = {"submodules": {}, "handles": [], "input_shapes": {}}

            # get list of submodules in layer
            layer_info[i]["submodules"] = get_layer_submodules(
                layer, submodule_types=[torch.nn.Linear], layer_name_filter=layer_name_filter
            )

            # add forward hook to submodules in layer
            def get_handler(layer_idx, submodule_name):
                def handler(_, inputs, output):
                    layer_info[layer_idx]["input_shapes"][submodule_name] = inputs[0].shape

                return handler

            for name, submodule in layer_info[i]["submodules"].items():
                layer_info[i]["handles"].append(submodule.register_forward_hook(get_handler(i, name)))

        # run a forward pass
        pytorch_model(**first_batch)

        # remove handles
        for info in layer_info.values():
            for handle in info["handles"]:
                handle.remove()
            del info["handles"]

        # convert submodules to trt modules
        logger.debug("Converting layers %d to %d...", min_layer, max_layer)
        for layer_index, info in layer_info.items():
            logger.debug("Converting layer %d...", layer_index)
            for name, shape in info["input_shapes"].items():
                inputs = torch.zeros(shape, dtype=torch.float16, device=device)
                # create trt module
                trt_module = compile_trt_model(info["submodules"][name], inputs, batch_size, seqlen)
                # get parent module
                parent_name = ".".join(name.split(".")[:-1])
                parent_module = get_attr(layers[layer_index], parent_name)
                # get submodule name
                module_name = name.split(".")[-1]
                # replace submodule with trt module
                setattr(parent_module, module_name, trt_module)
                # remove submodule from layer_info
                del info["submodules"][name]
                # TODO(jambayk): is the empty cache necessary? does it add processing time?
                # torch.cuda.empty_cache()
                # gc.collect()

        # restore cache
        pytorch_model.config.use_cache = use_cache

        # save save entire model to output_model_path
        output_model_path = Path(output_model_path).with_suffix(".pt")
        torch.save(pytorch_model, output_model_path)
        return PyTorchModelHandler(model_path=output_model_path)
