# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on the original implementation at
# https://github.com/IST-DASLab/sparsegpt
# https://arxiv.org/abs/2301.00774
# -------------------------------------------------------------------------
import logging
from typing import Any, Dict, List, Union

import torch

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModel
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pytorch.sparsegpt_utils import (
    SparseGPTModule,
    catch_layer_inputs,
    get_layer_submodules,
    get_layers,
    supported_models,
    validate_min_max_layers,
)

logger = logging.getLogger(__name__)


class SparseGPT(Pass):
    """Run SparseGPT on a Hugging Face PyTorch model.

    See https://arxiv.org/abs/2301.00774 for more details on the algorithm.

    This pass only supports PyTorchModel with hf_config. The transformers model type
    must be one of [bloom, gpt2, gpt_neox, llama, opt].
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "sparsity": PassConfigParam(
                type_=Union[float, List[int]],
                description=(
                    "Target sparsity. This can be a float or a list of two integers. Float is the target sparsity per"
                    " layer. List [n,m] applies semi-structured (n:m) sparsity patterns. Refer to"
                    " https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/"
                    " for more details on 2:4 sparsity pattern."
                ),
            ),
            "blocksize": PassConfigParam(
                type_=int, default_value=128, description="Blocksize to use for adaptive mask selection."
            ),
            "percdamp": PassConfigParam(
                type_=float,
                default_value=0.01,
                description="Percentage of the average Hessian diagonal to use for dampening. Must be in [0,1].",
            ),
            "min_layer": PassConfigParam(
                type_=int, default_value=None, description="Prune all layers with id >= min_layer."
            ),
            "max_layer": PassConfigParam(
                type_=int, default_value=None, description="Prune all layers with id < max_layer."
            ),
            "layer_name_filter": PassConfigParam(
                type_=Union[str, List[str]],
                default_value=None,
                description="Only prune layers whose name contains the given string(s).",
            ),
            # this is not the same as accelerator_spec.device which is the target device for inference
            # compute_device is the device we want to run the algorithm on, does not affect the final model
            # so accelerator_spec.device can be cpu but compute_device can be cuda for faster pass execution
            "compute_device": PassConfigParam(
                type_=str,
                default_value="auto",
                description=(
                    "Device to use for performing computations. Can be 'auto, 'cpu', 'cuda', 'cuda:0', etc. If 'auto',"
                    " will use cuda if available. Does not affect the final model."
                ),
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                required=True,
                description=(
                    "Data config to use for pruning weights. All samples in the data are expected to be of the"
                    " same length, most likely the max sequence length of the model."
                ),
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModel:
        model_type = model.model_attributes["model_type"]
        if model_type not in supported_models:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {supported_models}")

        # get sparsity mode and parameters
        if isinstance(config["sparsity"], float):
            assert 0 <= config["sparsity"] <= 1, "Sparsity must be in [0,1]."
        elif isinstance(config["sparsity"], list):
            assert len(config["sparsity"]) == 2, "Sparsity must be a float or a list of two integers."
        mode = "unstructured" if isinstance(config["sparsity"], float) else "structured"
        sparsity = config["sparsity"]
        n, m = sparsity if mode == "structured" else [0, 0]

        # get device to use for computations
        device = config["compute_device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Running SparseGPT on {device} with model_type: {model_type}, mode: {mode}, sparsity: {sparsity}")

        # load_data
        data_config = validate_config(config["data_config"], DataConfig)
        dataloader = data_config.to_data_container().create_dataloader(data_root)
        logger.debug(f"Data loaded. Number of batches: {len(dataloader)}")

        # load model
        pytorch_model = model.load_model()
        # we will update the model inplace
        # since the models are large, it is expensive to copy and maintain two copies
        # set model.model to None so that the input model doesn't use this same model object when it is loaded
        model.model = None
        # alternative is to copy the model and use the copy
        # pytorch_model = copy.deepcopy(model.model)
        pytorch_model.eval()
        use_cache = pytorch_model.config.use_cache
        pytorch_model.config.use_cache = False

        # get module list of layers
        layers = get_layers(pytorch_model, model_type)

        # get the inputs to the first layer
        inputs, attention_mask, extras = catch_layer_inputs(pytorch_model, model_type, dataloader, device)
        logger.debug(f"Inputs shape: {inputs.shape}")
        # place holder to store output from layer
        outputs = torch.zeros_like(inputs)

        # prune layers
        min_layer, max_layer = validate_min_max_layers(config["min_layer"], config["max_layer"], len(layers))
        layer_name_filter = config["layer_name_filter"] or []
        if isinstance(layer_name_filter, str):
            layer_name_filter = [layer_name_filter]
        # loop over layers
        logger.debug(f"Pruning layers {min_layer} to {max_layer}...")

        def get_handler(sparge_gpt_module):
            def handler(_, inputs, output):
                sparge_gpt_module.add_batch(inputs[0].data)

            return handler

        for i in range(min_layer, max_layer):
            logger.debug(f"Pruning layer {i}...")
            layer = layers[i]
            layer.to(device)

            # get list of submodules in layer
            submodules = get_layer_submodules(layer, layer_name_filter=layer_name_filter)
            # logger.debug(f"Submodules in layer {i}: {list(submodules.keys())}")

            # wrap submodules in layer with SparseGPTModule
            sparge_gpt_modules = {}
            for name, submodule in submodules.items():
                sparge_gpt_modules[name] = SparseGPTModule(submodule)

            # add forward hook to submodules in layer
            handles = []
            for name, submodule in submodules.items():
                handles.append(submodule.register_forward_hook(get_handler(sparge_gpt_modules[name])))

            # run layer
            for j in range(inputs.shape[0]):
                outputs[j] = layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, **extras)[0]

            # remove handler
            for handle in handles:
                handle.remove()

            # prune submodules in layer
            losses = {}
            for name, sparse_gpt_module in sparge_gpt_modules.items():
                loss = sparse_gpt_module.prune(
                    mode, sparsity, n, m, blocksize=config["blocksize"], percdamp=config["percdamp"]
                )
                losses[name] = loss
                sparse_gpt_module.free()
            logger.debug(f"Losses for layer {i}: {losses}")

            layer.to("cpu")
            if "cuda" in device:
                torch.cuda.empty_cache()

            inputs, outputs = outputs, inputs

        # save model
        pytorch_model.config.use_cache = use_cache
        pytorch_model.save_pretrained(output_model_path)

        # return PyTorchModel with updated model path
        model_config = model.to_json()["config"]
        model_config["model_path"] = output_model_path
        return PyTorchModel(**model_config)
