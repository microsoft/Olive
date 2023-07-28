# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on the original implementation at
# https://github.com/IST-DASLab/sparsegpt
# https://arxiv.org/abs/2301.00774
# -------------------------------------------------------------------------
import copy
import logging
from typing import Any, Dict, List, Union

import torch

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModel
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pytorch.sparsegpt_utils import SparseGPTModule, catch_layer_inputs, get_layer_submodules, get_layers

logger = logging.getLogger(__name__)


class SparseGPT(Pass):
    """Run SparseGPT on PyTorch model."""

    _requires_data_config = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            # TODO: `model_type` will be removed later when we generalize this pass to support other models.
            # it can be inferred from the input model.
            "model_type": PassConfigParam(
                type_=str,
                required=True,
                description="Transformer model type. Currently supported types are: opt.",
            ),
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
            "compute_device": PassConfigParam(
                type_=str,
                default_value="auto",
                description=(
                    "Device to use for performing computations. Can be 'auto, 'cpu', 'cuda', 'cuda:0', etc. If 'auto',"
                    " will use cuda if available. Does not affect the final model."
                ),
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModel:
        if config["model_type"] != "opt":
            raise ValueError(f"Unsupported model type: {config['model_type']}")
        model_type = config["model_type"]

        # get sparsity mode and parameters
        if isinstance(config["sparsity"], float):
            assert 0 <= config["sparsity"] <= 1, "Sparsity must be in [0,1]."
        elif isinstance(config["sparsity"], list):
            assert len(config["sparsity"]) == 2, "Sparsity must be a float or a list of two integers."
        mode = "unstructured" if isinstance(config["sparsity"], float) else "structured"
        sparsity = config["sparsity"]
        n, m = sparsity if mode == "structured" else [0, 0]
        logger.debug(f"Running SparseGPT with mode={mode}, sparsity={sparsity}")

        # get device to use for computations
        device = config["compute_device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Running SparseGPT on device: {device}")

        # load_data
        assert config["data_config"] is not None, "Data config is required for SparseGPT pass."
        dataloader = self._data_config.to_data_container().create_dataloader(data_root)

        # load model
        pytorch_model = model.load_model()
        pytorch_model = copy.deepcopy(pytorch_model)
        pytorch_model.eval()
        use_cache = pytorch_model.config.use_cache
        pytorch_model.config.use_cache = False

        # get module list of layers
        layers = get_layers(pytorch_model, model_type)

        # get the inputs to the first layer
        inputs, attention_mask = catch_layer_inputs(pytorch_model, model_type, dataloader, device)
        # place holder to store output from layer
        outputs = torch.zeros_like(inputs)

        # prune layers
        min_layer = config["min_layer"] or 0
        max_layer = config["max_layer"] or len(layers)
        layer_name_filter = config["layer_name_filter"] or []
        if isinstance(layer_name_filter, str):
            layer_name_filter = [layer_name_filter]
        # loop over layers
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

            def get_handler(sparge_gpt_module):
                def handler(_, input, output):
                    sparge_gpt_module.add_batch(input[0].data)

                return handler

            for name, submodule in submodules.items():
                handles.append(submodule.register_forward_hook(get_handler(sparge_gpt_modules[name])))

            # run layer
            for j in range(inputs.shape[0]):
                outputs[j] = layer(inputs[j].unsqueeze(0), attention_mask)[0]

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
            torch.cuda.empty_cache()

            inputs, outputs = outputs, inputs

        # save model
        pytorch_model.config.use_cache = use_cache
        pytorch_model.save_pretrained(output_model_path)

        # return PyTorchModel with updated model path
        model_config = model.to_json()["config"]
        model_config["model_path"] = output_model_path
        return PyTorchModel(**model_config)
