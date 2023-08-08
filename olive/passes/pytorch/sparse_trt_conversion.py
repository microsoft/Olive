# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on the original implementation at
# https://github.com/IST-DASLab/sparsegpt
# https://arxiv.org/abs/2301.00774
# -------------------------------------------------------------------------
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from olive.common.utils import tensor_data_to_device
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModel
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pytorch.sparsegpt_utils import _get_attr, get_layer_submodules, get_layers, seqlens

logger = logging.getLogger(__name__)


class SparseTRTConversion(Pass):
    """Convert a PyTorch model to a sparse fp16 Pytorch model with TRT Modules."""

    _requires_data_config = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModel:
        model_config = model.get_model_config()
        model_type = model_config.model_type
        if model_type not in seqlens:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {seqlens.keys()}")

        if not torch.cuda.is_available():
            raise ValueError("SparseTRTConversion requires a GPU to run.")
        device = "cuda"

        # load_data
        assert config["data_config"] is not None, "Data config is required for SparseTRTConversion."
        first_batch = self._data_config.to_data_container().get_first_batch(data_root_path=data_root)[0]
        first_batch = tensor_data_to_device(first_batch, device=device)
        batch_size = first_batch["input_ids"].shape[0]
        seqlen = seqlens[model_type]

        # load model
        pytorch_model = model.load_model()
        # we will update the model inplace
        # since the models are large, it is expensive to copy and maintain two copies
        # set model.model to None so that the input model doesn't use this same model object when it is loaded
        model.model = None
        # alternative is to copy the model and use the copy
        # pytorch_model = copy.deepcopy(model.model)
        pytorch_model.eval()
        # convert model to fp16 on GPU
        pytorch_model.to(dtype=torch.float16, device=device)
        # disable cache
        use_cache = pytorch_model.config.use_cache
        pytorch_model.config.use_cache = False

        # get module list of layers
        layers = get_layers(pytorch_model, model_type)

        # get layer information
        min_layer = config["min_layer"] or 0
        max_layer = config["max_layer"] or len(layers)
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
                def handler(_, input, output):
                    layer_info[layer_idx]["input_shapes"][submodule_name] = input[0].shape

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
        logger.debug(f"Converting layers {min_layer} to {max_layer}...")
        for layer_index, info in layer_info.items():
            logger.debug(f"Converting layer {layer_index}...")
            for name, shape in info["input_shapes"].items():
                input = torch.zeros(shape, dtype=torch.float16, device=device)
                # create trt module
                trt_module = self._compile_trt_model(info["submodules"][name], input, batch_size, seqlen)
                # get parent module
                parent_name = ".".join(name.split(".")[:-1])
                parent_module = (
                    _get_attr(layers[layer_index], ".".join(name.split(".")[:-1]))
                    if parent_name
                    else layers[layer_index]
                )
                # get submodule name
                module_name = name.split(".")[-1]
                # replace submodule with trt module
                setattr(parent_module, module_name, trt_module)
                # remove submodule from layer_info
                del info["submodules"][name]
                # TODO: is the empty cache necessary? does it add processing time?
                # torch.cuda.empty_cache()
                # gc.collect()

        # restore cache
        pytorch_model.config.use_cache = use_cache

        # save save entire model to output_model_path
        output_model_path = Path(output_model_path).with_suffix(".pt")
        torch.save(pytorch_model, output_model_path)
        return PyTorchModel(model_path=output_model_path)

    def _compile_trt_model(
        self, torch_module: torch.nn.Module, hidden_states: torch.Tensor, batch_size: int, seqlen: int
    ):
        """Compile a torch module to a TensorRT module.

        :param torch_module: The torch module to compile. Only torch.nn.Linear modules are supported currently.
        :param hidden_states: The input tensor to the torch module.
        :param batch_size: The batch size of the input tensor.
        :param seqlen: The maximum sequence length of the input tensor. seqlen dimension is treated as dynamic.
        """
        import tensorrt as trt
        import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
        from torch_tensorrt.fx import InputTensorSpec, TRTInterpreter, TRTModule, compile

        logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)

        if not isinstance(torch_module, torch.nn.Linear):
            raise TypeError("torch_module must be a torch.nn.Linear module. Other modules are not supported.")

        # whether the batch and seqlen dimensions are flattened
        flattened = len(hidden_states.shape) == 2

        # redirect stdout to avoid printing
        f = io.StringIO()
        with redirect_stdout(f):
            # compile torch module to TensorRT module
            low_model = compile(torch_module, [hidden_states]).eval()
        f.close()
        acc_model = acc_tracer.trace(low_model, [hidden_states])
        # get input specs
        hidden_size = hidden_states.shape[-1]
        if not flattened:
            shape = [batch_size, -1, hidden_size]
            shape_ranges = [
                ((batch_size, 1, hidden_size), (batch_size, 100, hidden_size), (batch_size, seqlen, hidden_size))
            ]
        else:
            shape = [-1, hidden_size]
            shape_ranges = [
                ((batch_size, hidden_size), (batch_size * 100, hidden_size), (batch_size * seqlen, hidden_size))
            ]
        input_specs = [InputTensorSpec(shape=shape, dtype=torch.float16, shape_ranges=shape_ranges)]
        # create TensorRT module
        interpreter = TRTInterpreter(
            acc_model, input_specs, explicit_batch_dimension=True, logger_level=trt.Logger.ERROR
        )
        result = interpreter.run(sparse_weights=True)
        trt_module = TRTModule(result.engine, result.input_names, result.output_names)
        return trt_module
