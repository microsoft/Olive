# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from typing import Any, Callable, Dict, List, Union

import torch

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import PyTorchModelHandler
from olive.model.handler.onnx import ONNXModelHandler
from olive.model.utils.onnx_utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.pass_config import PassConfigParam


class GptqQuantizer(Pass):
    """GPTQ quantization using Hugging Face Optimum and export model with onnxruntime optimized kernel."""

    _requires_user_script = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "nsamples": PassConfigParam(
                type_=int,
                default_value=128,
                description="number of samples in calibration dataset to apply quantization. Default value is 128",
            ),
            "bits": PassConfigParam(
                type_=int,
                default_value=4,
                description="quantization bits. Default value is 4",
            ),
            "layers_block_name": PassConfigParam(
                type_=str,
                default_value="model.layers",
                description="Block name to quantize. Default value is model.layers.",
            ),
            "outside_layer_modules": PassConfigParam(
                type_=List[str],
                default_value=None,
                description="Names of other nn modules that in the same level as the transformer layer block. "
                "Default value is None.",
            ),
            "inside_layer_modules": PassConfigParam(
                type_=List[List[str]],
                default_value=None,
                description="Names of linear layers in transformer layer module. Default value is None.",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Block size for quantization. Default value is 128.",
            ),
            "batch_size": PassConfigParam(
                type_=int,
                default_value=1,
                description="Batch size for quantization. Default value is 1.",
            ),
            "seed": PassConfigParam(
                type_=int,
                default_value=0,
                description="Random seed for sampling calibration dataset. Default value is 0.",
            ),
            "damp_percent": PassConfigParam(
                type_=float,
                default_value=0.01,
                description="Damping factor for quantization. Default value is 0.01.",
            ),
            "static_groups": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use static groups for quantization. Default value is False.",
            ),
            "true_sequential": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use true sequential for quantization. Default value is False.",
            ),
            "desc_act": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use descriptive activation for quantization. Default value is False.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Symmetric quantization. Default value is False.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                default_value=None,
                description="""
                    Data config for quantization. Default value is None.
                """,
            ),
            # TODO(trajep): consider to use data_config to implement the functionality of dataloader_func.
            "dataloader_func": PassConfigParam(
                type_=Union[Callable, str],
                default_value=None,
                description="""Function/function name to generate dataset for quantization.
                The returned datasets is a list of tokenized data
                (e.g. [{ 'input_ids': [ 1, 100, 15, ... ],'attention_mask': [ 1, 1, 1, ... ]},...]).
                Default is None.
                """,
            ),
            "dataloader_func_kwargs": PassConfigParam(
                type_=Dict[str, Any],
                default_value=None,
                description="Keyword arguments for dataloader_func. Default value is None.",
            ),
            "use_dynamo_exporter": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to use dynamo_export API to export ONNX model when export_optimum=False.",
            ),
            "target_opset": PassConfigParam(
                type_=int,
                default_value=13,
                description="The version of the default (ai.onnx) opset to target when export_optimum=False.",
            ),
            "export_device": PassConfigParam(
                type_=str,
                default_value="cpu",
                description="The version of the default (ai.onnx) opset to target when export_optimum=False.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        from auto_gptq import BaseQuantizeConfig
        from auto_gptq.modeling import BaseGPTQForCausalLM
        from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP

        from olive.passes.onnx.gptq_utils import QuantLinearORT

        if self.accelerator_spec.accelerator_type != Device.GPU:
            raise ValueError("Please use GPU to run gptq quantization.")

        dataset = None
        if config["dataloader_func"]:
            dataset = self._user_module_loader.call_object(
                config["dataloader_func"],
                **(config["dataloader_func_kwargs"] or {}),
            )
        elif config["data_config"]:
            data_config = validate_config(config["data_config"], DataConfig)
            dataloader = data_config.to_data_container().create_dataloader(data_root)
            dataset = [data[0] for data in dataloader]

        if (
            not dataset
            or not isinstance(dataset, list)
            or not isinstance(dataset[0], dict)
            or ("input_ids" not in dataset[0] or "attention_mask" not in dataset[0])
        ):
            raise ValueError(
                "Provided dataset is invalid. The returned datasets is a list of tokenized data "
                "(e.g. [{ 'input_ids': [ 1, 100, 15, ... ],'attention_mask': [ 1, 1, 1, ... ]},...])"
            )

        pytorch_model = model.load_model()
        quantize_config = BaseQuantizeConfig(
            bits=config["bits"],
            group_size=config["group_size"],
            damp_percent=config["damp_percent"],
            static_groups=config["static_groups"],
            true_sequential=config["true_sequential"],
            desc_act=config["desc_act"],
            sym=config["sym"],
        )

        def get_onnx_quant_linear(*args, **kwargs):
            return QuantLinearORT

        if hasattr(pytorch_model, "config") and pytorch_model.config.model_type in GPTQ_CAUSAL_LM_MODEL_MAP:
            model_type = pytorch_model.config.model_type
            model_class = GPTQ_CAUSAL_LM_MODEL_MAP[model_type]
            quantized_model = model_class(pytorch_model, False, quantize_config)
        else:
            quantized_model = BaseGPTQForCausalLM(pytorch_model, False, quantize_config)
            if not (config["layers_block_name"] and config["outside_layer_modules"] and config["inside_layer_modules"]):
                raise ValueError(
                    "Can't get layers_block_name to quantize automatically, "
                    "please set layers_block_name, outside_layer_modules and inside_layer_modules in config."
                )
            quantized_model.layers_block_name = config["layers_block_name"]
            quantized_model.outside_layer_modules = config["outside_layer_modules"]
            quantized_model.inside_layer_modules = config["inside_layer_modules"]

        import auto_gptq

        original = auto_gptq.modeling._utils.dynamically_import_QuantLinear  # pylint: disable=protected-access
        try:
            # Replace QuantLinear in autogptq with QuantLinearORT for quant linear layer packing
            auto_gptq.modeling._utils.dynamically_import_QuantLinear = (  # pylint: disable=protected-access
                get_onnx_quant_linear
            )

            # Autogpq quantize_model currently only support cuda device. It accepts model on cpu but
            # will move each block(layer) to cuda before quantization and move back to cpu when finished.
            quantized_model.quantize(dataset)
        finally:
            auto_gptq.modeling._utils.dynamically_import_QuantLinear = original  # pylint: disable=protected-access

        quantized_model = quantized_model.model
        assert self.accelerator_spec.accelerator_type == Device.GPU

        converted_onnx_model = OnnxConversion._export_pytorch_model(  # pylint: disable=protected-access
            quantized_model,
            model.get_dummy_inputs(),
            model.get_io_config(),
            config,
            config["export_device"],
            None,
            tempfile.tempdir,
        )
        output_model_path = resolve_onnx_path(output_model_path)
        output_model = model_proto_to_olive_model(converted_onnx_model, output_model_path, config)
        output_model.model_attributes = model.model_attributes
        return output_model
