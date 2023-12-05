# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import logging
import multiprocessing
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import onnx
import torch
from packaging import version

from olive.common.config_utils import validate_config
from olive.common.utils import find_submodules, resolve_torch_dtype, tensor_data_to_device
from olive.hardware import AcceleratorSpec
from olive.model import CompositeOnnxModel, DistributedOnnxModel, DistributedPyTorchModel, ONNXModel, PyTorchModel
from olive.model.hf_utils import HFModelLoadingArgs
from olive.model.model_config import IOConfig
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class TraceModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *input_data, **input_dict):
        if isinstance(self.model(*input_data, **input_dict), dict):
            return list(self.model(*input_data, **input_dict).values())
        return self.model(*input_data, **input_dict)


class OnnxConversion(Pass):
    """Convert a PyTorch model to ONNX model using torch.onnx.export on CPU."""

    _requires_user_script = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_opset": PassConfigParam(
                type_=int, default_value=13, description="The version of the default (ai.onnx) opset to target."
            ),
            "use_dynamo_exporter": PassConfigParam(
                type_=bool, default_value=False, description="Whether to use dynamo_export API to export ONNX model."
            ),
            "use_device": PassConfigParam(
                type_=str,
                description=(
                    "The device to use for conversion, e.g., 'cuda' or 'cpu'. If not specified, will use 'cpu' for"
                    " PyTorch model and 'cuda' for DistributedPyTorchModel."
                ),
            ),
            "torch_dtype": PassConfigParam(
                type_=str,
                description=(
                    "The dtype to cast the model to before conversion, e.g., 'float32' or 'float16'. If not specified,"
                    " will use the model as is."
                ),
            ),
            "parallel_jobs": PassConfigParam(
                type_=int,
                default=multiprocessing.cpu_count(),
                required=False,
                description="Number of parallel jobs. Defaulted to number of CPUs. Set it to 0 to disable.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> Union[CompositeOnnxModel, DistributedOnnxModel, ONNXModel]:
        # get the device to use for conversion
        # default to "cpu" for PyTorchModel and "cuda" for DistributedPyTorchModel
        device = config["use_device"] or "cpu"
        # get the dtype to use for conversion
        torch_dtype = resolve_torch_dtype(config["torch_dtype"]) if config["torch_dtype"] else None
        if torch_dtype == torch.float16 and device == "cpu":
            raise ValueError("Conversion to float16 is not supported for CPU.")

        if isinstance(model, DistributedPyTorchModel):
            if not config["use_device"]:
                device = "cuda"
            return self._convert_distributed_model_on_device(
                model, data_root, config, output_model_path, device, torch_dtype
            )

        # check if the model has components
        if model.components:
            onnx_models = []
            component_names = []
            for component_name in model.components:
                # convert each component
                component_model = model.get_component(component_name)
                component_output_path = str(Path(output_model_path).with_suffix("") / component_name)
                output_model_component = self._convert_model_on_device(
                    component_model, data_root, config, component_output_path, device, torch_dtype
                )
                # inherit model attributes from the input model if the output model does not have model attributes
                output_model_component.model_attributes = (
                    output_model_component.model_attributes or model.model_attributes
                )
                onnx_models.append(output_model_component)
                component_names.append(component_name)
            return CompositeOnnxModel(onnx_models, component_names)

        return self._convert_model_on_device(model, data_root, config, output_model_path, device, torch_dtype)

    @staticmethod
    def _export_pytorch_model(
        pytorch_model: torch.nn.Module,
        dummy_inputs,
        io_config,
        config: Dict[str, Any],
        device: Union[str, torch.device],
        torch_dtype: Optional[torch.dtype] = None,
        tempdir: Optional[Union[Path, str]] = None,
    ) -> onnx.ModelProto:
        """Export a torch.nn.Module to ONNX and return the loaded ONNX model.

        :param pytorch_model: the torch.nn.Module to export
        :param dummy_inputs: the dummy inputs to the model. Can be None if using dynamo_exporter
        :param io_config: the io_config for the model. This consists of the input and output names, and dynamic axes
        :param config: the config for the pass
        :param device: the device to use for conversion
        :param torch_dtype: the dtype to cast the model to before conversion
        :param tempdir: directory to use for temporary files
        """
        device = torch.device(device)
        use_gpu = device != torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # if pytorch_model is PeftModel, we need to get the base model
        # otherwise, the model forward has signature (*args, **kwargs) and torch.onnx.export ignores the dummy_inputs
        if importlib.util.find_spec("peft"):
            from peft import PeftModel

            if isinstance(pytorch_model, PeftModel):
                pytorch_model = pytorch_model.get_base_model()

        # put pytorch_model and dummy_inputs at the same device
        logger.debug(f"Converting model on device {device} with dtype {torch_dtype}.")
        pytorch_model.to(device)
        if torch_dtype:
            pytorch_model = pytorch_model.to(torch_dtype)

        dummy_inputs = tensor_data_to_device(dummy_inputs, device)
        if isinstance(pytorch_model, torch.jit.RecursiveScriptModule):
            pytorch_model = TraceModelWrapper(pytorch_model)

        onnx_model = None
        if config["use_dynamo_exporter"]:
            # available since torch==2.1.0
            torch_version = torch.__version__
            if version.parse(torch_version) < version.parse("2.1.0"):
                raise ImportError(
                    f"torch.onnx.dynamo_export is not available for torch version {torch_version}. "
                    "Please upgrade your torch version to 2.1.0 or above."
                )
            from torch.onnx import dynamo_export

            exported = dynamo_export(
                pytorch_model,
                *dummy_inputs,
                export_options=torch.onnx.ExportOptions(opset_version=config["target_opset"], dynamic_shapes=True),
            )
            onnx_model = exported.model_proto
        else:
            # Standard ONNX export

            # get input and output names, and dynamic axes
            assert (
                io_config is not None
            ), "Cannot get io_config for the model. Please specify io_config or hf_config for the model"
            io_config = validate_config(io_config, IOConfig)
            input_names = io_config.input_names
            output_names = io_config.output_names
            dynamic_axes = io_config.dynamic_axes

            # some dummy inputs might not be used in the model, so we need to remove them
            # this can happen when we are using an hf dataset to generate dummy inputs
            # only handle dict for now since we cannot get the name of the input from a list/tuple
            if isinstance(dummy_inputs, dict):
                dummy_input_keys = set(dummy_inputs.keys())

                # handle dummy inputs for model with past, which has past_key_values
                # match input names in `past_key_values.(hidden_layer_num).(key|value)` pattern
                for name, dm_input in dummy_inputs.items():
                    if isinstance(dm_input, list):
                        key_value_names = set(
                            [f"{name}.{idx}.key" for idx in range(len(dm_input))]
                            + [f"{name}.{idx}.value" for idx in range(len(dm_input))]
                        )
                        if key_value_names.issubset(set(input_names)):
                            dummy_input_keys.discard(name)

                unused_keys = dummy_input_keys - set(input_names)

                if unused_keys:
                    logger.debug(f"Removing unused dummy inputs: {unused_keys}")
                for key in unused_keys:
                    del dummy_inputs[key]

            # there might be multiple files created during export, so we need to track the dir
            # if there are other processes writing to the same dir, we might end up deleting files created by
            # other processes
            with tempfile.TemporaryDirectory(dir=tempdir, prefix="olive_tmp") as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                tmp_model_path = ONNXModel.resolve_path(tmp_dir_path)

                torch.onnx.export(
                    pytorch_model,
                    dummy_inputs,
                    tmp_model_path,
                    export_params=True,
                    opset_version=config["target_opset"],
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )
                onnx_model = onnx.load(tmp_model_path)
                # the model is loaded into memory, so it's safe to delete previously exported file(s)

            # Workaround as described under IOConfig.string_to_int_dim_params: change numeric dim_param to dim_value
            if io_config.string_to_int_dim_params:
                for tensor in onnx_model.graph.output:
                    for dim_proto in tensor.type.tensor_type.shape.dim:
                        if (
                            dim_proto.HasField("dim_param")
                            and dim_proto.dim_param in io_config.string_to_int_dim_params
                        ):
                            dim_value = int(dim_proto.dim_param)
                            dim_proto.Clear()
                            dim_proto.dim_value = dim_value

        # Reset to CPU so the resource consumed on GPU could be free.
        if use_gpu:
            pytorch_model.to("cpu")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return onnx_model

    @staticmethod
    def _load_pytorch_model(
        model: PyTorchModel, device: str, torch_dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.nn.Module, Optional[Dict]]:
        """Load the model and return the model and the model attributes.

        This method handles the following cases:
        1. model is not loaded from hf config, or the model loading args is not specified
            - load the model directly
        2. model is loaded from hf config, and the model loading args is specified
            - update model_loading_args.torch_dtype if torch_dtype is specified
            - if torch_dtype not specified, make sure the model loading args specify a dtype that is supported for
                conversion on the specified device
            - if quantization_method == "bitsandbytes" and load_in_4bit is True
                - remove quantization config from the model loading args
                - find quantized modules and add them to the model attributes
                - the onnx model must be quantized using OnnxBnb4Quantization pass after conversion
        Model attributes is None if the output model should inherit the model attributes from the input model.
        """
        if not model.is_model_loaded_from_hf_config() or not model.hf_config.model_loading_args:
            # if the model is not loaded from hf config, or the model loading args is not specified,
            # we can load the model directly
            return model.load_model(), None

        model_loading_args = model.hf_config.model_loading_args
        model_dtype = model_loading_args.get_torch_dtype()
        new_model_loading_args = deepcopy(model_loading_args.dict())
        new_model_attributes = model.model_attributes or {}
        if torch_dtype and torch_dtype != model_dtype:
            # if the model loading args specify a different dtype, update the model loading args
            logger.debug(
                f"Changing torch_dtype in model loading args from {model_loading_args.get_torch_dtype()} to"
                f" {torch_dtype}."
            )
            new_model_loading_args["torch_dtype"] = torch_dtype
            new_model_attributes["torch_dtype"] = str(torch_dtype).replace("torch.", "")
        elif model_dtype == torch.float16 and device == "cpu":
            logger.warning(
                "Loading model on CPU, but the model loading args specify dtype float16 which is not supported  for"
                " conversion on CPU. The dtype is changed to float32. If float16 model is desired, please specify"
                " use_device as 'cuda' or use OrtTransformerOptimization/OnnxFloatToFloat16 pass after conversion to"
                " convert the model to float16."
            )
            new_model_loading_args["torch_dtype"] = torch.float32
            new_model_attributes["torch_dtype"] = "float32"

        if (
            model_loading_args.quantization_method == "bitsandbytes"
            and model_loading_args.quantization_config["load_in_4bit"]
        ):
            logger.warning(
                "Bitsandbytes 4bit quantization is not supported for conversion. The quantization config is removed"
                " from the model loading args. Use OnnxBnb4Quantization pass after conversion to quantize the model."
            )
            new_model_loading_args["quantization_method"] = None
            new_model_loading_args["quantization_config"] = None
            new_model_attributes["quantization_config"] = model_loading_args.quantization_config
            if "quantized_modules" not in new_model_attributes:
                # find and add quantized modules to the model attributes
                # the QLoRA pass already adds quantized_modules to the model attributes, so this will not be executed
                # if the model was generated by QLoRA
                quantized_model = model.load_model()

                import bitsandbytes as bnb

                new_model_attributes["quantized_modules"] = find_submodules(quantized_model, bnb.nn.Linear4bit)
                # delete the quantized model to free up memory
                del quantized_model
                model.model = None

        # load the model with the updated model loading args
        new_hf_config = deepcopy(model.hf_config)
        new_hf_config.model_loading_args = HFModelLoadingArgs(**new_model_loading_args)
        return new_hf_config.load_model(model.model_path), new_model_attributes

    def _convert_model_on_device(
        self,
        model: PyTorchModel,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> ONNXModel:
        """Convert a PyTorchModel to an ONNXModel."""
        # load the model
        pytorch_model, model_attributes = self._load_pytorch_model(model, device, torch_dtype)
        pytorch_model.eval()

        # get dummy inputs
        dummy_inputs = model.get_dummy_inputs()
        io_config = None if config["use_dynamo_exporter"] else model.get_io_config()

        converted_onnx_model = OnnxConversion._export_pytorch_model(
            pytorch_model, dummy_inputs, io_config, config, device, torch_dtype, tempfile.tempdir
        )

        # save the model to the output path and return the model
        output_model_path = ONNXModel.resolve_path(output_model_path)
        output_model = model_proto_to_olive_model(converted_onnx_model, output_model_path, config)
        output_model.model_attributes = model_attributes
        return output_model

    @staticmethod
    def _export_ranked_model(params):
        """Export one rank of a DistributedPytorchModel to ONNX and save the model to the output path.

        :param params: a tuple of (pass_config, model_config, world_size, device, local_rank, output_dirpath)
            pass_config: the config for the pass
            model_config: the config for the DistributedPytorchModel
            device: the device to use for conversion
            torch_dtype: the dtype to cast the model to before conversion
            local_rank: the rank of the current process as well as the rank of the model to be converted
            output_dirpath: the path to the directory to save the model. The .onnx model will be saved in this
                directory with the name specified by DistributedOnnxModel.DEFAULT_RANKED_MODEL_NAME_FORMAT
        """
        pass_config, model_config, device, torch_dtype, local_rank, output_dirpath, tempdir = params

        # TODO(shaahji): Parameterize the replace/restore hooks to configure for different models
        from olive.passes.pytorch.tensor_parallel_llama2 import (
            replace_llama2_tensor_parallel_layers,
            restore_llama2_tensor_parallel_layers,
        )

        output_filename = DistributedOnnxModel.DEFAULT_RANKED_MODEL_NAME_FORMAT.format(local_rank)
        output_filepath = ONNXModel.resolve_path(output_dirpath, output_filename)

        try:
            restore_args = replace_llama2_tensor_parallel_layers()

            input_model = DistributedPyTorchModel(**model_config)
            olive_pytorch_model = input_model.load_model(local_rank)
            dummy_inputs = olive_pytorch_model.get_dummy_inputs()
            io_config = None if pass_config["use_dynamo_exporter"] else olive_pytorch_model.get_io_config()
            pytorch_model = olive_pytorch_model.prepare_session(rank=local_rank)

            with torch.no_grad():
                ranked_onnx_model = OnnxConversion._export_pytorch_model(
                    pytorch_model,
                    dummy_inputs,
                    io_config,
                    pass_config,
                    device,
                    torch_dtype,
                    tempdir,
                )
        finally:
            restore_llama2_tensor_parallel_layers(restore_args)

        # save the model to the output path and return the model
        model_proto_to_olive_model(ranked_onnx_model, output_filepath, pass_config)
        return 1  # Return 1 for success.

    def _convert_distributed_model_on_device(
        self,
        model: DistributedPyTorchModel,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> DistributedOnnxModel:
        """Convert a DistributedPyTorchModel to a DistributedOnnxModel."""
        pass_config = config
        model_config = model.to_json()["config"]
        world_size = model.num_ranks
        output_model_path = str(Path(output_model_path).with_suffix(""))
        use_gpu = torch.device(device) != torch.device("cpu")

        params = [
            (
                pass_config,
                model_config,
                torch.device("cuda", rank) if use_gpu else torch.device("cpu"),
                torch_dtype,
                rank,
                output_model_path,
                tempfile.tempdir,
            )
            for rank in range(world_size)
        ]

        max_parallel_jobs = min(world_size, config["parallel_jobs"] or multiprocessing.cpu_count())
        if max_parallel_jobs <= 1:
            results = [OnnxConversion._export_ranked_model(_) for _ in params]
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(processes=max_parallel_jobs) as pool:
                results = pool.map(OnnxConversion._export_ranked_model, params)

        if world_size != sum(results):
            raise RuntimeError("Failed to convert models")

        return DistributedOnnxModel(
            model_path=output_model_path,
            model_name_pattern=DistributedOnnxModel.DEFAULT_RANKED_MODEL_NAME_FORMAT,
            num_ranks=world_size,
            model_attributes=model.model_attributes,
        )


class OnnxOpVersionConversion(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        latest_opset_version = onnx.defs.onnx_opset_version()

        config = {
            "target_opset": PassConfigParam(
                type_=int,
                default_value=latest_opset_version,
                description="The version of the default (ai.onnx) opset to target. Default: latest opset version.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        # get current models's opset version
        model_proto = model.load_model()
        model_opset_version = model_proto.opset_import[0].version
        if model_opset_version == config["target_opset"]:
            logger.info(f"Model is already in target opset version {config['target_opset']}.")
            return model

        output_model_path = ONNXModel.resolve_path(output_model_path)
        model_proto = onnx.version_converter.convert_version(model_proto, config["target_opset"])
        return model_proto_to_olive_model(model_proto, output_model_path, config)
