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
from olive.model import (
    CompositeModelHandler,
    DistributedOnnxModelHandler,
    DistributedPyTorchModelHandler,
    HfFromPretrainedArgs,
    ONNXModelHandler,
    PyTorchModelHandler,
)
from olive.model.config import IoConfig
from olive.model.config.hf_config import HfConfig, get_model_type_from_hf_config
from olive.model.config.io_config import is_kv_cache_required
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.merge_decoders import merge_decoders
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


# TODO(jambayk): Consider conditional import of PeftModel so that we can type hint it.
def is_peft_model(model: torch.nn.Module) -> bool:
    """Check if the model is a PeftModel."""
    if importlib.util.find_spec("peft"):
        from peft import PeftModel

        return isinstance(model, PeftModel)
    return False


class OnnxConversion(Pass):
    """Convert a PyTorch model to ONNX model using torch.onnx.export on CPU."""

    _requires_user_script = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_opset": PassConfigParam(
                type_=int, default_value=13, description="The version of the default (ai.onnx) opset to target."
            ),
            "use_dynamo_exporter": PassConfigParam(
                type_=bool, default_value=False, description="Whether to use dynamo_export API to export ONNX model."
            ),
            "device": PassConfigParam(
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
            "merge_components": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to merge the converted components.",
            ),
            "merge_adapter_weights": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to merge adapter weights before conversion. "
                "After merging, the model structure is consistent with base model. "
                "That is useful if you cannot run conversion for some fine-tuned "
                "models with adapter weights",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> Union[CompositeModelHandler, DistributedOnnxModelHandler, ONNXModelHandler]:
        # get the device to use for conversion
        # default to "cpu" for PyTorchModelHandler and "cuda" for DistributedPyTorchModel
        device = config["device"] or "cpu"
        # get the dtype to use for conversion
        torch_dtype = resolve_torch_dtype(config["torch_dtype"]) if config["torch_dtype"] else None
        if torch_dtype == torch.float16 and device == "cpu":
            raise ValueError("Conversion to float16 is not supported for CPU.")

        if isinstance(model, DistributedPyTorchModelHandler):
            if not config["device"]:
                device = "cuda"
            return self._convert_distributed_model_on_device(
                model, data_root, config, output_model_path, device, torch_dtype
            )

        if not model.hf_config or not model.hf_config.components:
            return self._convert_model_on_device(model, data_root, config, output_model_path, device, torch_dtype)

        onnx_models = []
        component_names = []
        for component_name, component_model in model.get_hf_components():
            component_output_path = str(Path(output_model_path).with_suffix("") / component_name)
            output_model_component = self._convert_model_on_device(
                component_model, data_root, config, component_output_path, device, torch_dtype
            )
            # inherit model attributes from the input model if the output model does not have model attributes
            output_model_component.model_attributes = output_model_component.model_attributes or model.model_attributes
            onnx_models.append(output_model_component)
            component_names.append(component_name)

        if config["merge_components"] and len(onnx_models) == 2:
            merged_model = merge_decoders(onnx_models[0].model_path, onnx_models[1].model_path)
            return model_proto_to_olive_model(merged_model, resolve_onnx_path(output_model_path), config)

        return CompositeModelHandler(onnx_models, component_names)

    @staticmethod
    @torch.no_grad()
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
        if is_peft_model(pytorch_model):
            pytorch_model = pytorch_model.get_base_model()

        # put pytorch_model and dummy_inputs at the same device
        logger.debug("Converting model on device %s with dtype %s.", device, torch_dtype)
        pytorch_model.to(device)
        if torch_dtype:
            pytorch_model = pytorch_model.to(torch_dtype)

        dummy_inputs = tensor_data_to_device(dummy_inputs, device)
        if isinstance(pytorch_model, torch.jit.RecursiveScriptModule):
            pytorch_model = TraceModelWrapper(pytorch_model)

        # get input and output names, and dynamic axes
        assert (
            io_config is not None
        ), "Cannot get io_config for the model. Please specify io_config or hf_config for the model"
        io_config = validate_config(io_config, IoConfig)
        input_names = io_config.input_names

        # some dummy inputs might not be used in the model, so we need to remove them
        # this can happen when we are using an hf dataset to generate dummy inputs
        # only handle dict for now since we cannot get the name of the input from a list/tuple
        if isinstance(dummy_inputs, dict):
            dummy_input_keys = set(dummy_inputs.keys())

            # after the expansion, user should provide the correct input names for inference
            for name, dm_input in dummy_inputs.items():
                # the `past_key_values` is the argument name from huggingface model class
                # which is independent of the kv-related variables in input list provided by users
                # if user provided the kv-related variables, we should not remove
                # the `past_key_values` from dummy inputs. But if not, we should remove it.
                if (
                    name == "past_key_values"
                    and isinstance(dm_input, list)
                    and is_kv_cache_required(dm_input, io_config)
                ):
                    dummy_input_keys.discard(name)

            unused_keys = dummy_input_keys - set(input_names)

            if unused_keys:
                logger.debug("Removing unused dummy inputs: %s", unused_keys)

        onnx_model = None
        if config["use_dynamo_exporter"]:
            torch_version = torch.__version__
            if version.parse(torch_version) < version.parse("2.2.0"):
                raise ImportError(
                    f"torch.onnx.dynamo_export is not available for torch version {torch_version}. "
                    "Please upgrade your torch version to 2.2.0 or above."
                )
            from torch._dynamo import config
            from torch.onnx import dynamo_export

            config.capture_scalar_outputs = True

            if isinstance(dummy_inputs, dict):
                for key in unused_keys:
                    dummy_inputs[key] = None

            pytorch_model(*dummy_inputs.values())

            with tempfile.TemporaryDirectory(dir=tempdir, prefix="olive_tmp") as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                tmp_model_path = resolve_onnx_path(tmp_dir_path)

                dynamo_export(
                    pytorch_model,
                    *dummy_inputs.values(),
                    export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
                ).save(tmp_model_path)
                onnx.checker.check_model(tmp_model_path)
                onnx.shape_inference.infer_shapes_path(tmp_model_path)
                onnx_model = onnx.load(tmp_model_path)
        else:
            # Standard ONNX export
            if isinstance(dummy_inputs, dict):
                for key in unused_keys:
                    del dummy_inputs[key]

            output_names = io_config.output_names
            dynamic_axes = io_config.dynamic_axes

            # there might be multiple files created during export, so we need to track the dir
            # if there are other processes writing to the same dir, we might end up deleting files created by
            # other processes
            with tempfile.TemporaryDirectory(dir=tempdir, prefix="olive_tmp") as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                tmp_model_path = resolve_onnx_path(tmp_dir_path)

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

            # Workaround as described under IoConfig.string_to_int_dim_params: change numeric dim_param to dim_value
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
        model: PyTorchModelHandler, device: str, torch_dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.nn.Module, Optional[Dict]]:
        """Load the model and return the model and the model attributes.

        This method handles the following cases:
        1. model is not loaded from hf config, or the model loading args is not specified
            - load the model directly
        2. model is loaded from hf config, and the model loading args is specified
            - update from_pretrained_args.torch_dtype if torch_dtype is specified
            - if torch_dtype not specified, make sure the model loading args specify a dtype that is supported for
                conversion on the specified device
            - if quantization_method == "bitsandbytes" and load_in_4bit is True
                - remove quantization config from the model loading args
                - find quantized modules and add them to the model attributes
                - the onnx model must be quantized using OnnxBnb4Quantization pass after conversion
        Model attributes is None if the output model should inherit the model attributes from the input model.
        """
        if not model.is_model_loaded_from_hf_config() or not model.hf_config.from_pretrained_args:
            # if the model is not loaded from hf config, or the model loading args is not specified,
            # we can load the model directly
            return model.load_model(), None

        from_pretrained_args = model.hf_config.from_pretrained_args
        model_dtype = from_pretrained_args.get_torch_dtype()
        new_from_pretrained_args = deepcopy(from_pretrained_args.dict())
        new_model_attributes = model.model_attributes or {}
        if torch_dtype and torch_dtype != model_dtype:
            # if the model loading args specify a different dtype, update the model loading args
            logger.debug(
                "Changing torch_dtype in model loading args from %s to %s.",
                from_pretrained_args.get_torch_dtype(),
                torch_dtype,
            )
            new_from_pretrained_args["torch_dtype"] = torch_dtype
            new_model_attributes["torch_dtype"] = str(torch_dtype).replace("torch.", "")
        elif model_dtype == torch.float16 and device == "cpu":
            logger.warning(
                "Loading model on CPU, but the model loading args specify dtype float16 which is not supported  for"
                " conversion on CPU. The dtype is changed to float32. If float16 model is desired, please specify"
                " device as 'cuda' or use OrtTransformerOptimization/OnnxFloatToFloat16 pass after conversion to"
                " convert the model to float16."
            )
            new_from_pretrained_args["torch_dtype"] = torch.float32
            new_model_attributes["torch_dtype"] = "float32"

        if (
            from_pretrained_args.quantization_method == "bitsandbytes"
            and from_pretrained_args.quantization_config["load_in_4bit"]
        ):
            logger.warning(
                "Bitsandbytes 4bit quantization is not supported for conversion. The quantization config is removed"
                " from the model loading args. Use OnnxBnb4Quantization pass after conversion to quantize the model."
            )
            new_from_pretrained_args["quantization_method"] = None
            new_from_pretrained_args["quantization_config"] = None
            new_model_attributes["quantization_config"] = from_pretrained_args.quantization_config
            if "quantized_modules" not in new_model_attributes:
                # find and add quantized modules to the model attributes
                # the QLoRA pass already adds quantized_modules to the model attributes, so this will not be executed
                # if the model was generated by QLoRA
                quantized_model = model.load_model()

                # if PeftModel, need to unload adapter before finding quantized modules
                if is_peft_model(quantized_model):
                    quantized_model = quantized_model.unload()

                import bitsandbytes as bnb

                new_model_attributes["quantized_modules"] = find_submodules(quantized_model, bnb.nn.Linear4bit)

                # required for peft models since unloading changes the model
                # for others, do this to free gpu memory as quantized model is always on gpu
                del quantized_model
                model.model = None

        # load the model with the updated model loading args
        new_hf_config = deepcopy(model.hf_config)
        new_hf_config.from_pretrained_args = HfFromPretrainedArgs(**new_from_pretrained_args)
        return (
            PyTorchModelHandler(
                model_path=model.model_path, adapter_path=model.adapter_path, hf_config=new_hf_config
            ).load_model(),
            new_model_attributes,
        )

    def _convert_model_on_device(
        self,
        model: PyTorchModelHandler,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> ONNXModelHandler:
        """Convert a PyTorchModelHandler to an ONNXModelHandler."""
        # load the model
        pytorch_model, model_attributes = self._load_pytorch_model(model, device, torch_dtype)
        if config["merge_adapter_weights"] and is_peft_model(pytorch_model):
            logger.debug("Merging adapter weights into base model. This is specific to PeftModel.")
            pytorch_model = pytorch_model.merge_and_unload()
        pytorch_model.eval()

        # get dummy inputs
        dummy_inputs = model.get_dummy_inputs()
        io_config = model.get_io_config()

        converted_onnx_model = OnnxConversion._export_pytorch_model(
            pytorch_model, dummy_inputs, io_config, config, device, torch_dtype, tempfile.tempdir
        )

        # save the model to the output path and return the model
        output_model_path = resolve_onnx_path(output_model_path)
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

        hf_config = HfConfig(**model_config["hf_config"])
        model_type = get_model_type_from_hf_config(hf_config)

        if model_type == "llama":
            from olive.passes.pytorch.tensor_parallel_llama2 import (
                replace_llama2_tensor_parallel_layers as replace_tensor_parallel_layers,
            )
            from olive.passes.pytorch.tensor_parallel_llama2 import (
                restore_llama2_tensor_parallel_layers as restore_tensor_parallel_layers,
            )
        else:
            raise ValueError("Unsupported model type '{model_type}' for conversion pass")

        output_filename = DistributedOnnxModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT.format(local_rank)
        output_filepath = resolve_onnx_path(output_dirpath, output_filename)

        try:
            restore_args = replace_tensor_parallel_layers()

            input_model = DistributedPyTorchModelHandler(**model_config)

            if input_model.hf_config and input_model.hf_config.components:
                ranked_models = []
                for _, component_model in input_model.get_hf_components(local_rank):
                    dummy_inputs = component_model.get_dummy_inputs()
                    io_config = None if pass_config["use_dynamo_exporter"] else component_model.get_io_config()
                    pytorch_model = component_model.prepare_session(rank=local_rank)

                    ranked_component_modelproto = OnnxConversion._export_pytorch_model(
                        pytorch_model,
                        dummy_inputs,
                        io_config,
                        pass_config,
                        device,
                        torch_dtype,
                        tempdir,
                    )

                    ranked_models.append(ranked_component_modelproto)

                if len(ranked_models) == 2:
                    ranked_onnx_modelproto = merge_decoders(ranked_models[0], ranked_models[1])
                else:
                    raise RuntimeError("DistributedOnnxModelHandler can handle exactly 2 components.")
            else:
                olive_pytorch_model = input_model.load_model(local_rank)
                dummy_inputs = olive_pytorch_model.get_dummy_inputs()
                io_config = None if pass_config["use_dynamo_exporter"] else olive_pytorch_model.get_io_config()
                pytorch_model = olive_pytorch_model.prepare_session(rank=local_rank)

                ranked_onnx_modelproto = OnnxConversion._export_pytorch_model(
                    pytorch_model,
                    dummy_inputs,
                    io_config,
                    pass_config,
                    device,
                    torch_dtype,
                    tempdir,
                )

            # save the model to the output path
            model_proto_to_olive_model(ranked_onnx_modelproto, output_filepath, pass_config)
        finally:
            restore_tensor_parallel_layers(restore_args)  # pylint: disable=used-before-assignment

        return 1  # Return 1 for success.

    def _convert_distributed_model_on_device(
        self,
        model: DistributedPyTorchModelHandler,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> DistributedOnnxModelHandler:
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

        return DistributedOnnxModelHandler(
            model_path=output_model_path,
            model_name_pattern=DistributedOnnxModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT,
            num_ranks=world_size,
            model_attributes=model.model_attributes,
        )


class OnnxOpVersionConversion(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        # get current models's opset version
        model_proto = model.load_model()
        model_opset_version = model_proto.opset_import[0].version
        if model_opset_version == config["target_opset"]:
            logger.info("Model is already in target opset version %s.", config["target_opset"])
            return model

        output_model_path = resolve_onnx_path(output_model_path)
        model_proto = onnx.version_converter.convert_version(model_proto, config["target_opset"])
        return model_proto_to_olive_model(model_proto, output_model_path, config)
