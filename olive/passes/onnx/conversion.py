# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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
    DistributedHfModelHandler,
    DistributedOnnxModelHandler,
    HfModelHandler,
    ONNXModelHandler,
    PyTorchModelHandler,
)
from olive.model.config import IoConfig
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam, get_user_script_data_config

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

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            **get_external_data_config(),
            "target_opset": PassConfigParam(
                type_=int, default_value=14, description="The version of the default (ai.onnx) opset to target."
            ),
            "use_dynamo_exporter": PassConfigParam(
                type_=bool, default_value=False, description="Whether to use dynamo_export API to export ONNX model."
            ),
            "past_key_value_name": PassConfigParam(
                type_=str,
                default_value="past_key_values",
                description=(
                    "The arguments name to point to past key values. For model loaded from huggingface, "
                    "it is 'past_key_values'. Basically, it is used only when `use_dynamo_exporter` is True."
                ),
            ),
            "device": PassConfigParam(
                type_=str,
                description=(
                    "The device to use for conversion, e.g., 'cuda' or 'cpu'. If not specified, will use 'cpu' for"
                    " PyTorch model and 'cuda' for DistributedHfModel."
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
            "merge_adapter_weights": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to merge adapter weights before conversion. "
                    "After merging, the model structure is consistent with base model. "
                    "That is useful if you cannot run conversion for some fine-tuned "
                    "models with adapter weights"
                ),
            ),
            "save_metadata_for_token_generation": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to save metadata for token generation or not. "
                    "Includes config.json, generation_config.json, and tokenizer related files."
                ),
            ),
        }

    def _run_for_config(
        self,
        model: Union[DistributedHfModelHandler, HfModelHandler, PyTorchModelHandler],
        config: Dict[str, Any],
        output_model_path: str,
    ) -> Union[DistributedOnnxModelHandler, ONNXModelHandler]:
        output_model = self._run_for_config_internal(model, config, output_model_path)

        if isinstance(model, HfModelHandler) and config["save_metadata_for_token_generation"]:
            # output_model can only be an ONNXModelHandler
            output_dir = output_model.change_model_path_to_dir()

            output_model.model_attributes = model_attributes = output_model.model_attributes or {}
            model_attributes["additional_files"] = additional_files = model_attributes.get("additional_files", [])
            # quantization config is already popped from the model and included in model_attributes
            # don't want the information to be saved in metadata (issues with generation config save)
            additional_files.extend(model.save_metadata(str(output_dir), exclude_load_keys=["quantization_config"]))

        return output_model

    def _run_for_config_internal(
        self,
        model: Union[DistributedHfModelHandler, HfModelHandler, PyTorchModelHandler],
        config: Dict[str, Any],
        output_model_path: str,
    ) -> Union[DistributedOnnxModelHandler, ONNXModelHandler]:
        # get the device to use for conversion
        # default to "cpu" for PyTorchModelHandler and "cuda" for DistributedHfModel
        device = config["device"] or "cpu"
        # get the dtype to use for conversion
        torch_dtype = resolve_torch_dtype(config["torch_dtype"]) if config["torch_dtype"] else None
        if torch_dtype == torch.float16 and device == "cpu":
            logger.debug(
                "Converting model to float16 on CPU. This might fail for some models. If the conversion fails or model"
                " is incorrect, try converting the model on GPU or convert in float32 and use"
                " OrtTransformerOptimization/OnnxFloatToFloat16 pass after this pass."
            )

        if isinstance(model, DistributedHfModelHandler):
            if not config["device"]:
                device = "cuda"
            return self._convert_distributed_model_on_device(model, config, output_model_path, device, torch_dtype)

        return self._convert_model_on_device(model, config, output_model_path, device, torch_dtype)

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
        from olive.common.hf.peft import make_export_compatible_peft
        from olive.common.hf.quant import make_export_compatible_quant

        device = torch.device(device)
        use_gpu = device != torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("Converting model on device %s with dtype %s.", device, torch_dtype)
        pytorch_model.to(device)
        dummy_inputs = tensor_data_to_device(dummy_inputs, device)

        if isinstance(pytorch_model, torch.jit.RecursiveScriptModule):
            pytorch_model = TraceModelWrapper(pytorch_model)
        pytorch_model = make_export_compatible_peft(pytorch_model, merge_weights=config["merge_adapter_weights"])
        pytorch_model = make_export_compatible_quant(pytorch_model)
        # cast to dtype, want all modules including lora layers and quant linears in the same dtype
        if torch_dtype:
            pytorch_model = pytorch_model.to(torch_dtype)

        # get input and output names, and dynamic axes
        assert io_config is not None, "Cannot get io_config for the model."
        io_config = validate_config(io_config, IoConfig)

        onnx_model = None
        if config["use_dynamo_exporter"]:
            # Take the "release" version so that dev builds like 2.5.0dev1234 are treated as 2.5.0
            torch_version = version.parse(torch.__version__).release
            # The "legacy dynamo" is the torch.onnx_dynamo_export API
            legacy_dynamo_supported_version = version.parse("2.2.0").release
            # The new "dynamo" api is torch.onnx.export with dynamo=True
            # TODO(#1478): Change 2.6.0 back to 2.5.0 when dynamic_shapes are supported in Olive
            dynamo_supported_version = version.parse("2.6.0").release
            if torch_version < legacy_dynamo_supported_version:
                raise ImportError(
                    f"torch.onnx.dynamo_export is not available for torch version {torch_version}. "
                    "Please upgrade your torch version to 2.5.0 or above."
                )
            from torch._dynamo import config as dynamo_config

            dynamo_config.capture_scalar_outputs = True
            if isinstance(dummy_inputs, dict):
                dummy_kwargs = dummy_inputs
                dummy_inputs = ()
            else:
                dummy_kwargs = {}
                dummy_inputs = tuple(dummy_inputs)

            if torch_version < dynamo_supported_version:
                onnx_program = torch.onnx.dynamo_export(
                    pytorch_model,
                    *dummy_inputs,
                    **dummy_kwargs,
                    export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
                )
                onnx_model = onnx_program.model_proto
            else:
                onnx_program = torch.onnx.export(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                    pytorch_model,
                    dummy_inputs,
                    kwargs=dummy_kwargs,
                    opset_version=config["target_opset"],
                    input_names=io_config.input_names,
                    output_names=io_config.output_names,
                    dynamic_axes=io_config.dynamic_axes,
                    dynamo=True,
                    fallback=True,
                )
                assert onnx_program is not None
                onnx_model = onnx_program.model_proto
        else:
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
                    input_names=io_config.input_names,
                    output_names=io_config.output_names,
                    dynamic_axes=io_config.dynamic_axes,
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
    def _prepare_hf_model(
        model: HfModelHandler, device: str, torch_dtype: Optional[torch.dtype] = None
    ) -> HfModelHandler:
        """Prepare the HfModelHandler for conversion.

        This method handles the following cases:
        1. HfModelHandler with no load kwargs
            - no need to change the model
        2. HfModelHandler with load kwargs
            - update load_kwargs.torch_dtype if torch_dtype is specified
            - if torch_dtype not specified, make sure the load kwargs specify a dtype that is supported for
                conversion on the specified device
            - if quantization_method == "bitsandbytes" and load_in_4bit is True
                - remove quantization config from the load kwargs
                - find quantized modules and add them to the model attributes
                - the onnx model must be quantized using OnnxBnb4Quantization pass after conversion
        """
        from olive.common.hf.peft import is_peft_model

        if not model.load_kwargs:
            return model

        model_attributes = deepcopy(model.model_attributes or {})
        load_kwargs = model.load_kwargs
        model_dtype = load_kwargs.get_torch_dtype()
        new_load_kwargs = deepcopy(load_kwargs.dict())

        if torch_dtype and torch_dtype != model_dtype:
            # if the load kwargs specify a different dtype, update the load kwargs
            logger.debug(
                "Changing torch_dtype in load kwargs from %s to %s.",
                load_kwargs.get_torch_dtype(),
                torch_dtype,
            )
            new_load_kwargs["torch_dtype"] = torch_dtype
            model_attributes["torch_dtype"] = str(torch_dtype).replace("torch.", "")

        if load_kwargs.quantization_method == "bitsandbytes" and load_kwargs.quantization_config["load_in_4bit"]:
            logger.debug(
                "Bitsandbytes 4bit quantization is not supported for conversion. The quantization config is removed"
                " from the load kwargs. Use OnnxBnb4Quantization pass after conversion to quantize the"
                " model."
            )
            new_load_kwargs["quantization_method"] = None
            new_load_kwargs["quantization_config"] = None
            model_attributes["quantization_config"] = load_kwargs.quantization_config
            if "quantized_modules" not in model_attributes:
                # find and add quantized modules to the model attributes
                # the QLoRA pass already adds quantized_modules to the model attributes, so this will not be
                # executed if the model was generated by QLoRA
                quantized_model = model.load_model(cache_model=False)

                # if PeftModel, need to unload adapter before finding quantized modules
                if is_peft_model(quantized_model):
                    quantized_model = quantized_model.unload()

                import bitsandbytes as bnb

                model_attributes["quantized_modules"] = find_submodules(quantized_model, bnb.nn.Linear4bit)

        model_config = model.to_json()["config"]
        model_config["load_kwargs"] = new_load_kwargs
        model_config["model_attributes"] = model_attributes
        return HfModelHandler(**model_config)

    def _convert_model_on_device(
        self,
        model: Union[HfModelHandler, PyTorchModelHandler],
        config: Dict[str, Any],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> ONNXModelHandler:
        """Convert an HfModelHandler or PyTorchModelHandler to an ONNXModelHandler."""
        # prepare the model for conversion
        if isinstance(model, HfModelHandler):
            # optimum export config needs the loaded model to get io_config so we create a new model handler
            # which will be used to load the model and get the io_config
            model = self._prepare_hf_model(model, device, torch_dtype)

        # load the model
        pytorch_model = model.load_model(cache_model=False)
        pytorch_model.eval()

        # get dummy inputs
        dummy_inputs = self._get_dummy_inputs(model, config)
        io_config = model.io_config

        converted_onnx_model = OnnxConversion._export_pytorch_model(
            pytorch_model, dummy_inputs, io_config, config, device, torch_dtype, tempfile.tempdir
        )

        model_attributes = deepcopy(model.model_attributes or {})

        # add split information if present
        split_assignments = model_attributes.get("split_assignments")
        if split_assignments:
            split_assignment_str = ";".join([f"{k}={v}" for k, v in split_assignments.items()])
            onnx.helper.set_model_props(converted_onnx_model, {"split_assignments": split_assignment_str})

        # save the model to the output path and return the model
        output_model_path = resolve_onnx_path(output_model_path)
        output_model = model_proto_to_olive_model(converted_onnx_model, output_model_path, config)
        output_model.model_attributes = deepcopy(model.model_attributes or {})
        return output_model

    @staticmethod
    def _get_dummy_inputs(
        model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any]
    ) -> Union[Dict, Tuple]:
        """Get dummy inputs for the model."""
        return model.get_dummy_inputs(
            filter_hook=(
                model.merge_kv_cache_hook if config["use_dynamo_exporter"] else model.merge_kv_cache_to_tuple_hook
            ),
            filter_hook_kwargs={
                "past_kv_names": config["past_key_value_name"],
            },
        )

    @staticmethod
    def _export_ranked_model(params):
        """Export one rank of a DistributedHfModel to ONNX and save the model to the output path.

        :param params: a tuple of (pass_config, model_config, world_size, device, local_rank, output_dirpath)
            pass_config: the config for the pass
            model_config: the config for the DistributedHfModel
            device: the device to use for conversion
            torch_dtype: the dtype to cast the model to before conversion
            local_rank: the rank of the current process as well as the rank of the model to be converted
            output_dirpath: the path to the directory to save the model. The .onnx model will be saved in this
                directory with the name specified by DistributedOnnxModel.DEFAULT_RANKED_MODEL_NAME_FORMAT
        """
        pass_config, model_config, device, torch_dtype, local_rank, output_dirpath, tempdir = params

        model_type = model_config.get("model_attributes", {}).get("model_type")

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

            input_model = DistributedHfModelHandler(**model_config)

            olive_pytorch_model = input_model.load_model(local_rank)
            dummy_inputs = OnnxConversion._get_dummy_inputs(olive_pytorch_model, pass_config)
            io_config = None if pass_config["use_dynamo_exporter"] else olive_pytorch_model.io_config
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
        model: DistributedHfModelHandler,
        config: Dict[str, Any],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> DistributedOnnxModelHandler:
        """Convert a DistributedHfModel to a DistributedOnnxModel."""
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
        self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path)
        # since external data is saved in a separate file, we need to load the model to get the opset version
        model_proto = onnx.load(model.model_path, load_external_data=False)

        model_opset_version = model_proto.opset_import[0].version
        if model_opset_version == config["target_opset"]:
            logger.info("Model is already in target opset version %s.", config["target_opset"])
            return model

        converted_model_proto = onnx.version_converter.convert_version(model_proto, config["target_opset"])
        # copy the external data of original model to the new model
        dst_init_map = {init.name: init for init in converted_model_proto.graph.initializer}
        for src_init in model_proto.graph.initializer:
            if (
                src_init.name in dst_init_map
                and src_init.HasField("data_location")
                and src_init.data_location == onnx.TensorProto.EXTERNAL
            ):
                dst_init_map[src_init.name].CopyFrom(src_init)
        onnx.external_data_helper.load_external_data_for_model(
            converted_model_proto, str(Path(model.model_path).resolve().parent)
        )
        return model_proto_to_olive_model(converted_model_proto, output_model_path, config)
