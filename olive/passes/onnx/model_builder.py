# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Export a PyTorch model using the onnxruntime-genai package.
# --------------------------------------------------------------------------
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import onnx
import transformers

from olive.common.utils import IntEnumBase, StrEnumBase
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)


class ModelBuilder(Pass):
    """Converts a Huggingface generative PyTorch model to ONNX model using the Generative AI builder.

    See https://github.com/microsoft/onnxruntime-genai
    """

    class Precision(StrEnumBase):
        FP32 = "fp32"
        FP16 = "fp16"
        INT8 = "int8"
        INT4 = "int4"

    class AccuracyLevel(IntEnumBase):
        fp32 = 1
        fp16 = 2
        bf16 = 3
        int8 = 4

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=ModelBuilder.Precision,
                required=True,
                description="Precision of model.",
            ),
            "metadata_only": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description="Whether to export the model or generate required metadata only.",
            ),
            "search": PassConfigParam(
                type_=Dict[str, Any], required=False, description="Search options to use for generate loop."
            ),
            "int4_block_size": PassConfigParam(
                type_=int,
                required=False,
                description="Specify the block_size for int4 quantization. Acceptable values: 16/32/64/128/256.",
            ),
            "int4_accuracy_level": PassConfigParam(
                type_=ModelBuilder.AccuracyLevel,
                required=False,
                description="Specify the minimum accuracy level for activation of MatMul in int4 quantization.",
            ),
            "exclude_embeds": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description="Remove embedding layer from your ONNX model.",
            ),
            "exclude_lm_head": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description="Remove language modeling head from your ONNX model.",
            ),
            "enable_cuda_graph": PassConfigParam(
                type_=bool,
                default_value=None,  # Explicitly setting to None to differentiate between user intent and default.
                required=False,
                description=(
                    "The model can use CUDA graph capture for CUDA execution provider. "
                    "If enabled, all nodes being placed on the CUDA EP is the prerequisite "
                    "for the CUDA graph to be used correctly."
                ),
            ),
        }

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})
        precision = search_point.get("precision")

        # if device is GPU, but user choose CPU EP, the is_cpu should be True
        if (precision == ModelBuilder.Precision.FP16) and not (
            accelerator_spec.accelerator_type == Device.GPU
            and accelerator_spec.execution_provider != "CPUExecutionProvider"
        ):
            logger.info(
                "FP16 is not supported on CPU. Valid precision + execution"
                "provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, INT4 CPU, INT4 CUDA"
            )
            return False
        return True

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self,
        model: Union[HfModelHandler, ONNXModelHandler],
        config: Dict[str, Any],
        output_model_path: str,
    ) -> ONNXModelHandler:
        from onnxruntime_genai.models.builder import create_model

        precision = config["precision"]
        metadata_only = config["metadata_only"]

        if metadata_only:
            if not isinstance(model, ONNXModelHandler):
                raise ValueError("metadata_only option is available only with ONNXModel as input.")
        elif not isinstance(model, HfModelHandler):
            raise ValueError("model building is available only with HfModel as input.")

        Path(output_model_path).mkdir(parents=True, exist_ok=True)
        output_model_filepath = (
            Path(resolve_onnx_path(output_model_path))
            if not metadata_only
            else Path(resolve_onnx_path(output_model_path, model.onnx_file_name))
        )

        if self.accelerator_spec.execution_provider == "DmlExecutionProvider":
            target_execution_provider = "dml"
        elif self.accelerator_spec.execution_provider == "CUDAExecutionProvider":
            target_execution_provider = "cuda"
        elif self.accelerator_spec.execution_provider == "JsExecutionProvider":
            target_execution_provider = "web"
        else:
            target_execution_provider = "cpu"

        extra_args = {"filename": str(output_model_filepath.name)}
        if metadata_only:
            extra_args["config_only"] = True
            model_path = None
            input_path = str(model.get_resource("model_path"))
        else:
            model_path = model.model_name_or_path
            # provide the model path as input path, model builder uses input_path for quantized models
            input_path = model_path
            if model.adapter_path:
                extra_args["adapter_path"] = model.adapter_path

        if config.get("int4_block_size"):
            if int(config["int4_block_size"]) not in [16, 32, 64, 128, 256]:
                raise ValueError("Invalid int4_block_size. Accepted values: 16/32/64/128/256.")
            extra_args["int4_block_size"] = config["int4_block_size"]

        if config.get("int4_accuracy_level"):
            extra_args["int4_accuracy_level"] = config["int4_accuracy_level"].value

        # args that are only checked for presence, not value
        for arg in ["exclude_embeds", "exclude_lm_head"]:
            if config[arg]:
                extra_args[arg] = True

        # args that are checked for presence and value (if present)
        for arg in ["enable_cuda_graph"]:
            if config[arg] is not None:
                extra_args[arg] = "1" if config[arg] else "0"

        model_attributes = copy.deepcopy(model.model_attributes or {})

        try:
            create_model(
                model_name=model_path,
                input_path=input_path,
                output_dir=str(output_model_filepath.parent),
                precision=precision,
                execution_provider=target_execution_provider,
                # model builder uses the cache_dir both as hf cache and also to store intermediate files
                # not ideal, but we can't change this without changing the model builder
                cache_dir=transformers.utils.TRANSFORMERS_CACHE,
                **extra_args,
            )

            # add split information if present
            split_assignments = model_attributes.get("split_assignments")
            if split_assignments:
                # NOTE: currently the model builder renames modules to it's own naming convention
                # so the assignments for the renamed modules won't match
                split_assignment_str = ";".join([f"{k}={v}" for k, v in split_assignments.items()])

                # load the model and set the split_assignments as model properties
                # without the external data so that they can be used as is with the resaved model
                model_proto = onnx.load(output_model_filepath, load_external_data=False)
                onnx.helper.set_model_props(model_proto, {"split_assignments": split_assignment_str})
                onnx.save(model_proto, output_model_filepath)
        except Exception:
            # if model building fails, clean up the intermediate files in the cache_dir
            cache_dir = Path(transformers.utils.TRANSFORMERS_CACHE)
            if cache_dir.is_dir():
                for file in cache_dir.iterdir():
                    if file.suffix == ".bin":
                        file.unlink()
            raise

        # Override default search options with ones from user config
        genai_config_filepath = str(output_model_filepath.parent / "genai_config.json")
        with open(genai_config_filepath) as istrm:
            genai_config = json.load(istrm)

        genai_config["search"] = {**(genai_config.get("search") or {}), **(config.get("search") or {})}

        with open(genai_config_filepath, "w") as ostrm:
            json.dump(genai_config, ostrm, indent=4)

        # Save HfModel config
        if isinstance(model, HfModelHandler):
            model.get_hf_model_config().save_pretrained(str(output_model_filepath.parent))

        # add additional files generated by model builder to model_attributes
        model_attributes["generative"] = True
        additional_files = model_attributes.get("additional_files") or []
        if metadata_only:
            # add genai_config.json to additional_files since the model_builder creates copy of the other files
            # in the output directory leading to duplicate files in the additional_files list
            model_attributes["additional_files"] = [
                *additional_files,
                str(output_model_filepath.parent / "genai_config.json"),
            ]

            if isinstance(model, HfModelHandler):
                model_attributes["additional_files"].append(str(output_model_filepath.parent / "config.json"))
        else:
            model_attributes["additional_files"] = sorted(
                set(additional_files)
                # all files in the output directory except the model and model.data files
                | {str(fp) for fp in output_model_filepath.parent.iterdir()}
                - {str(output_model_filepath), str(output_model_filepath) + ".data"}
            )

        if metadata_only:
            output_model = copy.copy(model)
            output_model.model_attributes = model_attributes
        else:
            output_model = ONNXModelHandler(
                output_model_filepath.parent,
                onnx_file_name=output_model_filepath.name,
                model_attributes=model_attributes,
            )

        return output_model
