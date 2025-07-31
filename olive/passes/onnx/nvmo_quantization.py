# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Union

import onnxruntime as ort
import torch
from onnx import helper
from onnx.onnx_pb import ModelProto
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from olive.common.utils import StrEnumBase
from olive.constants import Precision, QuantAlgorithm
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OliveModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.search.search_parameter import Categorical

logger = logging.getLogger(__name__)


class NVModelOptQuantization(Pass):
    """Quantize ONNX model with Nvidia-ModelOpt."""

    DEFAULT_SETTINGS = MappingProxyType(
        {
            "dataset": "cnn",
            "calib_size": 32,
            "batch_size": 1,
            "use_fp16_calib_data": True,
            "add_position_ids": True,
            "int4_block_size": 128,
        }
    )

    class CalibrationMethod(StrEnumBase):
        AWQ_LITE = "awq_lite"
        AWQ_CLIP = "awq_clip"
        RTN = "rtn"
        RTN_DQ = "rtn_dq"

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=Precision,
                default_value=Precision.INT4,
                search_defaults=Categorical([Precision.FP8, Precision.INT8, Precision.INT4]),
                description="NVModelOpt Quantization mode.",
            ),
            "algorithm": PassConfigParam(
                type_=QuantAlgorithm,
                default_value=QuantAlgorithm.AWQ,
                search_defaults=Categorical([QuantAlgorithm.AWQ, QuantAlgorithm.RTN]),
                description="Algorithm of weight only quantization. Supports 'AWQ'.",
            ),
            "calibration_method": PassConfigParam(
                type_=NVModelOptQuantization.CalibrationMethod,
                default_value=NVModelOptQuantization.CalibrationMethod.AWQ_LITE,
                search_defaults=Categorical(list(NVModelOptQuantization.CalibrationMethod)),
                description="""
                Calibration method to be used for quantization.
                Choose from 'awq_lite', 'awq_clip', 'rtn', 'rtn_dq'
                """,
            ),
            "calibration_providers": PassConfigParam(
                type_=list,
                default_value=None,
                description="""
                List of execution providers to run the session during calibration.
                Choose from 'NvTensorRtRtx', 'cuda', 'dml' and 'cpu'.
                Default is None which uses from available providers.
                """,
            ),
            "tokenizer_dir": PassConfigParam(
                type_=str,
                default_value="",
                description="Tokenizer directory for calibration method.",
            ),
            "use_random_calibration_data": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to use random calibration data instead of actual calibration data.",
            ),
            "calibration_params": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value={},
                description="""
                Details about settings to be used for calibration e.g. calibration dataset,
                 samples size etc.
                """,
            ),
            "int4_block_size": PassConfigParam(
                type_=int,
                default_value=cls.DEFAULT_SETTINGS["int4_block_size"],
                description="Block size to use for INT4 weight-only quantization",
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=["/lm_head"],
                description="""
                List of nodes to exclude from quantization. By default, lm-head is excluded
                in quantization.
                """,
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        # Validate Precision
        if config.precision != Precision.INT4:
            logger.error("Only INT4 quantization is supported.")
            return False

        # Validate Algorithm
        if config.algorithm not in [QuantAlgorithm.AWQ, QuantAlgorithm.RTN]:
            logger.error("Only 'AWQ' and 'RTN' algorithms are supported.")
            return False

        # Validate Calibration
        if config.calibration_method not in list(NVModelOptQuantization.CalibrationMethod):
            logger.error("Calibration method must be from ['awq_lite', 'awq_clip', 'rtn', 'rtn_dq'].")
            return False

        use_random_calibration_data = config.use_random_calibration_data
        if not isinstance(use_random_calibration_data, bool):
            logger.error("'use_random_calibration_data' must be a boolean value.")
            return False

        tokenizer_dir = config.tokenizer_dir or ""
        if not use_random_calibration_data and not tokenizer_dir:
            logger.error("'tokenizer_dir' must be specified when 'use_random_calibration_data' is False.")
            return False

        # Optional: Validate 'tokenizer_dir' if necessary
        if not config.tokenizer_dir:
            logger.warning("Tokenizer directory 'tokenizer_dir' is not specified.")

        return True

    @staticmethod
    def get_execution_providers():
        available_eps = ort.get_available_providers()
        ep_list = None
        if "NvTensorRTRTXExecutionProvider" in available_eps:
            ep_list = ["NvTensorRtRtx", "cpu"]
        elif "CUDAExecutionProvider" in available_eps:
            ep_list = ["cuda", "cpu"]
        elif "DmlExecutionProvider" in available_eps:
            ep_list = ["dml", "cpu"]
        else:
            logger.warning("No execution provider is detected for NVIDIA GPU acceleration.")
            logger.warning("Available EPs = %s, falling back to CPU EP.", available_eps)
            ep_list = ["cpu"]
        return ep_list

    def prepare_input_shapes_string(self, batch_size, seq_len, past_seq_len, num_layers, num_kv_heads, head_dim):
        shapes = ""

        shapes += f"input_ids:{batch_size}x{seq_len}"
        shapes += f",attention_mask:{batch_size}x{seq_len}"

        for i in range(num_layers):
            key_name = f"past_key_values.{i}.key"
            value_name = f"past_key_values.{i}.value"
            shapes += f",{key_name}:{batch_size}x{num_kv_heads}x{past_seq_len}x{head_dim}"
            shapes += f",{value_name}:{batch_size}x{num_kv_heads}x{past_seq_len}x{head_dim}"

        return shapes

    def get_input_shapes_profile(self, model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)

        head_dim = config.hidden_size // config.num_attention_heads
        if hasattr(config, "head_dim") and config.head_dim is not None:
            head_dim = config.head_dim
        num_kv_heads = config.num_key_value_heads
        num_layers = config.num_hidden_layers

        min_shapes = self.prepare_input_shapes_string(1, 1, 0, num_layers, num_kv_heads, head_dim)
        max_shapes = self.prepare_input_shapes_string(1, 1024, 1024, num_layers, num_kv_heads, head_dim)
        opt_shapes = self.prepare_input_shapes_string(1, 512, 512, num_layers, num_kv_heads, head_dim)

        return min_shapes, max_shapes, opt_shapes

    def make_input_shapes_profile_for_ep_list(self, ep_list, config):
        # Input-shapes-profile will be used in provider-options for ORT session creation.
        # Provider options (even if {}) are needed for all EPs when we provide for any one of them.
        # Using empty shapes_profile for non-NvTensorRtRtx EPs.
        input_shapes_profile_sequence = []
        for ep in ep_list:
            if ep == "NvTensorRtRtx":
                min_shapes, max_shapes, opt_shapes = self.get_input_shapes_profile(config.tokenizer_dir)
                input_shapes_profile = {
                    "nv_profile_min_shapes": min_shapes,
                    "nv_profile_max_shapes": max_shapes,
                    "nv_profile_opt_shapes": opt_shapes,
                }
                input_shapes_profile_sequence.append(input_shapes_profile)
            else:
                input_shapes_profile_sequence.append({})

        return input_shapes_profile_sequence

    def get_calibration_param(self, calibration_params_config, param_name):
        if not calibration_params_config or (param_name not in calibration_params_config):
            return self.DEFAULT_SETTINGS[param_name]
        else:
            return calibration_params_config[param_name]

    def initialize_quant_config(self, config: type[BasePassConfig]) -> dict[str, Any]:
        random_calib = config.use_random_calibration_data
        if not random_calib and (config.algorithm != QuantAlgorithm.RTN):
            calib_inputs = self.get_calib_inputs(
                dataset_name=self.get_calibration_param(config.calibration_params, "dataset"),
                model_name=config.tokenizer_dir,
                cache_dir="./cache",
                calib_size=self.get_calibration_param(config.calibration_params, "calib_size"),
                batch_size=self.get_calibration_param(config.calibration_params, "batch_size"),
                block_size=512,
                device="cpu",
                use_fp16=self.get_calibration_param(config.calibration_params, "use_fp16_calib_data"),
                use_buffer_share=False,
                add_past_kv_inputs=True,
                max_calib_rows_to_load=128,
                add_position_ids=self.get_calibration_param(config.calibration_params, "add_position_ids"),
            )
        else:
            calib_inputs = None
            logger.warning("Not providing calibration data for quantization.")

        ep_list = config.calibration_providers or NVModelOptQuantization.get_execution_providers()

        input_shapes_profile = None
        if "NvTensorRtRtx" in ep_list and (config.algorithm != QuantAlgorithm.RTN):
            # NvTensorRtRtx EP uses (min, max, opt) profile for dynamic shapes in the model's inputs.
            input_shapes_profile = self.make_input_shapes_profile_for_ep_list(ep_list, config)

        logger.debug("===== Quantization Settings =====")
        logger.debug(
            "algo=%s, precision=%s, calib-method=%s",
            config.algorithm,
            config.precision,
            config.calibration_method,
        )
        logger.debug("tokenizer-dir=%s, int4-block-size=%d", config.tokenizer_dir, config.int4_block_size)
        logger.debug("dataset=%s", self.get_calibration_param(config.calibration_params, "dataset"))
        logger.debug("calib-size=%d", self.get_calibration_param(config.calibration_params, "calib_size"))
        logger.debug("batch-size=%d", self.get_calibration_param(config.calibration_params, "batch_size"))
        logger.debug(
            "use_fp16_calib_data=%s", self.get_calibration_param(config.calibration_params, "use_fp16_calib_data")
        )
        logger.debug("add_position_ids=%s", self.get_calibration_param(config.calibration_params, "add_position_ids"))
        logger.debug("calibration_eps=%s", ep_list)
        logger.debug("nodes-to-exclude=%s", config.nodes_to_exclude)
        logger.debug("input_shapes_profile is None? = %s", input_shapes_profile is None)
        logger.debug("=============================")

        # Return a dictionary containing necessary configuration for quantization
        return {
            "algorithm": config.algorithm,
            "precision": config.precision,
            "calibration_method": config.calibration_method,
            "tokenizer_dir": config.tokenizer_dir or "",
            "calibration_data_reader": calib_inputs,
            "calibration_eps": ep_list,
            "int4_block_size": config.int4_block_size,
            "nodes_to_exclude": config.nodes_to_exclude,
            "input_shapes_profile": input_shapes_profile,
        }

    def make_model_input(
        self,
        config,
        input_ids_arg,
        attention_mask_arg,
        add_past_kv_inputs,
        device,
        use_fp16,
        use_buffer_share,
        add_position_ids,
    ):
        input_ids = input_ids_arg
        attention_mask = attention_mask_arg

        if isinstance(input_ids_arg, list):
            input_ids = torch.tensor(input_ids_arg, device=device, dtype=torch.int64)
            attention_mask = torch.tensor(attention_mask_arg, device=device, dtype=torch.int64)

        inputs = {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
        }

        if add_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            inputs["position_ids"] = position_ids.contiguous()

        if add_past_kv_inputs:
            torch_dtype = torch.float16 if use_fp16 else torch.float32
            batch_size, _ = input_ids.shape
            max_sequence_length = config.max_position_embeddings
            num_heads, head_size = (
                config.num_key_value_heads,
                config.hidden_size // config.num_attention_heads,
            )

            if hasattr(config, "head_dim") and config.head_dim is not None:
                head_size = config.head_dim

            for i in range(config.num_hidden_layers):
                past_key = torch.zeros(
                    batch_size,
                    num_heads,
                    max_sequence_length if use_buffer_share else 0,
                    head_size,
                    device=device,
                    dtype=torch_dtype,
                )
                past_value = torch.zeros(
                    batch_size,
                    num_heads,
                    max_sequence_length if use_buffer_share else 0,
                    head_size,
                    device=device,
                    dtype=torch_dtype,
                )
                inputs.update(
                    {
                        f"past_key_values.{i}.key": past_key.contiguous(),
                        f"past_key_values.{i}.value": past_value.contiguous(),
                    }
                )

        return inputs

    def get_calib_inputs(
        self,
        dataset_name,
        model_name,
        cache_dir,
        calib_size,
        batch_size,
        block_size,
        device,
        use_fp16,
        use_buffer_share,
        add_past_kv_inputs,
        max_calib_rows_to_load,
        add_position_ids,
    ):
        # Access transformers and datasets from the instance variables

        config = AutoConfig.from_pretrained(
            model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=True
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = tokenizer.eos_token

        assert calib_size <= max_calib_rows_to_load, "calib size should be no more than max_calib_rows_to_load"

        from datasets import load_dataset

        if "cnn" in dataset_name:
            dataset2 = load_dataset("cnn_dailymail", name="3.0.0", split="train").select(range(max_calib_rows_to_load))
            column = "article"
        elif "pile" in dataset_name:
            dataset2 = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            column = "text"
        else:
            raise ValueError(f'dataset "{dataset_name}" not supported')

        dataset2 = dataset2[column][:calib_size]
        batch_encoded = tokenizer.batch_encode_plus(
            dataset2, return_tensors="pt", padding=True, truncation=True, max_length=block_size
        )
        batch_encoded = batch_encoded.to(device)
        batch_encoded_input_ids = batch_encoded["input_ids"]
        batch_encoded_attention_mask = batch_encoded["attention_mask"]

        calib_dataloader_input_ids = DataLoader(batch_encoded_input_ids, batch_size=batch_size, shuffle=False)
        calib_dataloader_attention_mask = DataLoader(batch_encoded_attention_mask, batch_size=batch_size, shuffle=False)

        if len(calib_dataloader_input_ids.dataset) != len(calib_dataloader_attention_mask.dataset):
            raise ValueError(
                f"Mismatch in dataset len: calib_dataloader_input_ids has {len(calib_dataloader_input_ids.dataset)} "
                f"items, calib_dataloader_attention_mask has {len(calib_dataloader_attention_mask.dataset)} items."
            )

        if len(calib_dataloader_input_ids) != len(calib_dataloader_attention_mask):
            raise ValueError(
                f"Mismatch in dataloader lengths: calib_dataloader_input_ids has {len(calib_dataloader_input_ids)} "
                f"items, while calib_dataloader_attention_mask has {len(calib_dataloader_attention_mask)} items."
            )

        number_of_batched_samples = calib_size // batch_size

        batched_input_ids = []
        for idx, data in enumerate(calib_dataloader_input_ids):
            batched_input_ids.append(data)
            if idx == (number_of_batched_samples - 1):
                break

        batched_attention_mask = []
        for idx, data in enumerate(calib_dataloader_attention_mask):
            batched_attention_mask.append(data)
            if idx == (number_of_batched_samples - 1):
                break

        batched_inputs_list = []
        for i in range(number_of_batched_samples):
            input_ids = batched_input_ids[i]
            attention_mask = batched_attention_mask[i]

            inputs = self.make_model_input(
                config,
                input_ids,
                attention_mask,
                add_past_kv_inputs,
                device,
                use_fp16,
                use_buffer_share,
                add_position_ids,
            )
            inputs = {input_name: torch_tensor.cpu().numpy() for input_name, torch_tensor in inputs.items()}
            batched_inputs_list.append(inputs)

        return batched_inputs_list

    def quantize_awq(self, model: Union[ModelProto, str], quant_config: dict[str, Any]) -> ModelProto:
        """Perform nvidia_awq quantization using ModelOpt's int4 quantize function.

        Args:
            model (ModelProto | str): The ONNX model or path to the model to quantize.
            quant_config (Dict[str, Any]): Configuration dictionary for quantization.

        Returns:
            ModelProto: The quantized ONNX model.

        """
        try:
            from modelopt.onnx.quantization.int4 import quantize as quantize_int4
        except ImportError:
            logger.exception(
                "Please ensure that 'modelopt' package is installed. Install it with 'pip install nvidia_modelopt'."
            )
            raise ImportError(
                "modelopt is not installed. Please install it using 'pip install nvidia_modelopt'. Exiting."
            ) from None

        logger.debug("Starting nvidia_awq quantization...")

        # Prepare calibration inputs
        calib_inputs = quant_config["calibration_data_reader"]

        # Perform quantization using ModelOpt's int4 quantize function
        quantized_model = quantize_int4(
            model,
            calibration_method=quant_config["calibration_method"],
            calibration_data_reader=calib_inputs,
            calibration_eps=quant_config["calibration_eps"],
            block_size=quant_config["int4_block_size"],
            nodes_to_exclude=quant_config["nodes_to_exclude"],
            input_shapes_profile=quant_config["input_shapes_profile"],
        )

        logger.debug("Completed nvidia_awq quantization.")
        return quantized_model

    def convert_opset_to_21_proto(self, model_proto: ModelProto) -> ModelProto:
        """Modify the model's opset to 21 if it's not already, operating on a ModelProto.

        Args:
            model_proto (ModelProto): The ONNX model proto to modify.

        Returns:
            ModelProto: The updated ONNX model proto with opset version 21.

        """
        current_opset = {opset.domain: opset.version for opset in model_proto.opset_import}

        default_domain_version = current_opset.get("", 0)
        if default_domain_version >= 21:
            logger.debug(
                "Model already uses opset version %s for the default domain. Skip conversion.", default_domain_version
            )
            return model_proto  # No conversion needed

        new_opset_imports = [
            helper.make_opsetid("", 21),  # Default domain with opset version 21
            helper.make_opsetid("com.microsoft", 1),  # Microsoft domain with version 1
        ]

        for domain, version in current_opset.items():
            if domain not in ["", "com.microsoft"]:
                new_opset_imports.append(helper.make_opsetid(domain, version))

        # Update the model's opset imports
        model_proto.ClearField("opset_import")
        model_proto.opset_import.extend(new_opset_imports)

        logger.debug("Model opset successfully converted to 21.")

        return model_proto

    def _run_for_config(
        self, model: OliveModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> OliveModelHandler:
        try:
            logger.debug("Loading the original ONNX model from %s.", model.model_path)
            quant_config = self.initialize_quant_config(config)

            # Perform quantization
            quantized_model_proto = self.quantize_awq(
                model=model.model_path,
                quant_config=quant_config,
            )

            # Convert opset to 21 if required
            converted_model_proto = self.convert_opset_to_21_proto(quantized_model_proto)

            output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

            logger.debug("Quantized and opset-converted model will be saved to %s", output_model_path)

            external_data_config = {
                "save_as_external_data": True,
                "all_tensors_to_one_file": True,
                "external_data_name": os.path.basename(output_model_path) + "_data",
                "size_threshold": 1024,
            }

            return model_proto_to_olive_model(converted_model_proto, output_model_path, external_data_config)

        except Exception:
            logger.exception("An error occurred during quantization and opset conversion")
            raise
