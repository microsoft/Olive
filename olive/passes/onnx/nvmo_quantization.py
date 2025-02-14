# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Dict, Type, Union

import onnx
import torch
from onnx import helper
from onnx.onnx_pb import ModelProto
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from olive.common.utils import StrEnumBase
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

    class Precision(StrEnumBase):
        FP8 = "fp8"
        INT8 = "int8"
        INT4 = "int4"

    class Algorithm(StrEnumBase):
        AWQ = "AWQ"

    class Calibration(StrEnumBase):
        AWQ_LITE = "awq_lite"
        AWQ_CLIP = "awq_clip"

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=NVModelOptQuantization.Precision,
                default_value="int4",
                search_defaults=Categorical(["fp8", "int8", "int4"]),
                description="NVModelOpt Quantization mode.",
            ),
            "algorithm": PassConfigParam(
                type_=NVModelOptQuantization.Algorithm,
                default_value="AWQ",
                search_defaults=Categorical(["AWQ"]),
                description="Algorithm of weight only quantization. Supports 'AWQ'.",
            ),
            "calibration": PassConfigParam(
                type_=NVModelOptQuantization.Calibration,
                default_value="awq_clip",
                search_defaults=Categorical(["awq_lite", "awq_clip"]),
                description="Calibration method for weight only quantization. Supports 'awq_lite' and 'awq_clip'.",
            ),
            "tokenizer_dir": PassConfigParam(
                type_=str,
                default_value="",
                description="Tokenizer directory for calibration method.",
            ),
            "random_calib_data": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to use random calibration data instead of actual calibration data.",
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: Type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        # Validate Precision
        if config.precision != NVModelOptQuantization.Precision.INT4:
            logger.error("Only INT4 quantization is supported.")
            return False

        # Validate Algorithm
        if config.algorithm not in [NVModelOptQuantization.Algorithm.AWQ.value]:
            logger.error("Only 'AWQ' algorithm is supported.")
            return False

        # Validate Calibration
        if config.calibration not in [
            NVModelOptQuantization.Calibration.AWQ_LITE.value,
            NVModelOptQuantization.Calibration.AWQ_CLIP.value,
        ]:
            logger.error("Calibration method must be either 'awq_lite' or 'awq_clip'.")
            return False

        random_calib = config.random_calib_data or False
        if not isinstance(random_calib, bool):
            logger.error("'random_calib_data' must be a boolean value.")
            return False

        tokenizer_dir = config.tokenizer_dir or ""
        if not random_calib and not tokenizer_dir:
            logger.error("'tokenizer_dir' must be specified when 'random_calib_data' is False.")
            return False

        # Optional: Validate 'tokenizer_dir' if necessary
        if not config.tokenizer_dir:
            logger.warning("Tokenizer directory 'tokenizer_dir' is not specified.")

        return True

    def initialize_quant_config(self, config: Type[BasePassConfig]) -> Dict[str, Any]:
        # Check if 'tokenizer_dir' is provided and not empty
        random_calib = config.random_calib_data or False
        if not random_calib:
            # Prepare calibration inputs only if tokenizer_dir is specified
            calib_inputs = self.get_calib_inputs(
                dataset_name="cnn",
                model_name=config.tokenizer_dir,
                cache_dir="./cache",
                calib_size=32,
                batch_size=1,
                block_size=512,
                device="cpu",
                use_fp16=True,
                use_buffer_share=False,
                add_past_kv_inputs=True,
                max_calib_rows_to_load=128,
                add_position_ids=True,
            )
        else:
            # If tokenizer_dir is empty, do not prepare calibration inputs
            calib_inputs = None
            logger.debug("No tokenizer directory specified. Skipping calibration input preparation.")

        # Return a dictionary containing necessary configuration for quantization
        return {
            "algorithm": config.algorithm or self.Algorithm.AWQ.value,
            "precision": config.precision or self.Precision.INT4.value,
            "calibration_method": config.calibration or self.Calibration.AWQ_CLIP.value,
            "tokenizer_dir": config.tokenizer_dir or "",
            "calibration_data_reader": calib_inputs,
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

    def quantize_awq(self, model: Union[ModelProto, str], quant_config: Dict[str, Any]) -> ModelProto:
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
        self, model: OliveModelHandler, config: Type[BasePassConfig], output_model_path: str
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

            onnx.save(converted_model_proto, output_model_path)
            logger.debug("Quantized and opset-converted model saved to %s", output_model_path)

            return model_proto_to_olive_model(converted_model_proto, output_model_path, config)

        except Exception:
            logger.exception("An error occurred during quantization and opset conversion")
            raise
