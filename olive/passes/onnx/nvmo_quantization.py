# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import onnx
from onnx import helper
from onnx.onnx_pb import ModelProto

from olive.common.utils import StrEnumBase
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OliveModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
from olive.strategy.search_parameter import Categorical

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
                required=True,
                description="Tokenizer directory for calibration method.",
            ),
        }

    def validate_search_point(
        self,
        search_point: Dict[str, Any],
        accelerator_spec: AcceleratorSpec,
        with_fixed_value: bool = False,
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})

        # Validate Precision
        if search_point.get("precision") != NVModelOptQuantization.Precision.INT4:
            logger.error("Only INT4 quantization is supported.")
            return False

        # Validate Algorithm
        if search_point.get("algorithm") not in [
            NVModelOptQuantization.Algorithm.AWQ.value,
        ]:
            logger.error("Only 'AWQ' algorithm is supported.")
            return False

        # Validate Calibration
        if search_point.get("calibration") not in [
            NVModelOptQuantization.Calibration.AWQ_LITE.value,
            NVModelOptQuantization.Calibration.AWQ_CLIP.value,
        ]:
            logger.error("Calibration method must be either 'awq_lite' or 'awq_clip'.")
            return False

        # Optional: Validate 'tokenizer_dir' if necessary
        if not search_point.get("tokenizer_dir"):
            logger.warning("Tokenizer directory 'tokenizer_dir' is not specified.")

        return True

    def initialize_quant_config(self, config: Dict[str, Any]):
        """Initialize the quantization configuration by setting up dependencies and calibration data."""
        # Import torch and DataLoader
        import torch
        from torch.utils.data import DataLoader

        self.torch = torch
        self.DataLoader = DataLoader

        # Import datasets
        try:
            from datasets import load_dataset

            self.load_dataset = load_dataset
        except ImportError:
            logger.exception(
                "The 'datasets' library is required but not installed. Please install it using 'pip install datasets'."
            )
            raise ImportError("datasets is not installed. Exiting.") from None

        # Import transformers
        from transformers import AutoConfig, AutoTokenizer

        self.AutoConfig = AutoConfig
        self.AutoTokenizer = AutoTokenizer

        # Determine the device
        device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

        # Prepare calibration inputs
        calib_inputs = self.get_calib_inputs(
            dataset_name="cnn",
            model_name=config["tokenizer_dir"],
            cache_dir="./cache",
            calib_size=32,
            batch_size=1,
            block_size=512,
            device=device,
            use_fp16=True,
            use_buffer_share=False,
            add_past_kv_inputs=True,
            max_calib_rows_to_load=128,
            add_position_ids=True,
        )

        # Return a dictionary containing necessary configuration for quantization
        return {
            "algorithm": config.get("algorithm", self.Algorithm.AWQ.value),
            "precision": config.get("precision", self.Precision.INT4.value),
            "calibration_method": config.get("calibration", self.Calibration.AWQ_CLIP.value),
            "tokenizer_dir": config["tokenizer_dir"],
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
        # Access torch from the instance variable
        torch = self.torch

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
            batch_size, sequence_length = input_ids.shape
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
        auto_config = self.AutoConfig
        auto_tokenizer = self.AutoTokenizer
        load_dataset = self.load_dataset

        config = auto_config.from_pretrained(
            model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=True
        )
        tokenizer = auto_tokenizer.from_pretrained(
            model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=True
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = tokenizer.eos_token

        assert calib_size <= max_calib_rows_to_load, "calib size should be no more than max_calib_rows_to_load"

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

        # Access DataLoader from the instance variable
        data_loader = self.DataLoader

        calib_dataloader_input_ids = data_loader(batch_encoded_input_ids, batch_size=batch_size, shuffle=False)
        calib_dataloader_attention_mask = data_loader(
            batch_encoded_attention_mask, batch_size=batch_size, shuffle=False
        )

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

        logger.info("Starting nvidia_awq quantization...")

        # Prepare calibration inputs
        calib_inputs = quant_config["calibration_data_reader"]

        # Perform quantization using ModelOpt's int4 quantize function
        quantized_model = quantize_int4(
            model,
            calibration_method=quant_config["calibration_method"],
            calibration_data_reader=calib_inputs,
        )

        logger.info("Completed nvidia_awq quantization.")
        return quantized_model

    def convert_opset_to_21(self, model_path: str, output_path: str) -> str:
        """Modify the model's opset to 21 if it's not already.

        Args:
            model_path (str): Path to the original ONNX model.
            output_path (str): Path to save the converted model.

        Returns:
            str: Path to the saved model.

        """
        # Load the original ONNX model
        model = onnx.load(model_path)

        current_opset = {opset.domain: opset.version for opset in model.opset_import}

        default_domain_version = current_opset.get("", 0)
        if default_domain_version >= 21:
            logger.info(
                "Model already uses opset version %s for the default domain. Skip conversion.", default_domain_version
            )
            return model_path  # No conversion needed

        logger.info("Converting model opset from %s to 21.", default_domain_version)

        new_opset_imports = [
            helper.make_opsetid("", 21),  # Default domain with opset version 21
            helper.make_opsetid("com.microsoft", 1),  # Microsoft domain with version 1
        ]

        for domain, version in current_opset.items():
            if domain not in ["", "com.microsoft"]:
                new_opset_imports.append(helper.make_opsetid(domain, version))

        # Create the updated model with new opset imports
        updated_model = onnx.helper.make_model(model.graph, opset_imports=new_opset_imports)

        # Define the external data file path (all tensors saved in one file)
        external_data_path = os.path.basename(output_path) + "_data"

        # Save the updated model with external data
        onnx.save(
            updated_model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path,
        )

        logger.info("Model opset successfully converted to 21 and saved to %s.", output_path)

        # Return the path to the saved model
        return output_path

    def _run_for_config(
        self, model: OliveModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> OliveModelHandler:

        temp_dir = tempfile.mkdtemp(prefix="modelopt_temp_")
        try:
            logger.info("Temporary directory created at %s.", temp_dir)

            temp_model_path = os.path.join(temp_dir, "model.onnx")
            converted_model_path = self.convert_opset_to_21(model.model_path, temp_model_path)
            if converted_model_path == model.model_path:
                logger.info("No opset conversion was necessary.")
            else:
                logger.info("Temporary model saved at %s.", converted_model_path)

            quant_config = self.initialize_quant_config(config)

            quantized_model = self.quantize_awq(
                model=converted_model_path,
                quant_config=quant_config,
            )

            output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
            olive_model = model_proto_to_olive_model(quantized_model, output_model_path, config)
            logger.info("Quantized model saved to %s", output_model_path)

            return olive_model

        finally:
            # Cleanup the temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info("Temporary directory %s has been cleaned up.", temp_dir)

                except Exception as cleanup_exc:
                    logger.warning("Failed to clean up temporary directory %s: %s", temp_dir, cleanup_exc)
