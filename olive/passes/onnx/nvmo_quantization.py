# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

from olive.common.config_utils import validate_config
from olive.common.utils import StrEnumBase
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OliveModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
from olive.strategy.search_parameter import Categorical

logger = logging.getLogger(__name__)
import onnx
from onnx import helper
import os
import shutil  # Import shutil for directory cleanup
import tempfile  # Import tempfile for creating unique temp directories


# static quantization specific config
_dataloader_config = {
    "data_config": PassConfigParam(
        type_=Union[DataConfig, Dict],
        required=True,
        description="Data config to load data for computing latency.",
    ),
}


# Function to modify the model's opset to 21 if it's not already
def convert_opset_to_21(model_path: str, output_path: str) -> str:
    # Load the original ONNX model
    model = onnx.load(model_path)

    # Check the current opset version
    current_opset = {opset.domain: opset.version for opset in model.opset_import}
    logger.debug(f"Current opset imports: {current_opset}")

    # Determine if the default domain has opset version 21
    default_domain_version = current_opset.get("", 0)
    if default_domain_version >= 21:
        logger.info(
            f"Model already uses opset version {default_domain_version} for the default domain. Skipping conversion."
        )
        return model_path  # No conversion needed

    # If not, proceed to convert to opset 21
    logger.info(f"Converting model opset from {default_domain_version} to 21.")

    # Create new opset imports with version 21
    new_opset_imports = [
        helper.make_opsetid("", 21),  # Default domain with opset version 21
        helper.make_opsetid("com.microsoft", 1),  # Microsoft domain with version 1
    ]

    # Optionally, retain other existing opset imports
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

    logger.info(f"Model opset successfully converted to 21 and saved to {output_path}.")

    # Return the path to the saved model
    return output_path


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
                searchable_values=Categorical(["fp8", "int8", "int4"]),
                description="NVModelOpt Quantization mode.",
            ),
            "algorithm": PassConfigParam(
                type_=NVModelOptQuantization.Algorithm,
                default_value="AWQ",
                searchable_values=Categorical(["AWQ"]),
                description="Algorithm of weight only quantization. Support 'AWQ'.",
            ),
            "calibration": PassConfigParam(
                type_=NVModelOptQuantization.Calibration,
                default_value="awq_clip",
                searchable_values=Categorical(["awq_lite", "awq_clip"]),
                description="Calibration method for weight only quantization. Supports 'awq_lite' and 'awq_clip'.",
            ),
            "tokenizer_dir": PassConfigParam(
                type_=str,
                required=True,
                default_value="",  # Updated default value
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

        # Optional: Validate 'hf' if necessary
        if not search_point.get("hf"):
            logger.warning("Tokenizer directory 'hf' is not specified.")

        return True

    def _run_for_config(
        self, model: OliveModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> OliveModelHandler:
        try:
            from onnxruntime.quantization.matmul_4bits_quantizer import (
                NVAWQWeightOnlyQuantConfig,
                QuantFormat,
                MatMul4BitsQuantizer,
            )
        except ImportError as exc:
            raise ImportError(
                "Please install `olive-ai[nvmo]` or `nvidia-modelopt[onnx]` to use INT4 AWQ quantization!"
            ) from exc

        # Step 1: Convert model's opset to 21 if necessary and save temporarily
        temp_dir = tempfile.mkdtemp(prefix="modelopt_temp_")  # Create a unique temp directory
        try:
            logger.info(f"Temporary directory created at {temp_dir}.")

            temp_model_path = os.path.join(temp_dir, "model.onnx")
            converted_model_path = convert_opset_to_21(model.model_path, temp_model_path)
            if converted_model_path == model.model_path:
                logger.info("No opset conversion was necessary.")
            else:
                logger.info(f"Temporary model saved at {converted_model_path}.")

            # Step 2: Quantize the model
            block_size = 128
            is_symmetric = False
            accuracy_level = None
            calibration_method = config.get("calibration", NVModelOptQuantization.Calibration.AWQ_CLIP.value)
            logger.info(f"Using calibration method: {calibration_method}")

            quant_config = NVAWQWeightOnlyQuantConfig(
                tokenizer_dir=config["tokenizer_dir"],
                calibration_method=calibration_method,
            )

            quant = MatMul4BitsQuantizer(
                model=converted_model_path,
                algo_config=quant_config,
                block_size=block_size,
                is_symmetric=is_symmetric,
                accuracy_level=accuracy_level,
                nodes_to_exclude=[],
                quant_format=QuantFormat.QDQ,
            )
            quant.process()
            q_model = quant.model.model
            logger.info("Model quantization completed successfully.")

            # Step 3: Save the quantized model to the output path
            output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
            olive_model = model_proto_to_olive_model(q_model, output_model_path, config)
            logger.info(f"Quantized model saved to {output_model_path}")

            return olive_model

        finally:
            # Cleanup the temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Temporary directory {temp_dir} has been cleaned up.")
                except Exception as cleanup_exc:
                    logger.warning(
                        f"Failed to clean up temporary directory {temp_dir}: {cleanup_exc}"
                    )
