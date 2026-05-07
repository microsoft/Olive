# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import logging
import shutil
import tempfile
from pathlib import Path

import onnx
from onnx.onnx_pb import ModelProto

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class NVModelOptGraphSurgery(Pass):
    """Perform graph surgeries on ONNX models using NVIDIA ModelOpt.

    This pass provides a scalable interface to all graph surgery operations
    available in ModelOpt. It uses ModelOpt's run_graph_surgery dispatcher,
    so any new surgery added to ModelOpt is automatically available here
    without code changes.

    Use get_available_surgeries() from modelopt.onnx.graph_surgery to see
    all available surgery types.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "surgery_type": PassConfigParam(
                type_=str,
                required=True,
                description=(
                    "Name of the graph surgery to perform. "
                    "Examples: 'replace-gqa', 'add-cross-kv', 'convert-bf16', 'transpose-dq'. "
                    "Run modelopt.onnx.graph_surgery.get_available_surgeries() to see all options."
                ),
            ),
            "surgery_params": PassConfigParam(
                type_=dict,
                default_value={},
                description=(
                    "Dictionary of surgery-specific parameters. "
                    "These are passed directly to the ModelOpt surgery function as keyword arguments. "
                    "Refer to ModelOpt documentation for each surgery's parameters."
                ),
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

        try:
            from modelopt.onnx.graph_surgery import get_available_surgeries
        except ImportError:
            logger.exception("modelopt is not installed. Install with 'pip install nvidia_modelopt'.")
            return False

        surgery_type = config.surgery_type
        available = get_available_surgeries()
        if surgery_type not in available:
            logger.error("Unknown surgery type: '%s'. Available: %s", surgery_type, available)
            return False

        return True

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        """Run the graph surgery on the model."""
        try:
            from modelopt.onnx.graph_surgery import run_graph_surgery
        except ImportError:
            raise ImportError("modelopt is not installed. Install with 'pip install nvidia_modelopt'.") from None

        surgery_type = config.surgery_type
        surgery_params = dict(config.surgery_params or {})

        logger.info("Starting ModelOpt graph surgery: %s", surgery_type)
        logger.debug("Surgery parameters: %s", surgery_params)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_path = Path(temp_dir) / "input_model.onnx"
                temp_output_path = Path(temp_dir) / "output_model.onnx"

                # Save input model to temp directory
                model_proto = model.load_model()
                onnx.save_model(
                    model_proto,
                    temp_input_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="input_model.onnx_data",
                    size_threshold=1024,
                )

                # Call ModelOpt's unified dispatcher
                result = run_graph_surgery(
                    surgery_name=surgery_type,
                    model_path=temp_input_path,
                    output_path=temp_output_path,
                    **surgery_params,
                )

                # Load modified model (without external data — we'll copy the file separately)
                if isinstance(result, ModelProto):
                    modified_model_proto = result
                    temp_ext_data_file = Path(temp_dir) / "output_model.onnx_data"
                    if temp_ext_data_file.exists():
                        modified_model_proto = onnx.load(temp_output_path, load_external_data=False)
                else:
                    modified_model_proto = onnx.load(temp_output_path, load_external_data=False)

                # Check for external data file
                temp_ext_data_file = Path(temp_dir) / "output_model.onnx_data"
                has_external_data = temp_ext_data_file.exists()

                # Resolve final output path
                output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
                output_dir = Path(output_model_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                output_ext_data_name = f"{Path(output_model_path).name}.data"

                if has_external_data:
                    # Copy external data file while temp dir still exists
                    output_ext_data_path = output_dir / output_ext_data_name
                    logger.info("Copying external data file to %s", output_ext_data_path)
                    shutil.copy2(temp_ext_data_file, str(output_ext_data_path))

                    # Update model references and save
                    from olive.passes.onnx.common import (
                        add_version_metadata_to_model_proto,
                        change_external_data_location,
                    )

                    change_external_data_location(modified_model_proto, output_ext_data_name)
                    modified_model_proto = add_version_metadata_to_model_proto(modified_model_proto)
                    onnx.save_model(modified_model_proto, str(output_model_path))

                    from olive.resource_path import LocalFolder

                    return ONNXModelHandler(
                        model_path=LocalFolder({"path": output_dir}),
                        onnx_file_name=Path(output_model_path).name,
                    )
                else:
                    external_data_config = {
                        "save_as_external_data": True,
                        "all_tensors_to_one_file": True,
                        "external_data_name": output_ext_data_name,
                        "size_threshold": 1024,
                    }
                    return model_proto_to_olive_model(modified_model_proto, output_model_path, external_data_config)

        except Exception:
            logger.exception("An error occurred during graph surgery: %s", surgery_type)
            raise
