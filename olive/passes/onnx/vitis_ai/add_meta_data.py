import json
import os
from pathlib import Path
from typing import Dict, Optional

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import *
from olive.passes.pass_config import BasePassConfig, PassConfigParam


class VitisAIAddMetaData(Pass):
    """Adds metadata to an ONNX model based on specified model attributes."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "config_meta_data_keys": PassConfigParam(
                type_=list,
                required=False,
                description="List of model attribute keys to extract as metadata",
                default_value=["architectures", "model_type"],
            ),
            "activation_type": PassConfigParam(
                type_=str,
                required=False,
                description="Activation dytpe",
            ),
            "weight_type": PassConfigParam(
                type_=str,
                required=False,
                description="weight dtype",
            ),
            "quant_type": PassConfigParam(type_=str, required=False, description="Quant dtype", default_value="NA"),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        if not hasattr(model, "model_attributes") or not model.model_attributes:
            raise ValueError("Model attributes are missing")

        model_details = model.model_attributes
        config_meta_data_keys = config.config_meta_data_keys

        # Verify required keys exist
        missing_keys = [key for key in config_meta_data_keys if key not in model_details]

        def get_attribute(key: str) -> Optional[str]:
            value = model_details.get(key)
            if isinstance(value, list):
                return ", ".join(map(str, value))
            return str(value) if value is not None else None

        # Prepare metadata
        metadata = {key: get_attribute(key) for key in config_meta_data_keys if get_attribute(key) is not None}

        if config.activation_type:
            metadata["activation_dtype"] = config.activation_type
        if config.weight_type:
            metadata["weight_dtype"] = config.weight_type
        if config.quant_type:
            metadata["quant_type"] = config.quant_type

        if not metadata:
            return model

        try:
            onnx_model = model.load_model()
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}") from e

        existing_metadata = {entry.key: idx for idx, entry in enumerate(onnx_model.metadata_props)}

        for key in metadata:
            if key in existing_metadata:
                del onnx_model.metadata_props[existing_metadata[key]]

        # Add validated metadata
        for key, value in metadata.items():
            entry = onnx_model.metadata_props.add()
            entry.key = key
            entry.value = str(value)

        # Save model
        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.onnx"

        try:
            onnx.save(onnx_model, str(model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to save modified model: {str(e)}") from e

        return ONNXModelHandler(model_path)


class VitisAIAddProviderGenAIConfig(Pass):
    """Add VitisAI provider configuration to GenAI pipeline elements in genai_config.json."""

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec) -> Dict[str, PassConfigParam]:
        return {
            "provider_options": PassConfigParam(
                type_=Dict,
                required=False,
                description="Custom options for VitisAI provider configuration",
                default_value={},
            )
        }

    def _run_for_config(
        self, model: CompositeModelHandler, config: BasePassConfig, output_model_path: str
    ) -> CompositeModelHandler:
        """Main entry point for the pass execution."""
        model_components_list = list(model.model_components)
        model_path = model_components_list[0].model_path
        genai_config_path = Path(model_path).parent / "genai_config.json"
        ouput_genai_config_path = Path(os.path.join(output_model_path, "genai_config.json"))

        if not genai_config_path.exists():
            print("genai-config not present")
            return model

        try:
            updated_config = self._update_genai_config(
                genai_config_path=genai_config_path, provider_options=config.provider_options
            )
            self._write_config(ouput_genai_config_path, updated_config)
        except Exception:
            raise
        return model

    def _update_genai_config(self, genai_config_path: Path, provider_options: Optional[Dict] = None) -> Dict:
        """Update and return the modified GenAI configuration."""
        config = self._read_config(genai_config_path)
        pipeline_config = self._get_pipeline_config(config)

        session_options = self._create_session_options(provider_options=provider_options or {})

        for component in ["context", "iterator"]:
            if component not in pipeline_config:
                raise KeyError(f"Missing required pipeline component: {component}")
            pipeline_config[component]["session_options"] = session_options

        return config

    def _read_config(self, config_path: Path) -> Dict:
        """Read and validate base configuration structure."""
        with config_path.open("r") as f:
            config = json.load(f)

        if not config.get("model", {}).get("decoder", {}).get("pipeline"):
            raise ValueError("Invalid GenAI config structure - missing decoder/pipeline")

        return config

    def _get_pipeline_config(self, config: Dict) -> Dict:
        """Extract and validate pipeline configuration."""
        pipeline = config["model"]["decoder"]["pipeline"]

        if not isinstance(pipeline, list) or len(pipeline) < 1:
            raise ValueError("Invalid pipeline configuration")

        return pipeline[0]

    def _create_session_options(self, provider_options: Dict) -> Dict:
        """Create standardized session options with VitisAI provider."""
        return {
            "log_id": "onnxruntime-genai",
            "provider_options": [{"VitisAI": provider_options}],
            "graph_optimization_level": "ORT_ENABLE_ALL",
        }

    def _write_config(self, config_path: Path, config: Dict):
        """Write updated configuration back to file."""
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
