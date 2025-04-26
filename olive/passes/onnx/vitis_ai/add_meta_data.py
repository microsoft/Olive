from pathlib import Path
from typing import Dict, List, Optional, Union

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import process_llm_pipeline, resave_model
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
        self, model: Union[ONNXModelHandler, CompositeModelHandler], config: BasePassConfig, output_model_path: str
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        if not hasattr(model, "model_attributes") or not model.model_attributes:
            raise ValueError("Model attributes are missing")

        model_details = model.model_attributes
        config_meta_data_keys = config.config_meta_data_keys

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

        if isinstance(model, ONNXModelHandler):
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

    def _create_session_options(self, provider_options: Dict) -> Dict:
        """Create standardized session options with VitisAI provider."""
        return {
            "log_id": "onnxruntime-genai",
            "provider_options": [{"VitisAI": provider_options}],
            "graph_optimization_level": "ORT_ENABLE_ALL",
        }

    def _run_for_config(
        self, model: CompositeModelHandler, config: BasePassConfig, output_model_path: str
    ) -> CompositeModelHandler:
        # Validate input model
        assert isinstance(model, CompositeModelHandler), "Requires CompositeModelHandler"
        model_components = list(model.model_components)
        assert all(isinstance(m, ONNXModelHandler) for m in model_components), "All components must be ONNX"
        assert len(model_components) >= 3, "Need at least embedding, transformer, and lm_head components"

        # Configure session and output paths
        session_options = self._create_session_options(provider_options=config.provider_options or {})
        output_path = Path(output_model_path).with_suffix("")

        # Define pipeline configuration using model component metadata
        pipeline = self._create_pipeline_config(model)

        # Process context and iterator
        return self._process_pipeline(model, pipeline, output_path, session_options)

    def _create_pipeline_config(self, model: CompositeModelHandler) -> Dict:
        """Create dynamic pipeline configuration based on component metadata"""
        return {
            "embeddings": model.model_component_names[0],
            "context": self._get_component_group(model, "context"),
            "iterator": self._get_component_group(model, "iterator"),
            "lm_head": model.model_component_names[-1],
        }

    def _get_component_group(self, model: CompositeModelHandler, pattern: str) -> List[str]:
        """Identify components by naming pattern or position"""
        # Example implementation - adjust based on your actual naming conventions
        return [name for name in model.model_component_names if pattern in name.lower()]

    def _process_pipeline(
        self, model: CompositeModelHandler, pipeline: Dict[str, List[str]], output_dir: Path, session_options: dict
    ) -> CompositeModelHandler:
        """Generic pipeline processor with multiple component support"""

        def component_processor(model_components: Dict, llm_pipeline: Dict, output_dir: Path) -> Dict:
            """Handle multiple context/iterator components"""
            processed = {"context": {}, "iterator": {}}

            for group in ["context", "iterator"]:
                for idx, comp_name in enumerate(llm_pipeline[group]):
                    # Generate filename preserving original component identity
                    base_name = (
                        f"{group}_{comp_name.split('_')[-1]}" if "_" in comp_name else group
                    )  ## little tricky, TODO: Revisit this logic
                    fname = f"{base_name}_{idx}.onnx"
                    dest_path = output_dir / fname

                    # Process and save component
                    resave_model(model_components[comp_name].model_path, dest_path, force_external_data=True)

                    processed[group][fname] = ONNXModelHandler(model_path=output_dir, onnx_file_name=fname)

            return processed

        return process_llm_pipeline(
            model, pipeline, component_processor, output_dir, group_session_options=session_options
        )
