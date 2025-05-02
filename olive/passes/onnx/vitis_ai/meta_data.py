#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Dict, List, Optional, Union

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model, process_llm_pipeline, resave_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam


class VitisAIAddMetaData(Pass):
    """Adds metadata to an ONNX model based on specified model attributes."""

    _accepts_composite_model = True

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

    def _add_meta_data(self, model: ONNXModelHandler, metadata: Dict):
        """Load the model and add the meta data to the model proto"""
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
        return onnx_model

    def _run_for_config(
        self, model: Union[ONNXModelHandler, CompositeModelHandler], config: BasePassConfig, output_model_path: str
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        if not (isinstance(model, ONNXModelHandler) or isinstance(model, CompositeModelHandler)):
            raise NotImplementedError

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
            onnx_model = self._add_meta_data(model, metadata)  # type: ignore
            output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
            return model_proto_to_olive_model(
                onnx_model, output_model_path, config
            )  # used in transformer optimizations, we can pass model save options in config

        elif isinstance(model, CompositeModelHandler):
            pipeline = self._create_pipeline_config(model)

            def process_context_iterator(component_models, llm_pipeline, output_dir):
                """Metadata will only be added to context and iterator models"""
                new_groups = {
                    "context": {},
                    "iterator": {},
                }
                is_split = len(llm_pipeline["context"]) > 1
                for idx, component_name in enumerate(llm_pipeline["context"]):
                    suffix = f"_{idx}" if is_split else ""

                    # resave the model with external data
                    intermediate_model_path = output_dir / f"transformer{suffix}.onnx"
                    resave_model(
                        component_models[component_name].model_path, intermediate_model_path, force_external_data=True
                    )

                    for pipeline_key in ["context", "iterator"]:
                        new_component_name = f"{pipeline_key}{suffix}"
                        component_proto = onnx.load(intermediate_model_path, load_external_data=False)
                        existing_metadata = {entry.key: idx for idx, entry in enumerate(component_proto.metadata_props)}
                        for key in metadata:
                            if key in existing_metadata:
                                del component_proto.metadata_props[existing_metadata[key]]
                        # Add validated metadata
                        for key, value in metadata.items():
                            entry = component_proto.metadata_props.add()
                            entry.key = key
                            entry.value = str(value)

                        # save the model with fixed shapes
                        component_model_path = output_dir / f"{new_component_name}.onnx"
                        onnx.save_model(component_proto, component_model_path)
                        new_groups[pipeline_key][new_component_name] = ONNXModelHandler(
                            model_path=output_dir, onnx_file_name=component_model_path.name
                        )

                    # delete the intermediate model
                    intermediate_model_path.unlink()

                return new_groups

            return process_llm_pipeline(model, pipeline, process_context_iterator, output_model_path)


class VitisAIUpdateGenAIProviderOpts(Pass):
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
        # TODO: Add option to load Session option from the config aswell
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
        return [name for name in model.model_component_names if pattern in name.lower()]

    def _process_pipeline(
        self, model: CompositeModelHandler, pipeline: Dict[str, List[str]], output_dir: Path, session_options: dict
    ) -> CompositeModelHandler:
        """Generic pipeline processor with multiple component support"""

        def process_context_iterator(component_models, llm_pipeline, output_dir):
            # Reference to process the context and iterator models : static_llm pass
            new_groups = {
                "context": {},
                "iterator": {},
            }
            is_split = (
                len(llm_pipeline["context"]) > 1
            )  # if no. of splits are >1  then there will be multiple context models in pipeline
            for idx, component_name in enumerate(llm_pipeline["context"]):
                suffix = f"_{idx}" if is_split else ""

                # resave the intermediate model with external data
                intermediate_model_path = output_dir / f"transformer{suffix}.onnx"
                resave_model(
                    component_models[component_name].model_path, intermediate_model_path, force_external_data=True
                )

                for key in ["context", "iterator"]:
                    new_component_name = f"{key}{suffix}"

                    component_proto = onnx.load(intermediate_model_path, load_external_data=False)

                    # save the model with fixed shapes
                    component_model_path = output_dir / f"{new_component_name}.onnx"
                    onnx.save_model(component_proto, component_model_path)
                    new_groups[key][new_component_name] = ONNXModelHandler(
                        model_path=output_dir, onnx_file_name=component_model_path.name
                    )

                # delete the intermediate model
                intermediate_model_path.unlink()

            return new_groups

        return process_llm_pipeline(
            model, pipeline, process_context_iterator, output_dir, group_session_options=session_options
        )
