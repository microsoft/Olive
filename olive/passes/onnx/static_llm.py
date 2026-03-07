# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path

import onnx

from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.hardware.constants import ExecutionProvider
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import (
    add_version_metadata_to_model_proto,
    fix_dim_params,
    process_llm_pipeline,
    resave_model,
    update_llm_pipeline_genai_config_gpu,
)
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class StaticLLM(Pass):
    """Convert a dynamic shaped LLM into a static shaped LLM.

    Expects a CompositeModelHandler with at least 3 components: embeddings, transformer layers, and lm_head.
    transformer layers can be split into multiple components. Each transformer layers component produces two
    new components:
        - context model (sequence length = context_length)
        - iterator model (sequence length = 1)
    embeddings and lm_head keep their original shapes.
    The output model has an attribute "llm_pipeline" that contains the mapping of the components with keys:
        - embeddings: name of the embeddings model
        - context: list of context model names
        - iterator: list of iterator model names
        - lm_head: name of the lm_head model
    """

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "batch_size": PassConfigParam(
                type_=int,
                default_value=1,
                description="Batch size of the model.",
            ),
            "context_length": PassConfigParam(
                type_=int,
                default_value=64,
                description="Input length of the context model.",
            ),
            "context_lengths": PassConfigParam(
                type_=list[int],
                default_value=None,
                description=(
                    "List of context lengths to generate static models QNN_GPU."
                    "If None or empty, falls back to single 'context_length'."
                ),
            ),
            "group_session_options": PassConfigParam(
                type_=dict,
                description=(
                    "Session options for the context and iterator models. Only used for models with genai_config."
                ),
            ),
        }

    def _run_for_config(self, model, config: type[BasePassConfig], output_model_path: str):
        if (
            self.accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider
            and self.accelerator_spec.accelerator_type == Device.GPU
        ):
            assert isinstance(model, ONNXModelHandler), "StaticLLM (qnn-gpu) requires a single ONNXModelHandler."
            return self._run_qnn_gpu(model, config, output_model_path)

        else:
            return self._run_generic(model, config, output_model_path)

    def _run_generic(self, model: CompositeModelHandler, config: type[BasePassConfig], output_model_path: str):
        assert isinstance(model, CompositeModelHandler), "StaticLLM pass only supports CompositeModelHandler"
        model_components = list(model.model_components)
        assert all(isinstance(m, ONNXModelHandler) for m in model_components), "All components must be ONNXModelHandler"
        assert len(model_components) >= 3, (
            "There should be at least 3 components in the model: embedding, transformer, and lm_head."
        )

        # only gqa models are supported for now
        assert (
            "GroupQueryAttention"
            in OnnxDAG(onnx.load(model_components[1].model_path, load_external_data=False)).get_node_op_types()
        ), "Only GQA models are supported for now."
        # get dimension params from embeddings model
        batch_size, sequence_length = OnnxDAG(
            onnx.load(model_components[0].model_path, load_external_data=False)
        ).get_io_shape("input_ids")
        assert isinstance(batch_size, str), "Batch size must be a symbolic dimension"
        assert isinstance(sequence_length, str), "Sequence length must be a symbolic dimension"

        output_model_path = Path(output_model_path).with_suffix("")

        # mapping from params to fixed values
        param_mapping_dict = {
            "context": {
                batch_size: config.batch_size,
                sequence_length: config.context_length,
            },
            "iterator": {batch_size: config.batch_size, sequence_length: 1},
        }

        # update the param mapping with the new shapes from the embeddings model
        for param_mapping in param_mapping_dict.values():
            self.fix_shape(
                onnx.load(model_components[0].model_path, load_external_data=False),
                param_mapping,
            )

        def process_context_iterator(component_models, llm_pipeline, output_dir):
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

                for key, param_mapping in param_mapping_dict.items():
                    new_component_name = f"{key}{suffix}"

                    component_proto = onnx.load(intermediate_model_path, load_external_data=False)
                    self.fix_shape(component_proto, param_mapping)

                    # save the model with fixed shapes
                    component_model_path = output_dir / f"{new_component_name}.onnx"
                    # Add olive version to metadata
                    add_version_metadata_to_model_proto(component_proto)
                    onnx.save_model(component_proto, component_model_path)
                    new_groups[key][new_component_name] = ONNXModelHandler(
                        model_path=output_dir, onnx_file_name=component_model_path.name
                    )

                # delete the intermediate model
                intermediate_model_path.unlink()

            return new_groups

        # need to update the genai_config with the new pipeline
        # only support GQA model with "past_seq_len" and "total_seq_len" inputs for now
        # ort-genai only supports static shaped models with sliding window with assumptions about
        # the model and kv-cache ios
        decoder_config_extra = {
            "inputs": {
                "past_sequence_length": "past_seq_len",
                "total_sequence_length": "total_seq_len",
            },
            "sliding_window": {
                "window_size": config.context_length,
                "pad_value": 0,
                "alignment": "left",
                "slide_key_value_cache": False,
            },
        }
        # dummy pipeline to get the context and iterator models
        pipeline = {
            "embeddings": model.model_component_names[0],
            "context": model.model_component_names[1:-1],
            "iterator": model.model_component_names[1:-1],
            "lm_head": model.model_component_names[-1],
        }

        return process_llm_pipeline(
            model,
            pipeline,
            process_context_iterator,
            output_model_path,
            decoder_config_extra=decoder_config_extra,
            group_session_options=config.group_session_options,
        )

    def _run_qnn_gpu(self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: Path):
        """QNN_GPU path: generate one or more static ONNX models for different context lengths.

        - If config.context_lengths is None/empty: use config.context_length (single model).
        - If config.context_lengths has 1 value: use that context length (single model).
        - If config.context_lengths has >1 values: generate multiple models and return CompositeModelHandler.
        """
        output_model_dir = Path(output_model_path).with_suffix("")
        model_path = Path(model.model_path)

        # --- Step 1: Load model (handle both single and external data) ---
        try:
            base_model_proto = onnx.load(model_path, load_external_data=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

        # --- Step 2: Get symbolic batch and sequence dims once ---
        batch_size, sequence_length = OnnxDAG(base_model_proto).get_io_shape("input_ids")
        if not (isinstance(batch_size, str) and isinstance(sequence_length, str)):
            raise ValueError("Input dimensions must be symbolic before static shape fixing.")

        # --- Determine which context lengths to use ---
        cfg_ctx_lengths = getattr(config, "context_lengths", None) or []
        ctx_lengths_list = [int(x) for x in cfg_ctx_lengths if x is not None]

        if not ctx_lengths_list:
            # Fall back to single context_length in config
            ctx_lengths_list = [int(config.context_length)]

        # If only one context length, we still treat it uniformly but return a single handler.
        multiple = len(ctx_lengths_list) > 1

        generated_handlers: dict[int, ONNXModelHandler] = {}
        generated_names: dict[int, str] = {}

        for ctx_len in ctx_lengths_list:
            # --- Clone base model proto for this variant ---
            model_proto = onnx.ModelProto()
            model_proto.CopyFrom(base_model_proto)

            # --- Step 3: Fix symbolic dimensions for this context length ---
            param_mapping = {batch_size: config.batch_size, sequence_length: ctx_len}
            self.fix_shape(model_proto, param_mapping)

            add_version_metadata_to_model_proto(model_proto)

            # --- Step 4: Save as external-data ONNX ---
            onnx_file_name = f"model_ctx{ctx_len}.onnx"
            output_model_file = Path(output_model_dir) / onnx_file_name
            external_data_file = Path(output_model_dir) / f"{onnx_file_name}.data"

            output_model_dir.mkdir(parents=True, exist_ok=True)
            onnx.save(
                model_proto,
                str(output_model_file),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_file.name,
                convert_attribute=False,
            )

            # Build handler for this static model
            new_model_attributes = deepcopy(model.model_attributes) or {}
            handler = ONNXModelHandler(
                model_path=output_model_dir,
                onnx_file_name=output_model_file.name,
                model_attributes=new_model_attributes,
            )

            # Store handler + a logical component name (e.g., ctx_128)
            generated_handlers[ctx_len] = handler
            generated_names[ctx_len] = f"ctx_{ctx_len}"

        # --- Step 5: Update genai_config.json ---
        # For single model: pipeline with one component.
        # For multiple models: pipeline with multiple components (composite).
        if not multiple:
            # Single context length
            ctx_len = ctx_lengths_list[0]
            handler = generated_handlers[ctx_len]

            decoder_config_extra = {
                "inputs": {
                    "past_sequence_length": "past_seq_len",
                    "total_sequence_length": "total_seq_len",
                },
                "sliding_window": {
                    "window_size": ctx_len,
                    "pad_value": 0,
                    "alignment": "left",
                    "slide_key_value_cache": False,
                },
            }

            handler = update_llm_pipeline_genai_config_gpu(
                model=handler,
                output_model_dir=output_model_dir,
                decoder_config_extra=decoder_config_extra,
                composite_components=None,
            )
            return handler

        # Multiple context lengths -> wrap in CompositeModelHandler and create composite pipeline
        components = []
        component_names = []
        for ctx_len, handler in sorted(generated_handlers.items(), key=lambda kv: kv[0]):
            components.append(handler)
            component_names.append(generated_names[ctx_len])

        new_model_attributes = deepcopy(model.model_attributes) or {}

        composite = CompositeModelHandler(
            model_components=components, model_component_names=component_names, model_attributes=new_model_attributes
        )

        # Build per-component sliding_window config keyed by name
        composite_decoder_extra = {
            "inputs": {
                "past_sequence_length": "past_seq_len",
                "total_sequence_length": "total_seq_len",
            },
            "sliding_window": {
                "window_size": max(ctx_lengths_list),
                "pad_value": 0,
                "alignment": "left",
                "slide_key_value_cache": False,
            },
        }

        composite = update_llm_pipeline_genai_config_gpu(
            model=composite,
            output_model_dir=output_model_dir,
            decoder_config_extra=composite_decoder_extra,
            composite_components=list(zip(component_names, components)),
        )

        return composite

    @staticmethod
    def fix_shape(model_proto: onnx.ModelProto, param_mapping: dict[str, int]):
        """Fix the shape of the model based on the param mapping.

        :param model_path: Path to the model.
        :param param_mapping: Mapping from params to fixed values. This gets updated with the output shapes of the new
            model.
        """
        original_shapes = {}
        dag = OnnxDAG(model_proto)
        for output_name in dag.get_output_names():
            original_shapes[output_name] = dag.get_io_shape(output_name)

        # fix dim params
        fix_dim_params(dag.model, param_mapping.keys(), param_mapping.values())

        # update the param mapping with the new shapes
        for output_name, original_shape in original_shapes.items():
            new_shape = dag.get_io_shape(output_name)

            for old_dim, new_dim in zip(original_shape, new_shape):
                if isinstance(old_dim, str) and isinstance(new_dim, int):
                    if old_dim in param_mapping:
                        assert param_mapping[old_dim] == new_dim, (
                            f"Param {old_dim} already exists with different value. Something is wrong."
                        )
                    param_mapping[old_dim] = new_dim
