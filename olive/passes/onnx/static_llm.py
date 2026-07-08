# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path

import onnx_ir as ir

from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.hardware.constants import ExecutionProvider
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import (
    add_version_metadata_to_ir_model,
    process_llm_pipeline,
    resave_model,
    update_llm_pipeline_genai_config_gpu,
)
from olive.passes.onnx.dynamic_to_fixed_shape import fix_dim_params
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


def _ir_io_shape(value: ir.Value) -> list:
    """Return the shape of an IR value as a list of ints (static dims) and strings (symbolic dims)."""
    if value.shape is None:
        return None
    return [dim.value if isinstance(dim, ir.SymbolicDim) else dim for dim in value.shape]


def _get_ir_input(ir_model: ir.Model, name: str) -> ir.Value:
    """Return the named graph input value of the model."""
    for graph_input in ir_model.graph.inputs:
        if graph_input.name == name:
            return graph_input
    raise ValueError(f"Input {name} was not found in graph inputs.")


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
        transformer_model = ir.load(model_components[1].model_path)
        assert any(node.op_type == "GroupQueryAttention" for node in transformer_model.graph.all_nodes()), (
            "Only GQA models are supported for now."
        )
        # get dimension params from embeddings model
        embedding_model = ir.load(model_components[0].model_path)
        input_ids = _get_ir_input(embedding_model, "input_ids")
        batch_size, sequence_length = _ir_io_shape(input_ids)
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
                ir.load(model_components[0].model_path),
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

                    # load lazily so the fixed-shape models share the intermediate external data file
                    component_ir = ir.load(intermediate_model_path)
                    self.fix_shape(component_ir, param_mapping)

                    # save the model with fixed shapes
                    component_model_path = output_dir / f"{new_component_name}.onnx"
                    # Add olive version to metadata
                    add_version_metadata_to_ir_model(component_ir)
                    ir.save(component_ir, component_model_path)
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
        output_model_dir = Path(output_model_path).with_suffix("")

        # --- Step 1: Load model (handle both single and external data) ---
        try:
            ir_model = model.load_ir_model()
            # load_ir_model() references external data lazily; materialize it so the model can be
            # re-saved into a fresh external data file under the output directory
            ir.external_data.load_to_model(ir_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

        # --- Step 2: Fix symbolic dimensions ---
        batch_size, sequence_length = _ir_io_shape(_get_ir_input(ir_model, "input_ids"))
        if not (isinstance(batch_size, str) and isinstance(sequence_length, str)):
            raise ValueError("Input dimensions must be symbolic before static shape fixing.")

        param_mapping = {batch_size: config.batch_size, sequence_length: config.context_length}
        self.fix_shape(ir_model, param_mapping)

        # --- Step 3: Save model as external-data format ---
        output_model_dir.mkdir(parents=True, exist_ok=True)
        output_model_file = output_model_dir / "model.onnx"
        external_data_file = output_model_dir / "model.onnx.data"

        ir.save(ir_model, output_model_file, external_data=external_data_file.name)

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

        input_model_path = model.model_path
        model_static = ONNXModelHandler(model_path=output_model_dir, onnx_file_name=output_model_file.name)

        return update_llm_pipeline_genai_config_gpu(
            model_static,
            output_model_dir,
            input_model_path,
            decoder_config_extra,
        )

    @staticmethod
    def fix_shape(ir_model: ir.Model, param_mapping: dict[str, int]):
        """Fix the shape of the model based on the param mapping.

        :param ir_model: The ONNX IR model to fix in place.
        :param param_mapping: Mapping from params to fixed values. This gets updated with the output shapes of the new
            model.
        """
        original_shapes = {output.name: _ir_io_shape(output) for output in ir_model.graph.outputs}

        # fix dim params
        fix_dim_params(ir_model, list(param_mapping.keys()), list(param_mapping.values()))

        # update the param mapping with the new shapes
        for output in ir_model.graph.outputs:
            original_shape = original_shapes[output.name]
            new_shape = _ir_io_shape(output)
            if original_shape is not None and new_shape is None:
                # keep output shapes stable even if symbolic shape inference cannot infer this output.
                # this preserves inter-model interface metadata used by compose.
                fallback_shape = [
                    param_mapping.get(dim, dim) if isinstance(dim, str) else dim for dim in original_shape
                ]
                output.shape = ir.Shape(fallback_shape)
                new_shape = fallback_shape
            if original_shape is None or new_shape is None:
                continue

            for old_dim, new_dim in zip(original_shape, new_shape):
                if isinstance(old_dim, str) and isinstance(new_dim, int):
                    if old_dim in param_mapping:
                        assert param_mapping[old_dim] == new_dim, (
                            f"Param {old_dim} already exists with different value. Something is wrong."
                        )
                    param_mapping[old_dim] = new_dim
