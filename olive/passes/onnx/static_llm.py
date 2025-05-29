# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import fix_dim_params, process_llm_pipeline, resave_model
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
            "group_session_options": PassConfigParam(
                type_=dict,
                description=(
                    "Session options for the context and iterator models. Only used for models with genai_config."
                ),
            ),
        }

    def _run_for_config(
        self, model: CompositeModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> CompositeModelHandler:
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
