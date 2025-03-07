# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, Type

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import fix_dim_params, resave_model
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
    """

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
        }

    def _run_for_config(
        self, model: CompositeModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> CompositeModelHandler:
        assert isinstance(model, CompositeModelHandler), "StaticLLM pass only supports CompositeModelHandler"
        model_components = list(model.model_components)
        assert all(isinstance(m, ONNXModelHandler) for m in model_components), "All components must be ONNXModelHandler"
        assert (
            len(model_components) >= 3
        ), "There should be at least 3 components in the model: embedding, transformer, and lm_head."

        # only gqa models are supported for now
        assert (
            "GroupQueryAttention"
            in OnnxDAG(onnx.load(model_components[1].model_path, load_external_data=False)).get_node_op_types()
        ), "Only GQA models are supported for now."

        output_model_path = Path(output_model_path).with_suffix("")

        new_components = {}
        # resave embeddings model
        embeddings_model_path = output_model_path / "embeddings.onnx"
        resave_model(model_components[0].model_path, embeddings_model_path)
        new_components["embeddings"] = ONNXModelHandler(
            model_path=output_model_path, onnx_file_name=embeddings_model_path.name
        )

        # get dimension params from embeddings model
        batch_size, sequence_length = OnnxDAG(onnx.load(embeddings_model_path, load_external_data=False)).get_io_shape(
            "input_ids"
        )
        assert isinstance(batch_size, str), "Batch size must be a symbolic dimension"
        assert isinstance(sequence_length, str), "Sequence length must be a symbolic dimension"

        # mapping from params to fixed values
        param_mapping_dict = {
            "context": {
                batch_size: config.batch_size,
                sequence_length: config.context_length,
            },
            "iterator": {batch_size: config.batch_size, sequence_length: 1},
        }
        pipeline = {"embeddings": "embeddings", "context": [], "iterator": []}

        # update the param mapping with the new shapes from the embeddings model
        for param_mapping in param_mapping_dict.values():
            self.fix_shape(
                onnx.load(embeddings_model_path, load_external_data=False),
                param_mapping,
            )

        is_split = len(model_components) > 3
        for idx, component in enumerate(model_components[1:-1]):
            suffix = f"_{idx}" if is_split else ""

            # resave the model with external data
            intermediate_model_path = output_model_path / f"transformer{suffix}.onnx"
            resave_model(component.model_path, intermediate_model_path, force_external_data=True)

            for key, param_mapping in param_mapping_dict.items():
                component_name = f"{key}{suffix}"

                component_proto = onnx.load(intermediate_model_path, load_external_data=False)
                self.fix_shape(component_proto, param_mapping)

                # save the model with fixed shapes
                component_model_path = output_model_path / f"{component_name}.onnx"
                onnx.save_model(component_proto, component_model_path)
                new_components[component_name] = ONNXModelHandler(
                    model_path=output_model_path, onnx_file_name=component_model_path.name
                )
                pipeline[key].append(component_name)

            # delete the intermediate model
            intermediate_model_path.unlink()

        # resave the lm_head model
        lm_head_model_path = output_model_path / "lm_head.onnx"
        resave_model(model_components[-1].model_path, lm_head_model_path)
        new_components["lm_head"] = ONNXModelHandler(
            model_path=output_model_path, onnx_file_name=lm_head_model_path.name
        )
        pipeline["lm_head"] = "lm_head"

        model_attributes = model.model_attributes or {}
        model_attributes["llm_pipeline"] = pipeline

        return CompositeModelHandler(
            list(new_components.values()), list(new_components.keys()), model_attributes=model_attributes
        )

    @staticmethod
    def fix_shape(model_proto: onnx.ModelProto, param_mapping: Dict[str, int]):
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
                        assert (
                            param_mapping[old_dim] == new_dim
                        ), f"Param {old_dim} already exists with different value. Something is wrong."
                    param_mapping[old_dim] = new_dim
