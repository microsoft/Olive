# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Type, Union

import numpy as np

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    copy_context_bin_files,
    get_context_bin_file_names,
    get_external_data_config,
    model_proto_to_file,
    resave_model,
)
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class ComposeOnnxModels(Pass):
    """Compose multiple ONNX models into a single model.

    This pass chains multiple ONNX models together by itertively connecting the output of the preceding model to the
    input of the next model. The final inputs and outputs are the set of all inputs and outputs of the models excluding
    those used to connect the models together.

    It also handles llm_pipeline models:
    - embeddings: the embeddings model is saved as is
    - context: the context model is composed of all models in the context group
    - iterator: the iterator model is composed of all models in the iterator group
    - lm_head: the lm_head model is saved as is
    """

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self,
        model: CompositeModelHandler,
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        assert isinstance(model, CompositeModelHandler), "ComposeOnnxModels pass only supports CompositeModelHandler"
        assert all(
            isinstance(m, ONNXModelHandler) for m in model.model_components
        ), "All components must be ONNXModelHandler"

        if llm_pipeline := (model.model_attributes or {}).get("llm_pipeline"):
            output_model_path = Path(output_model_path).with_suffix("")

            component_map = dict(model.get_model_components())
            new_component_models = {}
            new_llm_pipeline = {}

            # resave embeddings model
            embeddings_model_path = output_model_path / "embeddings.onnx"
            resave_model(component_map[llm_pipeline["embeddings"]].model_path, embeddings_model_path)
            new_component_models["embeddings"] = ONNXModelHandler(
                model_path=output_model_path, onnx_file_name=embeddings_model_path.name
            )
            new_llm_pipeline["embeddings"] = "embeddings"

            # compose the context and iterator models
            composed_suffix = (
                "_ctx" if get_context_bin_file_names(component_map[llm_pipeline["context"][0]].model_path) else ""
            )
            saved_cb_files = {}
            for group_name in ["context", "iterator"]:
                composed_name = f"{group_name}{composed_suffix}"
                new_component_models[composed_name] = self._get_composed_model(
                    [component_map[component_name].model_path for component_name in llm_pipeline[group_name]],
                    output_model_path / f"{composed_name}.onnx",
                    external_config=config.dict(),
                    saved_cb_files=saved_cb_files,
                    as_model_dir=True,
                )
                new_llm_pipeline[group_name] = [composed_name]

            # resave the lm_head model
            lm_head_model_path = output_model_path / "lm_head.onnx"
            resave_model(component_map[llm_pipeline["lm_head"]].model_path, lm_head_model_path)
            new_component_models["lm_head"] = ONNXModelHandler(
                model_path=output_model_path, onnx_file_name=lm_head_model_path.name
            )
            new_llm_pipeline["lm_head"] = "lm_head"

            new_model_attributes = deepcopy(model.model_attributes)
            new_model_attributes["llm_pipeline"] = new_llm_pipeline
            return CompositeModelHandler(
                list(new_component_models.values()),
                list(new_component_models.keys()),
                model_attributes=new_model_attributes,
            )

        return self._get_composed_model(
            [component.model_path for component in model.model_components],
            resolve_onnx_path(output_model_path),
            external_config=config.dict(),
        )

    @staticmethod
    def _get_composed_model(
        onnx_model_paths: list,
        output_model_path: Union[str, Path],
        external_config: dict,
        saved_cb_files: Optional[dict] = None,
        as_model_dir: bool = False,
    ) -> ONNXModelHandler:
        """Compose multiple ONNX models into a single model.

        :param onnx_model_paths: List of ONNX model paths.
        :param output_model_path: Path to save the composed ONNX model.
        :param external_config: Configuration for external data.
        :param saved_cb_files: Dictionary of saved context binary files.
        :param as_model_dir: Use model parent directory as output model_path.
        :return: Composed ONNX model.
        """
        dags = [OnnxDAG.from_model_path(path) for path in onnx_model_paths]

        seen_inputs = set()
        seen_outputs = set()
        for dag in dags:
            dag_inputs = set(dag.get_input_names())
            dag_outputs = set(dag.get_output_names())
            # avoid circular connection, model_2 output cannot be model_1 input
            assert dag_outputs.isdisjoint(
                seen_inputs
            ), f"Output names {dag_outputs.intersection(seen_inputs)} are already used as input names."
            # avoid reused output name
            assert dag_outputs.isdisjoint(
                seen_outputs
            ), f"Output names {dag_outputs.intersection(seen_outputs)} are already used as output names."

            # update seen inputs and outputs
            seen_inputs.update(dag_inputs)
            seen_outputs.update(dag_outputs)

        # will only keep the unused outputs
        # inputs will be automatically taken care of during compose
        final_outputs = seen_outputs - seen_inputs

        # compose
        composed_dag = dags.pop(0)
        while dags:
            dag = dags.pop(0)

            cd_input_names = set(composed_dag.get_input_names())
            cd_output_names = set(composed_dag.get_output_names())
            cd_initializer_names = set(composed_dag.get_initializer_names())
            for input_name in dag.get_input_names():
                if input_name in cd_input_names | cd_output_names:
                    assert dag.get_io_shape(input_name) == composed_dag.get_io_shape(
                        input_name
                    ), f"Input shape mismatch: {input_name}"
                    assert dag.get_io_dtype(input_name) == composed_dag.get_io_dtype(
                        input_name
                    ), f"Input dtype mismatch: {input_name}"
                    continue

                # will add to graph 0 for now
                # this will fail if the connection already exists in the graph = expected behavior.
                composed_dag.add_input(dag.get_input_proto(input_name), 0)

            for output_name in dag.get_output_names():
                composed_dag.add_output(dag.get_output_proto(output_name), 0)

            for init_name in dag.get_initializer_names():
                if init_name in cd_initializer_names:
                    np.testing.assert_array_equal(
                        dag.get_initializer_np_array(init_name), composed_dag.get_initializer_np_array(init_name)
                    ), f"Initializer mismatch: {init_name}"
                    continue

                composed_dag.add_initializer(dag.get_initializer_proto(init_name), 0)

            for intermediate_name in dag.get_intermediate_names():
                if value_info := dag.get_value_info_proto(intermediate_name):
                    composed_dag.add_value_info(value_info, 0)

            for node_name in dag.topological_sort():
                node = dag.get_node_proto(node_name)
                if composed_dag.has_node(node_name):
                    # there might be some dq nodes for initializers that are common between models
                    # since split model keeps dq with the consumer op
                    assert composed_dag.get_node_proto(node_name) == node, f"Node mismatch: {node_name}"
                    continue

                composed_dag.add_node(node, 0)

        for output_name in composed_dag.get_output_names():
            if output_name not in final_outputs:
                composed_dag.remove_output(output_name)

        # update the model graph
        composed_dag.update()

        # save the composed model
        output_model_path = Path(output_model_path)
        has_external_data = model_proto_to_file(composed_dag.model, output_model_path, **external_config)

        # copy over context binary files if any
        saved_cb_files = saved_cb_files if saved_cb_files is not None else {}
        for path in onnx_model_paths:
            has_external_data |= copy_context_bin_files(path, output_model_path.parent, saved_cb_files=saved_cb_files)

        return ONNXModelHandler(
            output_model_path.parent if (has_external_data or as_model_dir) else output_model_path,
            onnx_file_name=output_model_path.name if (has_external_data or as_model_dir) else None,
        )
