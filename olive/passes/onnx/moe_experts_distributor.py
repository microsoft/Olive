# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import logging
import multiprocessing
import os
import pprint
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Tuple, Type, Union

import numpy as np
import onnx
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

from olive.common.pydantic_v1 import validator
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import DistributedOnnxModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from onnxruntime.transformers.onnx_model import OnnxModel


logger = logging.getLogger(__name__)


class MoEExpertDistributionPatternMatcher:
    DEFAULT_DOMAIN: ClassVar[str] = "com.microsoft"
    JSON_SEPARATORS: ClassVar[Tuple] = (",", ": ")

    def __init__(self, world_size: int, input_filepath: str, debug=False):
        self.world_size = world_size
        self.input_filepath = input_filepath
        self.debug = debug

    @staticmethod
    def _dump_graph(node: Message, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as strm:
            json.dump(
                MessageToDict(node),
                fp=strm,
                indent=2,
                sort_keys=True,
                separators=MoEExpertDistributionPatternMatcher.JSON_SEPARATORS,
            )
            strm.flush()

    @staticmethod
    def _create_node_name(nodes: Dict[str, Message], op_type: str, prefix_a: str, prefix_b: str):
        prefix: str = ""
        last_slash: int = -1
        for i in range(min(len(prefix_a), len(prefix_b))):
            if prefix_a[i] == prefix_b[i]:
                prefix += prefix_a[i]
                if prefix_a[i] == "/":
                    last_slash = i
            else:
                break

        if last_slash > 0:
            prefix = prefix[: last_slash + 1]

        if not prefix.endswith("/"):
            prefix += "/"
        prefix += op_type

        suffix: int = 0
        name: str = prefix
        while True:
            if name not in nodes:
                return name

            name = f"{prefix}_{suffix}"
            suffix += 1

    @staticmethod
    def _replace_constant_value(constant: Message, value: int):
        for attr in constant.attribute:
            if attr.name == "value":
                attr.CopyFrom(
                    onnx.helper.make_attribute(
                        attr.name, onnx.numpy_helper.from_array(np.array([value], np.int64)), attr.doc_string
                    )
                )
                return

    @abstractmethod
    def identify_experts(self, output_dirpath: str):
        raise NotImplementedError

    @abstractmethod
    def distribute(
        self,
        experts: List[Any],
        num_experts: int,
        output_dirpath: str,
        use_external_data_format: bool = True,
        all_tensors_to_one_file: bool = True,
    ) -> List[str]:
        raise NotImplementedError


class MoEExpertDistributionPatternMatcherA(MoEExpertDistributionPatternMatcher):
    EXPERT_PATTERN: ClassVar[list] = [
        "Slice",  # 14
        "Concat",  # 13
        "MatMul",  # 12
        "Cast",  # 11
        "Relu",  # 10
        "MatMul",  # 9
        "Cast",  # 8
        "Slice",  # 7
        "Mul",  # 6
        "Div",  # 5
        "Add",  # 4
        "Gather",  # 3
        "Shape",  # 2
        "Reshape",  # 1
        "Reshape",  # 0
    ]

    def __init__(self, world_size: int, input_filepath: str, debug=False):
        # pylint: disable=useless-parent-delegation
        super().__init__(world_size, input_filepath, debug)

    @staticmethod
    def _create_ranked_model(
        nodes: Dict[str, Message],
        producers: Dict[str, Message],
        consumers: Dict[str, Message],
        world_size: int,
        experts: List[List[str]],
        num_experts: int,
        rank: int,
    ):
        experts_per_rank = num_experts // world_size
        start_index = rank * experts_per_rank
        end_index = start_index + experts_per_rank

        for expert in experts:
            div_node = nodes[expert[5]]
            concat_node = nodes[expert[13]]
            reshape_node = nodes[expert[1]]

            div_node_oname = div_node.output[0]
            reshape_node_oname = reshape_node.output[0]

            mul_nodes = [n for n in consumers[div_node_oname] if n.op_type == "Mul"]
            matmul_nodes = [producers[iname] for iname in concat_node.input if producers[iname].op_type == "MatMul"]
            slice_nodes = [n for n in consumers[reshape_node_oname] if n.op_type == "Slice"]

            slice_nodes[start_index].input[1] = slice_nodes[0].input[1]

            for i in range(num_experts):
                if (i < start_index) or (i >= end_index):
                    mul_nodes[i].input.remove(div_node_oname)
                    concat_node.input.remove(matmul_nodes[i].output[0])

                    for iname in list(slice_nodes[i].input):
                        slice_nodes[i].input.remove(iname)

    @staticmethod
    def _insert_collective_nodes(
        model: "OnnxModel", nodes: Dict[str, Message], world_size: int, experts: List[List[str]]
    ):
        from onnxruntime.transformers.onnx_model import OnnxModel

        for expert in experts:
            for i, j in [(0, 1), (-2, -1)]:
                prev_node = nodes[expert[i]]
                next_node = nodes[expert[j]]

                node_name = MoEExpertDistributionPatternMatcher._create_node_name(
                    nodes, "AllToAll", prev_node.name, next_node.name
                )
                output_name = node_name + "_output_0"
                alltoall = onnx.helper.make_node(
                    "AllToAll",
                    prev_node.output,
                    [output_name],
                    name=node_name,
                    domain=MoEExpertDistributionPatternMatcher.DEFAULT_DOMAIN,
                    group_size=world_size,
                )

                model.add_node(alltoall)
                nodes[node_name] = alltoall

                OnnxModel.replace_node_input(next_node, prev_node.output[0], output_name)

    @staticmethod
    def _fix_shapes(
        model: "OnnxModel",
        nodes: Dict[str, Message],
        producers: Dict[str, Message],
        consumers: Dict[str, Message],
        world_size: int,
        experts: List[List[str]],
        num_experts: int,
    ):
        experts_per_rank = num_experts // world_size

        for expert in experts:
            div_node = nodes[expert[5]]
            div_node_iname = div_node.input[1]
            div_node_oname = div_node.output[0]
            MoEExpertDistributionPatternMatcher._replace_constant_value(producers[div_node_iname], experts_per_rank)

            add_node = nodes[expert[4]]
            add_node_iname = add_node.input[1]
            MoEExpertDistributionPatternMatcher._replace_constant_value(producers[add_node_iname], experts_per_rank - 1)

            mul_nodes = [n for n in consumers[div_node_oname] if n.op_type == "Mul"]
            index = 1
            for mul_node in mul_nodes:
                if div_node_oname in mul_node.input:
                    MoEExpertDistributionPatternMatcher._replace_constant_value(producers[mul_node.input[1]], index)
                    index += 1

            reshape_node = nodes[expert[1]]
            for parent in model.get_parents(reshape_node, producers):
                if parent.op_type == "Concat":
                    concat_iname_0 = parent.input[0]
                    MoEExpertDistributionPatternMatcher._replace_constant_value(
                        producers[concat_iname_0], num_experts // experts_per_rank
                    )

                    concat_iname_1 = parent.input[1]
                    MoEExpertDistributionPatternMatcher._replace_constant_value(
                        producers[concat_iname_1], experts_per_rank
                    )

                    break

    @staticmethod
    def _generate_one(params: Tuple[Union[str, Path], Union[str, Path], int, List[List[str]], int, int, bool]) -> str:
        (
            input_filepath,
            output_dirpath,
            world_size,
            experts,
            num_experts,
            rank,
            use_external_data_format,
            all_tensors_to_one_file,
            debug,
        ) = params
        from onnxruntime.transformers.onnx_model import OnnxModel

        basename = DistributedOnnxModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT.format(rank)
        output_dirpath = Path(output_dirpath)

        model = OnnxModel(onnx.load_model(input_filepath))
        producers = model.output_name_to_node()
        consumers = model.input_name_to_nodes()
        nodes = {node.name: node for node in model.nodes()}

        MoEExpertDistributionPatternMatcherA._create_ranked_model(
            nodes, producers, consumers, world_size, experts, num_experts, rank
        )
        MoEExpertDistributionPatternMatcherA._insert_collective_nodes(model, nodes, world_size, experts)
        MoEExpertDistributionPatternMatcherA._fix_shapes(
            model, nodes, producers, consumers, world_size, experts, num_experts
        )

        model.prune_graph()
        output_filepath = str(output_dirpath / basename)
        model.save_model_to_file(
            output_filepath,
            use_external_data_format=use_external_data_format,
            all_tensors_to_one_file=all_tensors_to_one_file,
        )
        onnx.checker.check_model(model.model)

        if debug:
            MoEExpertDistributionPatternMatcher._dump_graph(model.model, str(output_dirpath / (basename + ".graph")))

        return output_filepath

    def identify_experts(self, output_dirpath: str):
        from onnxruntime.transformers.onnx_model import OnnxModel

        model = OnnxModel(onnx.load_model(self.input_filepath))
        producers = model.output_name_to_node()
        consumers = model.input_name_to_nodes()

        if self.debug:
            basename = Path(self.input_filepath).stem
            output_dirpath = Path(output_dirpath)
            OnnxModel.graph_topological_sort(model.model.graph)
            MoEExpertDistributionPatternMatcher._dump_graph(model.model, str(output_dirpath / f"{basename}.graph"))

        experts = []
        num_experts = None
        for node in model.get_nodes_by_op_type(MoEExpertDistributionPatternMatcherA.EXPERT_PATTERN[0]):
            path = model.match_parent_path(
                node, MoEExpertDistributionPatternMatcherA.EXPERT_PATTERN[1:], output_name_to_node=producers
            )
            if path:
                path.reverse()
                path.append(node)

                if self.debug:
                    pprint.pprint([n.name for n in path])  # noqa: T203

                div_node = path[5]
                div_node_oname = div_node.output[0]
                if not num_experts:
                    num_experts = len(consumers[div_node_oname])
                    if (num_experts % self.world_size) != 0:
                        raise f"""Number of expert paths on node "{div_node.name}" should be a multiple of
                                input world-size={self.world_size} but found {len(consumers[div_node_oname])}"""
                elif len(consumers[div_node_oname]) != num_experts:
                    raise f"""Inconsistent number of expert across layers.
                            expected={num_experts}, found={len(consumers[div_node_oname])}"""

                experts.append([n.name for n in path])

        return experts, num_experts

    def distribute(
        self,
        experts: List[Any],
        num_experts: int,
        output_dirpath: str,
        use_external_data_format: bool = True,
        all_tensors_to_one_file: bool = True,
        parallel_jobs: int = multiprocessing.cpu_count(),
    ) -> List[str]:
        params = [
            (
                self.input_filepath,
                output_dirpath,
                self.world_size,
                experts,
                num_experts,
                rank,
                use_external_data_format,
                all_tensors_to_one_file,
                self.debug,
            )
            for rank in range(self.world_size)
        ]

        max_parallel_jobs = min(self.world_size, parallel_jobs)
        if max_parallel_jobs <= 1:
            output_filepaths = [MoEExpertDistributionPatternMatcherA._generate_one(_) for _ in params]
        else:
            with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
                output_filepaths = pool.map(MoEExpertDistributionPatternMatcherA._generate_one, params)

        return output_filepaths


class MoEExpertsDistributor(Pass):
    """Split the input model (and insert necessary communication operations) to prepare for distributed inferencing."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "world_size": PassConfigParam(
                type_=int,
                default=2,
                required=True,
                description="Number of GPU nodes to distribute the model for. Must be greater than 1.",
            ),
            "parallel_jobs": PassConfigParam(
                type_=int,
                default=multiprocessing.cpu_count(),
                required=False,
                description="Number of parallel jobs. Defaulted to number of CPUs. Set it to 0 to disable.",
            ),
        }
        config.update(get_external_data_config())
        return config

    @staticmethod
    def _validate_world_size(v):
        if int(v) < 2:
            raise ValueError("world_size should be >= 2")

        return v

    @classmethod
    def _validators(cls) -> Dict[str, Callable]:
        return {"validate_distributor_config": validator("world_size", allow_reuse=True)(cls._validate_world_size)}

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> DistributedOnnxModelHandler:
        # huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
        # Disabling parallelism to avoid deadlocks...
        # To disable this warning, you can either:
        #     - Avoid using `tokenizers` before the fork if possible
        #     - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        matcher = MoEExpertDistributionPatternMatcherA(config.world_size, model.model_path)
        experts, num_experts = matcher.identify_experts(output_model_path)
        matcher.distribute(
            experts,
            num_experts,
            output_model_path,
            use_external_data_format=config.save_as_external_data,
            all_tensors_to_one_file=config.all_tensors_to_one_file,
            parallel_jobs=config.parallel_jobs or multiprocessing.cpu_count(),
        )
        return DistributedOnnxModelHandler(
            model_path=str(Path(output_model_path).with_suffix("")),
            model_name_pattern=DistributedOnnxModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT,
            num_ranks=config.world_size,
        )
