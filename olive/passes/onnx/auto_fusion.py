# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple, Union

import onnx
from packaging import version

from olive.common.config_utils import ParamCategory
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.utils import OnnxDAG
from olive.passes.pass_config import PassConfigParam

if TYPE_CHECKING:
    from olive.passes.onnx.auto_fusion_utils.fusion import FusionBase

logger = logging.getLogger(__name__)


class AutoFusion(Pass):
    """Automatically fuse nodes in an ONNX model using auto-generated custom operators."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "constant_overrides": PassConfigParam(
                type_=Dict[str, Dict[str, int]],
                description="Override default constants for the custom op builder. Dict of op type to constants.",
            ),
            "ort_headers_dir": PassConfigParam(
                type_=Union[Path, str],
                category=ParamCategory.PATH,
                description=(
                    "Path to the director with onnxruntime api headers. Only needed if not using a stable release of"
                    " onnxruntime. Should be '<onnxruntime-repo-dir>/include/onnxruntime/core/session' where"
                    " `<onnxruntime-repo-dir> is the path to a local clone of the ort github repository."
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime import __version__ as OrtVersion

        from olive.passes.onnx.auto_fusion_utils import DOMAIN, Builder

        if version.parse(OrtVersion) < version.parse("1.17.0"):
            raise RuntimeError("AutoFusion only supports ONNXRuntime version 1.17.0 or later")

        ort_headers_dir = config["ort_headers_dir"]
        if ort_headers_dir:
            assert Path(ort_headers_dir).is_dir(), f"ort_headers_dir: {ort_headers_dir} is not a directory."
        elif not Builder.check_ort_version(OrtVersion):
            raise RuntimeError(
                f"Cannot find a release for ONNXRuntime version {OrtVersion}. Please provide `ort_headers_dir`."
            )

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # get dag
        dag = OnnxDAG.from_model_path(model.model_path)
        # remove useless nodes
        dag.remove_redundant_cast_nodes()

        # get fusable candidates
        candidates = self.get_fusion_candidates(dag)

        fused_kernels = {}
        counter = Counter()
        for key, fusion in candidates:
            if not fusion.still_exists():
                continue
            if fusion.output_shape == [1]:
                # TODO(jambayk): fix issue with the Sqrt Div kernel with this shape
                continue

            node_names = fusion.get_node_names()
            fused_node, kernel_info = fusion.fuse()
            dag.replace_nodes(node_names, fused_node)

            # keep track of the fused kernels
            fused_kernels[key] = kernel_info
            counter[key] += 1
        logger.info(
            "Fusions: \n%s",
            "\n".join(f"{k_repr['ops']} ({key}): {counter[key]}" for key, k_repr in fused_kernels.items()),
        )
        dag.update()

        # update opset of model
        opset_import = dag.model.opset_import
        has_custom_domain = False
        for opset in opset_import:
            if opset.domain == DOMAIN:
                has_custom_domain = True
        if not has_custom_domain:
            opset_import.extend([onnx.helper.make_opsetid(DOMAIN, 1)])

        custom_op_lib = None
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Building custom op library...")
            builder = Builder(
                list(fused_kernels.values()),
                temp_dir,
                constant_overrides=config["constant_overrides"],
                ort_headers_dir=ort_headers_dir,
                ort_version=OrtVersion,
            )
            lib_path = builder.build()

            Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(lib_path, Path(output_model_path).parent / Path(lib_path).name)
            custom_op_lib = Path(lib_path).name

        # save the model to the output path and return the model
        return model_proto_to_olive_model(dag.model, output_model_path, config, custom_op_lib=custom_op_lib)

    @classmethod
    def _get_fusion_candidates_util(
        cls, dag: OnnxDAG, v: str, visited: Set[str], candidates_dict: Dict[str, List["FusionBase"]]
    ) -> None:
        """Find fusable candidates from all nodes reachable from v."""
        from olive.passes.onnx.auto_fusion_utils import get_fusion_class

        stack = [v]

        while stack:
            v = stack.pop()
            visited.add(v)

            for neighbor in dag.connections[v]:
                if neighbor not in visited:
                    stack.append(v)
                    stack.append(neighbor)
                    break
            else:
                fusion_cls = get_fusion_class(v, dag)
                if not fusion_cls or not fusion_cls(v, dag).can_fuse_more():
                    continue

                # keep track of all possible fusion starting from v
                fusions = []

                # try to fuse the node with its child
                child = dag.connections[v][0]
                fusion = fusion_cls(v, dag)
                if fusion.try_fuse_node(child):
                    fusions.append(fusion)

                # try to fuse the node with fusions of its child
                for child_candidates in candidates_dict.get(child, []):
                    fusion = fusion_cls(v, dag)
                    if fusion.try_concat_fusion(child_candidates):
                        fusions.append(fusion)

                if fusions:
                    candidates_dict[v] = fusions

    @classmethod
    def get_fusion_candidates(cls, dag: OnnxDAG) -> List[Tuple[Tuple, "FusionBase"]]:
        """Return fusable candidates in the graph.

        There will be overlap between candidates. For example, A -> B -> C and B -> C will both be returned.
        A -> B -> C and D -> C is also possible. The priority of the candidates during fusion will be determined
        by the fusion rules and heuristics.

        :param dag: The ONNX graph.
        :return: A dictionary of fusable candidates. Key is the base op and value is a tuple (op_names, op_types).
        """
        candidates_dict = {}
        visited = set()
        for v in dag.nodes:
            cls._get_fusion_candidates_util(dag, v, visited, candidates_dict)

        candidates = defaultdict(list)
        for fusions in candidates_dict.values():
            for fusion in fusions:
                fusion_hash = fusion.get_hash()
                candidates[(fusion_hash, len(fusion))].append((fusion_hash, fusion))
        candidates = dict(sorted(candidates.items(), key=lambda x: (x[0][1], len(x[1])), reverse=True))

        all_fusions = []
        for fusions in candidates.values():
            all_fusions.extend(fusions)
        return all_fusions
