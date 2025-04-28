from typing import TYPE_CHECKING, Optional

from onnx import GraphProto, TensorProto

from olive.common.utils import StrEnumBase

if TYPE_CHECKING:
    from onnxruntime.quantization import QuantFormat

MSFT_DOMAIN = "com.microsoft"


class OpType(StrEnumBase):
    """Enum for operator types."""

    MatMulNBits = "MatMulNBits"
    MatMul = "MatMul"
    Gather = "Gather"
    GatherBlockQuantized = "GatherBlockQuantized"


class Algorithm(StrEnumBase):
    DEFAULT = "DEFAULT"
    HQQ = "HQQ"
    RTN = "RTN"
    GPTQ = "GPTQ"


def get_initializer(name, graph_path: list[GraphProto]) -> tuple[TensorProto, GraphProto]:
    for gid in range(len(graph_path) - 1, -1, -1):
        graph = graph_path[gid]
        for tensor in graph.initializer:
            if tensor.name == name:
                return tensor, graph
    return None, None


class WeightOnlyQuantConfig:
    def __init__(
        self,
        algorithm: Algorithm,
        quant_format: "QuantFormat",
        op_types_to_quantize: Optional[tuple[str, ...]] = None,
        quant_axes: Optional[tuple[tuple[str, int], ...]] = None,
        customized_weight_config: Optional[dict] = None,
    ):
        """Configure Weight Only blockwise quantization parameters.

        Args:
            algorithm:
                weight only quantize algorithm name.
            quant_format:
                quantization format. QOperator or QDQ.
            op_types_to_quantize (optional):
                set of operator types to quantize. Default {MatMul}
            quant_axes (dict[str, int], optional):
                op:axis, which axis to quantize for an op. Default {MatMul: 0, Gather: 1}
            customized_weight_config:
                customized weight config for nodes if needed.
                If both customized_weight_config and nodes_to_exclude are set, nodes_to_exclude overwrites customized_weight_config.

        """
        self.algorithm = str(algorithm)
        self.quant_format = quant_format
        self.op_types_to_quantize = set(op_types_to_quantize) if op_types_to_quantize else {str(OpType.MatMul)}
        self.quant_axes = dict(quant_axes) if quant_axes else {str(OpType.MatMul): 0, str(OpType.Gather): 1}
        self.customized_weight_config = customized_weight_config
