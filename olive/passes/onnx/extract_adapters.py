# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

from olive.common.utils import WeightsFileFormat, save_weights
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from Olive.olive.constants import OpType
from olive.passes import Pass
from olive.passes.onnx.common import (
    DORA_NAME_PATTERNS_DYNAMO,
    DORA_NAME_PATTERNS_TORCHSCRIPT,
    LOHA_NAME_PATTERNS_DYNAMO,
    LOHA_NAME_PATTERNS_TORCHSCRIPT,
    LORA_NAME_PATTERNS_DYNAMO,
    LORA_NAME_PATTERNS_TORCHSCRIPT,
    AdapterType,
    get_adapter_name,
    get_external_data_config,
    model_has_adapters,
    model_has_adapters_from_dynamo,
    model_proto_to_olive_model,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExtractAdapters(Pass):
    """Extract adapter weights from ONNX model and save them as external weights file.

    If make_inputs is False, model proto is invalid after this pass as the adapter weights point to non-existent
    external files. Inference session must be created by first loading the adapter weights using
    SessionOptions.add_external_initializers.

    If make_inputs is True, the adapter weights are inputs to the model and must be provided during inference.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "adapter_type": PassConfigParam(
                type_=AdapterType,
                default_value=AdapterType.LORA,
                description=f"Type of adapter to extract. Valid values are {AdapterType.__members__.values()}.",
            ),
            "make_inputs": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Convert adapter weights to inputs. If false, the adapter weights will be set as initializers with"
                    " external data."
                ),
            ),
            "dynamic_lora_r": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether the model uses dynamic shape for lora_r. Only used if make_inputs is True. Valid only for"
                    " float modules."
                ),
            ),
            "optional_inputs": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Create default initializers (empty tensor with lora_r dimension set to 0) for the adapter weights,"
                    " if inputs not provided during inference. Only used if make_inputs is True. Valid only for float"
                    " modules."
                ),
            ),
            "save_format": PassConfigParam(
                type_=WeightsFileFormat,
                default_value=WeightsFileFormat.ONNX_ADAPTER,
                description="Format to save the weights in.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if not model_has_adapters(model.model_path, config.adapter_type):
            logger.info("Model does not have %s modules. Returning the original model.", config.adapter_type)
            return model

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        ir_model = model.load_ir_model()
        ir.external_data.load_to_model(ir_model)

        # dictionary to store adapter weights
        weights = {}

        if config.adapter_type in [AdapterType.LORA, AdapterType.DORA, AdapterType.LOHA]:
            if model_has_adapters_from_dynamo(model.model_path, config.adapter_type):
                weights = self._extract_adapter_from_dynamo(ir_model, adapter_type=config.adapter_type)
            else:
                if config.adapter_type == AdapterType.LORA:
                    weights = self._extract_adapter(ir_model, config, adapter_type=AdapterType.LORA)
                elif config.adapter_type == AdapterType.DORA:
                    weights = self._extract_adapter(ir_model, config, adapter_type=AdapterType.DORA)
                elif config.adapter_type == AdapterType.LOHA:
                    weights = self._extract_loha_adapter(ir_model, config)

        else:
            raise ValueError(f"Unsupported adapter type: {config.adapter_type}")

        if not weights:
            logger.info("No %s modules found in the model. Returning the original model.", config.adapter_type)
            return model

        if config.make_inputs:
            # create inputs for the weights
            for weight_name in weights:
                self._convert_initializer_to_input(ir_model, weight_name)
                self._make_dynamic_optional(ir_model, weights, weight_name, config)

        # save the weights
        weights_path = save_weights(weights, Path(output_model_path).parent / "adapter_weights", config.save_format)

        # save the model
        output_model = model_proto_to_olive_model(
            ir.to_proto(ir_model),
            output_model_path,
            config,
            external_initializers_file_name=weights_path.name if not config.make_inputs else None,
            constant_inputs_file_name=weights_path.name if config.make_inputs else None,
        )
        output_model.model_attributes = deepcopy(model.model_attributes) or {}
        # add adapter weights to the model attributes
        output_model.model_attributes["additional_files"] = additional_files = output_model.model_attributes.get(
            "additional_files", []
        )
        additional_files.append(str(weights_path))
        # save information about the weights in the model attributes
        weights_info = {name: [list(value.shape), str(value.dtype)] for name, value in weights.items()}
        if not config.make_inputs:
            output_model.model_attributes["external_initializers"] = weights_info
        else:
            output_model.model_attributes["constant_inputs"] = weights_info
        return output_model

    def _convert_initializer_to_input(self, model: ir.Model, initializer_name: str):
        """Convert a specific initializer to an input."""
        graph = model.graph

        # Check if the initializer exists
        if initializer_name not in graph.initializers:
            raise ValueError(f"Initializer '{initializer_name}' not found in graph")

        # Get the initializer
        initializer = graph.initializers[initializer_name]

        # Check if it's already an input
        if initializer in graph.inputs:
            return  # Already an input

        # Add to inputs
        graph.inputs.append(initializer)

    def _decompose_gemm(self, ir_model: ir.Model):
        """Decompose Gemm nodes into MatMul and Add nodes."""
        from onnxscript import rewriter
        from onnxscript.rewriter.rules.common import gemm_to_matmul_add_rule

        return rewriter.rewrite(ir_model, pattern_rewrite_rules=[gemm_to_matmul_add_rule])

    def _extract_adapter_from_dynamo(self, ir_model: ir.Model, adapter_type: AdapterType = AdapterType.LORA):
        """Extract adapter weights for LoRA, DoRA, or LoHa from an ONNX model.

        LoRA:
        lora_A -> MatMul -> ...
        lora_B -> MatMul -> ...

        DoRA:
        Besides LoRA A and LoRA B, DoRA also has a learnable magnitude vector M (dora_M):
                         W' = mV + dV = mV + mAB
        AB + dora_M -> Div -> ...

        LoHa:
        hada_w1_a + hada_w1_b -> MatMul -> ...
        hada_w2_a + hada_w2_b -> MatMul -> ...
        """
        if adapter_type == AdapterType.DORA:
            ir_model = self._decompose_gemm(ir_model)

        # dictionary to store adapter weights
        weights = {}

        # Get the appropriate patterns for the adapter type
        patterns = None
        if adapter_type == AdapterType.LORA:
            patterns = LORA_NAME_PATTERNS_DYNAMO
        elif adapter_type == AdapterType.DORA:
            patterns = DORA_NAME_PATTERNS_DYNAMO
        elif adapter_type == AdapterType.LOHA:
            patterns = LOHA_NAME_PATTERNS_DYNAMO
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

        to_rename = []
        for initializer in ir_model.graph.initializers.values():
            adapter_weight = get_adapter_name(initializer, patterns)
            if adapter_weight is None:
                continue

            to_rename.append((initializer, adapter_weight))

        for initializer, adapter_weight in to_rename:
            self._externalize_initializer(ir_model, weights, initializer, adapter_weight)

        return weights

    def _extract_adapter(
        self, ir_model: ir.Model, config: type[BasePassConfig], adapter_type: AdapterType = AdapterType.LORA
    ):
        """Extract adapter weights for either LoRA or Dora from an ONNX model.
        LoRA:
        output + default (lora_A) -> MatMul -> ...
        output + default_1 (lora_B) -> MatMul -> ...
        DoRA:
        Besides LoRA A and LoRA B, DoRA also has a learnable magnitude vector M (dora_M):
                         W' = mV + dV = mV + mAB
        AB + dora_M -> Div -> ...
        """
        # dictionary to store adapter weights
        weights = {}
        # keep track of float and quantized modules
        float_modules = set()
        quant_modules = set()

        # nodes to remove at the end
        nodes_to_remove = set()

        # lora and dora modules have different name patterns and valid ops
        patterns = None
        valid_ops = None
        if adapter_type == AdapterType.LORA:
            patterns = LORA_NAME_PATTERNS_TORCHSCRIPT
            valid_ops = {OpType.MatMul, OpType.MatMulNBits}
        if adapter_type == AdapterType.DORA:
            patterns = DORA_NAME_PATTERNS_TORCHSCRIPT
            valid_ops = {OpType.MatMul, OpType.MatMulNBits, OpType.Div}

        for node in ir_model.graph.all_nodes():
            node_name = node.name
            op_type = node.op_type
            if op_type not in valid_ops or not any(re.match(pattern, node_name) for pattern in patterns):
                # not a lora module
                continue

            # new name for the float weight
            new_weight_name = self._create_new_weight_name(node_name, adapter_type)
            # new names for quantized weight and parameters
            # zero point is optional if symmetric
            quantized_suffices = [".quant.weight", ".quant.scale", ".quant.zero_point"]
            new_quantized_names = [new_weight_name.replace(".weight", suffix) for suffix in quantized_suffices]

            if op_type == "Div":
                self._process_div_node(ir_model, node_name, weights, new_weight_name, float_modules)
            elif op_type == "MatMul":
                self._process_matmul_node(
                    ir_model,
                    node_name,
                    weights,
                    new_weight_name,
                    float_modules,
                    quant_modules,
                    new_quantized_names,
                    nodes_to_remove,
                )
            elif op_type == "MatMulNBits":
                self._process_matmulnbits_node(
                    ir_model, node, weights, new_weight_name, quant_modules, new_quantized_names
                )

        ir_model.graph.remove(nodes_to_remove)

        if config.make_inputs and quant_modules and config.dynamic_lora_r:
            # MatMulNBits has static K,N dimensions which are set as attributes
            # No use case for DequantizeLinear with dynamic lora_r
            logger.info("Quantized modules do not support dynamic_lora_r. Ignoring.")

        return weights

    def _extract_loha_adapter(self, ir_model: ir.Model, config: type[BasePassConfig]):
        """Extract LoHa adapter weights from all graphs in the ONNX model.

        This version supports both normal float initializers and QDQ (DequantizeLinear) chains.

        LoHa training adds 4 trainable initializers:
            hada_w1_a.default + hada_w1_b.default -> MatMul -> ...
            hada_w2_a.default + hada_w2_b.default -> MatMul -> ...

        Quantization (MatMulNBits) quantizes b initializers:
            hada_w1_a.default + hada_w1_b.default_Q4 + hada_w1_b.default_scales -> MatMulNBits -> ...
            hada_w2_a.default + hada_w2_b.default_Q4 + hada_w2_b.default_scales -> MatMulNBits -> ...

        QDQ quantizes b initializers:
            DequantizeLinear (x + x_scale + x_zero_point) + hada_w1_a.default -> MatMul -> ...
            DequantizeLinear (x + x_scale + x_zero_point) + hada_w2_a.default -> MatMul -> ...
        """
        weights = {}
        nodes_to_remove = set()
        float_modules = set()
        quant_modules = set()

        for initializer in ir_model.graph.initializers.values():
            old_initializer_name = initializer.name

            if any(re.match(pattern, initializer.name) for pattern in LOHA_NAME_PATTERNS_TORCHSCRIPT):
                new_initializer_name = self._create_new_weight_name(old_initializer_name, AdapterType.LOHA)
                consumer = initializer.consumers()[0]
                if consumer.op_type == OpType.MatMulNBits:
                    new_initializer_name = new_initializer_name + ".quant"
                    quant_modules.add(new_initializer_name)
                else:
                    float_modules.add(new_initializer_name.replace(".weight", ""))
                self._externalize_initializer(ir_model, weights, initializer, new_initializer_name)

                # check if the 2nd weight is quantized
                node_inputs = consumer.inputs
                if len(node_inputs) < 2:
                    continue
                sec_weight = node_inputs[1]
                if sec_weight.is_initializer():
                    continue
                sec_weight_new_name = sec_weight.name + ".weight"
                producer_op_type = sec_weight.producer().op_type

                if producer_op_type == OpType.DequantizeLinear:
                    quant_suffixes = [".quant.weight", ".quant.scale", ".quant.zero_point"]
                    new_quant_names = [sec_weight_new_name.replace(".weight", suf) for suf in quant_suffixes]
                    self._process_dequantizelinear(
                        ir_model,
                        consumer.name,
                        weights,
                        sec_weight,
                        sec_weight_new_name,
                        new_quant_names,
                        nodes_to_remove,
                    )
        ir_model.graph.remove(nodes_to_remove)

        if config.make_inputs and quant_modules and config.dynamic_lora_r:
            # No use case for DequantizeLinear with dynamic lora_r
            logger.info("Quantized modules do not support dynamic_lora_r. Ignoring.")

        return weights

    def _process_div_node(
        self,
        ir_model: ir.Model,
        node_name: str,
        weights: dict[str, "NDArray"],
        new_weight_name: str,
        float_modules: set[str],
    ):
        old_weight_name = ir_model.graph[node_name].inputs[0].name
        self._externalize_initializer(ir_model, weights, old_weight_name, new_weight_name)
        # add the module to the float modules
        float_modules.add(new_weight_name.replace(".weight", ""))

    def _process_matmul_node(
        self,
        ir_model: ir.Model,
        node_name: str,
        weights: dict[str, "NDArray"],
        new_weight_name: str,
        float_modules: set[str],
        quant_modules: set[str],
        new_quantized_names: list[str],
        nodes_to_remove: set[str],
    ):
        # float or QDQ quantized
        # original weight name
        old_weight: ir.Value = ir_model.graph[node_name].inputs[1]
        if old_weight.is_graph_input():
            # nothing to do here
            return
        if old_weight.is_initializer():
            self._externalize_initializer(ir_model, weights, old_weight.name, new_weight_name)

            # add the module to the float modules
            float_modules.add(new_weight_name.replace(".weight", ""))
        elif old_weight.producer().op_type == OpType.DequantizeLinear:
            self._process_dequantizelinear(
                ir_model, node_name, weights, old_weight, new_weight_name, new_quantized_names, nodes_to_remove
            )

            # add the module to the quant modules
            quant_modules.add(new_weight_name.replace(".weight", ".quant"))

    def _process_matmulnbits_node(
        self,
        ir_model: ir.Model,
        node: ir.Node,
        weights: dict[str, "NDArray"],
        new_weight_name: str,
        quant_modules: set[str],
        new_quantized_names: list[str],
    ):
        # weight is Nbits quantized
        # create empty initializers and change node inputs
        for old_input, new_input in zip(node.inputs[1:], new_quantized_names):
            self._externalize_initializer(ir_model, weights, old_input, new_input)

        # add the module to the quant modules
        quant_modules.add(new_weight_name.replace(".weight", ".quant"))

    def _process_dequantizelinear(
        self,
        ir_model: ir.Model,
        node_name: str,
        weights: dict[str, "NDArray"],
        old_weight: ir.Value,
        new_weight_name: str,
        new_quantized_names: list[str],
        nodes_to_remove: set[str],
    ):
        # weight is QDQ quantized
        # get the dequantize node
        old_dequantize_node = old_weight.producer()

        # zero point is optional so we keep track of used inputs
        used_inputs = []
        new_input_values = []
        # create new initializers for the dequantize node
        for old_input, new_input in zip(old_dequantize_node.inputs, new_quantized_names):
            initializer = self._externalize_initializer(ir_model, weights, old_input, new_input)
            used_inputs.append(new_input)
            new_input_values.append(initializer)

        # create a new dequantize node
        # NOTE: We could directly modify the original dequantize node but this assumes that the dequantize
        # node is not used elsewhere
        # this cannot be guaranteed (for instance, if the float model has lora modules with same weights,
        # they might all share the same dequantize node)
        new_node = ir.Node(
            domain=old_dequantize_node.domain,
            op_type=old_dequantize_node.op_type,
            inputs=new_input_values,
            attributes=old_dequantize_node.attributes,
            name=new_weight_name.replace("weight", "dequantize"),
            graph=ir_model.graph,
        )
        new_node.outputs[0].name = new_weight_name

        consumer_node = ir_model.graph[node_name]
        for i, inp in enumerate(consumer_node.inputs):
            if inp == old_weight:
                consumer_node.inputs[i] = new_node.outputs[0]
                break

        # add old dequantize node to remove
        nodes_to_remove.add(old_dequantize_node.name)

    @staticmethod
    def _create_new_weight_name(old_name: str, adapter_type: AdapterType = AdapterType.LORA) -> str:
        """Create new weight name based on old name.
        LORA: the new weight name is of the form model.layers.0.self_attn.q_proj.lora_A.quant.weight
        DORA: the new weight name is of the form model.layers.0.self_attn.q_proj.dora_A.weight for MatMul and
                model.layers.0.self_attn.q_proj.dora_M.weight for Mul
        """
        weight_name = old_name[1:] if old_name.startswith("/") else old_name
        op = weight_name.split("/")[-1]
        if adapter_type == AdapterType.LORA:
            return (
                weight_name.replace("/", ".")
                .replace("default.", "lora_A.")
                .replace("default_1.", "lora_B.")
                .replace("default_0.", "lora_A.")
                .replace("default_0_1.", "lora_B.")
                .replace(op, "weight")
            )
        if adapter_type == AdapterType.DORA:
            return (
                weight_name.replace("/", ".")
                .replace("default.default.", "dora_A.")  # For MatMul
                .replace("default.default_1.", "dora_B.")  # For MatMul
                .replace("default.", "dora_M.")  # For Div
                .replace(op, "weight")
            )
        if adapter_type == AdapterType.LOHA:
            return weight_name.replace("default", "weight")
        raise ValueError(f"Unsupported adapter type: {adapter_type}")

    @classmethod
    def _externalize_initializer(
        cls, ir_model: ir.Model, weights: dict[str, "NDArray"], initializer: ir.Value, new_name: str
    ):
        """Create a new initializer with the same shape and type as the old initializer.

        The initializer points to a dummy external location.
        Add the new initializer to the graph and store the weight in a dictionary.

        :param ir_model: IR model
        :param weights: dictionary to store the weights
        :param initializer: initializer to copy
        :param new_name: new initializer name
        """
        assert initializer.is_initializer(), f"{initializer.name} is not an initializer"

        old_name = initializer.name

        # Store the weight data
        weights[new_name] = initializer.const_value.numpy()

        # Rename the initializer
        initializer.name = new_name

        # Update the initializers dictionary
        if old_name in ir_model.graph.initializers:
            ir_model.graph.initializers.pop(old_name)
            ir_model.graph.initializers[new_name] = initializer

        # Create external tensor
        external_tensor = ir.ExternalTensor(
            location="dummy-location.bin",
            offset=None,
            length=None,
            dtype=initializer.const_value.dtype,
            shape=initializer.const_value.shape,
            name=new_name,
            base_dir="",
        )

        initializer.const_value = external_tensor
        return initializer

    def _make_dynamic_optional(
        self, model: ir.Model, weights: dict[str, "NDArray"], name: str, config: type[BasePassConfig]
    ):
        """Make the input dynamic and optional."""
        if "lora_magnitude_vector" in name:
            # magnitude vector's shape is independent of lora_r, so we do nothing
            return

        # Determine which dimension should be made dynamic based on pattern in name
        dim_idx = 1
        if "lora_A" in name:
            dim_idx = 1
        elif "lora_B" in name:
            dim_idx = 0
        elif "hada_w1_a" in name or "hada_w2_a" in name:
            dim_idx = 0  # For the first matrix in Hadamard products
        elif "hada_w1_b" in name or "hada_w2_b" in name:
            dim_idx = 1  # For the second matrix in Hadamard products

        # make the input dynamic
        if config.dynamic_lora_r:
            self._make_input_dim_dynamic(model, name, dim_idx, "lora_r")

        # create default initializer with the lora_r dimension set to 0
        if config.optional_inputs:
            shape = list(weights[name].shape)
            shape[dim_idx] = 0
            zero_array = np.zeros(shape, dtype=weights[name].dtype)
            initializer_value = model.graph.initializers[name]
            initializer_value.const_value = ir.Tensor(zero_array)
            model.graph.inputs.append(initializer_value)

    def _make_input_dim_dynamic(self, model: ir.Model, input_name: str, dim_idx: int, dim_param: str):
        """Make a dimension of an input dynamic."""
        # Find the input value
        input_value = None
        for inp in model.graph.inputs:
            if inp.name == input_name:
                input_value = inp
                break

        if input_value is None:
            raise ValueError(f"{input_name} is not an input.")

        if input_value.shape is None:
            raise ValueError(f"Input {input_name} does not have shape information.")

        if dim_idx >= len(input_value.shape):
            raise ValueError(
                f"Input {input_name} has rank {len(input_value.shape.dims)} but trying to access dim {dim_idx}."
            )

        # Create new shape with symbolic dimension
        new_dims = list(input_value.shape)
        if isinstance(new_dims[dim_idx], ir.SymbolicDim) and new_dims[dim_idx].value is not None:
            raise ValueError(f"Can't replace existing dynamic dim {new_dims[dim_idx].value} with {dim_param}")

        new_dims[dim_idx] = ir.SymbolicDim(dim_param)
        input_value.shape = ir.Shape(new_dims)
