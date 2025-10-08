# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

from olive.common.utils import WeightsFileFormat, save_weights
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    DORA_NAME_PATTERNS,
    LOHA_NAME_PATTERNS,
    LORA_NAME_PATTERNS,
    AdapterType,
    get_adapter_name,
    get_external_data_config,
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
        logger.warning("=== ExtractAdapters Pass START ===")
        logger.warning(f"Input model path: {model.model_path}")
        logger.warning(f"Output model path: {output_model_path}")
        logger.warning(f"Adapter type: {config.adapter_type}")
        logger.warning(f"make_inputs: {config.make_inputs}")
        logger.warning(f"save_format: {config.save_format}")
        
        # Validate input model
        if model is None:
            logger.error("Input model is None!")
            return None

        if not hasattr(model, 'model_path') or not model.model_path:
            logger.error("Input model has no valid model_path!")
            return None

        logger.warning(f"Input model type: {type(model)}")
        logger.warning(f"Input model attributes: {getattr(model, 'model_attributes', 'None')}")
        
        try:
            output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
            logger.warning(f"Resolved output model path: {output_model_path}")
        except Exception as e:
            logger.error(f"Error resolving output model path: {e}")
            return None

        try:
            logger.warning("Loading IR model...")
            ir_model = model.load_ir_model()
            logger.warning(f"IR model loaded successfully, type: {type(ir_model)}")

            logger.warning("Loading external data to model...")
            ir.external_data.load_to_model(ir_model)
            logger.warning("External data loaded")

            # Log model basic info
            if hasattr(ir_model, 'graph') and ir_model.graph:
                logger.warning(f"Number of initializers in graph: {len(ir_model.graph.initializers)}")
                logger.warning(f"Number of inputs in graph: {len(ir_model.graph.inputs)}")
                logger.warning(f"Number of outputs in graph: {len(ir_model.graph.outputs)}")

                # Log first few initializer names
                init_names = list(ir_model.graph.initializers.keys())[:10]
                logger.warning(f"First 10 initializer names: {init_names}")
            else:
                logger.error("IR model has no valid graph structure!")
                return None

        except Exception as e:
            logger.error(f"Error loading IR model: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full error stack: {traceback.format_exc()}")
            return None

        # dictionary to store adapter weights
        weights = {}

        try:
            logger.warning(f"Starting to extract {config.adapter_type} adapter weights...")
            if config.adapter_type in [AdapterType.LORA, AdapterType.DORA, AdapterType.LOHA]:
                weights = self._extract_adapter(ir_model, adapter_type=config.adapter_type)
                logger.warning(f"Number of extracted weights: {len(weights)}")
                if weights:
                    logger.warning(f"Weight names: {list(weights.keys())}")
                    # Log weight shapes
                    for name, weight in weights.items():
                        logger.warning(f"Weight {name}: shape={weight.shape}, dtype={weight.dtype}")
                else:
                    logger.warning("No weights extracted!")
            else:
                logger.error(f"Unsupported adapter type: {config.adapter_type}")
                raise ValueError(f"Unsupported adapter type: {config.adapter_type}")
        except Exception as e:
            logger.error(f"Error extracting adapter weights: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full error stack: {traceback.format_exc()}")
            return None

        if not weights:
            logger.warning("No %s modules found in the model. Returning the original model.", config.adapter_type)
            logger.warning("=== ExtractAdapters Pass END (returning original model) ===")
            return model

        try:
            if config.make_inputs:
                logger.warning("Starting to convert weights to inputs...")
                # create inputs for the weights
                for weight_name in weights:
                    logger.warning(f"Processing weight: {weight_name}")
                    self._convert_initializer_to_input(ir_model, weight_name)
                    self._make_dynamic_optional(ir_model, weights, weight_name, config)
                logger.warning("Weights conversion to inputs completed")
        except Exception as e:
            logger.error(f"Error converting weights to inputs: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full error stack: {traceback.format_exc()}")
            return None

        try:
            logger.warning("Starting to save weights file...")
            weights_path = save_weights(weights, Path(output_model_path).parent / "adapter_weights", config.save_format)
            logger.warning(f"Weights file saved successfully: {weights_path}")
        except Exception as e:
            logger.error(f"Error saving weights file: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full error stack: {traceback.format_exc()}")
            return None

        try:
            weights_file_name = weights_path.name if hasattr(weights_path, 'name') else Path(weights_path).name
            external_init_file = weights_file_name if not config.make_inputs else None
            constant_inputs_file = weights_file_name if config.make_inputs else None
            logger.warning("Starting to save model...")
            # save the model
            output_model = model_proto_to_olive_model(
                ir.to_proto(ir_model),
                output_model_path,
                config,
                external_initializers_file_name=external_init_file,
                constant_inputs_file_name=constant_inputs_file,
            )

            if output_model is None:
                logger.error("model_proto_to_olive_model returned None!")
                return None

            logger.warning(f"Output model created successfully, type: {type(output_model)}")
            logger.warning(f"Output model path: {getattr(output_model, 'model_path', 'None')}")
            
        except Exception as e:
            logger.error(f"Error creating output model: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            logger.error(f"Full error stack: {traceback_str}")
            return None

        try:
            logger.warning("Starting to set model attributes...")
            output_model.model_attributes = deepcopy(model.model_attributes) or {}
            logger.warning(f"Copied model attributes: {output_model.model_attributes}")

            # add adapter weights to the model attributes
            output_model.model_attributes["additional_files"] = additional_files = output_model.model_attributes.get(
                "additional_files", []
            )
            additional_files.append(str(weights_path))
            logger.warning(f"Added additional files: {additional_files}")

            # save information about the weights in the model attributes
            weights_info = {name: [list(value.shape), str(value.dtype)] for name, value in weights.items()}
            logger.warning(f"Weights info: {weights_info}")

            if not config.make_inputs:
                output_model.model_attributes["external_initializers"] = weights_info
                logger.warning("Weights info saved as external_initializers")
            else:
                output_model.model_attributes["constant_inputs"] = weights_info
                logger.warning("Weights info saved as constant_inputs")

            logger.warning(f"Final model attributes: {output_model.model_attributes}")

        except Exception as e:
            logger.error(f"Error setting model attributes: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full error stack: {traceback.format_exc()}")
            return None

        logger.warning("=== ExtractAdapters Pass COMPLETED SUCCESSFULLY ===")
        logger.warning(f"Returned model type: {type(output_model)}")
        logger.warning(f"Returned model path: {getattr(output_model, 'model_path', 'None')}")
        return output_model

    def _convert_initializer_to_input(self, model: ir.Model, initializer_name: str):
        """Convert a specific initializer to an input."""
        logger.warning(f"Converting initializer to input: {initializer_name}")

        graph = model.graph

        # Check if the initializer exists
        if initializer_name not in graph.initializers:
            logger.error(f"Initializer '{initializer_name}' not found in graph!")
            logger.warning(f"Available initializers: {list(graph.initializers.keys())[:10]}...")
            raise ValueError(f"Initializer '{initializer_name}' not found in graph")

        # Get the initializer
        initializer = graph.initializers[initializer_name]
        logger.warning(f"Found initializer: {initializer_name}, type: {type(initializer)}")

        # Check if it's already an input
        if initializer in graph.inputs:
            logger.warning(f"Initializer {initializer_name} already an input, skipping")
            return  # Already an input

        # Add to inputs
        graph.inputs.append(initializer)
        logger.warning(f"Successfully added {initializer_name} to input list, current number of inputs: {len(graph.inputs)}")

    def _decompose_gemm(self, ir_model: ir.Model):
        """Decompose Gemm nodes into MatMul and Add nodes."""
        from onnxscript import rewriter
        from onnxscript.rewriter.rules.common import gemm_to_matmul_add_rule

        return rewriter.rewrite(ir_model, pattern_rewrite_rules=[gemm_to_matmul_add_rule])

    def _extract_adapter(self, ir_model: ir.Model, adapter_type: AdapterType = AdapterType.LORA):
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
        logger.warning(f"=== Starting to extract {adapter_type} adapter weights ===")

        if adapter_type == AdapterType.DORA:
            logger.warning("DoRA type, need to decompose Gemm nodes first...")
            try:
                ir_model = self._decompose_gemm(ir_model)
                logger.warning("Gemm node decomposition completed")
            except Exception as e:
                logger.error(f"Error decomposing Gemm nodes: {e}")
                raise

        # dictionary to store adapter weights
        weights = {}

        # Get the appropriate patterns for the adapter type
        patterns = None
        if adapter_type == AdapterType.LORA:
            patterns = LORA_NAME_PATTERNS
            logger.warning(f"Using LoRA patterns: {patterns}")
        elif adapter_type == AdapterType.DORA:
            patterns = DORA_NAME_PATTERNS
            logger.warning(f"Using DoRA patterns: {patterns}")
        elif adapter_type == AdapterType.LOHA:
            patterns = LOHA_NAME_PATTERNS
            logger.warning(f"Using LoHa patterns: {patterns}")
        else:
            logger.error(f"Unsupported adapter type: {adapter_type}")
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

        logger.warning(f"Starting to scan initializers in model, total: {len(ir_model.graph.initializers)}")

        to_rename = []
        matched_count = 0
        for i, initializer in enumerate(ir_model.graph.initializers.values()):
            if i < 10:  # Only logging details of first 10
                logger.warning(f"Checking initializer [{i}]: {initializer.name}")

            adapter_weight = get_adapter_name(initializer, patterns)
            if adapter_weight is None:
                if i < 10:
                    logger.warning(f"  -> Does not match any adapter pattern")
                continue

            logger.warning(f"Found matching adapter weight: {initializer.name} -> {adapter_weight}")
            to_rename.append((initializer, adapter_weight))
            matched_count += 1

        logger.warning(f"Total found {matched_count} matching adapter weights")
        
        if not to_rename:

            # Log some initializer names for debugging
            init_names = list(ir_model.graph.initializers.keys())[:20]
            logger.warning("No matching adapter weights found!")
            logger.warning("Possible reasons:")
            logger.warning("1. Model has no adapter weights")
            logger.warning("2. Adapter weight naming pattern does not match expectations")
            logger.warning("3. Wrong adapter type selected")
            logger.warning(f"First 20 initializer names in model: {init_names}")
            return weights

        logger.warning("Starting to process matching adapter weights...")
        for i, (initializer, adapter_weight) in enumerate(to_rename):
            logger.warning(f"Processing weight [{i+1}/{len(to_rename)}]: {initializer.name} -> {adapter_weight}")

            try:
                old_name = initializer.name

                # Store the weight data
                if hasattr(initializer, 'const_value') and initializer.const_value is not None:
                    weight_data = initializer.const_value.numpy()
                    weights[adapter_weight] = weight_data
                    logger.warning(f"  Weight data extracted successfully: shape={weight_data.shape}, dtype={weight_data.dtype}")
                else:
                    logger.error(f"  Initializer {old_name} has no valid const_value")
                    continue

                # Rename the initializer
                initializer.name = adapter_weight

                # Update the initializers dictionary
                if old_name in ir_model.graph.initializers:
                    ir_model.graph.initializers.pop(old_name)
                    ir_model.graph.initializers[adapter_weight] = initializer
                    logger.warning(f"  Initializer dictionary updated successfully")
                else:
                    logger.warning(f"  Initializer {old_name} not in dictionary")

                # Create external tensor
                external_tensor = ir.ExternalTensor(
                    location="dummy-location.bin",
                    offset=None,
                    length=None,
                    dtype=initializer.const_value.dtype,
                    shape=initializer.const_value.shape,
                    name=adapter_weight,
                    base_dir="",
                )

                initializer.const_value = external_tensor
                logger.warning(f"  External tensor created successfully")

            except Exception as e:
                logger.error(f"Error processing weight {initializer.name}: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Full error stack: {traceback.format_exc()}")
                continue

        logger.warning(f"=== Adapter weight extraction completed, successfully extracted {len(weights)} weights ===")
        return weights

    def _make_dynamic_optional(
        self, model: ir.Model, weights: dict[str, "NDArray"], name: str, config: type[BasePassConfig]
    ):
        """Make the input dynamic and optional."""
        logger.warning(f"Processing dynamic optional input: {name}")

        if "lora_magnitude_vector" in name:
            # magnitude vector's shape is independent of lora_r, so we do nothing
            logger.warning(f"Skipping magnitude vector: {name}")
            return

        # Determine which dimension should be made dynamic based on pattern in name
        dim_idx = 1
        if "lora_A" in name:
            dim_idx = 1
            logger.warning(f"LoRA A weight, using dimension index: {dim_idx}")
        elif "lora_B" in name:
            dim_idx = 0
            logger.warning(f"LoRA B weight, using dimension index: {dim_idx}")
        elif "hada_w1_a" in name or "hada_w2_a" in name:
            dim_idx = 0  # For the first matrix in Hadamard products
            logger.warning(f"LoHa w1_a/w2_a weight, using dimension index: {dim_idx}")
        elif "hada_w1_b" in name or "hada_w2_b" in name:
            dim_idx = 1  # For the second matrix in Hadamard products
            logger.warning(f"LoHa w1_b/w2_b weight, using dimension index: {dim_idx}")

        # make the input dynamic
        if config.dynamic_lora_r:
            logger.warning(f"Setting dynamic lora_r dimension: {name}, dim_idx={dim_idx}")
            try:
                self._make_input_dim_dynamic(model, name, dim_idx, "lora_r")
                logger.warning(f"Successfully set dynamic dimension: {name}")
            except Exception as e:
                logger.error(f"Error setting dynamic dimension: {e}")
                raise
        else:
            logger.warning(f"Skipping dynamic dimension setting (dynamic_lora_r=False): {name}")

        # create default initializer with the lora_r dimension set to 0
        if config.optional_inputs:
            logger.warning(f"Creating default initializer for optional input: {name}")
            try:
                shape = list(weights[name].shape)
                original_shape = shape.copy()
                shape[dim_idx] = 0
                logger.warning(f"Original shape: {original_shape}, New shape: {shape}")

                zero_array = np.zeros(shape, dtype=weights[name].dtype)
                logger.warning(f"Creating zero array: shape={zero_array.shape}, dtype={zero_array.dtype}")

                initializer_value = model.graph.initializers[name]
                initializer_value.const_value = ir.Tensor(zero_array)
                model.graph.inputs.append(initializer_value)
                logger.warning(f"Successfully created optional input: {name}")
            except Exception as e:
                logger.error(f"Error creating optional input: {e}")
                raise
        else:
            logger.warning(f"Skipping optional input creation (optional_inputs=False): {name}")

    def _make_input_dim_dynamic(self, model: ir.Model, input_name: str, dim_idx: int, dim_param: str):
        """Make a dimension of an input dynamic."""
        logger.warning(f"Setting input dimension to dynamic: {input_name}, dim_idx={dim_idx}, dim_param={dim_param}")

        # Find the input value
        input_value = None
        for inp in model.graph.inputs:
            if inp.name == input_name:
                input_value = inp
                break

        if input_value is None:
            logger.error(f"{input_name} is not an input!")
            logger.warning(f"Current input list: {[inp.name for inp in model.graph.inputs]}")
            raise ValueError(f"{input_name} is not an input.")

        logger.warning(f"Found input: {input_name}, type: {type(input_value)}")

        if input_value.shape is None:
            logger.error(f"Input {input_name} has no shape information!")
            raise ValueError(f"Input {input_name} does not have shape information.")

        logger.warning(f"Input shape: {input_value.shape}, length: {len(input_value.shape)}")

        if dim_idx >= len(input_value.shape):
            logger.error(f"Input {input_name} has rank {len(input_value.shape)}, but trying to access dimension {dim_idx}")
            raise ValueError(
                f"Input {input_name} has rank {len(input_value.shape.dims)} but trying to access dim {dim_idx}."
            )

        # Create new shape with symbolic dimension
        new_dims = list(input_value.shape)
        logger.warning(f"Original dimensions: {new_dims}")

        if isinstance(new_dims[dim_idx], ir.SymbolicDim) and new_dims[dim_idx].value is not None:
            logger.error(f"Cannot replace existing dynamic dimension {new_dims[dim_idx].value} with {dim_param}")
            raise ValueError(f"Can't replace existing dynamic dim {new_dims[dim_idx].value} with {dim_param}")

        new_dims[dim_idx] = ir.SymbolicDim(dim_param)
        input_value.shape = ir.Shape(new_dims)
        logger.warning(f"New dimensions: {new_dims}")
        logger.warning(f"Successfully set dynamic dimension: {input_name}[{dim_idx}] = {dim_param}")
