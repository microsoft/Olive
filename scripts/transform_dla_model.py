import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Callable

import onnx

from olive.passes.onnx.dla_transforms import transform_add_intermediate_tensors_to_outputs

logger = logging.getLogger(__name__)


# pylint: disable=W0621


def setup_logging(model_path):
    model_path = Path(model_path)
    model_name = model_path.stem
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"transform_{model_name}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def execute_shape_inference(input_model, output_model):
    try:
        logger.info("Running shape inference on %s", input_model)
        model = onnx.load(input_model)

        # Run shape inference
        try:
            # Try symbolic shape inference first if available
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

            inferred_model = SymbolicShapeInference.infer_shapes(
                model, int_max=2**31 - 1, auto_merge=True, guess_output_rank=False, verbose=3
            )
            logger.info("Symbolic shape inference completed successfully")
        except ImportError:
            # Fall back to standard ONNX shape inference
            logger.info("Using standard ONNX shape inference (symbolic shape inference not available)")
            inferred_model = onnx.shape_inference.infer_shapes(model)
            logger.info("Standard shape inference completed successfully")

        # Save the inferred model
        onnx.save(inferred_model, output_model)
        logger.info("Shape inference completed successfully. Output saved to %s", output_model)

    except Exception:
        logger.exception("Error running shape inference")
        # If shape inference fails, just copy the original model
        logger.info("Falling back to original model without shape inference")
        model = onnx.load(input_model)
        onnx.save(model, output_model)


def all_tensors_are_4d(model_path):
    """Check if all tensors in the model are already 4D."""
    try:
        model = onnx.load(model_path)
        # Check all value_info, inputs, and outputs
        for value_info in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
            if hasattr(value_info.type, "tensor_type") and hasattr(value_info.type.tensor_type, "shape"):
                shape = value_info.type.tensor_type.shape
                if hasattr(shape, "dim") and len(shape.dim) != 4:
                    return False
        return True
    except Exception:
        return False


def get_available_transforms() -> dict[str, Callable]:
    """Dynamically load transform functions from transforms.py."""
    try:
        # Import the transforms module
        transforms = importlib.import_module("olive.passes.onnx.dla_transforms")

        # Find all functions starting with 'transform_'
        return {
            name: func
            for name, func in inspect.getmembers(transforms, inspect.isfunction)
            if name.startswith("transform_")
        }

    except ImportError:
        logger.exception("transforms.py module not found. Make sure it exists in the same directory.")
        return {}
    except Exception:
        logger.exception("Error loading transforms")
        return {}


def apply_transforms(
    model, transform_sequence: list[str], transform_functions: dict[str, Callable], options=None
) -> onnx.ModelProto:
    """Apply a sequence of transforms to the model."""
    if options is None:
        options = {}
    for transform_name in transform_sequence:
        logger.info("Applying transform: %s", transform_name)

        # Check if the transform exists in our function map
        if transform_name not in transform_functions:
            logger.warning("Transform '%s' not found in transforms.py - skipping", transform_name)
            continue

        # Get the function
        transform_func = transform_functions[transform_name]

        # Check function signature and apply accordingly
        sig = inspect.signature(transform_func)
        param_names = list(sig.parameters.keys())

        if len(param_names) == 1:
            # onnxscript transform functions returns transformed model
            if transform_name in ["transform_reshape_reducesum", "transform_reshape_clip_reducesum"]:
                model = transform_func(model)
            else:
                transform_func(model)
        elif len(param_names) > 1:
            # Handle special cases
            if transform_name == "transform_remove_qdq" and "keep_clip_after_inputs" in options:
                # Pass the keep_clip_after_inputs flag
                keep_clip_after_inputs = options.get("keep_clip_after_inputs", False)
                transform_func(model, keep_clip_after_inputs)
            elif transform_name == "transform_add_intermediate_tensors_to_outputs":
                # This function requires intermediate_tensor_to_add parameter
                logger.warning("%s requires additional parameters, skipping", transform_name)
                continue

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform ONNX model for 4D operations")
    parser.add_argument("--original_model", type=str, required=True, help="Path to the input ONNX model")
    parser.add_argument("--transformed_model", type=str, required=True, help="Path to save the transformed ONNX model")
    parser.add_argument(
        "--replace_qdq_with_clip", action="store_true", help="Whether to replace QDQ nodes with Clip nodes"
    )
    parser.add_argument(
        "--keep_clip_after_inputs",
        action="store_true",
        help="Whether to replace QDQ nodes after inputs with Clip nodes (this will effectly limits input range). This option is only valid when transform_remove_qdq is in transform_sequence",
    )
    parser.add_argument(
        "--debug_outputs", action="store_true", help="Add intermediate tensors as outputs for debugging"
    )
    parser.add_argument(
        "--transform_sequence",
        nargs="+",
        required=True,
        help="Sequence of transforms to apply, excluding qdq transform, which is controlled by --replace_qdq_with_clip flag",
    )
    parser.add_argument("--save_logging", action="store_true", help="Save log into logs directory")

    args = parser.parse_args()
    original_model = args.original_model
    transformed_model = args.transformed_model
    transform_sequence = args.transform_sequence
    debug_outputs = args.debug_outputs
    if args.save_logging:
        setup_logging(original_model)

    options = {"keep_clip_after_inputs": args.keep_clip_after_inputs}

    if args.replace_qdq_with_clip:
        transform_sequence.insert(0, "transform_qdq_to_clip")
        transformed_model = transformed_model.replace(".onnx", "_clipped.onnx")
        # replace transform_reshape_reducesum with transform_reshape_clip_reducesum
        if "transform_reshape_reducesum" in transform_sequence:
            transform_sequence[transform_sequence.index("transform_reshape_reducesum")] = (
                "transform_reshape_clip_reducesum"
            )
    else:
        transform_sequence.insert(0, "transform_remove_qdq")

    original_model_file_name = Path(original_model).name
    transform_functions = get_available_transforms()

    # Some models needs shape inference before transforming (unknown output shape error)
    # Create proper path for shape inferenced model
    shape_inferenced_original_model = str(
        Path(original_model).parent / original_model_file_name.replace(".onnx", "_shape_inferenced.onnx")
    )
    execute_shape_inference(original_model, shape_inferenced_original_model)

    if all_tensors_are_4d(shape_inferenced_original_model):
        logger.info("All tensors are 4D")
        sys.exit(0)

    if args.debug_outputs:
        intermediate_tensor_to_add = set()
        debug_original_model = onnx.load(shape_inferenced_original_model)
        debug_transformed_model_name = original_model.replace(".onnx", "_debug.onnx")
        transform_add_intermediate_tensors_to_outputs(debug_original_model, intermediate_tensor_to_add)
        onnx.save(debug_original_model, debug_transformed_model_name)

    model = onnx.load(shape_inferenced_original_model)

    model = apply_transforms(model, transform_sequence, transform_functions, options)

    onnx.save(model, transformed_model)

    shape_inferenced_transformed_model = transformed_model.replace(".onnx", "_shape_inferenced.onnx")
    execute_shape_inference(transformed_model, shape_inferenced_transformed_model)

    if debug_outputs:
        model = onnx.load(shape_inferenced_transformed_model)
        debug_transformed_model = transformed_model.replace(".onnx", "_debug.onnx")
        transform_add_intermediate_tensors_to_outputs(model, intermediate_tensor_to_add)
        onnx.save(model, debug_transformed_model)

    sys.exit(0)
