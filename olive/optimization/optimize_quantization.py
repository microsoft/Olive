import logging
from shutil import copy

import onnxruntime as ort
from onnxruntime import quantization
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ort.set_default_logger_severity(4)


def quantization_optimize(optimization_config):
    logger.info("ONNX model quantization started")
    base_dir = os.path.dirname(optimization_config.model_path)
    unquantized_model = os.path.join(base_dir, "unquantized_model.onnx")
    copy(optimization_config.model_path, unquantized_model)
    try:
        quantization.quantize_dynamic(unquantized_model, optimization_config.model_path)
        default_ep = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
        ort.InferenceSession(optimization_config.model_path, providers=[default_ep])
        logger.info("ONNX model quantized successfully")
    except Exception as e:
        logger.info("Quantization optimization failed with error {}. Original model will be used for optimization.".format(e))
        copy(unquantized_model, optimization_config.model_path)
