import logging
from shutil import copy
import subprocess

import onnxruntime as ort
import os
import tempfile

from ..constants import OLIVE_LOG_LEVEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ort.set_default_logger_severity(4)


def transformer_optimize(optimization_config):
    if optimization_config.transformer_args and "model_type" in optimization_config.transformer_args:
        optimize_by_model_type(optimization_config)
    else:
        for model_type in (["bert", "gpt2", "bert_tf", "bert_keras"]):
            optimize_by_model_type(optimization_config, model_type)


def optimize_by_model_type(optimization_config, model_type=None):
    middle_path = os.path.join(tempfile.mkdtemp(), 'middle_optimized.onnx')
    cmd = "python -m onnxruntime.transformers.optimizer --input %s --output %s " % (optimization_config.model_path, middle_path)
    if optimization_config.transformer_args:
        cmd += optimization_config.transformer_args
    if model_type:
        cmd += " --model_type {}".format(model_type)
    logger.info("Running TransformersOptimizer with command {}".format(cmd))

    if OLIVE_LOG_LEVEL == "INFO":
        ret = subprocess.run(cmd, shell=True)
    elif OLIVE_LOG_LEVEL == "WARNING":
        ret = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    else:
        ret = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if ret.returncode == 0:
        try:
            default_ep = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
            ort.InferenceSession(middle_path, providers=[default_ep])
            logger.info("Transformers optimization finished with success")
            copy(middle_path, optimization_config.model_path)
        except Exception:
            logger.info("Invalid model after transformer optimization. Original model will be used.")
    else:
        logger.info("Transformers optimization failed. Original model will be used for optimization.")
