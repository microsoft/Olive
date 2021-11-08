import logging
import os
import sys
import pkg_resources
import numpy as np

ONNXRUNTIME_VERSION = "1.9.0"
ONNX_VERSION = "1.7.0"
PYTORCH_VERSION = "1.7"
TENSORFLOW_VERSION = "1.15"
FRAMEWORK_TENSORFLOW = "tensorflow"
FRAMEWORK_PYTORCH = "pytorch"

WARMUP_NUM = 10
TEST_NUM = 20
OLIVE_RESULT_PATH = "olive_opt_result"
SUB_PROCESS_NAME_PREFIX = "concurrency_subprocess"
ONNX_MODEL_PATH = "res.onnx"
QUERY_COUNT =500
MILLI_SEC = 1000

ONNX_TO_NP_TYPE_MAP = {
    "tensor(bool)": np.bool,
    "tensor(int)": np.int32,
    'tensor(int32)': np.int32,
    'tensor(int8)': np.int8,
    'tensor(uint8)': np.uint8,
    'tensor(int16)': np.int16,
    'tensor(uint16)': np.uint16,
    'tensor(uint64)': np.uint64,
    "tensor(int64)": np.int64,
    'tensor(float16)': np.float16,
    "tensor(float)": np.float32,
    'tensor(double)': np.float64,
    'tensor(string)': np.string_,
}


EP_TO_PROVIDER_TYPE_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dnnl": "DnnlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider"
}

EXECUTION_MODE_MAP = {
    "sequential": 0,
    "parallel": 1
}

ORT_OPT_LEVEL_MAP = {
    "disable": 0,
    "basic": 1,
    "extended": 2,
    "all": 99
}

STR_TO_NP_TYPE_MAP = {
    "bool": np.bool,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint': np.uint32,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'int8': np.int8,
    'int16': np.int16,
    'int': np.int32,
    'int32': np.int32,
    "int64": np.int64,
    'float16': np.float16,
    'float': np.float32,
    "float32": np.float32,
    'float64': np.float64,
    'string': np.string_,
}

# setup requirenemts for conversion
SETUP_REQUIREMENTS_CONVERSION = {
    "tensorflow_1.11": ["tensorflow==1.11.0", "pandas==0.23.4", "onnx=={}".format(ONNX_VERSION), "tf2onnx==1.7.2"],
    "tensorflow_1.12": ["tensorflow==1.12.0", "pandas==0.23.4", "onnx=={}".format(ONNX_VERSION), "tf2onnx==1.7.2"],
    "tensorflow_1.13": ["tensorflow==1.13.1", "pandas==0.23.4", "onnx=={}".format(ONNX_VERSION), "tf2onnx==1.7.2"],
    "tensorflow_1.14": ["tensorflow==1.14.0", "pandas==0.23.4", "onnx=={}".format(ONNX_VERSION), "tf2onnx==1.7.2"],
    "tensorflow_1.15": ["tensorflow==1.15.0", "pandas==0.23.4", "onnx=={}".format(ONNX_VERSION), "tf2onnx==1.7.2"],
    "pytorch_1.3": ["onnx=={}".format(ONNX_VERSION), "torch==1.3.0+cpu", "torchvision==0.4.1+cpu"],
    "pytorch_1.4": ["onnx=={}".format(ONNX_VERSION), "torch==1.4.0+cpu", "torchvision==0.5.0+cpu"],
    "pytorch_1.5": ["onnx=={}".format(ONNX_VERSION), "torch==1.5.1+cpu", "torchvision==0.6.1+cpu"],
    "pytorch_1.6": ["onnx=={}".format(ONNX_VERSION), "torch==1.6.0+cpu", "torchvision==0.7.0+cpu"],
    "pytorch_1.7": ["onnx=={}".format(ONNX_VERSION), "torch==1.7.0+cpu", "torchvision==0.8.1+cpu"]
}

LOGGING_LEVEL_MAP = {
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}

OLIVE_LOG_LEVEL = os.getenv("OLIVE_LOG_LEVEL").upper() if os.getenv("OLIVE_LOG_LEVEL") else "ERROR"


MODEL_MOUNT_DIR = "/mnt/model"
OPT_IMG_NAME = "olive_optimization"
CVT_IMG_NAME = "olive_conversion"
OLIVE_MOUNT_DIR = "/mnt/olive"
MCR_PREFIX = "mcr.microsoft.com/olive"

PYTHON_PATH = sys.executable

def get_packages_dict():
    installed_packages = pkg_resources.working_set
    installed_packages_dict = {}
    for i in installed_packages:
        installed_packages_dict[i.key] = i.version
    return installed_packages_dict

INSTALLED_PACKAGES_DICT = get_packages_dict()
