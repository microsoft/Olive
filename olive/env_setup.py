import argparse
import subprocess

from .constants import ONNXRUNTIME_VERSION, SETUP_REQUIREMENTS_CONVERSION, PYTORCH_VERSION, TENSORFLOW_VERSION, PYTHON_PATH, INSTALLED_PACKAGES_DICT
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_packages(onnxruntime_version=None, use_gpu=False, model_framework=None, framework_version=None):
    # install packages for throughput tuning
    try:
        import mlperf_loadgen
    except ImportError:
        install_cmd = "{} -m pip install mlperf_loadgen --extra-index-url https://olivewheels.azureedge.net/test".format(PYTHON_PATH)
        logger.info(install_cmd)
        subprocess.run(install_cmd, stdout=subprocess.PIPE, shell=True, check=True)
        logger.info("loadgen package installed with success")

    # install onnxruntime packages for optimization
    ort_package = "onnxruntime_gpu_tensorrt" if use_gpu else "onnxruntime_openvino_dnnl"
    ort_version = onnxruntime_version if onnxruntime_version else ONNXRUNTIME_VERSION
    if onnxruntime_version or use_gpu:
        try:
            import onnxruntime
            if (onnxruntime.__version__ != onnxruntime_version) or (use_gpu and "CUDAExecutionProvider" not in onnxruntime.get_available_providers()):
                raise ImportError
        except ImportError:
            install_cmd = "{} -m pip install {}=={} --extra-index-url https://olivewheels.azureedge.net/test".format(
                PYTHON_PATH, ort_package, ort_version)
            logger.info(install_cmd)
            subprocess.run(install_cmd, stdout=subprocess.PIPE, shell=True, check=True)
            logger.info("{}=={} installed with success".format(ort_package, ort_version))
    else:
        try:
            import onnxruntime
        except ImportError:
            install_cmd = "{} -m pip install {}=={} --extra-index-url https://olivewheels.azureedge.net/test".format(
                PYTHON_PATH, ort_package, ort_version)
            logger.info(install_cmd)
            subprocess.run(install_cmd, stdout=subprocess.PIPE, shell=True, check=True)
            logger.info("{}=={} installed with success".format(ort_package, ort_version))

    # install packages for conversion
    if model_framework:
        model_framework = model_framework.lower()
        if model_framework == "pytorch":
            if framework_version not in ["1.3", "1.4", "1.5", "1.6", "1.7"]:
                framework_version = PYTORCH_VERSION
                logger.info(
                    "PyTorch framework version can only be selected in {}. framework_version={} will be used".format(
                        ["1.3", "1.4", "1.5", "1.6", "1.7"], PYTORCH_VERSION))
        elif model_framework == "tensorflow":
            if framework_version not in ["1.11", "1.12", "1.13", "1.14", "1.15"]:
                framework_version = TENSORFLOW_VERSION
                logger.info(
                    "TensorFlow framework version can only be selected in {}. framework_version={} will be used".format(
                        ["1.11", "1.12", "1.13", "1.14", "1.15"], TENSORFLOW_VERSION))
        else:
            raise Exception("model_framework should be selected from pytorch or tensorflow")

        requirements = SETUP_REQUIREMENTS_CONVERSION.get("{}_{}".format(model_framework, framework_version))
        install_cmd = "{} -m pip install {} -f https://download.pytorch.org/whl/torch_stable.html".format(
            PYTHON_PATH, " ".join(requirements))
        logger.info(install_cmd)
        subprocess.run(install_cmd, stdout=subprocess.PIPE, shell=True, check=True)


def install_server_dependencies():
    install_cmd = "{} -m pip install pandas netron flask flask-cors redis celery[redis] flower".format(PYTHON_PATH)
    subprocess.run(install_cmd, stdout=subprocess.PIPE, shell=True, check=True)

