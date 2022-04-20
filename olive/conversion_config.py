import os

from .constants import ONNX_MODEL_PATH
from .constants import STR_TO_NP_TYPE_MAP
from .conversion.io_schema import IOSchema, IOSchemaLoader
from .conversion.opset_version import get_max_opset_version
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversionConfig:
    def __init__(self,
                 model_path,
                 model_root_path=None,
                 inputs_schema=[],
                 outputs_schema=[],
                 model_framework=None,
                 onnx_opset=12,
                 onnx_model_path=ONNX_MODEL_PATH,
                 sample_input_data_path=None):
        self.model_path = model_path
        self.model_root_path = model_root_path
        self.inputs_schema = self._complete_schema(inputs_schema)
        self.outputs_schema = self._complete_schema(outputs_schema)
        self.model_framework = model_framework
        self.onnx_model_path = onnx_model_path
        self.onnx_opset = onnx_opset
        self.sample_input_data_path = sample_input_data_path
        self.model_file_base_dir = self._get_model_base_dir()
        self._validate_opset_version()

    def _get_model_base_dir(self):
        if os.path.isdir(self.model_path):
            self.model_file_base_dir = self.model_path
        else:
            self.model_file_base_dir = os.path.dirname(self.model_path)
        return self.model_file_base_dir

    def _complete_schema(self, original_schema):
        result_schema = []
        if original_schema:
            for i in original_schema:
                name = i.get(IOSchemaLoader.NAME_KEY, None)
                dtype = i.get(IOSchemaLoader.DTYPE_KEY, None)
                dtype = None if (not dtype) else dtype.lower()
                if dtype and (dtype == "undefined" or dtype.isspace() or dtype == ""):
                    dtype = None
                if dtype and (not (dtype in STR_TO_NP_TYPE_MAP)):
                    raise Exception(f"dtype should be one of the followings: {STR_TO_NP_TYPE_MAP.keys()}")

                shape = i.get(IOSchemaLoader.SHAPE_KEY, None)
                if shape and len(shape) == 0:
                    shape = None

                schema = IOSchema(name, dtype, shape)
                result_schema.append(schema)

        return result_schema

    def _validate_opset_version(self):
        default_max_version = get_max_opset_version()
        input_opset = self.onnx_opset

        if input_opset is None or input_opset > default_max_version:
            input_opset = default_max_version
            logger.warning("Input opset is illegal, so set it to default_max_version")
        logger.info("Opset version {} will ne used".format(input_opset))

        self.onnx_opset = input_opset
