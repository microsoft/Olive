from abc import abstractmethod
import logging

import numpy as np
import onnxruntime as ort

from ..constants import FRAMEWORK_TENSORFLOW, FRAMEWORK_PYTORCH, STR_TO_NP_TYPE_MAP, ONNX_TO_NP_TYPE_MAP
from ..util import if_enumerable, load_npz_file, reorder_npz_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ort.set_default_logger_severity(4)

class BaseConverter:

    def __init__(self, conversion_config):
        self.conversion_config = conversion_config
        self.original_input_names = self.get_original_input_names()
        self.original_output_names = self.get_original_output_names()

    def get_original_input_names(self):
        res = []
        for i in self.conversion_config.inputs_schema:
            res.append(i.name)
        return res

    def get_original_output_names(self):
        res = []
        if self.conversion_config.outputs_schema:
            for o in self.conversion_config.outputs_schema:
                res.append(o.name)
        return res

    def convert(self):
        self._validate_config()
        logger.info("Converting model into ONNX")
        try:
            self._convert()
        except Exception as e:
            raise Exception("Conversion failed with error: {}".format(e))
        self._verify_correctness()

    @staticmethod
    def create_converter(conversion_config):
        converter = None
        framework = conversion_config.model_framework.lower()
        if framework == FRAMEWORK_TENSORFLOW:
            from .tensorflow_converter import TensorflowConverter
            converter = TensorflowConverter(conversion_config)
        elif framework == FRAMEWORK_PYTORCH:
            from .pytorch_converter import PyTorchConverter
            converter = PyTorchConverter(conversion_config)
        return converter

    def _verify_correctness(self):
        logger.info("Validate ONNX model")
        test_data = self._generate_inputs()
        self._verify_onnx_models(test_data)

    def _generate_inputs(self):
        if self.conversion_config.sample_input_data_path:
            npz_data = load_npz_file(self.conversion_config.sample_input_data_path)
            return reorder_npz_data(npz_data, self.original_input_names)
        else:
            dims_list = []
            type_list = []
            if self.conversion_config.inputs_schema:
                for schema in self.conversion_config.inputs_schema:
                    if schema.shape:
                        dims_list.append(schema.shape)
                    if schema.dtype:
                        type_list.append(STR_TO_NP_TYPE_MAP[schema.dtype])

            onnx_session = ort.InferenceSession(self.conversion_config.onnx_model_path, providers=["CPUExecutionProvider"])
            if len(dims_list) == 0:
                dims_list = [i.shape for i in onnx_session.get_inputs()]
            if len(type_list) == 0:
                type_list = [ONNX_TO_NP_TYPE_MAP[i.type] for i in onnx_session.get_inputs()]
            return self.generate_random_inputs_from_schema(dims_list, type_list, num_of_data=1)

    @staticmethod
    def generate_random_inputs_from_schema(input_shapes, input_types, num_of_data=1):
        assert len(input_shapes) == len(input_types)
        num_of_input = len(input_shapes)
        test_data = []
        for i in range(0, num_of_input):

            _shape = [1 if (x is None or (type(x) is str and ('unk' in x or x == 'N'))) else x for x in input_shapes[i]]
            shape = [1 if (type(x) is int and x < 0) else x for x in _shape]

            # generate values
            vals = np.random.random_sample(shape).astype(input_types[i])
            for _ in range(num_of_data - 1):
                v = np.random.random_sample(shape).astype(input_types[i])
                vals = np.append(vals, v, axis=0)

            test_data.append(vals)
        return test_data

    def _verify_onnx_models(self, test_data):
        try:
            # get original result(it's converted into ONNX output format)
            original_results = self._get_original_model_inference_result(test_data)
            original_results_flatten = self.flatten_list(original_results)

            # verify ONNX model
            onnx_results = self.get_onnx_model_inference_result(test_data)
            onnx_results_flatten = self.flatten_list(onnx_results)
            self.compare_with_output(original_results_flatten, onnx_results_flatten)
        except AssertionError as e:
            raise Exception("Test result is not acceptable, err: {}".format(e))

    def flatten_list(self, data):
        if isinstance(data, list) or isinstance(data, tuple):
            data_in_list_format = []
            for i in data:
                list_format = self.flatten_list(i)

                for j in list_format:
                    data_in_list_format.append(j)

            return data_in_list_format
        return [data]

    def compare_with_output(self, desired_outputs, onnx_outputs, decimal=3):
        assert len(onnx_outputs) == len(desired_outputs)

        for onnx_result, desired_result in zip(onnx_outputs, desired_outputs):
            if if_enumerable(onnx_result):
                for idx, _ in enumerate(onnx_result):
                    onnx_ele = onnx_result[idx]
                    desired_ele = desired_result[idx]
                    if isinstance(onnx_ele, dict):
                        for dict_key in onnx_ele.keys():
                            np.testing.assert_almost_equal(onnx_ele.get(dict_key),
                                                           desired_ele.get(dict_key), decimal=decimal)
                    else:
                        np.testing.assert_almost_equal(onnx_ele, desired_ele, decimal=decimal)
            else:
                np.testing.assert_almost_equal(onnx_result, desired_result, decimal=decimal)

    def get_onnx_model_inference_result(self, test_data):
        onnx_session = ort.InferenceSession(self.conversion_config.onnx_model_path, providers=["CPUExecutionProvider"])
        onnx_input_names = self.original_input_names
        onnx_output_names = self.original_output_names
        input_dict = {v: test_data[i] for i, v in enumerate(onnx_input_names)}
        onnx_output_list = onnx_session.run(onnx_output_names, input_dict)
        return onnx_output_list

    @abstractmethod
    def _get_model_loader(self):
        raise NotImplementedError("Not implemented yet")

    @abstractmethod
    def _convert(self):
        raise NotImplementedError("Not implemented yet.")

    @abstractmethod
    def _validate_config(self):
        raise NotImplementedError("Not implemented yet.")

    @abstractmethod
    def _get_original_model_inference_result(self, test_data):
        raise NotImplementedError("Not implemented yet")