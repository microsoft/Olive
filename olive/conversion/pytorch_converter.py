from collections import OrderedDict
import copy
import sys

from packaging import version
import torch
import torch.jit

from .base_converter import BaseConverter
from .io_schema import IOSchemaLoader
from ..constants import OLIVE_LOG_LEVEL
from ..util import load_npz_file
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PYTORCH_DATA_TYPE_MAP = {
    "float": torch.float,
    "float32": torch.float32,
    "float64": torch.float64,
    "double": torch.double,
    "int64": torch.int64,
    "int32": torch.int32,
    "int": torch.int
}


class PyTorchConverter(BaseConverter):

    def __init__(self, conversion_config):
        super(PyTorchConverter, self).__init__(conversion_config)
        self.pytorch_model_loader = None  # type: Optional[PytorchModelLoader]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device {}".format(self.device))

    def _convert(self):
        log_verbose = True if OLIVE_LOG_LEVEL == "INFO" else False

        dummy_input = self._create_dummy_input()
        self._load_model()

        # prepare output info if the model is torchscript
        dummy_output = None
        if isinstance(self.pytorch_model_loader.pytorch_model, torch.jit.ScriptModule):
            if not IOSchemaLoader.validate_schema_properties(self.conversion_config.outputs_schema,
                                                             [IOSchemaLoader.SHAPE_KEY]):
                raise Exception("To convert torchscript PyTorch model outputs_schema needs shape information.")
            dummy_output = PyTorchConverter._create_dummy_data_from_schema(self.conversion_config.outputs_schema, self.device)
        
        # convert model
        if version.parse(torch.__version__) >= version.parse('1.10.21.11.0'):
            torch.onnx.export(self.pytorch_model_loader.pytorch_model,
                            dummy_input,
                            self.conversion_config.onnx_model_path,
                            input_names=self.original_input_names,
                            output_names=self.original_output_names,
                            opset_version=self.conversion_config.onnx_opset,
                            dynamic_axes=self._get_dynamic_axes(),
                            verbose=log_verbose)
        else:
            torch.onnx.export(self.pytorch_model_loader.pytorch_model,
                dummy_input,
                self.conversion_config.onnx_model_path,
                input_names=self.original_input_names,
                output_names=self.original_output_names,
                opset_version=self.conversion_config.onnx_opset,
                dynamic_axes=self._get_dynamic_axes(),
                example_outputs=dummy_output,
                verbose=log_verbose)

    def _load_model(self):
        self.pytorch_model_loader = PytorchModelLoader(conversion_config=self.conversion_config, device=self.device)
        self.pytorch_model_loader.load_model()

    def _create_dummy_input(self):
        dummy_input = []
        if self.conversion_config.sample_input_data_path:
            test_data = load_npz_file(self.conversion_config.sample_input_data_path)
            for schema in self.conversion_config.inputs_schema:
                dummy_input.append(torch.from_numpy(test_data.get(schema.name)).to(self.device))
        if len(dummy_input) == 0:
            dummy_input = PyTorchConverter._create_dummy_data_from_schema(
                self.conversion_config.inputs_schema, self.device)
        dummy_input = tuple(dummy_input)
        return dummy_input

    @staticmethod
    def _create_dummy_data_from_schema(io_schema, device):
        dummy_output = []
        for schema in io_schema:
            shape = copy.deepcopy(schema.shape)  # avoid modifying original output schema
            for i, v in enumerate(shape):
                if v < 0:
                    shape[i] = 1

            dummy_output.append(torch.zeros(
                shape, dtype=PyTorchConverter._get_pytorch_type(schema.dtype)).to(device))

        return tuple(dummy_output)

    def _validate_config(self):
        # Check for PyTorch converter customize arguments
        properties_to_validate = [IOSchemaLoader.SHAPE_KEY, IOSchemaLoader.NAME_KEY]
        if not self.conversion_config.inputs_schema or \
                (not IOSchemaLoader.validate_schema_properties(self.conversion_config.inputs_schema,
                                                               properties_to_validate)):
            raise Exception("To convert PyTorch model, inputs_schema needs shape and name information.")

    def _get_dynamic_axes(self):
        dynamic_axes = {}
        io_schema_list = [self.conversion_config.inputs_schema, self.conversion_config.outputs_schema]
        for schema_list in io_schema_list:
            for schema in schema_list:
                idx = []

                if schema.shape is None:
                    continue

                for i, shape in enumerate(schema.shape):
                    if (not isinstance(shape, int)) or (isinstance(shape, int) and shape < 0):
                        idx.append(i)

                if len(idx) > 0:
                    dynamic_axes[schema.name] = idx

        return dynamic_axes

    def _get_original_model_inference_result(self, test_data):
        return self.pytorch_model_loader.inference(test_data)

    @staticmethod
    def _get_pytorch_type(data_type):
        if data_type is None:
            return torch.float
        if data_type in PYTORCH_DATA_TYPE_MAP.keys():
            return PYTORCH_DATA_TYPE_MAP.get(data_type)
        else:
            raise ValueError("unknown data type {}".format(data_type))


class PytorchModelLoader:

    def __init__(self, conversion_config, device):
        self.conversion_config = conversion_config
        self.model_path = self.conversion_config.model_path
        self.device = device
        self.model_root_path = self.conversion_config.model_root_path
        self.pytorch_model = None

    def load_model(self):
        # append model_root_dir and model_root_path to sys.path to make sure all python scripts are searchable
        sys.path.append(self.conversion_config.model_file_base_dir)
        if self.model_root_path is not None:
            sys.path.append(self.model_root_path)

        try:
            # load the PyTorch model(we don't support state_dict format model)
            self.pytorch_model = torch.load(self.model_path, map_location=self.device)
        except (RuntimeError, ModuleNotFoundError) as e:
            # if torch.load is failed, try to load the model with torch.jit.load
            # torch 1.3 only supports loading scriptModel by torch.jit.load
            logger.warning("Got exception during loading PyTorch model, "
                            "try to load with torch.jit.load, exception: {}".format(e))
            self.pytorch_model = torch.jit.load(self.model_path, map_location=self.device)
        if isinstance(self.pytorch_model, OrderedDict):
            raise ValueError("state_dict format is not support")
        self.pytorch_model.to(self.device)
        self.pytorch_model.eval()

    def inference(self, test_data):
        test_data_tensor_list = []
        for data in test_data:
            data_to_tensor = torch.from_numpy(data)
            test_data_tensor_list.append(data_to_tensor.to(self.device))

        with torch.no_grad():
            pytorch_output = self.pytorch_model(*test_data_tensor_list)
        pytorch_output_onnx_format = PytorchModelLoader._convert_to_onnx_format(pytorch_output)

        return pytorch_output_onnx_format

    @staticmethod
    def _convert_to_onnx_format(pytorch_output):
        if isinstance(pytorch_output, torch.Tensor):
            output_to_cpu = pytorch_output.cpu()
            return output_to_cpu.numpy()
        elif isinstance(pytorch_output, OrderedDict) or isinstance(pytorch_output, dict):
            to_list = [v for _, v in pytorch_output.items()]
            return PytorchModelLoader._convert_to_onnx_format(to_list)
        elif isinstance(pytorch_output, tuple) or isinstance(pytorch_output, list):
            data = []
            for i in pytorch_output:
                items = PytorchModelLoader._convert_to_onnx_format(i)
                data.append(items)
            return data
