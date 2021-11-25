import logging
from shutil import copy

import numpy as np
from onnx import _load_bytes, load_model_from_string
from onnx.external_data_helper import ExternalDataInfo, _sanitize_path, _get_all_tensors
import onnxruntime as ort
import os

from .constants import OLIVE_RESULT_PATH, EP_TO_PROVIDER_TYPE_MAP, ONNX_TO_NP_TYPE_MAP, WARMUP_NUM, TEST_NUM, \
    EXECUTION_MODE_MAP, ORT_OPT_LEVEL_MAP
from .util import load_npz_file, is_npz_format

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizationConfig:

    def __init__(self,
                 model_path,
                 inputs_spec=None,
                 output_names=None,
                 providers_list=None,
                 quantization_enabled=False,
                 transformer_enabled=False,
                 transformer_args=None,
                 sample_input_data_path=None,
                 concurrency_num=1,
                 kmp_affinity=["respect,none"],
                 omp_max_active_levels=["1"],
                 result_path=OLIVE_RESULT_PATH,
                 warmup_num=WARMUP_NUM,
                 test_num=TEST_NUM,
                 inter_thread_num_list=[None],
                 intra_thread_num_list=[None],
                 ort_opt_level_list=["all"],
                 execution_mode_list=None,
                 omp_wait_policy_list=None,
                 trt_fp16_enabled=False,
                 throughput_tuning_enabled=False,
                 max_latency_percentile=None,
                 max_latency_sec=None,
                 threads_num=None,
                 dynamic_batching_size=1,
                 min_duration_sec=10):

        self.model_path = model_path
        self.inputs_spec = inputs_spec
        self.providers_list = providers_list
        self.quantization_enabled = quantization_enabled
        self.transformer_enabled = transformer_enabled
        self.transformer_args = transformer_args
        self.sample_input_data_path = sample_input_data_path
        self.concurrency_num = concurrency_num
        self.kmp_affinity = kmp_affinity
        self.omp_max_active_levels = omp_max_active_levels
        self.result_path = result_path
        self.warmup_num = warmup_num
        self.test_num = test_num
        self.inter_thread_num_list = inter_thread_num_list
        self.intra_thread_num_list = intra_thread_num_list
        self.ort_opt_level_list = ort_opt_level_list
        self.execution_mode_list = execution_mode_list
        self.trt_fp16_enabled = trt_fp16_enabled
        self.output_names = output_names
        self.throughput_tuning_enabled = throughput_tuning_enabled
        self.max_latency_percentile = max_latency_percentile
        self.max_latency_sec = max_latency_sec
        self.dynamic_batching_size = dynamic_batching_size
        self.threads_num = threads_num
        self.min_duration_sec = min_duration_sec
        if omp_wait_policy_list:
            self.omp_wait_policy_list = omp_wait_policy_list
            if "ACTIVE" in [i.upper() for i in self.omp_wait_policy_list] and self.concurrency_num > 1:
                logger.warning("Concurrent optimization with OMP_WAIT_POLICY=ACTIVE may take long time")
        else:
            self.omp_wait_policy_list = ["PASSIVE"] if self.concurrency_num > 1 else ["ACTIVE", "PASSIVE"]

        self._ort_opt_level_map()
        self._execution_mode_map()
        self._validate_model_path()
        self._duplicate_model_for_tuning()
        self._validate_providers_list()
        if self.throughput_tuning_enabled:
            self._validate_throughput_config()
        if self.inputs_spec is None:
            self.inputs_spec = self._generate_inputs_spec()
        self.inference_input_dict = self._generate_input_data()

    def _validate_throughput_config(self):
        if not (self.max_latency_percentile and self.max_latency_sec):
            raise ValueError("max_latency_percentile and max_latency_sec are needed for throughput tuning")
        if not self.threads_num:
            raise ValueError("threads_num is needed for throughput tuning")

    def _ort_opt_level_map(self):
        ort_opt_level_list = []
        for ort_opt_level in self.ort_opt_level_list:
            if str(ort_opt_level.lower()) not in ORT_OPT_LEVEL_MAP.keys():
                raise KeyError("failed in mapping ort opt level {}".format(ort_opt_level))
            ort_opt_level_list.append(ORT_OPT_LEVEL_MAP[str(ort_opt_level.lower())])
        self.ort_opt_level_list = ort_opt_level_list

    def _execution_mode_map(self):
        execution_mode_list = []
        if self.execution_mode_list:
            for execution_mode in self.execution_mode_list:
                if execution_mode.lower() not in EXECUTION_MODE_MAP.keys():
                    raise KeyError("failed in mapping execution mode {}".format(execution_mode))
                else:
                    execution_mode_list.append(ort.ExecutionMode(EXECUTION_MODE_MAP[execution_mode.lower()]))
        else:
            execution_mode_list = [ort.ExecutionMode.ORT_SEQUENTIAL, ort.ExecutionMode.ORT_PARALLEL]

        self.execution_mode_list = execution_mode_list

    def _validate_model_path(self):
        logger.info("Checking the model file...")
        if not self.model_path:
            raise ValueError("Model path must be provided")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Can't find the model file, please check the model_path")
        suffix = os.path.splitext(self.model_path)[-1]
        if not suffix.lower() == ".onnx":
            raise ValueError("File ends with .onnx is required for ONNX model")

    def _generate_inputs_spec(self):
        shapes_list = []
        execution_provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
        onnx_session = ort.InferenceSession(self.model_path, providers=[execution_provider])
        dims_list, names_list = zip(*[(i.shape, i.name) for i in onnx_session.get_inputs()])

        for dims in dims_list:
            # get shape
            # regard unk__32 and None as 1
            shape = [1 if (x is None or (type(x) is str)) else x for x in dims]
            shapes_list.append(shape)
        inputs_spec = dict(zip(names_list, shapes_list))

        return inputs_spec

    def _validate_providers_list(self):
        available_providers = ort.get_available_providers()
        provider_test_list = []

        if self.providers_list:
            for p in self.providers_list:
                ep_name = EP_TO_PROVIDER_TYPE_MAP.get(p.lower())
                if ep_name in available_providers:
                    provider_test_list.append(ep_name)
                else:
                    logger.info("Provider {} not found in available provider list".format(p))
        else:
            provider_test_list = available_providers

        if provider_test_list:
            logger.info("Providers will be tested for optimization: {}".format(provider_test_list))
            self.providers_list = provider_test_list
        else:
            raise ValueError("No providers available for test")

    def _generate_input_data(self):
        if self.sample_input_data_path:
            if is_npz_format(self.sample_input_data_path):
                input_dict = load_npz_file(self.sample_input_data_path)
            else:
                raise ValueError("Sample input data should be saved in file ends with .npz")
        else:
            input_dict = {}
            execution_provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
            inputs = ort.InferenceSession(self.model_path, providers=[execution_provider]).get_inputs()
            input_types = []
            for i in range(0, len(inputs)):
                if inputs[i].type in ONNX_TO_NP_TYPE_MAP.keys():
                    input_types.append(ONNX_TO_NP_TYPE_MAP[inputs[i].type])
                else:
                    raise KeyError("failed in mapping operator {} which has type {}".format(
                        inputs[i].name, inputs[i].type))

            for i in range(0, len(inputs)):
                shape = [1 if (type(x) is int and x < 0) else x for x in self.inputs_spec[inputs[i].name]]
                # generate values
                vals = np.random.random_sample(shape).astype(input_types[i])
                input_dict[inputs[i].name] = vals

        return input_dict

    def _duplicate_model_for_tuning(self):
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)
        model_dir = os.path.dirname(self.model_path)

        s = _load_bytes(self.model_path)
        model_proto = load_model_from_string(s)
        tensors = _get_all_tensors(model_proto)
        for tensor in tensors:
            info = ExternalDataInfo(tensor)
            file_location = _sanitize_path(info.location)
            if file_location:
                copy(os.path.join(model_dir, file_location), self.result_path)

        optimized_model_path = os.path.join(self.result_path, "optimized_model.onnx")
        copy(self.model_path, optimized_model_path)
        self.model_path = optimized_model_path

