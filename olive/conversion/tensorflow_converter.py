import logging

from enum import Enum
import numpy as np
import os
import tensorflow as tf
import tf2onnx
from tf2onnx import tf_loader, optimizer
from tf2onnx.tfonnx import tf_optimize, process_tf_graph
from typing import Optional

from .base_converter import BaseConverter
from .io_schema import IOSchemaLoader
from ..constants import OLIVE_LOG_LEVEL, LOGGING_LEVEL_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if OLIVE_LOG_LEVEL in LOGGING_LEVEL_MAP.keys():
    tf2onnx.logging.set_level(LOGGING_LEVEL_MAP.get(OLIVE_LOG_LEVEL))
else:
    tf2onnx.logging.set_level(logging.ERROR)
    logger.warning("OLIVE_LOG_LEVEL is invalid. tf2onnx set_level=ERROR will be used.")


class TensorflowConverter(BaseConverter):
    def __init__(self, conversion_config):
        super(TensorflowConverter, self).__init__(conversion_config)
        self.tf_sess = None
        self.model_loader = None  # type: Optional[TensorflowModelLoader]

    def _convert(self):
        logger.info("Converting...")
        self._restore_session_from_model_file()

        tf.reset_default_graph()
        graph_def = tf_optimize(self.original_input_names, self.original_output_names,
                                self.tf_sess.graph_def, fold_constant=True)
        tf.import_graph_def(graph_def, name="")
        onnx_graph = process_tf_graph(tf.get_default_graph(), opset=self.conversion_config.onnx_opset,
                                      input_names=self.original_input_names, output_names=self.original_output_names)
        try:
            opt_graph = optimizer.optimize_graph(onnx_graph)
        except Exception as e:
            opt_graph = None
            logger.warning("Failed to optimize ONNX graph, original un-optimized graph will be used, e = {}".format(e))
        onnx_graph = opt_graph if opt_graph is not None else onnx_graph
        model_proto = onnx_graph.make_model("onnx-proto")
        with open(self.conversion_config.onnx_model_path, "wb") as f:
            f.write(model_proto.SerializeToString())

    def _restore_session_from_model_file(self):
        self.model_loader = TensorflowModelLoader(conversion_config=self.conversion_config,
                                                  original_input_names=self.original_input_names,
                                                  original_output_names=self.original_output_names)
        self.model_loader.load_model()
        self.original_input_names, self.original_output_names, self.tf_sess = \
            self.model_loader.original_input_names, self.model_loader.original_output_names, self.model_loader.tf_sess

    def _validate_config(self):
        # check if the model_file is a checkpoint or not
        self.conversion_config.model_path = self._get_tf_file_automatically()

        # checkpoint and graphdef format model must provide input and output name
        model_type = TensorflowModelLoader.get_model_type(self.conversion_config.model_path)
        if _ModelType.CHECKPOINT == model_type or _ModelType.FROZEN_PB == model_type:
            validate_schema_properties = IOSchemaLoader.validate_schema_properties
            properties = [IOSchemaLoader.NAME_KEY]
            if not (validate_schema_properties(self.conversion_config.inputs_schema, properties)
                    and validate_schema_properties(self.conversion_config.outputs_schema, properties)):
                raise Exception("Please provide inputs_schema and outputs_schema to convert TensorFlow graphdef or checkpoint models.")

    def _get_tf_file_automatically(self):
        file_exists = {".meta": False, ".index": False, ".data": False, "checkpoint": False}
        if os.path.isdir(self.conversion_config.model_path):
            meta_file_path = None
            for root, dirs, files in os.walk(self.conversion_config.model_path):
                for f in files:
                    if '.meta' in f:
                        file_exists['.meta'] = True
                        meta_file_path = os.path.join(root, f)
                    elif '.index' in f:
                        file_exists['.index'] = True
                    elif '.data' in f:
                        file_exists['.data'] = True
                    elif 'checkpoint' in f:
                        file_exists['checkpoint'] = True

                # this might be a checkpoint model, we update the model_file with the path of *.meta
                if all(v is True for v in file_exists.values()):
                    return meta_file_path

        return self.conversion_config.model_path

    def _get_original_model_inference_result(self, test_data):
        return self.model_loader.inference(test_data)


class _ModelType(Enum):
    SAVED_MODEL = 1
    FROZEN_PB = 2
    CHECKPOINT = 3


class TensorflowModelLoader:
    def __init__(self, conversion_config, original_input_names, original_output_names):
        self.model_path = conversion_config.model_path
        self.original_input_names = original_input_names
        self.original_output_names = original_output_names
        self.tf_sess = None  # type: Optional[tf.Session]

    def load_model(self):
        load_method = None
        model_type = TensorflowModelLoader.get_model_type(self.model_path)
        if _ModelType.FROZEN_PB == model_type:
            load_method = tf_loader.from_graphdef
        elif _ModelType.CHECKPOINT == model_type:
            load_method = tf_loader.from_checkpoint
        elif _ModelType.SAVED_MODEL == model_type:
            load_method = tf_loader.from_saved_model

        if model_type == _ModelType.SAVED_MODEL:
            graph_def, ins, outs = load_method(self.model_path, None, None)
            if not self.original_input_names:
                logger.info("Get model input from saved model")
                self.original_input_names = ins
            if not self.original_output_names:
                logger.info("Get model output from saved model")
                self.original_output_names = outs
        else:
            try:
                graph_def, _, _ = load_method(self.model_path, self.original_input_names, self.original_output_names)
            except Exception as e:
                logger.error("Load TensorFlow model failed, error: {}".format(e))
                raise

        graph = tf.import_graph_def(graph_def, name="")
        self.tf_sess = tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False))

    @staticmethod
    def get_model_type(model_path):
        if model_path.endswith(".meta"):
            return _ModelType.CHECKPOINT
        elif model_path.endswith(".pb") or model_path.endswith(".graphdef"):
            return _ModelType.FROZEN_PB
        else:
            return _ModelType.SAVED_MODEL

    def inference(self, test_data):
        input_dict_for_tf = {v: test_data[i] for i, v in enumerate(self.original_input_names)}
        model_res = self.tf_sess.run(self.original_output_names, input_dict_for_tf)

        tf_output_onnx_format = []
        for o in model_res:
            if not isinstance(o, np.ndarray):
                logger.warning("TensorFlow output got type {}".format(type(o)))
                tf_output_onnx_format.append(np.array(o))
            else:
                tf_output_onnx_format.append(o)

        return tf_output_onnx_format
