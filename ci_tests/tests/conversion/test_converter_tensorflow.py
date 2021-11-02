import logging
import os
import shutil
import uuid

import numpy as np
import pytest
import tensorflow as tf

from olive.conversion.io_schema import IOSchemaLoader
from olive.conversion_config import ConversionConfig
from olive.convert import convert
from ci_tests.tests.conversion.tensorflow_to_onnx_example import create_and_train_mnist
from ci_tests.tests.conversion.util import save_test_data_to_disk

TF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'output')
TF_INPUT_DATA = os.path.join(TF_MODEL_PATH, 'tf.input.data.npz')

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp-tensorflow', 'output')
logging.info(f"all test files are put under {test_dir}")


def train_models():
    # save tf model to "saved model" format
    logging.info("please wait for a while, because the script will train MNIST from scratch")
    tf.reset_default_graph()
    sess_tf, saver, input_tensor, output_tensor, test_data = create_and_train_mnist()
    logging.info("save tensorflow in format \"saved_model\"")

    save_path = os.path.join(TF_MODEL_PATH, "saved_model")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    tf.saved_model.simple_save(sess_tf, save_path, {input_tensor.name: input_tensor},
                               {output_tensor.name: output_tensor})

    # save tf model to "frozen graph" format
    logging.info("save tensorflow in format \"frozen graph\"")
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess_tf, sess_tf.graph_def, [output_tensor.name[:-2]])
    save_path = os.path.join(TF_MODEL_PATH, "mnist_frozen.pb")
    with open(save_path, "wb") as file:
        file.write(frozen_graph.SerializeToString())

    # save tf model to "checkpoint" format
    logging.info("save tensorflow in format \"checkpoint\"")
    save_path = os.path.join(TF_MODEL_PATH, "ckpt/model.ckpt")
    save_path = saver.save(sess_tf, save_path)

    # input_tensor.name = input:0, output_tensor.name = result:0
    logging.info(f"input_tensor.name = {input_tensor.name}, output_tensor.name = {output_tensor.name}, "
                 f"save_path = {save_path}")

    # save test data
    value_list = [test_data]
    name_list = [input_tensor.name]
    data = dict(zip(name_list, value_list))
    np.savez(TF_INPUT_DATA, **data)

    return input_tensor, output_tensor


def prepare_conversion():
    # only train model if model output folder is not exist
    if not os.path.exists(TF_MODEL_PATH):
        os.mkdir(TF_MODEL_PATH)
        logging.info("Training model and generating test data")
        input_tensor, output_tensor = train_models()

        inputs_schema = [{IOSchemaLoader.NAME_KEY: input_tensor.name}]
        outputs_schema = [{IOSchemaLoader.NAME_KEY: output_tensor.name}]
    else:
        logging.info("Using existing model and data")

        inputs_schema = [{IOSchemaLoader.NAME_KEY: "input:0"}]
        outputs_schema = [{IOSchemaLoader.NAME_KEY: "result:0"}]

    return inputs_schema, outputs_schema


SCHEMAS = prepare_conversion()


def test_saved_model():
    logging.info("converting... saved_model")
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=os.path.join(TF_MODEL_PATH, "saved_model"),
                                  model_framework="tensorflow", onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


def test_saved_model_with_npz():
    logging.info("converting... saved_model")
    inputs_schema = SCHEMAS[0]
    outputs_schema = SCHEMAS[1]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=os.path.join(TF_MODEL_PATH, "saved_model"), inputs_schema=inputs_schema,
                                  outputs_schema=outputs_schema, sample_input_data_path=TF_INPUT_DATA,
                                  model_framework="tensorflow", 
                                  onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


def test_frozen_graph():
    logging.info("converting... frozen_graph")
    inputs_schema = SCHEMAS[0]
    outputs_schema = SCHEMAS[1]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=os.path.join(TF_MODEL_PATH, "mnist_frozen.pb"),
                                  inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_framework="tensorflow", 
                                  onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


def test_frozen_graph_with_negative_shape():
    logging.info("converting... frozen_graph")
    inputs_schema = [{IOSchemaLoader.NAME_KEY: "input:0",
                      IOSchemaLoader.SHAPE_KEY: [-1, 784]}]
    outputs_schema = SCHEMAS[1]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=os.path.join(TF_MODEL_PATH, "mnist_frozen.pb"),
                                  inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_framework="tensorflow", 
                                  onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


def test_checkpoint():
    # test the checkpoint model with .meta file
    logging.info("converting... checkpoint")
    inputs_schema = SCHEMAS[0]
    outputs_schema = SCHEMAS[1]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=os.path.join(TF_MODEL_PATH, "ckpt", 'model.ckpt.meta'),
                                  inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_framework="tensorflow", 
                                  onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


def test_checkpoint_with_folder():
    # test the checkpoint model with checkpoint folder
    logging.info("converting... checkpoint")
    inputs_schema = SCHEMAS[0]
    outputs_schema = SCHEMAS[1]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"

    cvt_config = ConversionConfig(model_path=os.path.join(TF_MODEL_PATH, "ckpt"), inputs_schema=inputs_schema,
                                  outputs_schema=outputs_schema,
                                  model_framework="tensorflow", 
                                  onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


def test_multiple_inputs():
    # prepare convert parameters
    inputs_schema = []
    input_names = ["title_lengths:0", "title_encoder:0", "ratings:0", "query_lengths:0",
                   "passage_lengths:0", "features:0", "encoder:0", "decoder:0", "Placeholder:0"]
    for name in input_names:
        inputs_schema.append({IOSchemaLoader.NAME_KEY: name})
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "output_identity:0"}, {IOSchemaLoader.NAME_KEY: "loss_identity:0"}]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"

    # generate input data
    data_path = os.path.join(TF_MODEL_PATH, 'test_data.npz')
    save_test_data_to_disk(data_path)

    cvt_config = ConversionConfig(model_path=os.path.join(os.path.dirname(__file__), 'data', 'full_doran_frozen.pb'),
                                  inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  sample_input_data_path=data_path,
                                  model_framework="tensorflow", 
                                  onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


def get_tf_version():
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze

    x = freeze.freeze()
    for p in x:
        pkg = p.split("==")
        if pkg[0] == "tensorflow":
            return pkg[1]

    raise Exception("TensorFlow package missing")


@pytest.mark.skipif(get_tf_version() < "1.14", reason="infogan model requires TensorFlow 1.14.x")
def test_infogan_pb():
    logging.info("converting... infogan model")

    # convert
    inputs_schema = [{IOSchemaLoader.NAME_KEY: "zc_vectors:0",
                      IOSchemaLoader.DTYPE_KEY: "float32",
                      IOSchemaLoader.SHAPE_KEY: [10, 74]},
                     {IOSchemaLoader.NAME_KEY: "is_training_generator:0",
                      IOSchemaLoader.DTYPE_KEY: "bool",
                      IOSchemaLoader.SHAPE_KEY: []}]
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "generator/layer_3/Sigmoid:0"}]

    test_data_path = os.path.join(os.path.dirname(__file__), "data", "infogan_inputs.npz")
    data = np.load(test_data_path, allow_pickle=True)
    for x in np.ndenumerate(data):
        real_data1 = np.array(x[1]['zc_vectors:0'], dtype=np.float32)
        real_data2 = np.array(x[1]['is_training_generator:0'])
        data = [real_data1, real_data2]
        break

    name_list = ["zc_vectors:0", "is_training_generator:0"]
    test_data = dict(zip(name_list, data))
    npz_file_path = os.path.join(os.path.dirname(__file__), "data", "test_infogan_inputs.npz")
    np.savez(npz_file_path, **test_data)

    onnx_model_path = f"{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=os.path.join(os.path.dirname(__file__), 'data', "infogan_after_tf_optimize.pb"),
                                  inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  sample_input_data_path=npz_file_path,
                                  model_framework="tensorflow", 
                                  onnx_model_path=onnx_model_path)
    convert(cvt_config)

    # make sure the first input of the converted ONNX model is dynamic
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    assert 'unk' in sess.get_inputs()[0].shape[0]
    assert 74 == sess.get_inputs()[0].shape[1]
    os.remove(onnx_model_path)
