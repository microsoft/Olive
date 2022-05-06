import os
import shutil
import sys
import uuid

import pytest

from olive.conversion.io_schema import IOSchemaLoader
from olive.conversion_config import ConversionConfig
from olive.convert import convert
from ci_tests.tests.conversion.constants import TEST_INPUT_DIR, CUSTOMIZED_MODEL_MULTIPLE_INPUTS_OUTPUTS_DATA, \
    CUSTOMIZED_MODEL_MULTIPLE_INPUTS_DATA, PRETRAINED_MODEL_VIDEO_DATA, CUSTOMIZED_MODEL_MULTIPLE_OUTPUTS_DATA
import ci_tests.tests.conversion.data.sample.multi_file_model.src.multi_file_model as mf
import ci_tests.tests.conversion.data.sample.sample_model.sample_model as sm
from ci_tests.tests.conversion.util import prepare_test_data, prepare_model, get_classification_params

prepare_test_data()


@pytest.mark.parametrize(('model_name'), get_classification_params())
def test_pretrained_model_classification(model_name):
    # prepare model
    model_path = os.path.join(TEST_INPUT_DIR, model_name, f"_{str(uuid.uuid4())}.pth")
    prepare_model(model_path, model_name)

    # test
    inputs_schema = [{IOSchemaLoader.SHAPE_KEY: [1, 3, 244, 244],
                     IOSchemaLoader.DTYPE_KEY: "float",
                     IOSchemaLoader.NAME_KEY: "input_0"}]
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "output_0"}]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_framework="pytorch", onnx_model_path=onnx_model_path,
                                  onnx_opset=11)
    convert(cvt_config)
    os.remove(onnx_model_path)


@pytest.mark.video
def test_pretrained_model_video_r2plus1d_18():
    # prepare model
    model_path = os.path.join(TEST_INPUT_DIR, 'video_r2plus1d_18', f"_{str(uuid.uuid4())}.pth")
    prepare_model(model_path, "video_r2plus1d_18", 1)

    # test
    inputs_schema = [{IOSchemaLoader.SHAPE_KEY: [2, 3, 4, 112, 112],
                     IOSchemaLoader.DTYPE_KEY: "float",
                     IOSchemaLoader.NAME_KEY: "input_0"}]
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "output_0"}]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  sample_input_data_path=PRETRAINED_MODEL_VIDEO_DATA, onnx_opset=11,
                                  model_framework="pytorch", onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


@pytest.mark.video
def test_pretrained_model_video_mc3_18():
    # prepare model
    model_path = os.path.join(TEST_INPUT_DIR, 'video_mc3_18', f"_{str(uuid.uuid4())}.pth")
    prepare_model(model_path, "video_mc3_18", 1)

    # test
    inputs_schema = [{IOSchemaLoader.SHAPE_KEY: [2, 3, 4, 112, 112],
                     IOSchemaLoader.DTYPE_KEY: "float",
                     IOSchemaLoader.NAME_KEY: "input_0"}]
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "output_0"}]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  sample_input_data_path=PRETRAINED_MODEL_VIDEO_DATA, onnx_opset=11,
                                  model_framework="pytorch", onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


@pytest.mark.video
def test_pretrained_model_video_r3d_18():
    # prepare model
    model_path = os.path.join(TEST_INPUT_DIR, 'video_r3d_18', f"_{str(uuid.uuid4())}.pth")
    prepare_model(model_path, "video_r3d_18", 1)

    # test
    inputs_schema = [{IOSchemaLoader.SHAPE_KEY: [2, 3, 4, 112, 112],
                     IOSchemaLoader.DTYPE_KEY: "float",
                     IOSchemaLoader.NAME_KEY: "input_0"}]
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "output_0"}]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  sample_input_data_path=PRETRAINED_MODEL_VIDEO_DATA, onnx_opset=11,
                                  model_framework="pytorch", onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


@pytest.mark.customized_model
def test_customized_model_multiple_inputs():
    # prepare model
    model_path = os.path.join(TEST_INPUT_DIR, 'customized_model_multiple_inputs', f"_{str(uuid.uuid4())}.pth")
    inputs_schema = [
        {
            IOSchemaLoader.SHAPE_KEY: [10, 3, 244, 244],
            IOSchemaLoader.DTYPE_KEY: "float",
            IOSchemaLoader.NAME_KEY: "multiple_in_0"
        },
        {
            IOSchemaLoader.SHAPE_KEY: [10, 3, 244, 244],
            IOSchemaLoader.DTYPE_KEY: "float",
            IOSchemaLoader.NAME_KEY: "multiple_in_1"
        },
        {
            IOSchemaLoader.SHAPE_KEY: [10, 3, 244, 244],
            IOSchemaLoader.DTYPE_KEY: "float",
            IOSchemaLoader.NAME_KEY: "multiple_in_2"
        }
    ]
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "multiple_out_0"}]
    prepare_model(model_path, "customized_model_multiple_inputs", len(inputs_schema))
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    # test
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  sample_input_data_path=CUSTOMIZED_MODEL_MULTIPLE_INPUTS_DATA, onnx_opset=11,
                                  model_framework="pytorch", onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


@pytest.mark.customized_model
def test_customized_model_multiple_outputs():
    # prepare model
    model_path = os.path.join(TEST_INPUT_DIR, 'customized_model_multiple_outputs', f"_{str(uuid.uuid4())}.pth")
    inputs_schema = [
        {
            IOSchemaLoader.SHAPE_KEY: [10, 3, 244, 244],
            IOSchemaLoader.DTYPE_KEY: "float",
            IOSchemaLoader.NAME_KEY: "multiple_in_0"
        }
    ]
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "multiple_out_0"}, {IOSchemaLoader.NAME_KEY: "multiple_out_1"}]
    prepare_model(model_path, "customized_model_multiple_outputs", len(inputs_schema))
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    # test
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  sample_input_data_path=CUSTOMIZED_MODEL_MULTIPLE_OUTPUTS_DATA,
                                  model_framework="pytorch", onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


@pytest.mark.customized_model
def test_customized_model_multiple_inputs_and_outputs():
    # prepare model
    model_path = os.path.join(TEST_INPUT_DIR, 'customized_model_multiple_inputs_outputs', f"_{str(uuid.uuid4())}.pth")
    inputs_schema = [
        {
            IOSchemaLoader.SHAPE_KEY: [10, 3, 244, 244],
            IOSchemaLoader.DTYPE_KEY: "float",
            IOSchemaLoader.NAME_KEY: "multiple_in_0"
        },
        {
            IOSchemaLoader.SHAPE_KEY: [10, 3, 244, 244],
            IOSchemaLoader.DTYPE_KEY: "float",
            IOSchemaLoader.NAME_KEY: "multiple_in_1"
        }
    ]
    outputs_schema = [{IOSchemaLoader.NAME_KEY: "multiple_out_0"}, {IOSchemaLoader.NAME_KEY: "multiple_out_1"}]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    prepare_model(model_path, "customized_model_multiple_inputs_outputs", len(inputs_schema))

    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  sample_input_data_path=CUSTOMIZED_MODEL_MULTIPLE_INPUTS_OUTPUTS_DATA,
                                  model_framework="pytorch", onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)


@pytest.mark.customized_model_with_script
@pytest.mark.parametrize('use_sample_data', [True, False])
@pytest.mark.parametrize('use_dynamic_input_schema', [True, False])
def test_customized_model_with_dynamic_input(use_sample_data, use_dynamic_input_schema):
    model_dir = os.path.join(TEST_INPUT_DIR, "test_customized_model_with_dynamic_input")
    model_path = os.path.join(model_dir, "sample_model.pth")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # prepare model and sample input data
    shutil.copytree(os.path.join(os.path.dirname(__file__), 'data', 'sample', 'sample_model'),
                    model_dir)
    sample_input_data_path = os.path.join(model_dir, 'data.npz')
    sm.save_sample_input_data(sample_input_data_path)

    # prepare model
    inputs_schema = [{
        IOSchemaLoader.NAME_KEY: sm.INPUT_NAME,
        IOSchemaLoader.SHAPE_KEY: sm.get_input_shape(use_dynamic_input_schema),
        IOSchemaLoader.DTYPE_KEY: "float"}]
    outputs_schema = [{
        IOSchemaLoader.NAME_KEY: sm.OUTPUT_NAME,
        IOSchemaLoader.SHAPE_KEY: sm.get_output_shape(use_dynamic_input_schema)}]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"

    if use_sample_data:
        cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                      sample_input_data_path=sample_input_data_path,onnx_opset=11,
                                      model_framework="pytorch", onnx_model_path=onnx_model_path)
    else:
        cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,onnx_opset=11,
                                      model_framework="pytorch", onnx_model_path=onnx_model_path)

    convert(cvt_config)
    os.remove(onnx_model_path)


@pytest.mark.customized_model_with_script
def test_customized_model_with_sub_dir():
    model_dir = os.path.join(TEST_INPUT_DIR, "test_customized_model_with_sub_dir")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # prepare model
    shutil.copytree(os.path.join(os.path.dirname(__file__), 'data', 'sample', "multi_file_model"),
                    os.path.join(model_dir))

    model_path = os.path.join(model_dir, "data/multi_file_model.pth")

    # prepare model
    inputs_schema = []
    for name, shape in zip(mf.get_input_names(), mf.get_input_shapes()):
        inputs_schema.append({
            IOSchemaLoader.NAME_KEY: name,
            IOSchemaLoader.SHAPE_KEY: shape
        })

    outputs_schema = []
    for name in mf.get_output_names():
        outputs_schema.append({IOSchemaLoader.NAME_KEY: name})
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"

    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_root_path=os.path.join(model_dir, 'src'), model_framework="pytorch", onnx_opset=11,
                                   onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)

    # exclude script if it's already appended during conversion
    for p in sys.path:
        if p.startswith(model_dir):
            sys.path.remove(p)


@pytest.mark.script_module
def test_script_module():
    # prepare model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'scriptmodule.pt')
    inputs_schema = [
        {
            IOSchemaLoader.SHAPE_KEY: [1],
            IOSchemaLoader.NAME_KEY: "input_0"
        }
    ]
    outputs_schema = [
        {
            IOSchemaLoader.SHAPE_KEY: [1],
            IOSchemaLoader.NAME_KEY: "output_0"
        }
    ]
    onnx_model_path = f"_{str(uuid.uuid4())}.onnx"
    # test
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  onnx_opset=11, model_framework="pytorch", onnx_model_path=onnx_model_path)
    convert(cvt_config)
    os.remove(onnx_model_path)
