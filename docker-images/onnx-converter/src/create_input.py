# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import re
import shutil
import numpy as np
import onnxruntime as rt
import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto


# Randomly generate input tensor given its shape
def create_tensor(name, dims, tensor_name, path, data_type=np.float32, vals=None):
    if vals is None:
        vals = np.random.random_sample(dims).astype(data_type)
    tensor = numpy_helper.from_array(vals)
    tensor.name = tensor_name

    with open(os.path.join(path, name), 'wb') as f:
        f.write(tensor.SerializeToString())


TYPE_MAP = {
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
}


def search_for_existing_test_data(user_data_path, output_data_path):
    if not os.path.exists(user_data_path):
        print('No data found under user provided folder. Generating random input data.')
        return
    regex = re.compile("test_data*")
    test_folder_name = 'test_data_set_'
    test_folder_idx = 0
    for f in os.listdir(user_data_path):
        if regex.match(f):
            cur_user_data_path = os.path.join(user_data_path, f)
            for inputFiles in os.listdir(cur_user_data_path):
                if inputFiles.endswith('.pb') == False:
                    print('No data found under user provided folder. Generating random input data.')
                    return
            print('Test data .pb files found under {}. '.format(cur_user_data_path))
            # Copy the test data to the ONNX output dir. Overwrite if already exists.
            cur_test_folder_path = os.path.join(output_data_path, test_folder_name + str(test_folder_idx))
            print('copying {} to {}'.format(cur_user_data_path,
                                            os.path.join(output_data_path, test_folder_name + str(test_folder_idx))))
            if os.path.exists(cur_test_folder_path):
                shutil.rmtree(cur_test_folder_path)
            shutil.copytree(cur_user_data_path, cur_test_folder_path)
            test_folder_idx += 1


def generate_inputs(input_model_path, src_test_data, output_model_path):
    # Create a test folder path
    output_test_data_dir = os.path.dirname(os.path.abspath(output_model_path))
    test_path = os.path.join(output_test_data_dir, 'test_data_set_0')

    regex = re.compile("test_data*")
    # Find and copy over the user specified test data
    if src_test_data:
        search_for_existing_test_data(src_test_data, output_test_data_dir)
    else:
        potential_data_dir = os.path.dirname(os.path.abspath(input_model_path))
        search_for_existing_test_data(potential_data_dir, output_test_data_dir)

    if not os.path.exists(test_path):
        os.mkdir(test_path)
        os.chmod(test_path, 0o644)
    # Check if test folder and test data already exists
    regex = re.compile("test_data*")
    for f in os.listdir(output_test_data_dir):
        if regex.match(f):
            user_data_path = os.path.join(output_test_data_dir, f)
            for inputFiles in os.listdir(user_data_path):
                if inputFiles.endswith('.pb'):
                    print("Test data .pb files already exist. Skipping dummy input generation. ")
                    return test_path

    # Get input names from converted model
    sess = rt.InferenceSession(output_model_path)

    #########################
    # Let's see the input name and shape.
    print("%s inputs: " % output_model_path)
    inputs = sess.get_inputs()
    for i in range(0, len(inputs)):
        print("input name: %s, shape: %s, type: %s" % (inputs[i].name, inputs[i].shape, inputs[i].type))
        # If the input has None dimensions, replace with 1
        shape_corrected = [1 if x == None else x for x in inputs[i].shape]
        if inputs[i].type == "tensor(string)" or not all(isinstance(dim, int) for dim in shape_corrected):
            shutil.rmtree(test_path)
            raise ValueError(
                "Cannot auto generate inputs. Please provide your own input .pb files under output_onnx_path folder. ")
        # Create random input and write to .pb
        create_tensor("input_%s.pb" % i, shape_corrected, inputs[i].name, test_path, TYPE_MAP.get(inputs[i].type))

    print("Randomized input .pb file generated at ", test_path)
    return test_path
