import os
import re
import numpy as np
import onnxruntime as rt
import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto

def create_tensor(name, dims, tensor_name, path, data_type=np.float32, vals=None):
	if not vals:
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

def generate_inputs(model):
    
    # Create a test folder path
    model_dir = os.path.dirname(os.path.abspath(model))

    # Check if test folder and test data already exists
    regex = re.compile("test_data*")
    for f in os.listdir(model_dir):
        if regex.match(f):
            test_path = os.path.join(model_dir, f)
            for inputFiles in os.listdir(test_path):
                if inputFiles.endswith('.pb') and inputFiles.startswith("input"):
                    print("Input.pb already exists. Skipping dummy input generation. ")
                    return test_path    
                    
    test_path = os.path.join(model_dir, "test_data_set_0")
    if not os.path.exists(test_path):
        os.mkdir(test_path)
        os.chmod(test_path, 0o644)

    # Get the input model
    sess = rt.InferenceSession(model)

    #########################
    # Let's see the input name and shape.
    print("%s inputs: " % model)
    inputs = sess.get_inputs()
    for i in range(0, len(inputs)):
        print("input name: %s, shape: %s, type: %s" % (inputs[i].name, inputs[i].shape, inputs[i].type))
        # If the input has None dimensions, replace with 1
        shape_corrected = [1 if x == None else x for x in inputs[i].shape]
        if inputs[i].type == "tensor(string)":
            raise ValueError("Cannot auto generate string inputs. Please provide your own input .pb file. ")
        # Create random input and write to .pb
        create_tensor("input_%s.pb" % i, shape_corrected, inputs[i].name, test_path, TYPE_MAP.get(inputs[i].type))

    print("Randomized input .pb file generated at ", test_path)
    return test_path
