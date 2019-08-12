import argparse
import os
import numpy as np
import onnxruntime as rt
import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        required=True,
        help="The path of the onnx model.")
    args = parser.parse_args()

    return args

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

def main():
    args = get_args()

    # Get the input model
    sess = rt.InferenceSession(args.model)

    # Create a test folder path
    test_path = os.path.join(os.path.dirname(args.model), "test_data_set_0")

    # Check if test folder and test data already exists
    if os.path.exists(test_path):
        for f in os.listdir(test_path):
            if f.endswith('.pb'):
                print("Input.pb already exists. Skipping dummy input generation. ")
                with open('/output.txt', 'w') as output:
                    output.write(test_path)
                return
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    #########################
    # Let's see the input name and shape.
    print("%s inputs: " % args.model)
    inputs = sess.get_inputs()
    for i in range(0, len(inputs)):
        print("input name: %s, shape: %s, type: %s" % (inputs[i].name, inputs[i].shape, inputs[i].type))
        # If the input has None dimensions, replace with 1
        shape_corrected = [1 if x == None else x for x in inputs[i].shape]
        if inputs[i].type == "tensor(string)":
            raise ValueError("Cannot auto generate string inputs. Please provide your own input .pb file. ")
        # Create random input and write to .pb
        create_tensor("input_%s.pb" % i, shape_corrected, inputs[i].name, test_path, TYPE_MAP.get(inputs[i].type))
    
    with open('/output.txt', 'w') as f:
        f.write(test_path)
    print("Randomized input .pb file generated at ", test_path)


if __name__ == "__main__":
    main()