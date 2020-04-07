import argparse
import sys
import os
import pickle
import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto

def convert_data_to_pb(pickle_path, output_folder="test_data_set_0", is_input=True):
    """
    Convert pickle test data file to ONNX .pb files.
    Args:
        pickle_path: The path to your pickle file. The pickle file should contain
        a dictionary with the following format: 
            {
                input_name_1: test_data_1,
                input_name_2: test_data_2,
                ...
            }
        output_folder: The folder to store .pb files. The folder should be empty 
        and its name starts with test_data_*. Default is "test_data_set_0". 
    """
    extension = pickle_path.split(".")[1]
    if extension == "pb":
        print("Test Data already in .pb format. ")
        return
    try:
        test_data_dict = pickle.load(open(pickle_path, "rb"))
    except:
        raise ValueError("Cannot load test data with pickle. ")
    # Type check for the pickle file. Expect a dictionary with input names as keys 
    # and data as values.
    if type(test_data_dict) is not dict:
        raise ValueError("Data type error. Expect a dictionary with input names as keys and data as values.")

    # Create a test_data_set folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_prefix = "input_"
    if is_input == False:
        file_prefix = "output_"
    idx = 0
    for name, data in test_data_dict.items():
        tensor = numpy_helper.from_array(data)
        tensor.name = name
        pb_file_name = file_prefix + idx + ".pb"
        pb_file_location = os.path.join(output_folder, pb_file_name)
        with open(pb_file_location, 'wb') as f:
            f.write(tensor.SerializeToString())
            print("Successfully store input {} in {}".format(name, pb_file_location))
        idx += 1
    
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data", 
        type=str,
        help="A pickle file storing a dictionary with input names and data. ")
    parser.add_argument("--output_folder", 
        required=False,
        default="test_data_set_0",
        help="An output folder to store the output .pb files. Default: test_data_set_0. ")
    parser.add_argument("--is_input", 
        required=False,
        default=True,
        help="If the pickle file specifies input data or output data. ")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    convert_data_to_pb(args.test_data, args.output_folder, args.is_input)

if __name__ == "__main__":
    main()