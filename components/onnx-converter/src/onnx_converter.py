import argparse
import subprocess
from pathlib import Path
from shutil import copyfile
from check_model import get_extension, check_model
from create_input import generate_inputs
import coremltools
import onnxmltools
from onnxmltools.convert.common.data_types import *
import json
import os
import pprint

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", 
        required=False,
        help="A JSON file specifying the run specs. ")
    parser.add_argument(
        "--model", 
        required=False,
        help="The path of the model to be converted.")
    parser.add_argument(
        "--output_onnx_path", 
        required=False,
        help="The desired path to store the converted .onnx file"
    )
    parser.add_argument(
        "--model_type", 
        required=False,
        help="The type of original model. \
            Available types are caffe, cntk, coreml, keras, libsvm, lightgbm, mxnet, pytorch, scikit-learn, tensorflow and xgboost"
    )
    parser.add_argument(
        "--model_inputs_names", 
        required=False,
        help="Optional. The model's input names. Required for tensorflow frozen models and checkpoints. "
    )
    parser.add_argument(
        "--model_outputs_names", 
        required=False,
        help="Optional. The model's output names. Required for tensorflow frozen models checkpoints. "
    )
    parser.add_argument(
        "--model_params", 
        required=False,
        help="Optional. The params of the model. "
    )
    parser.add_argument(
        "--model_input_shapes", 
        required=False,
        type=shape_type,
        help="Optional. List of tuples. The input shape(s) of the model. Each dimension separated by ','. "
    )
    parser.add_argument(
        "--initial_types",
        required=False,
        help="Optional. List of tuples. Specifies the initial types for onnxmltools. "
    )
    parser.add_argument(
        "--target_opset", 
        required=False,
        default="7",
        help="Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3."
    )
    parser.add_argument(
        "--caffe_model_prototxt", 
        required=False,
        help="Optional. prototxt file for caffe models. "
    )
    args = parser.parse_args()
    return args

class ConverterParamsFromJson():
    def __init__(self):
        with open(get_args().input_json) as f:
            loaded_json = json.load(f)

        # Check the required inputs
        if loaded_json.get("model") == None:
            raise ValueError("Please specified \"model\" in the input json. ")
        if loaded_json.get("model_type") == None:
            raise ValueError("Please specified \"model_type\" in the input json. ")
        if loaded_json.get("output_onnx_path") == None:
            raise ValueError("Please specified \"output_onnx_path\" in the input json. ")
        self.model = loaded_json["model"]
        self.model_type = loaded_json["model_type"]
        self.output_onnx_path = loaded_json["output_onnx_path"]
        self.model_inputs_names = loaded_json["model_inputs_names"] if loaded_json.get("model_inputs_names") else None
        self.model_outputs_names = loaded_json["model_outputs_names"] if loaded_json.get("model_outputs_names") else None
        self.model_params = loaded_json["model_params"] if loaded_json.get("model_params") else None
        self.model_input_shapes = shape_type(loaded_json["model_input_shapes"]) if loaded_json.get("model_input_shapes") else None
        self.initial_types = eval(loaded_json["initial_types"]) if loaded_json.get("initial_types") else None
        self.target_opset = loaded_json["target_opset"] if loaded_json.get("target_opset") else "7"
        self.caffe_model_prototxt = loaded_json["caffe_model_prototxt"] if loaded_json.get("caffe_model_prototxt") else None

def shape_type(s):
    import ast
    if s == None or len(s) == 0:
        return
    try:
        shapes_list = list(ast.literal_eval(s))
        if isinstance(shapes_list[0], tuple) == False:
            # Nest the shapes list to make it a list of tuples
            return [tuple(shapes_list)]
        return shapes_list
    except:
        raise argparse.ArgumentTypeError("Model input shapes must be a list of tuple. Each dimension separated by ','. ")

def caffe2onnx(args):
    caffe_model = args.model
    # Convert Caffe model to CoreML 
    if args.caffe_model_prototxt != None and len(args.caffe_model_prototxt)> 0:
        caffe_model = (args.model, args.caffe_model_prototxt)
    coreml_model = coremltools.converters.caffe.convert(caffe_model)

    # Name and path for intermediate coreml model
    output_coreml_model = 'model.mlmodel'

    # Save CoreML model
    coreml_model.save(output_coreml_model)

    # Load a Core ML model
    coreml_model = coremltools.utils.load_spec(output_coreml_model)

    # Convert the Core ML model into ONNX
    onnx_model = onnxmltools.convert_coreml(coreml_model, target_opset=int(args.target_opset))

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def cntk2onnx(args):
    import cntk
    # Load your CNTK model
    cntk_model = cntk.Function.load(args.model, device=cntk.device.cpu())

    # Convert the CNTK model into ONNX
    cntk_model.save(args.output_onnx_path, format=cntk.ModelFormat.ONNX)

def coreml2onnx(args):
    # Load your CoreML model
    coreml_model = coremltools.utils.load_spec(args.model)

    # Convert the CoreML model into ONNX
    onnx_model = onnxmltools.convert_coreml(coreml_model, 
        initial_types = args.initial_types,
        target_opset=int(args.target_opset))

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def keras2onnx(args):
    import keras
    # Load your Keras model
    keras_model = keras.models.load_model(args.model)

    # Convert the Keras model into ONNX
    onnx_model = onnxmltools.convert_keras(keras_model, 
        initial_types = args.initial_types,
        target_opset=int(args.target_opset))

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def libsvm2onnx(args):
    from svmutil import svm_load_model
    import pickle
    if get_extension(args.model) == "pkl":
        with open(args.model, "rb") as f:
            model = pickle.loads(f)
    else:
        model = args.model
    # Load your LibSVM model
    libsvm_model = svm_load_model(model)
    # Convert the LibSVM model into ONNX
    onnx_model = onnxmltools.convert.convert_libsvm(libsvm_model, 
        initial_types = args.initial_types, 
        target_opset=int(args.target_opset))
    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def lightgbm2onnx(args):
    import lightgbm as lgb
    from onnxmltools.convert.common.data_types import FloatTensorType
    # Check for required arguments
    if not args.initial_types:
        raise ValueError("Please provide --initial_types to convert scikit learn models.")
    import pickle
    if get_extension(args.model) == "pkl":
        with open(args.model, "rb") as f:
            lgb_model = pickle.load(f)
    else:
        # Load your LightGBM model
        lgb_model = lgb.Booster(model_file=args.model)

    # Convert the LightGBM model into ONNX
    onnx_model = onnxmltools.convert_lightgbm(lgb_model, 
        initial_types = args.initial_types,
        target_opset=int(args.target_opset))

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def mxnet2onnx(args):
    import mxnet as mx
    import numpy as np
    from mxnet.contrib import onnx as onnx_mxnet
    # MXNet model format check 
    if get_extension(args.model) != 'json':
        raise ValueError("Please provide a valid .json model file for MXNet model conversion. ")
    if args.model_params == None:
        raise ValueError("Please provide a valid model params file for MXNet model conversion. ")
    if args.model_input_shapes == None:
        raise ValueError("Please provide a list of valid model input shapes for MXNet model conversion. ")
    # Convert your MXNet model into ONNX and save as protobuf
    onnx_mxnet.export_model(args.model, args.model_params, args.model_input_shapes, np.float32, args.output_onnx_path)

def pytorch2onnx(args):
    # PyTorch exports to ONNX without the need for an external converter
    import torch
    from torch.autograd import Variable
    import torch.onnx
    import torchvision
    # Create input with the correct dimensions of the input of your model
    if args.model_input_shapes == None:
        raise ValueError("Please provide --model_input_shapes to convert Pytorch models.")
    dummy_model_input = []
    if len(args.model_input_shapes) == 1:
        dummy_model_input = Variable(torch.randn(*args.model_input_shapes))
    else:
        for shape in args.model_input_shapes:
            dummy_model_input.append(Variable(torch.randn(*shape)))

    # load the PyTorch model
    model = torch.load(args.model, map_location="cpu")

    # export the PyTorch model as an ONNX protobuf
    torch.onnx.export(model, dummy_model_input, args.output_onnx_path)

def sklearn2onnx(args):
    from sklearn.externals import joblib
    from skl2onnx import convert_sklearn
    # Check for required arguments
    if not args.initial_types:
        raise ValueError("Please provide --initial_types to convert scikit learn models.")
    # Load your sklearn model
    skl_model = joblib.load(args.model)
    
    # Convert the sklearn model into ONNX
    onnx_model = onnxmltools.convert_sklearn(skl_model, 
        initial_types = args.initial_types,
        target_opset=int(args.target_opset))
    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def tf2onnx(args): 
    if get_extension(args.model) == "pb":
        if not args.model_inputs_names and not args.model_outputs_names:
            raise ValueError("Please provide --model_inputs_names and --model_outputs_names to convert Tensorflow graphdef models.")
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--input", args.model, 
            "--output", args.output_onnx_path, 
            "--inputs", args.model_inputs_names,
            "--outputs", args.model_outputs_names, 
            "--opset", args.target_opset, 
            "--fold_const"])
    elif get_extension(args.model) == "meta":
        if not args.model_inputs_names and not args.model_outputs_names:
            raise ValueError("Please provide --model_inputs_names and --model_outputs_names to convert Tensorflow checkpoint models.")
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--checkpoint", args.model, 
            "--output", args.output_onnx_path, 
            "--inputs", args.model_inputs_names,
            "--outputs", args.model_outputs_names, 
            "--opset", args.target_opset, 
            "--fold_const"])
    else:
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--saved-model", args.model, 
            "--output", args.output_onnx_path, 
            "--opset", args.target_opset,
            "--fold_const"])

def xgboost2onnx(args):
    import xgboost as xgb
    # Check for required arguments
    if not args.initial_types:
        raise ValueError("Please provide --initial_types to convert scikit learn models.")
    import pickle
    if get_extension(args.model) == "pkl":
        with open(args.model, "rb") as f:
            xgb_model = pickle.load(f)
    else:
        # Load your XGBoost model
        xgb_model = xgb.Booster(model_file=args.model)
    # Convert the XGBoost model into ONNX
    onnx_model = onnxmltools.convert.convert_xgboost(xgb_model, 
        initial_types = args.initial_types, 
        target_opset=int(args.target_opset))
    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

suffix_format_map = {
    "h5": "keras",
    "keras": "keras",
    "mlmodel": "coreml",
}

converters = {
    "caffe": caffe2onnx,
    "cntk": cntk2onnx,
    "coreml": coreml2onnx,
    "keras": keras2onnx,
    "libsvm": libsvm2onnx,
    "lightgbm": lightgbm2onnx,
    "mxnet": mxnet2onnx,    
    "scikit-learn": sklearn2onnx,
    "pytorch": pytorch2onnx,
    "tensorflow": tf2onnx,
    "xgboost": xgboost2onnx
}

output_template = {
    "output_onnx_path": "", # The output path where the converted .onnx file is stored. 
    "conversion_status": "", # SUCCEED, FAILED
    "correctness_verified": "", # SUCCEED, NOT SUPPORTED, FAILED
    "input_folder": "", 
    "error_message": ""
}

def convert_models(args):
    # Quick format check
    model_extension = get_extension(args.model)
    if (args.model_type == "onnx" or model_extension == "onnx"):
        print("Input model is already ONNX model. Skipping conversion.")
        if args.model != args.output_onnx_path:
            copyfile(args.model, args.output_onnx_path)
        return
    
    if converters.get(args.model_type) == None:
        raise ValueError('Model type {} is not currently supported. \n\
            Please select one of the following model types -\n\
                caffe, cntk, coreml, keras, libsvm, lightgbm, mxnet, pytorch, scikit-learn, tensorflow or xgboost'.format(args.model_type))
    
    suffix = suffix_format_map.get(model_extension)

    if suffix != None and suffix != args.model_type:
        raise ValueError('model with extension {} do not come from {}'.format(model_extension, args.model_type))

    # Find the corresponding converter for current model
    converter = converters.get(args.model_type)
    # Run converter
    converter(args)

def main():        
    args = get_args()
    if args.input_json != None and len(args.input_json) > 0:
        args = ConverterParamsFromJson()
    else:
        if not args.model or len(args.model) == 0:
            raise ValueError("Please specify the required argument \"model\" either in a json file or by --model")
        if not args.model_type or len(args.model_type) == 0:
            raise ValueError("Please specify the required argument \"model_type\" either in a json file or by --model_type")
        if not args.output_onnx_path or len(args.output_onnx_path) == 0:
            raise ValueError("Please specify the required argument \"output_onnx_path\" either in a json file or by --ouptut_onnx_path")
        if args.initial_types and len(args.initial_types) > 0:
            args.initial_types = eval(args.initial_types)
    # Create a test folder path
    output_dir = os.path.dirname(os.path.abspath(args.output_onnx_path))
    output_json_path = os.path.join(output_dir, "output.json")
    print("\n-------------\nModel Conversion\n")
    try:
        convert_models(args)
    except Exception as e:
        print("Conversion error occurred. Abort. ")
        output_template["conversion_status"] = "FAILED"
        output_template["correctness_verified"] = "FAILED"
        output_template["error_message"] = e
        print("\n-------------\nMODEL CONVERSION SUMMARY (.json file generated at %s )\n" % output_json_path)
        pprint.pprint(output_template)
        with open(output_json_path, "w") as f:
            json.dump(output_template, f, indent=4)
        raise e

    output_template["conversion_status"] = "SUCCESS"
    output_template["output_onnx_path"] = args.output_onnx_path

    # Dump output path to output.txt for kubeflow pipeline use
    with open('/output.txt', 'w') as f:
        f.write(args.output_onnx_path)

    print("\n-------------\nMODEL INPUT GENERATION(if needed)\n")
    # Generate random inputs for the model if input files are not provided
    inputs_path = generate_inputs(args.output_onnx_path)
    output_template["input_folder"] = inputs_path

    print("\n-------------\nMODEL CORRECTNESS VERIFICATION\n")
    # Test correctness
    verify_status = check_model(args.model, args.output_onnx_path, inputs_path, args.model_type, args.model_inputs_names, args.model_outputs_names)
    output_template["correctness_verified"] = verify_status
    print("\n-------------\nMODEL CONVERSION SUMMARY (.json file generated at %s )\n" % output_json_path)
    pprint.pprint(output_template)
    with open(output_json_path, "w") as f:
        json.dump(output_template, f, indent=4)

if __name__ == "__main__":
    main()