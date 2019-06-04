import argparse
import subprocess
from pathlib import Path
from shutil import copyfile
from check_model import get_extension, check_model
from create_input import generate_inputs
import coremltools
import onnxmltools

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        required=True,
        help="The path of the model to be converted.")
    parser.add_argument(
        "--output_onnx_path", 
        required=True,
        help="The desired path to store the converted .onnx file"
    )
    parser.add_argument(
        "--model_type", 
        required=True,
        help="The type of original model. \
            Available types are caffe, cntk, coreml, keras, libsvm, lightgbm, mxnet, pytorch, scikit-learn, tensorflow and xgboost"
    )
    parser.add_argument(
        "--model_inputs", 
        required=False,
        help="Optional. The model's input names. Required for tensorflow frozen models and checkpoints. "
    )
    parser.add_argument(
        "--model_outputs", 
        required=False,
        help="Optional. The model's output names. Required for tensorflow frozen models checkpoints. "
    )
    parser.add_argument(
        "--inputs_as_nchw", 
        required=False,
        help="Optional. The model's input names. Required for tensorflow frozen models and checkpoints. "
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
        "--target_opset", 
        required=False,
        default="7",
        help="Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3."
    )
    args = parser.parse_args()

    return args

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
    # Convert Caffe model to CoreML 
    coreml_model = coremltools.converters.caffe.convert(args.model)

    # Name and paht for intermediate coreml model
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
    onnx_model = onnxmltools.convert_coreml(coreml_model, target_opset=int(args.target_opset))

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def keras2onnx(args):
    import keras
    # Load your Keras model
    keras_model = keras.models.load_model(args.model)

    # Convert the Keras model into ONNX
    onnx_model = onnxmltools.convert_keras(keras_model, target_opset=int(args.target_opset))

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
    onnx_model = onnxmltools.convert.convert_libsvm(libsvm_model, target_opset=int(args.target_opset))
    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def lightgbm2onnx(args):
    import lightgbm as lgb
    from onnxmltools.convert.common.data_types import FloatTensorType
    import pickle
    if get_extension(args.model) == "pkl":
        with open(args.model, "rb") as f:
            lgb_model = pickle.load(f)
    else:
        # Load your LightGBM model
        lgb_model = lgb.Booster(model_file=args.model)

    # Convert the LightGBM model into ONNX
    onnx_model = onnxmltools.convert_lightgbm(lgb_model, 
        initial_types=[('input', FloatTensorType(shape=[1, 'None']))],
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
    # dummy_model_input = Variable(torch.randn(*args.model_input_shapes))
    # load the PyTorch model
    model = torch.load(args.model, map_location="cpu")

    # export the PyTorch model as an ONNX protobuf
    torch.onnx.export(model, dummy_model_input, args.output_onnx_path)

def sklearn2onnx(args):
    from sklearn.externals import joblib
    from skl2onnx import convert_sklearn
    # Load your sklearn model
    skl_model = joblib.load(args.model)
    # Convert the sklearn model into ONNX
    onnx_model = onnxmltools.convert_sklearn(skl_model, target_opset=int(args.target_opset))
    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

def tf2onnx(args): 
    if get_extension(args.model) == "pb":
        if not args.model_inputs and not args.model_outputs:
            raise ValueError("Please provide --model_inputs and --model_outputs to convert Tensorflow graphdef models.")
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--input", args.model, 
            "--output", args.output_onnx_path, 
            "--inputs", args.model_inputs,
            "--outputs", args.model_outputs, 
            "--opset", args.target_opset])
    elif get_extension(args.model) == "meta":
        if not args.model_inputs and not args.model_outputs:
            raise ValueError("Please provide --model_inputs and --model_outputs to convert Tensorflow checkpoint models.")
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--checkpoint", args.model, 
            "--output", args.output_onnx_path, 
            "--inputs", args.model_inputs,
            "--outputs", args.model_outputs, 
            "--opset", args.target_opset])
    else:
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--saved-model", args.model, 
            "--output", args.output_onnx_path, 
            "--opset", args.target_opset])

def xgboost2onnx(args):
    import xgboost as xgb
    from onnxmltools.convert.common.data_types import FloatTensorType
    import pickle
    if get_extension(args.model) == "pkl":
        with open(args.model, "rb") as f:
            xgb_model = pickle.load(f)
    else:
        # Load your XGBoost model
        xgb_model = xgb.Booster(model_file=args.model)
    # Convert the XGBoost model into ONNX
    onnx_model = onnxmltools.convert.convert_xgboost(xgb_model, 
        initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
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
def main():
    args = get_args()
    
    # Quick format check
    model_extension = get_extension(args.model)
    if (args.model_type == "onnx" or model_extension == "onnx"):
        print("Input model is already ONNX model. Skipping conversion.")
        if args.model != args.output_onnx_path:
            copyfile(args.model, args.output_onnx_path)
        with open('/output.txt', 'w') as f:
            f.write(args.output_onnx_path)
        return
    
    if converters.get(args.model_type) == None:
        raise ValueError('Model type {} is not currently supported. \n\
            Please select one of the following model types -\n\
                caffe, cntk, coreml, keras, libsvm, lightgbm, mxnet, pytorch, scikit-learn, tensorflow or xgboost'.format(args.model_type))
    
    suffix = suffix_format_map.get(model_extension)

    if suffix != None and suffix != args.model_type:
        raise ValueError('model with extension {} do not come from {}'.format(model_extension, args.model_type))
    converter = converters.get(args.model_type)
    converter(args)

    with open('/output.txt', 'w') as f:
        f.write(args.output_onnx_path)
    
    # Generate random inputs for the model if input files are not provided
    inputs_path = generate_inputs(args.output_onnx_path)

    # Test correctness
    check_model(args.model, args.output_onnx_path, inputs_path, args.model_type, args.model_inputs, args.model_outputs)
    
if __name__ == "__main__":
    main()