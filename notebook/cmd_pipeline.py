import argparse
import onnxpipeline
from pathlib import PurePosixPath, PureWindowsPath
import os.path as osp

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        required=True,
        help="The path of the model to be converted.")
    parser.add_argument(
        "--model_type", 
        required=True,
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
        type=str,
        help="Optional. List of tuples. The input shape(s) of the model. Each dimension separated by ','. "
    )
    parser.add_argument(
        "--target_opset", 
        required=False,
        default=7,
        help="Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3."
    )

    parser.add_argument(
        "--initial_types",
        required=False,
        help="Optional. List of tuples. Specifies the initial types for onnxmltools. "
    )

    parser.add_argument(
        "--caffe_model_prototxt", 
        required=False,
        help="Optional. prototxt file for caffe models. "
    )    

    parser.add_argument(
        "--input_json", 
        required=False,
        help="Optional. Provide a JSON file with arguments."
    )

    parser.add_argument("--result",
                        help="Optional. Result folder.")

    parser.add_argument('--nvidia', action="store_true", default=False,
                        help="Optional. Add it for enabling GPU. ")

    args = parser.parse_args()

    return args

def win_path_to_linux_relative(path):
    return osp.relpath(path).replace("\\", "/")


def main():
    args = get_args()
    pipeline = onnxpipeline.Pipeline()
    model=pipeline.convert_model(model_type=args.model_type, model=win_path_to_linux_relative(args.model), model_input_shapes=args.model_input_shapes,
        model_inputs_names=args.model_inputs_names, model_outputs_names=args.model_outputs_names,
        model_params=args.model_params, target_opset=args.target_opset, input_json=args.input_json,
        initial_types=args.initial_types, caffe_model_prototxt=args.caffe_model_prototxt)
    if 'Runtimes' in pipeline.client.info() and 'nvidia' in pipeline.client.info()['Runtimes']:
        pipeline.perf_test(model=model, result=win_path_to_linux_relative(args.result), runtime=args.nvidia)
    else:
        if args.nvidia:
            print('Not support Nvidia in local machine. Need to be installed.')
        args.nvidia = False
        pipeline.perf_test(model=model, result=args.result, runtime=args.nvidia)
  
if __name__== "__main__":
    main()
