import argparse
import onnxpipeline

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

    parser.add_argument("--result",
                        help="Optional. Result folder.")

    parser.add_argument("--runtime",
                        help="Optional. Type 'nvidia' for enabling GPU, otherwise ''. ")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    pipeline = onnxpipeline.Pipeline()
    model=pipeline.convert_model(model_type=args.model_type, model=args.model, model_input_shapes=args.model_input_shapes,
        model_inputs_names=args.model_inputs_names, model_outputs_names=args.model_outputs_names,
        model_params=args.model_params, target_opset=args.target_opset)
    pipeline.perf_test(model=model, result=args.result, runtime=args.runtime)
  
if __name__== "__main__":
    main()
