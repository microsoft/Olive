## ONNX Converter Image

This image is used to convert models from major model frameworks to ONNX, generate input files if not provided, and then test the converted models' correctness. 
Supported frameworks are - 
   - CNTK
   - CoreML
   - Keras
   - scikit-learn
   - Tensorflow
   - PyTorch


## How to Run 

### Azure Registry

A pre-built version of the image is available at Azure Registry. Once you have docker installed, you can easily pull and run the image on Linux as well as on Windows. 

With the correct credentials, you can pull the image directly using 
```
docker pull ziylregistry.azurecr.io/onnx-converter
```

Upon success, run Docker onnx-converter image by
```
docker run ziylregistry.azurecr.io/onnx-converter --model <model_path> --output_onnx_path <output_path_to_.onnx> --model_type <input_model_framework_name> [optional args]
```

### Run With Docker
First build image with docker. Under this directory, run
```
docker build -t onnx-converter .
```
Then, you can run the docker image with customized parameters. 

```
docker run onnx-converter --model <model_path> --output_onnx_path <output_path_to_.onnx> --model_type <input_model_framework_name> [optional args]
```

### Run With Python
Alternatively, you can run without docker by 
```
python onnx_converter.py --model <model_path> --output_onnx_path <output_path_to_.onnx> --model_type <input_model_framework_name> [optional args]
```
For detailed description of all available parameters, refer to the following. 

## Run parameters

`--input_json`: A JSON file that contains all neccessary run specs. For example:
```
{
    "model": "/mnist/",
    "model_type": "tensorflow",
    "output_onnx_path": "mnist.onnx"
}
```

`--model`: Required or specified in input json. The path of the model that needs to be converted.

`--output_onnx_path`: Required or specified in input json. The path to store the converted ONNX model. Should end with ".onnx". e.g. "/newdir/output.onnx". A cleaned directory is recommended. 

`--model_type`: Required or specified in input json. The name of original model framework. Available types are cntk, coreml, keras, scikit-learn, tensorflow and pytorch.

`--target_opset`: Optional. The opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3. Latest Opset is recommanded. Refer to [ONNX Opset](https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix) for the latest Opset. 

`--model_inputs_names`: Required for tensorflow frozen models and checkpoints. The model's input names. 

`--model_outputs_names`: Required for tensorflow frozen models and checkpoints. The model's output names. 

`--model_input_shapes`: Required for PyTorch models. List of tuples. The input shape(s) of the model. Each dimension separated by ','.

`--initial_types`: Required for scikit-learn. List of tuples.

`--caffe_model_prototxt`: Required for Caffe models. The path of the .prototxt file for caffe model.
 

