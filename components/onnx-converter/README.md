## Onnx-converter Image

This image is used to convert models from major model frameworks to onnx, generate input files if not provided, and then test the converted models' correctness. 
Supported frameworks are - 
   caffe, cntk, coreml, keras, libsvm, mxnet, scikit-learn, tensorflow and pytorch.


## How to Run 
First build image with docker
```
docker build -t onnx-converter .
```
Upon success, you should see you can run the docker image with customized parameters. 

```
docker run onnx-converter --model [YOUR MODEL FILE] --output_onnx_path [OUTPUT PATH TO STORE CONVERTED .ONNX] --model_type [TYPE OF THE INPUT MODEL]
```

For detailed description of all available parameters, refer to the following. 
## Run parameters

**model**: string
   
   Required. The path of the model that needs to be converted.

**output_onnx_path**: string
   
   Required. The path to store the converted onnx model. Should end with ".onnx". e.g. output.onnx

**model_type**: string
   
   Required. The name of original model framework. 
   
   Available types are caffe, cntk, coreml, keras, libsvm, mxnet, scikit-learn, tensorflow and pytorch.

**model_inputs**: string
   
   Optional. The model's input names. Required for tensorflow frozen models and checkpoints.

**model_outputs**: string
   
   Optional. The model's output names. Required for tensorflow frozen models checkpoints.

**model_params**: string

Optional. The params of the model if needed.

**model_input_shapes**: list of tuple
   
   Optional. List of tuples. The input shape(s) of the model. Each dimension separated by ','.

**target_opset**: int

Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3. Defaults to 7. 