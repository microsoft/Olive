Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# OLive Command Line Tool

This repository shows how to deploy and use OLive on Windows by running commands.

## Prerequisites
Install [Docker](https://docs.docker.com/install/).

## How to use
### For Windows
```bash
..\utils\build.sh
pip install docker
```

Use cmd and type as below:
```bash
python cmd_pipeline.py --model [model_path] --model_type [model_type] --result [result_directory_path] [--other_parameters] [other parameters' value]
```
**IMPORTANT:** Any path in the parameter must be under the current directory (/cmd-tool).


1. --model_type: Required. support caffe, cntk, keras, scikit-learn, tensorflow and pytorch.

2. --model_path: Required. provide the local path of the model. 

3. --result: Optional. The directory path for results.

4. --target_opset: Latest Opset is recommanded. Refer to [ONNX Opset](https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix) for the latest Opset. 

5. --nvidia: Optional. Use this boolean flag to enable GPU if you have one.

6. --model_inputs_names: Required for tensorflow frozen models and checkpoints. The model's input names.  

7. --model_outputs_names: Required for tensorflow frozen models and checkpoints. The model's output names. 

8. --model_input_shapes: Required for Pytorch models. List of tuples. The input shape(s) of the model. Each dimension separated by ','.  

9. --initial_types: Required for scikit-learn. List of tuples.

10. --caffe_model_prototxt: Required for Caffe models. Prototxt files for caffe models. 

11. --input_json: A JSON file that contains all neccessary run specs. For example:
```
{
    "model": "/mnist/",
    "model_type": "tensorflow",
    "output_onnx_path": "mnist.onnx"
}
```

Details of other parameters can be referenced "Convert model to ONNX section" in [onnx-pipeline.ipynb](../notebook/onnx-pipeline.ipynb)

For example:
```bash
python cmd_pipeline.py --model pytorch/saved_model.pb --model_type pytorch --model_input_shapes (3,3,224,224) --result result/
```

Then all the result JSONs will be produced under /result.
Also print the logs for the process in the terminal. Check if there is any error.
