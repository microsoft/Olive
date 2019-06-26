Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# Onnx Pipeline

This repository shows how to deploy and use Onnx pipeline with dockers including convert model, generate input and performance test.

# Prerequisites
### For Windows
```bash
build.sh
pip install docker
```

### For Linux
```bash
sh build.sh
pip install docker
```
Install Jupyter Notebook and use the Notebook [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)

# Convert model to ONNX and Performance test tool
This command is used to convert model from major model frameworks to onnx and then performance test that onnx model.

Supported frameworks are - caffe, cntk, coreml, keras, libsvm, mxnet, scikit-learn, tensorflow and pytorch.

### For Windows
Use cmd and type as below:
```bash
python cmd_pipeline.py --model [model_path] --model_type [model_type] --result [result_directory_path] --runtime [runtime] [--other_parameters] [other parameters' value]
```

1. --model_type: Required. support caffe, cntk, keras, scikit-learn, tensorflow and pytorch.

2. --model_path: Required. provide the local path of the model.

3. --result: Optional. The directory path for results.

4. --runtime: Optional. type 'nvidia' for enabling GPU, otherwise ''. 

5. --model_inputs_names: Optional. The model's input names. Required for tensorflow frozen models and checkpoints. 

6. --model_outputs_names: Optional. The model's output names. Required for tensorflow frozen models checkpoints.

7. --model_params: Optional. The params of the model.

8. --model_input_shapes: Optional. List of tuples. The input shape(s) of the model. Each dimension separated by ','. 

9. --target_opset: Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.

Details of other parameters can be referenced here [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)

For example:
```bash
python cmd_pipeline.py --model pytorch/saved_model.pb --model_type pytorch --model_input_shapes '(3,3,224,224)' --runtime ''
```

Then all the result JSONs will be produced under result/


### For Linux
Use Jupyter Notebook and see [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)
