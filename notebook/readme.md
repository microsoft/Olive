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

1. [model_type]: Required. support caffe, cntk, keras, scikit-learn, tensorflow and pytorch.

2. [model_path]: Required. provide the local path of the model.

3. [result_directory_path]: Optional. The directory path for results.

4. [runtime]: Optional. type 'nvidia' for enabling GPU, otherwise ''. 

Details of other parameters can be referenced here [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)

For example:
```bash
python cmd_pipeline.py --model pytorch/saved_model.pb --model_type pytorch --model_input_shapes '(3,3,224,224)' --runtime ''
```

Then all the result JSONs will be produced under result/


### For Linux
Use Jupyter Notebook and see [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)
