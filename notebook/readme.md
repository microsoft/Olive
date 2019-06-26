Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# Onnx Pipeline

This repository shows how to deploy and use Onnx pipeline with dockers including convert model, generate input and performance test.

# Prerequisites
### For Windows
```bash
build.bat
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
python -c "import onnxpipeline; p=onnxpipeline.Pipeline(); model=p.convert_model(model_type='[model_type]', model='[model_path]', [other_parameters]='[parameter_values]'); p.perf_test(model=model, result='[result_directory_path]', runtime='[runtime]')"
```

1. [model_type]: support caffe, cntk, keras, scikit-learn, tensorflow and pytorch.

2. [model_path]: provide the local path of the model.

3. [result_directory_path]: The directory path for results.

4. [runtime]: type 'nvidia' for enabling GPU, otherwise ''. 


Details of other parameters can be referenced here [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)

For example:
```bash
python -c "import onnxpipeline; p=onnxpipeline.Pipeline(); model=p.convert_model(model_type='pytorch', model='pytorch/saved_model.pb',model_input_shapes='(1,3,224,224)'); p.perf_test(model=model, result='result/', runtime='')"
```

Then all the result JSONs will be produced under result/


### For Linux
Use Jupyter Notebook and see [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)
