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

### For linux
```bash
sh build.sh
pip install docker
```
Install Jupyter Notebook and use the Notebook [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)

# Convert model to ONNX
This image is used to convert model from major model frameworks to onnx. Supported frameworks are - caffe, cntk, coreml, keras, libsvm, mxnet, scikit-learn, tensorflow and pytorch.
You can run the docker image with customized parameters.

### For Windows
Use cmd and type as below:
```bash
python -c "import onnxpipeline ; p=onnxpipeline.Pipeline(); model=p.convert_model(model_type='[model_type]', model='[model_path]', [other_parameters]='[parameter_values]')"
```

[model_type]: support caffe, cntk, keras, scikit-learn, tensorflow and pytorch.
[model_path]: provide the local path of the model.

Details of the parameters can be referenced here [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)

For example:
```bash
python -c "import onnxpipeline ; p=onnxpipeline.Pipeline(); model = p.convert_model(model_type='pytorch', model='pytorch/saved_model.pb',model_input_shapes='(1,3,224,224)')"
```

Once convert model succuessfully, it would print a onnx model path for performance test in the next step.
For instance:
```bash
/mnt/model/test/model.onnx
```


### For linux
Use Jupyter Notebook and see [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)

# Performance test tool

### For Windows

Given the onnx model path which produced by the previous step, it can output the JSONs for performance test.

```bash
python -c "import onnxpipeline ; p=onnxpipeline.Pipeline(); p.perf_test(model='[onnx_model_path]', result='[result_directory_path]', runtime='')"
```

[onnx_model_path]: The path of the onnx model that wants to be performed. (produced by the previous step)
[result_directory_path]: The directory path for results.
[runtime]: type 'nvidia' for enabling GPU, otherwise ''. 


For example:

```bash
python -c "import onnxpipeline ; p=onnxpipeline.Pipeline(); p.perf_test(model='/mnt/model/test/model.onnx', result='result', runtime='')"
```

### For linux
Use Jupyter Notebook and see [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)
