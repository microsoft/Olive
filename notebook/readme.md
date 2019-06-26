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

### For Windows
Use cmd and type as below:
```bash
python -c "import onnxpipeline ; p=onnxpipeline.Pipeline(); model=p.convert_model(model_type='[model_type]', model='[model_path]', [other_parameters]='[parameter_values]')"
```

The description of parameters can be referenced here [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)

For example:
```bash
python -c "import onnxpipeline ; p=onnxpipeline.Pipeline(); p.convert_model(model_type='pytorch', model='saved_model.pb',model_input_shapes='(1,3,224,224)')"
```
### For linux
Use Jupyter Notebook and see [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)
