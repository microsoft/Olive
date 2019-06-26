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

# Convert model to ONNX

### For Windows
Use cmd and type as below:
```bash
python -c "import onnxpipeline ; p = onnxpipeline.Pipeline('cntk'); model=p.convert_model(model_type='cntk', model='ResNet50_ImageNet_Caffe.model')"
```
### For linux
Use Jupyter Notebook and see [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)
