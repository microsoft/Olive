Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# OLive Notebook Demo

This repository provides quick demo and visualization of how to deploy and use OLive in Jupyter notebook.

## Prerequisites
Install [Docker](https://docs.docker.com/install/).

### Windows
```bash
build.sh
pip install docker
pip install jupyter
```

### Linux
```bash
sh build.sh
pip install docker
pip install jupyter
```

## Start Notebook
Type the below in command line, and then choose [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb) to use the notebook.

```
jupyter notebook
```

## Convert model to ONNX and Performance test tool
This command is used to convert model from major model frameworks to onnx and then performance test that onnx model.

**IMPORTANT:** Any path in the notebook must be under the current directory (/cmd-tool).

Supported frameworks are - tensorflow, pytorch, cntk, coreml, keras and scikit-learn.

Use Jupyter Notebook and see [onnx-pipeline.ipynb](https://github.com/liuziyue/onnx-pipeline/blob/master/notebook/onnx-pipeline.ipynb)
