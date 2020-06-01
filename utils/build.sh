# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pull onnx-converter and perf-tuning docker images from mcr
docker pull mcr.microsoft.com/onnxruntime/onnx-converter
docker pull mcr.microsoft.com/onnxruntime/perf-tuning

# install python dependency modules
python -m pip install docker pandas onnx
