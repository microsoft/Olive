# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN pip install azure-ai-ml \
            azure-identity \
            azureml-defaults \
            azureml-dataprep \
            onnxruntime \
            openvino \
            openvino-dev[tensorflow,onnx] \
            tensorflow \
            olive-ai==0.1.0

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /olive
