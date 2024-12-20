# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

RUN apt-get -y update && ACCEPT_EULA=Y apt-get -y upgrade
RUN pip install azure-ai-ml \
            azure-identity \
            azureml-dataprep \
            onnxruntime \
            openvino \
            openvino-dev[tensorflow,onnx] \
            tensorflow \
            onnxconverter_common \
            olive-ai

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /olive
