# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN pip install onnxruntime \
            datasets \
            torchvision \
            transformers \
            openvino \
            openvino-dev \
            onnxconverter_common \
            olive-ai==0.1.0

WORKDIR /olive