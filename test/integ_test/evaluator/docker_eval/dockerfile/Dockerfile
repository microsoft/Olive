# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN apt-get -y update && apt-get -y upgrade
RUN pip install onnxruntime \
            datasets \
            torchvision \
            transformers \
            openvino \
            openvino-dev \
            onnxconverter_common \
            git+https://github.com/microsoft/Olive.git \
            --no-cache-dir

WORKDIR /olive
