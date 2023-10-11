# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN apt-get -y update && ACCEPT_EULA=Y apt-get -y upgrade
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install onnxruntime \
            datasets \
            transformers \
            openvino \
            openvino-dev \
            onnxconverter_common \
            git+https://github.com/microsoft/Olive.git \
            --no-cache-dir

WORKDIR /olive
