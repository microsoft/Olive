# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

RUN apt-get -y update && ACCEPT_EULA=Y apt-get -y upgrade
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install onnxruntime \
            datasets \
            git+https://github.com/microsoft/Olive.git \
            onnxconverter_common \
            openvino \
            openvino-dev \
            pandas \
            plotly \
            psutil \
            transformers \
            --no-cache-dir

WORKDIR /olive
