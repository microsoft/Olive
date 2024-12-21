# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION

RUN apt-get update && \
    apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libnvinfer-lean10 \
    python3-libnvinfer-lean \
    libnvinfer-dispatch10 \
    python3-libnvinfer-dispatch \
    tensorrt-libs \
    tensorrt-dev \
    libnvinfer-lean-dev \
    libnvinfer-dispatch-dev \
    python3-libnvinfer \
    unzip \
    docker.io
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

COPY . /olive
WORKDIR /olive
RUN pip install -e .
