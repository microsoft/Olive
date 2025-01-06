# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION
ARG TENSORRT_VERSION=10.0.1.6-1+cuda12.4

RUN apt-get update && \
    apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libnvinfer10=${TENSORRT_VERSION} \
    libnvinfer-dev=${TENSORRT_VERSION} \
    libnvinfer-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-vc-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
    libnvonnxparsers-dev=${TENSORRT_VERSION} \
    libnvinfer-plugin10=${TENSORRT_VERSION} \
    libnvinfer-vc-plugin10=${TENSORRT_VERSION} \
    libnvonnxparsers10=${TENSORRT_VERSION} \
    libnvinfer-headers-dev=${TENSORRT_VERSION} \
    libnvinfer-lean10=${TENSORRT_VERSION} \
    python3-libnvinfer-lean=${TENSORRT_VERSION} \
    libnvinfer-dispatch10=${TENSORRT_VERSION} \
    python3-libnvinfer-dispatch=${TENSORRT_VERSION} \
    tensorrt-libs=${TENSORRT_VERSION} \
    tensorrt-dev=${TENSORRT_VERSION} \
    libnvinfer-lean-dev=${TENSORRT_VERSION} \
    libnvinfer-dispatch-dev=${TENSORRT_VERSION} \
    python3-libnvinfer=${TENSORRT_VERSION} \
    unzip \
    docker.io
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

COPY . /olive
WORKDIR /olive
RUN pip install -e .
