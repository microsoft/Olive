# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION

RUN apt-get update && \
    apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    unzip \
    docker.io
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

COPY . /olive
WORKDIR /olive
RUN python -m venv olive-venv
RUN . olive-venv/bin/activate && \
    pip install --upgrade setuptools && \
    pip install -e .
