#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Installing setuptools to build Olive from source
uv pip install setuptools

# Requires installation of uv
uv pip install -r ../requirements.txt

# Require installation of Olive dependencies
uv pip install -r ../../../requirements.txt

# Disable CUDA extension build
export BUILD_CUDA_EXT=0

# Install AutoGPTQ from source
uv pip install --no-build-isolation git+https://github.com/PanQiWei/AutoGPTQ.git

# Install GptqModel from source
# Note: Commit hash corresponds to commit which fixes Gemma 3 memory leak issue. See README.md for additional details.
uv pip install --no-build-isolation git+https://github.com/ModelCloud/GPTQModel.git@558449bed3ef2653c36041650d30da6bbbca440d

# Install onnxruntime-qnn without installing onnxruntime
uv pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
uv pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
