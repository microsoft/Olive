#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eux

# This script creates a python environment in $QNN_SDK_ROOT/olive-pyenv or $SNPE_ROOT/olive-pyenv

# Usage: ./create_python_env.sh -v/--version <python_version> --sdk <snpe/qnn>
while [[ "$#" -gt 0 ]]; do
    key="$1"

    case $key in
        -v|--version)
            PY_VERSION="$2"
            shift # Shift past argument
            ;;
        --sdk)
            SDK="$2"
            shift # Shift past argument
            ;;
        *)
            # Handle unknown arguments
            echo "Unknown argument: $key"
            exit 1
        ;;
    esac
    shift # Shift to the next argument
done


if [ "$SDK" == "snpe" ]; then
    SDK_ROOT=$SNPE_ROOT
elif [ "$SDK" == "qnn" ]; then
    SDK_ROOT=$QNN_SDK_ROOT
else
    echo "Unknown SDK: $SDK"
    exit 1
fi

PY_ENV_NAME=olive-pyenv
FILES_DIR=$SDK_ROOT/python-env-setup
rm -rf "$FILES_DIR"
mkdir "$FILES_DIR"

# Install conda if not already installed
if ! command -v conda; then
    # check if CONDA_INSTALLER is set, if not download the latest miniconda installer
    if [ -z ${CONDA_INSTALLER+x} ]; then
        CONDA_INSTALLER="$FILES_DIR"/install_conda.sh
        curl -fsSL -o "$CONDA_INSTALLER" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi
    # Install conda
    sh "$CONDA_INSTALLER" -b -p "$FILES_DIR"/miniconda
    CONDA=$FILES_DIR/miniconda/bin/conda
else
    CONDA=conda
fi

# Create python environment
$CONDA create -y -p "$FILES_DIR"/$PY_ENV_NAME python="$PY_VERSION"

# if PIP_EXTRA_ARGS is set, then use it else use ""
PIP_EXTRA_ARGS=${PIP_EXTRA_ARGS:-""}

# Install snpe requirements
"$FILES_DIR"/$PY_ENV_NAME/bin/python -m pip install --upgrade pip "$PIP_EXTRA_ARGS"
if [ "$PY_VERSION" == "3.6" ]; then
    "$FILES_DIR"/$PY_ENV_NAME/bin/python -m pip install onnx==1.11.0 onnx-simplifier packaging tensorflow==1.15.0 pyyaml pandas==1.1.5 numpy==1.18.5 "$PIP_EXTRA_ARGS"
elif [ "$PY_VERSION" == "3.8" ]; then
    "$FILES_DIR"/$PY_ENV_NAME/bin/python -m pip install onnx onnx-simplifier packaging tensorflow==2.10.1 pyyaml pandas==1.1.5 numpy==1.23.5 "$PIP_EXTRA_ARGS"
else
    echo "Unsupported python version: $PY_VERSION, only 3.6 and 3.8 are supported"
    exit 1
fi


rm -rf "${SDK_ROOT:?}"/$PY_ENV_NAME
mv "$FILES_DIR"/$PY_ENV_NAME "$SDK_ROOT"/$PY_ENV_NAME

# Remove all unnecessary files
rm -rf "$FILES_DIR"
