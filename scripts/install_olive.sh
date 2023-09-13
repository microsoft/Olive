#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eoux pipefail

INSTALL_DEV_MODE=$1
DEVICE=$2

# Upgrade pip
echo "Upgrading pip"
python -m pip install --upgrade pip

# install torch cpu or gpu
if [[ "$DEVICE" == "cpu" ]]; then
    echo "Installing torch cpu"
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing torch gpu"
    python -m pip install torch torchvision torchaudio
fi

# Install olive
if [[ "$INSTALL_DEV_MODE" == "True" ]]; then
    echo "Installing olive in dev mode"
    set +u
    python -m pip install -e .$INSTALL_EXTRAS
    set -u
else
    echo "Installing olive"
    set +u
    python -m pip install .$INSTALL_EXTRAS
    set -u
fi
