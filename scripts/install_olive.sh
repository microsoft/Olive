#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eoux pipefail

PIPELINE=$1
INSTALL_DEV_MODE=$2

# Create virtual environment
if [[ "$PIPELINE" == "True" ]]; then
    echo "Creating virtual environment for pipeline"
    set +x
    python -m venv olive-venv
    source olive-venv/bin/activate
    set -x
else
    echo "Using active python environment"
fi

# Upgrade pip
echo "Upgrading pip"
python -m pip install --upgrade pip

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
