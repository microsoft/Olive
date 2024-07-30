# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import transformers

MIN_TRANSFORMERS_VERSION = "4.30.2"

# check transformers version
assert (
    transformers.__version__ >= MIN_TRANSFORMERS_VERSION
), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."
