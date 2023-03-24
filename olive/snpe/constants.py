# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum


class SNPEDevice(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    DSP = "dsp"
    AIP = "aip"


class InputType(str, Enum):
    DEFAULT = "default"
    IMAGE = "image"
    OPAQUE = "opaque"


class InputLayout(str, Enum):
    NCDHW = "NCDHW"
    NDHWC = "NDHWC"
    NCHW = "NCHW"
    NHWC = "NHWC"
    NFC = "NFC"
    NCF = "NCF"
    NTF = "NTF"
    TNF = "TNF"
    NF = "NF"
    NC = "NC"
    F = "F"
    NONTRIVIAL = "NONTRIVIAL"


class PerfProfile(str, Enum):
    SYSTEM_SETTINGS = "system_settings"
    POWER_SAVER = "power_saver"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"
    BURST = "burst"


class ProfilingLevel(str, Enum):
    OFF = "off"
    BASIC = "basic"
    MODERATE = "moderate"
    DETAILED = "detailed"
